"""
Pretraining of the features: episodes of a pretraining policy (random, or a policy trained on other features, see
hag.rl.experiment.pretraining_episodes), bounds of the MinMax scaling of the observations and of the filter bank
features, and reservoirs. The reservoirs share the same input matrix and bias (one block of units per input feature, as the
default input matrix of reservoirpy's HAGReservoir) and only differ by W: random, rescaled to a spectral radius, for
the ESN; learned by HAG on the pretraining episodes; zero for the random projection "proj" (control of HAG).
"""
import copy
import math

import numpy as np
from reservoirpy.mat_gen import block_input, random_sparse, uniform
from reservoirpy.utils.random import rand_generator

from hag.models.jax_hag_reservoir import HAGReservoir
from hag.rl.envs import hidden_state, make_env
from hag.rl.features import FeaturePipeline, make_filter_bank

FILTER_BANK_PARAMS = ("decomposition", "n_filters", "slowest")  # parameters of make_filter_bank
INPUT_PARAMS = ("input_scaling", "bias_scaling", "bias_dist", "lr")  # parameters of the inputs of the reservoirs
# parameters of the plasticity of HAG (arguments of HAGReservoir)
HAG_PARAMS = ("homeostasis", "target", "spread", "weight_increment", "min_window", "max_window", "use_full_instance",
              "max_partners", "intrinsic_saturation", "intrinsic_coef")
HYBRIDS = ("hag_on_esn", "hag_from_esn", "esn_hag")  # reservoirs combining an ESN and HAG
MIN_NEURONS_PER_INPUT = 4  # minimal size of the block of units of an input feature (many inputs: more units)


class FeaturePolicy:
    """Policy of a stable-baselines3 model trained on the features of a pipeline, acting on the raw observations of one
    environment (stochastic actions by default, to keep exploring)."""

    def __init__(self, model, pipeline: FeaturePipeline, deterministic: bool = False):
        self.model = model
        self.pipeline = copy.deepcopy(pipeline)
        self.pipeline.set_n_envs(1)
        self.deterministic = deterministic

    def reset(self):
        """Start of a new episode."""
        self.pipeline.reset()

    def __call__(self, observation: np.ndarray):
        features = self.pipeline.step(np.asarray(observation)[None, :])
        return self.model.predict(features, deterministic=self.deterministic)[0][0]


def collect_episodes(name: str, n_steps: int, seed: int = 0, policy=None, hidden: bool = False):
    """Observations (T, D) of the episodes of a policy (e.g. FeaturePolicy; random if None) on the benchmark name (see
    hag.rl.envs.BENCHMARKS), until n_steps steps. hidden: also return the hidden states (T, H) of the episodes (see
    hag.rl.envs.hidden_state), returns (episodes, hidden states)."""
    env = make_env(name)
    env.action_space.seed(seed)
    episodes, hiddens, steps = [], [], 0
    while steps < n_steps:
        observation, _ = env.reset(seed=seed + len(episodes))
        if policy is not None:
            policy.reset()
        observations, states, done = [observation], [hidden_state(env)] if hidden else [], False
        while not done:
            action = env.action_space.sample() if policy is None else policy(observation)
            observation, _, terminated, truncated, _ = env.step(action)
            observations.append(observation)
            if hidden:
                states.append(hidden_state(env))
            done = terminated or truncated
        episodes.append(np.asarray(observations, dtype=float))
        hiddens.append(np.asarray(states, dtype=float))
        steps += len(observations)
    env.close()
    return (episodes, hiddens) if hidden else episodes


def scaling_bounds(episodes: list) -> tuple:
    """(low, high) of the observations of the episodes."""
    observations = np.concatenate(episodes)
    return observations.min(axis=0), observations.max(axis=0)


def input_matrices(units: int, n_inputs: int, seed: int, input_scaling: float, bias_scaling: float,
                   bias_dist: str = "foldnorm") -> tuple:
    """(Win, bias) of the reservoirs, drawn as reservoirpy's HAGReservoir would with this seed: block input matrix (one
    block of units per input feature, uniform weights in [0, 1)) and bias |N(0.1, 0.1)| ("foldnorm", default of
    HAGReservoir) or uniform in [-1, 1] ("uniform": thresholds spread over the input range), times the scalings."""
    Win_rng, _, bias_rng, _ = rand_generator(seed).spawn(4)
    Win = block_input(units, n_inputs, input_scaling=input_scaling, seed=Win_rng)
    if bias_dist == "foldnorm":
        bias = random_sparse(units, dist="foldnorm", c=1.0, scale=0.1, input_scaling=bias_scaling, seed=bias_rng)
    elif bias_dist == "uniform":
        bias = random_sparse(units, dist="uniform", loc=-1.0, scale=2.0, input_scaling=bias_scaling, seed=bias_rng)
    else:
        raise ValueError(f"Unknown bias distribution {bias_dist!r}: 'foldnorm' or 'uniform'.")
    return np.asarray(Win, dtype=float), np.ravel(np.asarray(bias, dtype=float))


def build_reservoir(kind: str, sequences: list, units: int, seed: int, params: dict,
                    min_per_input: int = MIN_NEURONS_PER_INPUT) -> dict:
    """Reservoir {"W", "Win", "bias", "lr"} on the input features of sequences (list of (T, n_inputs)). units is
    rounded up to a multiple of the number of input features (one block per feature, of at least MIN_NEURONS_PER_INPUT
    units). The reservoirs share the same
    Win and bias (params "input_scaling", "bias_scaling", "bias_dist") and leak rate ("lr"):
      - "hag": W learned by HAG (JAX version of HAGReservoir, hag.models.jax_hag_reservoir) from an empty matrix, the
        other params being arguments of HAGReservoir,
      - "esn": random W (uniform), params "sr" (spectral radius) and "rc_connectivity",
      - "proj": W = 0 (random projection of the input features, control of HAG), the other params being ignored,
    and the hybrids of an ESN and HAG (HYBRIDS), whose params are those of the ESN and, in params["hag"], those of
    HAG:
      - "hag_on_esn": the random W of the ESN, fixed, plus connections learned by HAG on top of it (HAG's plasticity
        only changes its own connections, but runs with the dynamics of the whole reservoir),
      - "hag_from_esn": W learned by HAG from the random W of the ESN instead of an empty matrix (HAG can weaken and
        strengthen all its connections),
      - "esn_hag": an ESN and a HAG reservoir side by side (block-diagonal W), sharing the units of the ESN of the same
        size (half of the units of each input block each, the ESN taking the extra one of odd blocks), each with its
        own inputs (params of the inputs of the ESN and of HAG).
    min_per_input: minimal number of units per input feature."""
    params = dict(params)
    hag = dict(params.pop("hag", {}))  # (hybrids)
    if kind == "esn_hag":
        n_inputs = sequences[0].shape[1]
        per_input = max(math.ceil(units / n_inputs), min_per_input)  # (units per input of the ESN of the same size)
        esn = build_reservoir("esn", sequences, n_inputs * math.ceil(per_input / 2), seed, params, min_per_input=1)
        hag = build_reservoir("hag", sequences, n_inputs * (per_input // 2), seed,
                              {key: value for key, value in hag.items() if key in INPUT_PARAMS + HAG_PARAMS},
                              min_per_input=1)
        lr = [np.full(len(r["W"]), r["lr"], dtype=float) for r in (esn, hag)]
        W = np.zeros((len(esn["W"]) + len(hag["W"]),) * 2)
        W[:len(esn["W"]), :len(esn["W"])], W[len(esn["W"]):, len(esn["W"]):] = esn["W"], hag["W"]
        return {"W": W, "Win": np.vstack([esn["Win"], hag["Win"]]), "bias": np.concatenate([esn["bias"], hag["bias"]]),
                "lr": np.concatenate(lr)}
    lr = params.pop("lr", 1.0)
    n_inputs = sequences[0].shape[1]
    units = n_inputs * max(math.ceil(units / n_inputs), min_per_input)
    Win, bias = input_matrices(units, n_inputs, seed, params.pop("input_scaling", 1.0), params.pop("bias_scaling", 1.0),
                               params.pop("bias_dist", "foldnorm"))
    if kind == "hag":
        node = HAGReservoir(W=np.zeros((units, units)), Win=Win, bias=bias, lr=lr, seed=seed, **params)
        W = node.fit(sequences).W
    elif kind in ("esn", "hag_on_esn", "hag_from_esn"):
        W = uniform(units, units, sr=params.pop("sr", 0.9), connectivity=params.pop("rc_connectivity", 0.1), seed=seed)
        W = W.toarray() if hasattr(W, "toarray") else W
        plasticity = {key: value for key, value in hag.items() if key in HAG_PARAMS}
        if kind == "hag_on_esn":
            node = HAGReservoir(W=np.zeros((units, units)), W_fixed=W, Win=Win, bias=bias, lr=lr, seed=seed,
                                **plasticity)
            W = W + np.asarray(node.fit(sequences).W)
        elif kind == "hag_from_esn":
            node = HAGReservoir(W=W, Win=Win, bias=bias, lr=lr, seed=seed, **plasticity)
            W = node.fit(sequences).W
    elif kind == "proj":
        W = np.zeros((units, units))
    else:
        raise ValueError(f"Unknown reservoir {kind!r}: 'hag', 'esn', 'proj' or {HYBRIDS}.")
    return {"W": np.asarray(W, dtype=float), "Win": Win, "bias": bias, "lr": lr}


def make_pipeline(condition: str, episodes: list, units: int, seed: int, params: dict = None,
                  include_input: bool = False) -> FeaturePipeline:
    """Feature pipeline of a condition: "obs" (scaled observations), "filterbank", "filterbank+esn", "filterbank+hag"
    or "filterbank+proj", fitted on the pretraining episodes. params: parameters of the filter bank (FILTER_BANK_PARAMS,
    see hag.rl.features.make_filter_bank; defaults if missing) and of the reservoir (see build_reservoir)."""
    params = dict(params or {})
    bank_params = {key: params.pop(key) for key in FILTER_BANK_PARAMS if key in params}
    low, high = scaling_bounds(episodes)
    bank = None if condition == "obs" else make_filter_bank(**bank_params)
    pipeline = FeaturePipeline(low, high, bank=bank).fit_feature_scaling(episodes)
    if condition in ("obs", "filterbank"):
        return pipeline
    kind = condition.split("+")[1]
    sequences = [pipeline.transform_episode(episode) for episode in episodes]
    reservoir = build_reservoir(kind, sequences, units, seed, params)
    with_reservoir = FeaturePipeline(low, high, bank=bank, reservoir=reservoir, include_input=include_input)
    with_reservoir.feature_low, with_reservoir.feature_scale = pipeline.feature_low, pipeline.feature_scale
    return with_reservoir
