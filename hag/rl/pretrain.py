"""
Pretraining of the features: episodes of a pretraining policy (random, or a policy trained on other features, see
hag.rl.train.pretraining_episodes), bounds of the MinMax scaling of the observations and of the filter bank features,
and reservoirs. The reservoirs share the same input matrix and bias (one block of units per input feature, as the
default input matrix of reservoirpy's HAGReservoir) and only differ by W: random, rescaled to a spectral radius, for
the ESN; learned by HAG on the pretraining episodes; zero for the random projection "proj" (control of HAG).
"""
import copy
import math

import numpy as np
from reservoirpy.mat_gen import block_input, random_sparse, uniform
from reservoirpy.utils.random import rand_generator

from hag.models.jax_hag_reservoir import HAGReservoir
from hag.rl.envs import make_env
from hag.rl.features import FeaturePipeline, make_filter_bank

FILTER_BANK_PARAMS = ("decomposition", "n_filters", "slowest")  # parameters of make_filter_bank
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


def collect_episodes(name: str, n_steps: int, seed: int = 0, policy=None) -> list:
    """Observations (T, D) of the episodes of a policy (e.g. FeaturePolicy; random if None) on the benchmark name (see
    hag.rl.envs.BENCHMARKS), until n_steps steps."""
    env = make_env(name)
    env.action_space.seed(seed)
    episodes, steps = [], 0
    while steps < n_steps:
        observation, _ = env.reset(seed=seed + len(episodes))
        if policy is not None:
            policy.reset()
        observations, done = [observation], False
        while not done:
            action = env.action_space.sample() if policy is None else policy(observation)
            observation, _, terminated, truncated, _ = env.step(action)
            observations.append(observation)
            done = terminated or truncated
        episodes.append(np.asarray(observations, dtype=float))
        steps += len(observations)
    env.close()
    return episodes


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


def build_reservoir(kind: str, sequences: list, units: int, seed: int, params: dict) -> dict:
    """Reservoir {"W", "Win", "bias", "lr"} on the input features of sequences (list of (T, n_inputs)). units is
    rounded up to a multiple of the number of input features (one block per feature, of at least MIN_NEURONS_PER_INPUT
    units). The reservoirs share the same
    Win and bias (params "input_scaling", "bias_scaling", "bias_dist") and leak rate ("lr"):
      - "hag": W learned by HAG (JAX version of HAGReservoir, hag.models.jax_hag_reservoir) from an empty matrix, the
        other params being arguments of HAGReservoir,
      - "esn": random W (uniform), params "sr" (spectral radius) and "rc_connectivity",
      - "proj": W = 0 (random projection of the input features, control of HAG), the other params being ignored."""
    params = dict(params)
    lr = params.pop("lr", 1.0)
    n_inputs = sequences[0].shape[1]
    units = n_inputs * max(math.ceil(units / n_inputs), MIN_NEURONS_PER_INPUT)
    Win, bias = input_matrices(units, n_inputs, seed, params.pop("input_scaling", 1.0), params.pop("bias_scaling", 1.0),
                               params.pop("bias_dist", "foldnorm"))
    if kind == "hag":
        node = HAGReservoir(W=np.zeros((units, units)), Win=Win, bias=bias, lr=lr, seed=seed, **params)
        W = node.fit(sequences).W
    elif kind == "esn":
        W = uniform(units, units, sr=params.pop("sr", 0.9), connectivity=params.pop("rc_connectivity", 0.1), seed=seed)
        W = W.toarray() if hasattr(W, "toarray") else W
    elif kind == "proj":
        W = np.zeros((units, units))
    else:
        raise ValueError(f"Unknown reservoir {kind!r}: 'hag', 'esn' or 'proj'.")
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
