"""
Settings shared by the reinforcement learning methods (hag.rl.ppo, hag.rl.lspi) and their hyperparameter searches:
benchmark (ENV_ID, set with the environment variable HAG_RL_ENV, see hag.rl.envs.BENCHMARKS for the benchmarks and
their budgets), conditions (preprocessings of the observations), seeds, reservoir size, default features, and the
pretraining episodes on which the features are fitted.

Conditions:
  - "obs": scaled observations (no memory),
  - "filterbank": causal filter bank (multi time scale features, see hag.rl.features),
  - "filterbank+esn": random reservoir driven by the filter bank,
  - "filterbank+hag": HAG reservoir (W learned by HAG, unsupervised) driven by the filter bank,
  - "filterbank+proj": control of HAG, its filter bank, input matrix and bias with W = 0 (random projection).
The reservoirs share the same input matrix and bias. The features are fitted on pretraining episodes (PRETRAIN_STEPS
steps), then frozen:
  - PRETRAIN_POLICY = False: episodes of a random policy,
  - PRETRAIN_POLICY = True: two-phase pretraining. PPO (MLP) is first trained for PRETRAIN_TIMESTEPS on the
    "filterbank" features (fitted on random episodes), then the pretraining episodes are collected with this policy
    (stochastic actions, to keep exploring): they look like the episodes seen during training. These episodes are the
    same for all the conditions and methods.
"""
import copy
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from hag.rl.envs import BENCHMARKS, make_env
from hag.rl.pretrain import FeaturePolicy, collect_episodes, make_pipeline
from hag.rl.utils import BENCHMARK_NAME
from hag.rl.wrappers import FeatureVecEnv

# =============================== PARAMETERS ===============================
ENV_ID = BENCHMARK_NAME    # benchmark (environment variable HAG_RL_ENV)
BENCHMARK = BENCHMARKS[ENV_ID]
CONDITIONS = ["obs", "filterbank", "filterbank+esn", "filterbank+hag", "filterbank+proj"]
SEEDS = range(10)

# features
PRETRAIN_STEPS = 50_000     # steps of the pretraining episodes, to fit the features
PRETRAIN_POLICY = True      # two-phase pretraining: episodes of a policy trained on "filterbank" (else random policy)
PRETRAIN_TIMESTEPS = BENCHMARK.hpo_timesteps  # PPO budget of the pretraining policy
PRETRAIN_NET_ARCH = dict(pi=[64, 64], vf=[64, 64])  # stable-baselines3's default MLP, for the pretraining policy
UNITS = 100                 # reservoir size, as in Léger et al. 2023 (Evolving Reservoirs for Meta Reinforcement
                            # Learning), rounded up to a multiple of the number of input features (see
                            # hag.rl.pretrain.build_reservoir: at least MIN_NEURONS_PER_INPUT units per input)
# filter bank (see hag.rl.features.make_filter_bank) of the pretraining policy, and used if no hyperparameter
# optimization
FILTER_BANK_PARAMS = dict(decomposition="ema", n_filters=8, slowest=0.02)
# parameters of the features of the conditions (filter bank and reservoir, see hag.rl.pretrain.make_pipeline), used
# if no hyperparameter optimization
DEFAULT_PARAMS = {
    "obs": {},
    "filterbank": FILTER_BANK_PARAMS,
    "esn": dict(FILTER_BANK_PARAMS, sr=0.9, rc_connectivity=0.1, input_scaling=0.5, bias_scaling=0.1, lr=1.0),
    "hag": dict(FILTER_BANK_PARAMS, homeostasis="mean", target=0.5, spread=0.1, weight_increment=0.02, min_window=5,
                max_window=20, input_scaling=0.5, bias_scaling=0.1, lr=1.0),
}
# ===========================================================================

# name of the results and of the hyperparameter optimization studies
RESULTS_NAME = ENV_ID


def condition_kind(condition: str) -> str:
    """Kind of features of a condition ("obs", "filterbank", "esn", "hag", or a hybrid of both, see
    hag.rl.pretrain.HYBRIDS; the control "proj" takes HAG's)."""
    kind = condition.split("+")[1] if "+" in condition else condition
    return "hag" if kind == "proj" else kind


def train_ppo(pipeline, seed, total_timesteps, callback=None, n_envs=1, net_arch=PRETRAIN_NET_ARCH, **ppo_params):
    """PPO trained from scratch on the features of pipeline, on n_envs environments, with policy and value networks
    net_arch and hyperparameters ppo_params (default: those of stable-baselines3). Rewards normalized for some
    benchmarks (BENCHMARK.normalize_reward)."""
    train_env = FeatureVecEnv(make_vec_env(lambda: make_env(ENV_ID), n_envs=n_envs, seed=seed), copy.deepcopy(pipeline))
    if BENCHMARK.normalize_reward:
        train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True)
    model = PPO("MlpPolicy", train_env, seed=seed, device="cpu", verbose=0, policy_kwargs=dict(net_arch=net_arch),
                **ppo_params)
    return model.learn(total_timesteps, callback=callback)


def pretraining_episodes(seed, hidden=False):
    """Pretraining episodes of a seed (see PRETRAIN_POLICY). hidden: also return their hidden states (see
    hag.rl.pretrain.collect_episodes), returns (episodes, hidden states)."""
    episodes = collect_episodes(ENV_ID, PRETRAIN_STEPS, seed=seed, hidden=hidden)
    if not PRETRAIN_POLICY:
        return episodes
    start = time.time()
    pipeline = make_pipeline("filterbank", episodes[0] if hidden else episodes, UNITS, seed, FILTER_BANK_PARAMS)
    model = train_ppo(pipeline, seed, PRETRAIN_TIMESTEPS)
    # reset seeds different from those of the random episodes
    episodes = collect_episodes(ENV_ID, PRETRAIN_STEPS, seed=20_000 + seed, policy=FeaturePolicy(model, pipeline),
                                hidden=hidden)
    lengths = np.array([len(episode) for episode in (episodes[0] if hidden else episodes)])
    print(f"[pretrain] {ENV_ID} seed {seed}: {len(lengths)} episodes of the pretraining policy, length "
          f"{lengths.mean():.0f} ± {lengths.std():.0f} (min {lengths.min()}, max {lengths.max()}) | "
          f"{round(time.time() - start)}s", flush=True)
    return episodes
