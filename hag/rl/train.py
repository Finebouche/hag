"""
PPO (stable-baselines3's default hyperparameters, learning rate searched by hag.rl.hpo), with linear policy and value
function (NET_ARCH: linear readouts of the features, as the readout of a reservoir), on a partially observable
benchmark (ENV_ID, set with the environment variable HAG_RL_ENV, see hag.rl.envs.BENCHMARKS for the benchmarks and
their budgets) with different preprocessings of the observations:
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
    same for all the conditions, whose PPO is then trained from scratch.
For each condition and seed: returns of the training episodes (sample efficiency; returns of the environment, also when
PPO's rewards are normalized) and final evaluation of the policy. With
USE_HPO_PARAMS, each condition uses the best parameters (features and learning rate) of the hyperparameter optimization
(hag/rl/hpo.py) if available: "obs" and "filterbank" studies, best of the "esn_*" studies for the ESN, best of the
"hag_mean_*" and "hag_variance_*" studies for HAG (and its control "proj").
Results: <RL_RESULTS>/rl_<name>.csv, learning curves in <RL_RESULTS>/rl_curves/ (name: RESULTS_NAME, RL_RESULTS: see
hag.rl.utils).

The runs (seed, condition) are parallelized over N_WORKERS processes.

Run from the repository root:  HAG_RL_ENV=<benchmark> python -m hag.rl.train [condition ...]
"""
import copy
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize

from hag.rl.utils import BENCHMARK_NAME, N_WORKERS, RL_RESULTS
from hag.rl.envs import BENCHMARKS, make_env
from hag.rl.pretrain import FeaturePolicy, collect_episodes, make_pipeline
from hag.rl.wrappers import FeatureVecEnv

# =============================== PARAMETERS ===============================
ENV_ID = BENCHMARK_NAME    # benchmark (environment variable HAG_RL_ENV)
BENCHMARK = BENCHMARKS[ENV_ID]
CONDITIONS = ["obs", "filterbank", "filterbank+esn", "filterbank+hag", "filterbank+proj"]
SEEDS = range(10)
TOTAL_TIMESTEPS = BENCHMARK.train_timesteps
N_EVAL_EPISODES = 50

# features
PRETRAIN_STEPS = 50_000     # steps of the pretraining episodes, to fit the features
PRETRAIN_POLICY = True      # two-phase pretraining: episodes of a policy trained on "filterbank" (else random policy)
PRETRAIN_TIMESTEPS = BENCHMARK.hpo_timesteps  # PPO budget of the pretraining policy
UNITS = 100                 # reservoir size, as in Léger et al. 2023 (Evolving Reservoirs for Meta Reinforcement
                            # Learning), rounded up to a multiple of the number of input features (see
                            # hag.rl.pretrain.build_reservoir: at least MIN_NEURONS_PER_INPUT units per input)
# filter bank (see hag.rl.features.make_filter_bank) of the pretraining policy, and used if no hyperparameter
# optimization
FILTER_BANK_PARAMS = dict(decomposition="ema", n_filters=8, slowest=0.02)
# parameters of the features (filter bank and reservoir, see hag.rl.pretrain.make_pipeline), used if no hyperparameter
# optimization (with the default learning rate of PPO)
DEFAULT_PARAMS = {
    "obs": {},
    "filterbank": FILTER_BANK_PARAMS,
    "esn": dict(FILTER_BANK_PARAMS, sr=0.9, rc_connectivity=0.1, input_scaling=0.5, bias_scaling=0.1, lr=1.0),
    "hag": dict(FILTER_BANK_PARAMS, homeostasis="mean", target=0.5, spread=0.1, weight_increment=0.02, min_window=5,
                max_window=20, input_scaling=0.5, bias_scaling=0.1, lr=1.0),
}
USE_HPO_PARAMS = True

# PPO: stable-baselines3's default hyperparameters, except the learning rate (searched), on one environment (its
# default: the rollouts are n_steps = 2048 steps per environment, 8 environments would give 8 times fewer updates)
# and with linear readouts (no hidden layer) as networks of the policy and of the value function, as the readout of a
# reservoir. Rewards normalized for some benchmarks (BENCHMARK.normalize_reward).
NET_ARCH = dict(pi=[], vf=[])
N_ENVS = 1
N_EVAL_ENVS = 8
# PPO of the pretraining policy (it only collects the episodes): stable-baselines3's default MLP and hyperparameters
PRETRAIN_PPO = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))
# ===========================================================================

# name of the results and of the hyperparameter optimization studies (hag.rl.hpo)
RESULTS_NAME = ENV_ID


class EpisodeReturns(BaseCallback):
    """Returns and end timesteps of the training episodes (returns of the environment, recorded by the Monitor wrapper
    of make_vec_env: not affected by the normalization of the rewards)."""

    def __init__(self):
        super().__init__()
        self.returns, self.timesteps = [], []

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "episode" in info:
                self.returns.append(info["episode"]["r"])
                self.timesteps.append(self.num_timesteps)
        return True


def condition_params(condition):
    """Parameters of a condition (features and learning rate of PPO): best of the hyperparameter optimization if
    available (and USE_HPO_PARAMS), else DEFAULT_PARAMS. The control "proj" takes the parameters of HAG."""
    kind = condition.split("+")[1] if "+" in condition else condition
    kind = "hag" if kind == "proj" else kind
    if USE_HPO_PARAMS:
        from hag.rl.hpo import best_params  # (avoids a circular import)
        best = best_params(kind)
        if best is not None:
            return best
    return DEFAULT_PARAMS[kind]


def train_ppo(pipeline, seed, total_timesteps, callback=None, n_envs=N_ENVS, net_arch=NET_ARCH, **ppo_params):
    """PPO trained from scratch on the features of pipeline, on n_envs environments, with policy and value networks
    net_arch and hyperparameters ppo_params (default: those of stable-baselines3)."""
    train_env = FeatureVecEnv(make_vec_env(lambda: make_env(ENV_ID), n_envs=n_envs, seed=seed), copy.deepcopy(pipeline))
    if BENCHMARK.normalize_reward:
        train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True)
    model = PPO("MlpPolicy", train_env, seed=seed, device="cpu", verbose=0, policy_kwargs=dict(net_arch=net_arch),
                **ppo_params)
    return model.learn(total_timesteps, callback=callback)


def pretraining_episodes(seed):
    """Pretraining episodes of a seed (see PRETRAIN_POLICY)."""
    episodes = collect_episodes(ENV_ID, PRETRAIN_STEPS, seed=seed)
    if not PRETRAIN_POLICY:
        return episodes
    start = time.time()
    pipeline = make_pipeline("filterbank", episodes, UNITS, seed, FILTER_BANK_PARAMS)
    model = train_ppo(pipeline, seed, PRETRAIN_TIMESTEPS, **PRETRAIN_PPO)
    # reset seeds different from those of the random episodes
    episodes = collect_episodes(ENV_ID, PRETRAIN_STEPS, seed=20_000 + seed, policy=FeaturePolicy(model, pipeline))
    lengths = np.array([len(episode) for episode in episodes])
    print(f"[pretrain] {ENV_ID} seed {seed}: {len(episodes)} episodes of the pretraining policy, length "
          f"{lengths.mean():.0f} ± {lengths.std():.0f} (min {lengths.min()}, max {lengths.max()}) | "
          f"{round(time.time() - start)}s", flush=True)
    return episodes


def run(condition, seed, episodes, params=None, total_timesteps=TOTAL_TIMESTEPS, evaluate=True, save_curves=True,
        check=None):
    """Train PPO with the features of a condition, fitted on the pretraining episodes, and return the results of the
    run. params: parameters of the features and "learning_rate" of PPO (default: condition_params; stable-baselines3's
    learning rate if missing). check: function called with the feature pipeline before the training (e.g. to stop a
    run whose reservoir is not suitable by raising an exception)."""
    start = time.time()
    params = condition_params(condition) if params is None else params
    features = {key: value for key, value in params.items() if key != "learning_rate"}
    ppo_params = {key: value for key, value in params.items() if key == "learning_rate"}
    pipeline = make_pipeline(condition, episodes, UNITS, seed, features)
    if check is not None:
        check(pipeline)
    eval_env = FeatureVecEnv(make_vec_env(lambda: make_env(ENV_ID), n_envs=N_EVAL_ENVS, seed=10_000 + seed),
                             copy.deepcopy(pipeline))

    returns = EpisodeReturns()
    model = train_ppo(pipeline, seed, total_timesteps, callback=returns, **ppo_params)
    eval_mean, eval_std = (evaluate_policy(model, eval_env, n_eval_episodes=N_EVAL_EPISODES, deterministic=True)
                           if evaluate else (np.nan, np.nan))

    if save_curves:
        curves = RL_RESULTS / "rl_curves"
        curves.mkdir(parents=True, exist_ok=True)
        np.savez(curves / f"{RESULTS_NAME}_{condition}_seed{seed}.npz", returns=returns.returns,
                 timesteps=returns.timesteps)
    train = np.asarray(returns.returns)
    W = None if pipeline.reservoir is None else pipeline.reservoir["W"]
    return {"env": ENV_ID, "condition": condition, "seed": seed, "n_features": pipeline.n_features,
            "connections_per_neuron": np.nan if W is None else np.count_nonzero(W) / len(W),
            "eval_return_mean": eval_mean, "eval_return_std": eval_std,
            "train_return_mean": float(train.mean()) if len(train) else np.nan,  # whole training: sample efficiency
            "train_return_last10%": float(train[-max(1, len(train) // 10):].mean()) if len(train) else np.nan,
            "time_s": round(time.time() - start), "params": params}


def main(conditions):
    output = RL_RESULTS / f"rl_{RESULTS_NAME}.csv"
    rows = []
    with ProcessPoolExecutor(N_WORKERS, mp_context=get_context("spawn")) as pool:
        episodes = dict(zip(SEEDS, pool.map(pretraining_episodes, SEEDS)))
        futures = [pool.submit(run, condition, seed, episodes[seed]) for seed in SEEDS for condition in conditions]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(f"[rl] {ENV_ID} {row['condition']:16s} seed {row['seed']}: eval {row['eval_return_mean']:6.1f} ± "
                  f"{row['eval_return_std']:5.1f} | train mean {row['train_return_mean']:6.1f}, last 10% "
                  f"{row['train_return_last10%']:6.1f} | {row['n_features']} features | {row['time_s']}s", flush=True)
            # saved after each run, so that an interrupted study keeps its results
            output.parent.mkdir(parents=True, exist_ok=True)
            rows.sort(key=lambda row: (row["seed"], conditions.index(row["condition"])))
            pd.DataFrame(rows).to_csv(output, index=False)

    results = pd.DataFrame(rows)
    metrics = ["eval_return_mean", "train_return_mean", "train_return_last10%"]
    summary = results.groupby("condition")[metrics].agg(["mean", "std"])
    print(summary.round(1).to_string())
    print(f"Results saved to {output}")


if __name__ == "__main__":
    main(sys.argv[1:] or CONDITIONS)
