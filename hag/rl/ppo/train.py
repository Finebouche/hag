"""
PPO (stable-baselines3's default hyperparameters, learning rate searched by hag.rl.ppo.hpo), with linear policy and
value function (NET_ARCH: linear readouts of the features, as the readout of a reservoir), on the benchmark and
conditions of hag.rl.experiment (features fitted on its pretraining episodes).
For each condition and seed: returns of the training episodes (sample efficiency; returns of the environment, also when
PPO's rewards are normalized) and final evaluation of the policy. With USE_HPO_PARAMS, each condition uses the best
parameters (features and learning rate) of the hyperparameter optimization (hag.rl.ppo.hpo) if available: "obs" and
"filterbank" studies, best of the "esn_*" studies for the ESN, best of the "hag_mean_*" and "hag_variance_*" studies
for HAG (and its control "proj").
Results: <RL_RESULTS>/rl_<name>.csv, learning curves in <RL_RESULTS>/rl_curves/ (name: RESULTS_NAME, RL_RESULTS: see
hag.rl.utils).

The runs (seed, condition) are parallelized over N_WORKERS processes.

Run from the repository root:  HAG_RL_ENV=<benchmark> python -m hag.rl.ppo.train [condition ...]
"""
import copy
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from hag.rl.envs import make_env
from hag.rl.experiment import (BENCHMARK, CONDITIONS, DEFAULT_PARAMS, ENV_ID, RESULTS_NAME, SEEDS, UNITS,
                               condition_kind, pretraining_episodes, train_ppo)
from hag.rl.pretrain import make_pipeline
from hag.rl.utils import N_WORKERS, RL_RESULTS
from hag.rl.wrappers import FeatureVecEnv

# =============================== PARAMETERS ===============================
TOTAL_TIMESTEPS = BENCHMARK.train_timesteps
N_EVAL_EPISODES = 50
USE_HPO_PARAMS = True
# PPO: stable-baselines3's default hyperparameters, except the learning rate (searched), on one environment (its
# default: the rollouts are n_steps = 2048 steps per environment, 8 environments would give 8 times fewer updates)
# and with linear readouts (no hidden layer) as networks of the policy and of the value function, as the readout of a
# reservoir. Rewards normalized for some benchmarks (BENCHMARK.normalize_reward).
NET_ARCH = dict(pi=[], vf=[])
N_ENVS = 1
N_EVAL_ENVS = 8
# ===========================================================================


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
    kind = condition_kind(condition)
    if USE_HPO_PARAMS:
        from hag.rl.ppo.hpo import best_params  # (avoids a circular import)
        best = best_params(kind)
        if best is not None:
            return best
    return DEFAULT_PARAMS[kind]


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
    model = train_ppo(pipeline, seed, total_timesteps, callback=returns, n_envs=N_ENVS, net_arch=NET_ARCH,
                      **ppo_params)
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
