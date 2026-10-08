"""
DQN (stable-baselines3's default hyperparameters but the size of the replay buffer, learning rate searched by
hag.rl.dqn.hpo), with a linear Q-function (NET_ARCH: one linear readout of the features per action, as the readout of a
reservoir), on the benchmark and conditions of hag.rl.experiment (features fitted on its pretraining episodes), instead
of PPO (hag.rl.ppo): the readout is learned by gradient on the temporal difference error of off-policy samples (replay
buffer), as LSPI's (hag.rl.lspi) is in closed form. Benchmarks with discrete actions only. Rewards are not normalized
(replay buffer).
For each condition and seed: returns of the training episodes (sample efficiency) and final evaluation of the greedy
policy. Parameters of a condition (condition_params): best ones of the hyperparameter optimization of DQN
(hag.rl.dqn.hpo: features and learning rate), else the features of the best ones of PPO (hag.rl.ppo.hpo) with
stable-baselines3's learning rate, else the default ones (hag.rl.experiment.DEFAULT_PARAMS).

Results: <RL_RESULTS>/dqn_<name>.csv (columns of hag.rl.ppo.train), learning curves in <RL_RESULTS>/rl_curves/.
Run from the repository root:  HAG_RL_ENV=<benchmark> python -m hag.rl.dqn.train [condition ...]
"""
import copy
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from hag.rl.envs import make_env
from hag.rl.experiment import (BENCHMARK, CONDITIONS, DEFAULT_PARAMS, ENV_ID, RESULTS_NAME, SEEDS, UNITS,
                               condition_kind, pretraining_episodes)
from hag.rl.pretrain import make_pipeline
from hag.rl.ppo.train import EpisodeReturns
from hag.rl.utils import N_WORKERS, RL_RESULTS
from hag.rl.wrappers import FeatureVecEnv

# =============================== PARAMETERS ===============================
TOTAL_TIMESTEPS = BENCHMARK.train_timesteps
N_EVAL_EPISODES = 50
USE_HPO_PARAMS = True
NET_ARCH = []               # hidden layers of the Q-network: none, linear readouts
# stable-baselines3's defaults but the replay buffer (1M by default: 8 GB with 1000 features), and the parameters
# searched by hag.rl.dqn.hpo
DQN_PARAMS = dict(buffer_size=100_000)
SEARCHED = ("learning_rate",)
N_EVAL_ENVS = 8
# ===========================================================================


def condition_params(condition):
    """Parameters of a condition (features and learning rate of DQN): best of the hyperparameter optimization of DQN if
    available (and USE_HPO_PARAMS), else the features of the best ones of PPO, else DEFAULT_PARAMS. The control "proj"
    takes the parameters of HAG."""
    kind = condition_kind(condition)
    if USE_HPO_PARAMS:
        from hag.rl.dqn import hpo  # (avoids a circular import)
        from hag.rl.ppo import hpo as ppo_hpo
        best = hpo.best_params(kind)
        if best is not None:
            return best
        best = ppo_hpo.best_params(kind)
        if best is not None:
            return {key: value for key, value in best.items() if key != "learning_rate"}
    return DEFAULT_PARAMS[kind]


def train_dqn(pipeline, seed, total_timesteps, callback=None, net_arch=NET_ARCH, **dqn_params):
    """DQN trained from scratch on the features of pipeline, with Q-network net_arch and hyperparameters dqn_params
    (default: DQN_PARAMS, then those of stable-baselines3)."""
    train_env = FeatureVecEnv(make_vec_env(lambda: make_env(ENV_ID), n_envs=1, seed=seed), copy.deepcopy(pipeline))
    model = DQN("MlpPolicy", train_env, seed=seed, device="cpu", verbose=0, policy_kwargs=dict(net_arch=net_arch),
                **dict(DQN_PARAMS, **dqn_params))
    return model.learn(total_timesteps, callback=callback)


def run(condition, seed, episodes, params=None, total_timesteps=TOTAL_TIMESTEPS, evaluate=True, save_curves=True,
        check=None, units=UNITS, net_arch=NET_ARCH):
    """Train DQN with the features of a condition, fitted on the pretraining episodes, and return the results of the
    run. params: parameters of the features and of DQN (SEARCHED; default: condition_params). check: function called
    with the feature pipeline before the training (e.g. to stop a run whose reservoir is not suitable by raising an
    exception). units: reservoir size, net_arch: hidden layers of the Q-network (default: linear readouts)."""
    start = time.time()
    params = condition_params(condition) if params is None else params
    features = {key: value for key, value in params.items() if key not in SEARCHED}
    dqn_params = {key: value for key, value in params.items() if key in SEARCHED}
    pipeline = make_pipeline(condition, episodes, units, seed, features)
    if check is not None:
        check(pipeline)
    eval_env = FeatureVecEnv(make_vec_env(lambda: make_env(ENV_ID), n_envs=N_EVAL_ENVS, seed=10_000 + seed),
                             copy.deepcopy(pipeline))

    returns = EpisodeReturns()
    model = train_dqn(pipeline, seed, total_timesteps, callback=returns, net_arch=net_arch, **dqn_params)
    eval_mean, eval_std = (evaluate_policy(model, eval_env, n_eval_episodes=N_EVAL_EPISODES, deterministic=True)
                           if evaluate else (np.nan, np.nan))

    if save_curves:
        curves = RL_RESULTS / "rl_curves"
        curves.mkdir(parents=True, exist_ok=True)
        np.savez(curves / f"dqn_{RESULTS_NAME}_{condition}_seed{seed}.npz", returns=returns.returns,
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
    output = RL_RESULTS / f"dqn_{RESULTS_NAME}.csv"
    rows = []
    with ProcessPoolExecutor(N_WORKERS, mp_context=get_context("spawn")) as pool:
        episodes = dict(zip(SEEDS, pool.map(pretraining_episodes, SEEDS)))
        futures = [pool.submit(run, condition, seed, episodes[seed]) for seed in SEEDS for condition in conditions]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(f"[dqn] {ENV_ID} {row['condition']:16s} seed {row['seed']}: eval {row['eval_return_mean']:7.3f} ± "
                  f"{row['eval_return_std']:5.3f} | train mean {row['train_return_mean']:7.3f}, last 10% "
                  f"{row['train_return_last10%']:7.3f} | {row['n_features']} features | {row['time_s']}s",
                  flush=True)
            # saved after each run, so that an interrupted study keeps its results
            output.parent.mkdir(parents=True, exist_ok=True)
            rows.sort(key=lambda row: (row["seed"], conditions.index(row["condition"])))
            pd.DataFrame(rows).to_csv(output, index=False)

    results = pd.DataFrame(rows)
    metrics = ["eval_return_mean", "train_return_mean", "train_return_last10%"]
    print(results.groupby("condition")[metrics].agg(["mean", "std"]).round(3).to_string())
    print(f"Results saved to {output}")


if __name__ == "__main__":
    main(sys.argv[1:] or CONDITIONS)
