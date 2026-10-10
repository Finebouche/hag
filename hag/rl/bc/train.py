"""
Behavioral cloning of an expert by a ridge readout of the features, with DAgger (Ross et al., 2011, "A reduction of
imitation learning and structured prediction to no-regret online learning"), on the benchmark and conditions of
hag.rl.experiment (features fitted on its pretraining episodes): supervised, no reinforcement learning. The readout is
told the right action at each step, so its return is what a linear readout of the features can reach when its
learning is not the problem (an upper bound of the RL methods with linear readouts, hag.rl.ppo, hag.rl.lspi,
hag.rl.nac). Benchmarks with discrete actions only.

Expert (expert): PPO (stable-baselines3's MLP and default hyperparameters, EXPERT_TIMESTEPS) on the expert observations
(hag.rl.envs.make_expert_env: observation and hidden state, the task is then Markov), one per seed, saved in
<RL_RESULTS>/experts/ (reused if present).
Readout: action = argmax of a ridge regression of the one-hot expert actions on the standardized features
(RidgeClassifierCV, regularization chosen by generalized cross-validation among ALPHAS).
Rounds: round 0 collects STEPS_PER_ROUND steps of the expert's episodes (behavioral cloning), each next round
STEPS_PER_ROUND steps of the episodes of the greedy readout of the previous round, labeled by the expert (DAgger); the
readout is fitted on all the samples. The readout of each round is evaluated on N_SELECTION_EPISODES episodes
("selection_curve", "bc_return": round 0, "greedy_return_mean": mean over the rounds), the best one on N_EVAL_EPISODES
episodes (other seeds). Parameters of a condition (condition_params): features of the best ones of PPO
(hag.rl.ppo.hpo), else the default ones (hag.rl.experiment.DEFAULT_PARAMS): no hyperparameter optimization of its own,
the regularization being chosen by cross-validation.

Results: <RL_RESULTS>/bc_<name>.csv (columns of hag.rl.ppo.train and of BC).
Run from the repository root:  HAG_RL_ENV=<benchmark> python -m hag.rl.bc.train [condition ...]
"""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifierCV
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize

from hag.rl.envs import make_expert_env
from hag.rl.experiment import (BENCHMARK, CONDITIONS, DEFAULT_PARAMS, ENV_ID, RESULTS_NAME, SEEDS, UNITS,
                               condition_kind, pretraining_episodes)
from hag.rl.lspi.train import Features
from hag.rl.pretrain import make_pipeline
from hag.rl.utils import N_WORKERS, RL_RESULTS

# =============================== PARAMETERS ===============================
EXPERT_TIMESTEPS = BENCHMARK.train_timesteps
N_ROUNDS = 5                # round 0: behavioral cloning, next ones: DAgger
STEPS_PER_ROUND = 20_000
ALPHAS = np.logspace(-3, 4, 8)
USE_HPO_PARAMS = True
N_SELECTION_EPISODES = 10
N_EVAL_EPISODES = 50
# ===========================================================================


def expert(seed) -> dict:
    """Expert of a seed, trained if not saved yet: {"path": its file, "return_mean", "return_std": its evaluation on
    N_EVAL_EPISODES episodes}. The path is that of the current RL_RESULTS (the results of a Slurm job are copied to a
    temporary folder, different for each job)."""
    folder = RL_RESULTS / "experts"
    path, info = folder / f"{RESULTS_NAME}_seed{seed}.zip", folder / f"{RESULTS_NAME}_seed{seed}.json"
    if not info.exists():
        start = time.time()
        env = make_vec_env(lambda: make_expert_env(ENV_ID), n_envs=1, seed=seed)
        if BENCHMARK.normalize_reward:
            env = VecNormalize(env, norm_obs=False, norm_reward=True)
        model = PPO("MlpPolicy", env, seed=seed, device="cpu", verbose=0).learn(EXPERT_TIMESTEPS)
        folder.mkdir(parents=True, exist_ok=True)
        model.save(path)
        eval_env = make_vec_env(lambda: make_expert_env(ENV_ID), n_envs=8, seed=10_000 + seed)
        mean, std = evaluate_policy(model, eval_env, n_eval_episodes=N_EVAL_EPISODES, deterministic=True)
        info.write_text(json.dumps(dict(return_mean=float(mean), return_std=float(std))))
        print(f"[bc] {ENV_ID} expert seed {seed}: return {mean:.3f} ± {std:.3f} | {round(time.time() - start)}s",
              flush=True)
    return dict(json.loads(info.read_text()), path=str(path))


class Readout:
    """Greedy policy of a ridge classifier of the expert actions (argmax of its linear scores)."""

    def __init__(self, X, labels):
        classes = np.unique(labels)
        if len(classes) == 1:  # (the expert always takes the same action)
            self.coef, self.intercept, self.classes = np.zeros((1, X.shape[1])), np.zeros(1), classes
        else:
            model = RidgeClassifierCV(alphas=ALPHAS).fit(X, labels)
            self.coef, self.intercept, self.classes = np.atleast_2d(model.coef_), np.atleast_1d(model.intercept_), \
                model.classes_
            if len(self.classes) == 2:  # (one score: positive for the second class)
                self.coef, self.intercept = np.vstack([-self.coef, self.coef]), np.hstack([-self.intercept,
                                                                                            self.intercept])

    def __call__(self, x) -> int:
        return int(self.classes[np.argmax(self.coef @ x + self.intercept)])


def play(env, features: Features, model, readout, n: int, seed: int, record: bool = True) -> tuple:
    """Episodes of the expert model (if readout is None) or of the greedy readout, until n steps (or n episodes if not
    record), the actions of the expert being the labels. Returns the features, the labels and the returns of the
    episodes."""
    X, labels, returns = [], [], []
    k = 0
    while (len(X) < n) if record else (k < n):
        observation, _ = env.reset(seed=seed + k)
        features.reset()
        done, total = False, 0.0
        while not done:
            x = features(observation[:env.n_agent])
            label = int(model.predict(observation, deterministic=True)[0]) if record or readout is None else None
            if record:
                X.append(x)
                labels.append(label)
            action = label if readout is None else readout(x)
            observation, reward, terminated, truncated, _ = env.step(action)
            total, done = total + reward, terminated or truncated
        returns.append(total)
        k += 1
    return np.asarray(X), np.asarray(labels), returns


def condition_params(condition):
    """Parameters of the features of a condition: features of the best parameters of PPO if available (and
    USE_HPO_PARAMS), else DEFAULT_PARAMS. The control "proj" takes the parameters of HAG."""
    kind = condition_kind(condition)
    if USE_HPO_PARAMS:
        from hag.rl.ppo import hpo as ppo_hpo
        best = ppo_hpo.best_params(kind)
        if best is not None:
            return best
    return DEFAULT_PARAMS[kind]


def run(condition, seed, episodes, params=None, save_curves=True, check=None, units=UNITS, expert_info=None):
    """Behavioral cloning (DAgger) on the features of a condition, fitted on the pretraining episodes, and return the
    results of the run. params: parameters of the features (default: condition_params; others ignored). check: function
    called with the feature pipeline before the training. units: reservoir size. expert_info: expert of the seed
    (default: expert(seed))."""
    start = time.time()
    params = condition_params(condition) if params is None else params
    feature_params = {key: value for key, value in params.items() if key not in ("learning_rate", "ridge", "gamma")}
    pipeline = make_pipeline(condition, episodes, units, seed, feature_params)
    if check is not None:
        check(pipeline)
    features = Features(pipeline, episodes)
    expert_info = expert(seed) if expert_info is None else expert_info
    model = PPO.load(expert_info["path"], device="cpu")
    env = make_expert_env(ENV_ID)
    X, labels, readout, curve, best, train_returns = None, None, None, [], (-np.inf, None, -1), []
    for round_ in range(N_ROUNDS):
        new_X, new_labels, returns = play(env, features, model, readout, STEPS_PER_ROUND,
                                          seed=1_000_000 * seed + 10_000 * round_)
        train_returns += returns
        X = new_X if X is None else np.concatenate([X, new_X])
        labels = new_labels if labels is None else np.concatenate([labels, new_labels])
        readout = Readout(X, labels)
        _, _, selection = play(env, features, model, readout, N_SELECTION_EPISODES, seed=500_000 + 100 * round_,
                               record=False)
        curve.append(float(np.mean(selection)))
        if curve[-1] > best[0]:
            best = (curve[-1], readout, round_)
    _, _, evaluation = play(env, features, model, best[1], N_EVAL_EPISODES, seed=10_000 + seed, record=False)

    if save_curves:
        curves = RL_RESULTS / "rl_curves"
        curves.mkdir(parents=True, exist_ok=True)
        np.savez(curves / f"bc_{RESULTS_NAME}_{condition}_seed{seed}.npz", selection=curve)
    W = None if pipeline.reservoir is None else pipeline.reservoir["W"]
    return {"env": ENV_ID, "condition": condition, "seed": seed, "n_features": pipeline.n_features,
            "connections_per_neuron": np.nan if W is None else np.count_nonzero(W) / len(W),
            "eval_return_mean": float(np.mean(evaluation)), "eval_return_std": float(np.std(evaluation)),
            "train_return_mean": float(np.mean(train_returns)),  # episodes collected (expert, then readouts)
            "bc_return": curve[0], "greedy_return_mean": float(np.mean(curve)), "best_round": best[2],
            "selection_curve": curve, "expert_return_mean": expert_info["return_mean"],
            "time_s": round(time.time() - start), "params": feature_params}


def main(conditions):
    output = RL_RESULTS / f"bc_{RESULTS_NAME}.csv"
    rows = []
    with ProcessPoolExecutor(N_WORKERS, mp_context=get_context("spawn")) as pool:
        experts = [pool.submit(expert, seed) for seed in SEEDS]
        episodes = dict(zip(SEEDS, pool.map(pretraining_episodes, SEEDS)))
        experts = dict(zip(SEEDS, [future.result() for future in experts]))
        futures = [pool.submit(run, condition, seed, episodes[seed], expert_info=experts[seed])
                   for seed in SEEDS for condition in conditions]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(f"[bc] {ENV_ID} {row['condition']:16s} seed {row['seed']}: eval {row['eval_return_mean']:7.3f} ± "
                  f"{row['eval_return_std']:5.3f} | BC {row['bc_return']:7.3f}, best round {row['best_round']} | "
                  f"expert {row['expert_return_mean']:7.3f} | {row['time_s']}s", flush=True)
            output.parent.mkdir(parents=True, exist_ok=True)
            rows.sort(key=lambda row: (row["seed"], conditions.index(row["condition"])))
            pd.DataFrame(rows).to_csv(output, index=False)
    results = pd.DataFrame(rows)
    metrics = ["eval_return_mean", "bc_return", "greedy_return_mean", "expert_return_mean"]
    print(results.groupby("condition")[metrics].agg(["mean", "std"]).round(3).to_string())
    print(f"Results saved to {output}")


if __name__ == "__main__":
    main(sys.argv[1:] or CONDITIONS)
