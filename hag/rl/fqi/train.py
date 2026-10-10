"""
Fitted Q-iteration (Ernst et al., 2005, "Tree-based batch mode reinforcement learning") with extra-trees, on the
benchmark and conditions of hag.rl.experiment (features fitted on its pretraining episodes): batch Q-learning reusing
all the samples, as LSPI (hag.rl.lspi), but with a nonlinear Q-function. Benchmarks with discrete actions only.

Q(x, a) is one ExtraTreesRegressor on [x, one-hot(a)], x being the features (standardized with the statistics of the
pretraining episodes, see hag.rl.lspi.train.Features). Each iteration collects STEPS_PER_ITERATION steps with the
epsilon-greedy policy of the current Q (random at the first iteration; epsilon decreasing linearly over the iterations),
adds them to the samples, then runs Q_ITERATIONS steps of Q-iteration on them, Q <- regression of
r + gamma max_a' Q(x', a') (no bootstrap at the end of a terminated episode), each on at most MAX_FIT_SAMPLES samples
drawn at random. The greedy policy of each iteration is evaluated on N_SELECTION_EPISODES episodes
("greedy_return_mean": mean over the iterations), the best one on N_EVAL_EPISODES episodes (other seeds).
Parameters of a condition (condition_params): best ones of the hyperparameter optimization of FQI (hag.rl.fqi.hpo:
features, "gamma" and "min_samples_leaf"), else the features of the best ones of PPO (hag.rl.ppo.hpo) with GAMMA and
MIN_SAMPLES_LEAF, else the default ones (hag.rl.experiment.DEFAULT_PARAMS).

Results: <RL_RESULTS>/fqi_<name>.csv (columns of hag.rl.ppo.train and of FQI).
Run from the repository root:  HAG_RL_ENV=<benchmark> python -m hag.rl.fqi.train [condition ...]
"""
import copy
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

from hag.rl.envs import make_env
from hag.rl.experiment import (CONDITIONS, DEFAULT_PARAMS, ENV_ID, RESULTS_NAME, SEEDS, UNITS, condition_kind,
                               pretraining_episodes)
from hag.rl.lspi.train import Features
from hag.rl.pretrain import make_pipeline
from hag.rl.utils import N_WORKERS, RL_RESULTS

# =============================== PARAMETERS ===============================
N_ITERATIONS = 20
STEPS_PER_ITERATION = 10_000
Q_ITERATIONS = 5            # Q-iteration steps per iteration (on all the samples)
MAX_FIT_SAMPLES = 20_000    # samples of each regression (drawn at random among all the samples)
N_ESTIMATORS = 20
GAMMA = 0.99                # discount (default, searched by hag.rl.fqi.hpo)
MIN_SAMPLES_LEAF = 10       # of the trees (default, searched)
FQI_PARAMS = ("gamma", "min_samples_leaf")
USE_HPO_PARAMS = True
EPSILON = (0.5, 0.05)       # exploration of the first and last iterations
N_SELECTION_EPISODES = 10
N_EVAL_EPISODES = 50
# ===========================================================================


class QFunction:
    """Q(x, a) of extra-trees on [x, one-hot(a)] (zero before the first fit)."""

    def __init__(self, n_actions: int, min_samples_leaf: int, seed: int):
        self.n_actions = n_actions
        self.model = ExtraTreesRegressor(n_estimators=N_ESTIMATORS, min_samples_leaf=min_samples_leaf,
                                         max_features="sqrt", n_jobs=1, random_state=seed)
        self.fitted = False

    def inputs(self, X: np.ndarray, actions: np.ndarray) -> np.ndarray:
        return np.concatenate([X, np.eye(self.n_actions)[actions]], axis=1)

    def all_actions(self, X: np.ndarray) -> np.ndarray:
        """Q(x, a) of all the actions, shape (len(X), n_actions)."""
        if not self.fitted:
            return np.zeros((len(X), self.n_actions))
        rows = np.repeat(X, self.n_actions, axis=0)
        actions = np.tile(np.arange(self.n_actions), len(X))
        return self.model.predict(self.inputs(rows, actions)).reshape(len(X), self.n_actions)

    def fit(self, X: np.ndarray, actions: np.ndarray, targets: np.ndarray):
        self.model.fit(self.inputs(X, actions), targets)
        self.fitted = True


def play(env, features: Features, Q: QFunction, epsilon: float, n: int, rng, seed: int, record: bool = True) -> tuple:
    """Episodes of the epsilon-greedy policy of Q (random if Q is not fitted) until n steps (or n episodes if not
    record). Returns the samples (X, actions, rewards, X', terminal) and the returns of the episodes."""
    X, actions, rewards, X_next, terminal, returns = [], [], [], [], [], []
    steps, k = 0, 0
    while (steps < n) if record else (k < n):
        observation, _ = env.reset(seed=seed + k)
        features.reset()
        x, done, total = features(observation), False, 0.0
        while not done:
            if not Q.fitted or rng.random() < epsilon:
                action = int(rng.integers(Q.n_actions))
            else:
                action = int(np.argmax(Q.all_actions(x[None, :])[0]))
            observation, reward, terminated, truncated, _ = env.step(action)
            x_next = features(observation)
            if record:
                X.append(x), actions.append(action), rewards.append(reward), X_next.append(x_next)
                terminal.append(terminated)
            x, total, done, steps = x_next, total + reward, terminated or truncated, steps + 1
        returns.append(total)
        k += 1
    samples = None if not record else (np.asarray(X, dtype=np.float32), np.asarray(actions),
                                       np.asarray(rewards, dtype=float), np.asarray(X_next, dtype=np.float32),
                                       np.asarray(terminal))
    return samples, returns


def q_iteration(Q: QFunction, samples: tuple, gamma: float, rng):
    """One step of Q-iteration on (at most MAX_FIT_SAMPLES of) the samples."""
    X, actions, rewards, X_next, terminal = samples
    rows = rng.choice(len(X), size=min(len(X), MAX_FIT_SAMPLES), replace=False)
    targets = rewards[rows] + gamma * (~terminal[rows]) * Q.all_actions(X_next[rows]).max(axis=1)
    Q.fit(X[rows], actions[rows], targets)


def condition_params(condition):
    """Parameters of a condition (features, "gamma" and "min_samples_leaf"): best of the hyperparameter optimization of
    FQI if available (and USE_HPO_PARAMS), else the features of the best ones of PPO, else DEFAULT_PARAMS. The controls
    take the parameters of their model (see hag.rl.experiment.condition_kind)."""
    kind = condition_kind(condition)
    if USE_HPO_PARAMS:
        from hag.rl.fqi import hpo  # (avoids a circular import)
        from hag.rl.ppo import hpo as ppo_hpo
        for best in (hpo.best_params(kind), ppo_hpo.best_params(kind)):
            if best is not None:
                return {key: value for key, value in best.items() if key != "learning_rate"}
    return DEFAULT_PARAMS[kind]


def run(condition, seed, episodes, params=None, save_curves=True, check=None, units=UNITS):
    """FQI on the features of a condition, fitted on the pretraining episodes, and return the results of the run.
    params: parameters of the features, "gamma" and "min_samples_leaf" (default: condition_params; GAMMA and
    MIN_SAMPLES_LEAF if missing). check: function called with the feature pipeline before the training. units:
    reservoir size."""
    start = time.time()
    params = condition_params(condition) if params is None else params
    fqi = dict(dict(gamma=GAMMA, min_samples_leaf=MIN_SAMPLES_LEAF),
               **{key: params[key] for key in FQI_PARAMS if key in params})
    feature_params = {key: value for key, value in params.items() if key not in FQI_PARAMS + ("learning_rate",)}
    pipeline = make_pipeline(condition, episodes, units, seed, feature_params)
    if check is not None:
        check(pipeline)
    features = Features(pipeline, episodes)
    env = make_env(ENV_ID)
    rng = np.random.default_rng(seed)
    Q = QFunction(env.action_space.n, int(fqi["min_samples_leaf"]), seed)
    samples, train_returns, curve, best = None, [], [], (-np.inf, None, -1)
    for iteration in range(N_ITERATIONS):
        epsilon = EPSILON[0] + (EPSILON[1] - EPSILON[0]) * iteration / max(1, N_ITERATIONS - 1)
        new, returns = play(env, features, Q, epsilon, STEPS_PER_ITERATION, rng,
                            seed=1_000_000 * seed + 10_000 * iteration)
        train_returns += returns
        samples = new if samples is None else tuple(np.concatenate([s, n]) for s, n in zip(samples, new))
        for _ in range(Q_ITERATIONS):
            q_iteration(Q, samples, fqi["gamma"], rng)
        _, selection = play(env, features, Q, 0.0, N_SELECTION_EPISODES, rng, seed=500_000 + 100 * iteration,
                            record=False)
        curve.append(float(np.mean(selection)))
        if curve[-1] > best[0]:
            best = (curve[-1], copy.deepcopy(Q), iteration)
    _, evaluation = play(env, features, best[1], 0.0, N_EVAL_EPISODES, rng, seed=10_000 + seed, record=False)

    if save_curves:
        curves = RL_RESULTS / "rl_curves"
        curves.mkdir(parents=True, exist_ok=True)
        np.savez(curves / f"fqi_{RESULTS_NAME}_{condition}_seed{seed}.npz", returns=train_returns, selection=curve)
    W = None if pipeline.reservoir is None else pipeline.reservoir["W"]
    returns = np.asarray(train_returns)
    return {"env": ENV_ID, "condition": condition, "seed": seed, "n_features": pipeline.n_features,
            "connections_per_neuron": np.nan if W is None else np.count_nonzero(W) / len(W),
            "eval_return_mean": float(np.mean(evaluation)), "eval_return_std": float(np.std(evaluation)),
            "train_return_mean": float(returns.mean()),
            "train_return_last10%": float(returns[-max(1, len(returns) // 10):].mean()),
            "greedy_return_mean": float(np.mean(curve)), "best_iteration": best[2], "selection_curve": curve,
            "time_s": round(time.time() - start), "params": dict(feature_params, **fqi)}


def main(conditions):
    output = RL_RESULTS / f"fqi_{RESULTS_NAME}.csv"
    rows = []
    with ProcessPoolExecutor(N_WORKERS, mp_context=get_context("spawn")) as pool:
        episodes = dict(zip(SEEDS, pool.map(pretraining_episodes, SEEDS)))
        futures = [pool.submit(run, condition, seed, episodes[seed]) for seed in SEEDS for condition in conditions]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(f"[fqi] {ENV_ID} {row['condition']:16s} seed {row['seed']}: eval {row['eval_return_mean']:7.3f} ± "
                  f"{row['eval_return_std']:5.3f} | train mean {row['train_return_mean']:7.3f} | best iteration "
                  f"{row['best_iteration']} | {row['time_s']}s", flush=True)
            output.parent.mkdir(parents=True, exist_ok=True)
            rows.sort(key=lambda row: (row["seed"], conditions.index(row["condition"])))
            pd.DataFrame(rows).to_csv(output, index=False)
    results = pd.DataFrame(rows)
    metrics = ["eval_return_mean", "greedy_return_mean", "train_return_mean", "train_return_last10%"]
    print(results.groupby("condition")[metrics].agg(["mean", "std"]).round(3).to_string())
    print(f"Results saved to {output}")


if __name__ == "__main__":
    main(sys.argv[1:] or CONDITIONS)
