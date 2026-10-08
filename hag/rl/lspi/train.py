"""
Least-squares policy iteration (LSPI, Lagoudakis & Parr, 2003) on the benchmark and conditions of hag.rl.experiment
(features fitted on its pretraining episodes), instead of PPO (hag.rl.ppo): the readout of the features is learned in
closed form (regularized least squares, as the ridge readout of HAG's studies on the classification datasets) rather
than by gradient. Benchmarks with discrete actions only.

Q(x, a) = w_a · [x, 1], x being the features (standardized with the statistics of the pretraining episodes): one linear
readout per action. Each iteration collects STEPS_PER_ITERATION steps with the epsilon-greedy policy of the current Q
(random at the first iteration; epsilon decreasing linearly over the iterations), adds them to the samples, then
solves LSTD-Q on all the samples for the greedy policy (INNER_ITERATIONS times, policy iteration on the samples):
    A = sum phi(x, a) (phi(x, a) - gamma phi(x', pi(x')))^T,  b = sum phi(x, a) r,  w = (A + ridge n I)^-1 b
(phi(x, a): [x, 1] in the block of the action a; no bootstrap at the end of a terminated episode). The greedy policy of
each iteration is evaluated on N_SELECTION_EPISODES episodes ("greedy_return_mean": mean over the iterations, the
sample efficiency of the greedy policies); the best one is evaluated on N_EVAL_EPISODES episodes (other seeds).
Parameters of a condition (condition_params): best ones of the hyperparameter optimization of LSPI (hag.rl.lspi.hpo:
features, "ridge" and "gamma"), else the features of the best ones of PPO (hag.rl.ppo.hpo) with RIDGE and GAMMA, else
the default ones (hag.rl.experiment.DEFAULT_PARAMS).

Results: <RL_RESULTS>/lspi_<name>.csv (columns of hag.rl.ppo.train and of LSPI).
Run from the repository root:  HAG_RL_ENV=<benchmark> python -m hag.rl.lspi.train [condition ...]
"""
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

import numpy as np
import pandas as pd

from hag.rl.envs import make_env
from hag.rl.experiment import (CONDITIONS, DEFAULT_PARAMS, ENV_ID, RESULTS_NAME, SEEDS, UNITS, condition_kind,
                               pretraining_episodes)
from hag.rl.pretrain import make_pipeline
from hag.rl.probe import episode_features
from hag.rl.utils import N_WORKERS, RL_RESULTS

# =============================== PARAMETERS ===============================
N_ITERATIONS = 20
STEPS_PER_ITERATION = 10_000
INNER_ITERATIONS = 3        # LSTD-Q solves per iteration (policy iteration on the samples)
GAMMA = 0.99                # discount (default, searched by hag.rl.lspi.hpo)
RIDGE = 1e-3                # regularization of LSTD-Q, relative to the number of samples (default, searched)
LSPI_PARAMS = ("ridge", "gamma")  # parameters of LSPI in the parameters of a condition
USE_HPO_PARAMS = True
EPSILON = (0.5, 0.05)       # exploration of the first and last iterations
N_SELECTION_EPISODES = 10
N_EVAL_EPISODES = 50
# ===========================================================================


class Features:
    """Standardized features [x, 1] of the observations of one environment (state of the pipeline, reset at the start
    of each episode)."""

    def __init__(self, pipeline, episodes: list):
        self.pipeline = pipeline
        self.pipeline.set_n_envs(1)
        X = np.concatenate([episode_features(pipeline, episode) for episode in episodes])
        self.mean, self.std = X.mean(axis=0), X.std(axis=0) + 1e-8

    def reset(self):
        self.pipeline.reset()

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        x = (self.pipeline.step(np.asarray(observation, dtype=float)[None, :])[0] - self.mean) / self.std
        return np.append(x, 1.0)


def play(env, features: Features, W, epsilon: float, n_steps: int, rng, seed: int, record: bool = True) -> tuple:
    """Episodes of the epsilon-greedy policy of Q(x, a) = W[a] · x (random if W is None) until n_steps steps (or
    n_steps episodes if not record). Returns the samples (X, actions, rewards, X', terminal) and the returns of the
    episodes."""
    n_actions = env.action_space.n
    X, actions, rewards, X_next, terminal, returns = [], [], [], [], [], []
    steps, k = 0, 0
    while (steps < n_steps) if record else (k < n_steps):
        observation, _ = env.reset(seed=seed + k)
        features.reset()
        x, done, total = features(observation), False, 0.0
        while not done:
            if W is None or rng.random() < epsilon:
                action = int(rng.integers(n_actions))
            else:
                action = int(np.argmax(W @ x))
            observation, reward, terminated, truncated, _ = env.step(action)
            x_next = features(observation)
            if record:
                X.append(x); actions.append(action); rewards.append(reward); X_next.append(x_next)
                terminal.append(terminated)
            x, total, done, steps = x_next, total + reward, terminated or truncated, steps + 1
        returns.append(total)
        k += 1
    samples = None if not record else (np.asarray(X, dtype=np.float32), np.asarray(actions),
                                       np.asarray(rewards, dtype=float), np.asarray(X_next, dtype=np.float32),
                                       np.asarray(terminal))
    return samples, returns


def lstdq(X, actions, rewards, X_next, terminal, W, n_actions: int, ridge: float = RIDGE,
          gamma: float = GAMMA) -> np.ndarray:
    """LSTD-Q for the greedy policy of W (actions of the next states), returns the new W (n_actions, F)."""
    F = X.shape[1]
    next_actions = np.argmax(X_next @ W.T, axis=1)
    A, b = np.zeros((n_actions * F, n_actions * F)), np.zeros(n_actions * F)
    for a in range(n_actions):
        rows = actions == a
        Xa = X[rows].astype(float)
        A[a * F:(a + 1) * F, a * F:(a + 1) * F] += Xa.T @ Xa
        b[a * F:(a + 1) * F] = Xa.T @ rewards[rows]
        for a_next in range(n_actions):
            pairs = rows & (next_actions == a_next) & ~terminal
            if pairs.any():
                A[a * F:(a + 1) * F, a_next * F:(a_next + 1) * F] -= gamma * (X[pairs].astype(float).T
                                                                              @ X_next[pairs].astype(float))
    A[np.diag_indices_from(A)] += ridge * len(X)
    return np.linalg.solve(A, b).reshape(n_actions, F)


def condition_params(condition):
    """Parameters of a condition (features, "ridge" and "gamma"): best of the hyperparameter optimization of LSPI if
    available (and USE_HPO_PARAMS), else the features of the best ones of PPO, else DEFAULT_PARAMS. The control "proj"
    takes the parameters of HAG."""
    kind = condition_kind(condition)
    if USE_HPO_PARAMS:
        from hag.rl.lspi import hpo  # (avoids a circular import)
        from hag.rl.ppo import hpo as ppo_hpo
        for best in (hpo.best_params(kind), ppo_hpo.best_params(kind)):
            if best is not None:
                return {key: value for key, value in best.items() if key != "learning_rate"}
    return DEFAULT_PARAMS[kind]


def run(condition, seed, episodes, params=None, save_curves=True, check=None, units=UNITS):
    """LSPI on the features of a condition, fitted on the pretraining episodes, and return the results of the run.
    params: parameters of the features, "ridge" and "gamma" (default: condition_params; RIDGE and GAMMA if missing).
    check: function called with the feature pipeline before the training (e.g. to stop a run whose reservoir is not
    suitable by raising an exception). units: reservoir size."""
    start = time.time()
    params = condition_params(condition) if params is None else params
    lspi = dict(dict(ridge=RIDGE, gamma=GAMMA), **{key: params[key] for key in LSPI_PARAMS if key in params})
    feature_params = {key: value for key, value in params.items() if key not in LSPI_PARAMS + ("learning_rate",)}
    pipeline = make_pipeline(condition, episodes, units, seed, feature_params)
    if check is not None:
        check(pipeline)
    features = Features(pipeline, episodes)
    env = make_env(ENV_ID)
    n_actions = env.action_space.n
    rng = np.random.default_rng(seed)
    samples, train_returns, curve, best = None, [], [], (-np.inf, None, -1)
    W = None
    for iteration in range(N_ITERATIONS):
        epsilon = EPSILON[0] + (EPSILON[1] - EPSILON[0]) * iteration / max(1, N_ITERATIONS - 1)
        new, returns = play(env, features, W, epsilon, STEPS_PER_ITERATION, rng,
                            seed=1_000_000 * seed + 10_000 * iteration)
        train_returns += returns
        samples = new if samples is None else tuple(np.concatenate([s, n]) for s, n in zip(samples, new))
        W = np.zeros((n_actions, len(samples[0][0]))) if W is None else W
        for _ in range(INNER_ITERATIONS):
            W = lstdq(*samples, W, n_actions, **lspi)
        _, selection = play(env, features, W, 0.0, N_SELECTION_EPISODES, rng, seed=500_000 + 100 * iteration,
                            record=False)
        curve.append(float(np.mean(selection)))
        if curve[-1] > best[0]:
            best = (curve[-1], W.copy(), iteration)
    _, evaluation = play(env, features, best[1], 0.0, N_EVAL_EPISODES, rng, seed=10_000 + seed, record=False)

    if save_curves:
        curves = RL_RESULTS / "rl_curves"
        curves.mkdir(parents=True, exist_ok=True)
        np.savez(curves / f"lspi_{RESULTS_NAME}_{condition}_seed{seed}.npz", returns=train_returns,
                 selection=curve)
    W_res = None if pipeline.reservoir is None else pipeline.reservoir["W"]
    returns = np.asarray(train_returns)
    return {"env": ENV_ID, "condition": condition, "seed": seed, "n_features": pipeline.n_features,
            "connections_per_neuron": np.nan if W_res is None else np.count_nonzero(W_res) / len(W_res),
            "eval_return_mean": float(np.mean(evaluation)), "eval_return_std": float(np.std(evaluation)),
            "train_return_mean": float(returns.mean()),
            "train_return_last10%": float(returns[-max(1, len(returns) // 10):].mean()),
            "greedy_return_mean": float(np.mean(curve)), "best_iteration": best[2], "selection_curve": curve,
            "time_s": round(time.time() - start), "params": dict(feature_params, **lspi)}


def main(conditions):
    output = RL_RESULTS / f"lspi_{RESULTS_NAME}.csv"
    rows = []
    with ProcessPoolExecutor(N_WORKERS, mp_context=get_context("spawn")) as pool:
        episodes = dict(zip(SEEDS, pool.map(pretraining_episodes, SEEDS)))
        futures = [pool.submit(run, condition, seed, episodes[seed])
                   for seed in SEEDS for condition in conditions]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(f"[lspi] {ENV_ID} {row['condition']:16s} seed {row['seed']}: eval "
                  f"{row['eval_return_mean']:7.3f} ± {row['eval_return_std']:5.3f} | train mean "
                  f"{row['train_return_mean']:7.3f}, last 10% {row['train_return_last10%']:7.3f} | best iteration "
                  f"{row['best_iteration']} | {row['time_s']}s",
                  flush=True)
            output.parent.mkdir(parents=True, exist_ok=True)
            rows.sort(key=lambda row: (row["seed"], conditions.index(row["condition"])))
            pd.DataFrame(rows).to_csv(output, index=False)
    results = pd.DataFrame(rows)
    metrics = ["eval_return_mean", "greedy_return_mean", "train_return_mean", "train_return_last10%"]
    print(results.groupby("condition")[metrics].agg(["mean", "std"]).round(3).to_string())
    print(f"Results saved to {output}")


if __name__ == "__main__":
    main(sys.argv[1:] or CONDITIONS)
