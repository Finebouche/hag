"""
Natural actor-critic with an LSTD-Q(lambda) critic (Peters & Schaal, 2008, "Natural actor-critic", Neurocomputing 71),
on the benchmark and conditions of hag.rl.experiment (features fitted on its pretraining episodes), instead of PPO
(hag.rl.ppo): a policy (actor) updated along the natural gradient, given in closed form by a least-squares critic, as
LSPI's (hag.rl.lspi) Q-function. Benchmarks with discrete actions only.

Policy: softmax of linear scores, pi(a | x) ~ exp(theta_a · [x, 1]), x being the features (standardized with the
statistics of the pretraining episodes, see hag.rl.lspi.train.Features), theta starting at zero (uniform policy).
Critic: Q(x, a) = v · [x, 1] + w · grad_theta log pi(a | x), the state value plus the advantage in the compatible
features of the policy, whose weights w are the natural gradient. Each update collects at least STEPS_PER_UPDATE steps
of complete episodes of the policy, solves LSTD-Q(lambda) on them,
    z_t = lambda z_{t-1} + phi_t,  A = sum z_t (phi_t - gamma phi'_t)^T,  b = sum z_t r_t,
    [v, w] = (A + ridge n I)^-1 b
(phi_t = [[x_t, 1], grad log pi(a_t | x_t)], phi'_t = [[x_{t+1}, 1], 0], zero at the end of a terminated episode; the
traces z restart at each episode), then moves the policy by theta <- theta + "step_size" * w / ||w||, until
TOTAL_TIMESTEPS environment steps. The greedy policy is evaluated N_SELECTIONS times during the training
("selection_curve"), the best one on N_EVAL_EPISODES episodes (other seeds).
Parameters of a condition (condition_params): best ones of the hyperparameter optimization of NAC (hag.rl.nac.hpo:
features, "step_size", "lam", "ridge"), else the features of the best ones of PPO (hag.rl.ppo.hpo) with the defaults,
else the default ones (hag.rl.experiment.DEFAULT_PARAMS).

Results: <RL_RESULTS>/nac_<name>.csv (columns of hag.rl.ppo.train and of NAC).
Run from the repository root:  HAG_RL_ENV=<benchmark> python -m hag.rl.nac.train [condition ...]
"""
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

import numpy as np
import pandas as pd

from hag.rl.envs import make_env
from hag.rl.experiment import (BENCHMARK, CONDITIONS, DEFAULT_PARAMS, ENV_ID, RESULTS_NAME, SEEDS, UNITS,
                               condition_kind, pretraining_episodes)
from hag.rl.lspi.train import Features
from hag.rl.pretrain import make_pipeline
from hag.rl.utils import N_WORKERS, RL_RESULTS

# =============================== PARAMETERS ===============================
TOTAL_TIMESTEPS = BENCHMARK.train_timesteps
STEPS_PER_UPDATE = 10_000
GAMMA = 0.99
LAM = 0.9                   # lambda of the eligibility traces of LSTD-Q(lambda) (default, searched)
RIDGE = 1e-3                # regularization of LSTD-Q, relative to the number of samples (default, searched)
STEP_SIZE = 0.1             # norm of the update of theta (default, searched)
NAC_PARAMS = ("step_size", "lam", "ridge")
USE_HPO_PARAMS = True
N_SELECTIONS = 20
N_SELECTION_EPISODES = 10
N_EVAL_EPISODES = 50
# ===========================================================================


def softmax(scores: np.ndarray) -> np.ndarray:
    e = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def play(env, features: Features, theta: np.ndarray, n: int, rng, seed: int, greedy: bool = False) -> tuple:
    """Episodes of the policy theta (stochastic, or greedy) until n steps (n episodes if greedy). Returns the episodes
    (list of (X (T, F + 1), actions (T,), rewards (T,), X_next (T, F + 1), terminated)) and their returns."""
    episodes, returns, steps, k = [], [], 0, 0
    while (k < n) if greedy else (steps < n):
        observation, _ = env.reset(seed=seed + k)
        features.reset()
        x, done, X, actions, rewards, X_next = features(observation), False, [], [], [], []
        terminated = False
        while not done:
            scores = theta @ x
            action = int(np.argmax(scores)) if greedy else int(rng.choice(len(scores), p=softmax(scores)))
            observation, reward, terminated, truncated, _ = env.step(action)
            x_next = features(observation)
            X.append(x), actions.append(action), rewards.append(reward), X_next.append(x_next)
            x, done = x_next, terminated or truncated
        episodes.append((np.asarray(X), np.asarray(actions), np.asarray(rewards, dtype=float), np.asarray(X_next),
                         terminated))
        returns.append(float(np.sum(rewards)))
        steps, k = steps + len(X), k + 1
    return episodes, returns


def natural_gradient(episodes: list, theta: np.ndarray, gamma: float, lam: float, ridge: float) -> np.ndarray:
    """Natural gradient w (shape of theta) of the policy theta, from LSTD-Q(lambda) on episodes of the policy."""
    n_actions, F = theta.shape
    d = F + n_actions * F
    A, b, n = np.zeros((d, d)), np.zeros(d), 0
    for X, actions, rewards, X_next, terminated in episodes:
        pi = softmax(X @ theta.T)
        grad_log = (np.eye(n_actions)[actions] - pi)[:, :, None] * X[:, None, :]  # (T, n_actions, F)
        phi = np.concatenate([X, grad_log.reshape(len(X), -1)], axis=1)
        phi_next = np.zeros_like(phi)
        phi_next[:, :F] = X_next
        if terminated:
            phi_next[-1] = 0.0
        z, Z = np.zeros(d), np.empty_like(phi)
        for t in range(len(phi)):
            z = lam * z + phi[t]
            Z[t] = z
        A += Z.T @ (phi - gamma * phi_next)
        b += Z.T @ rewards
        n += len(X)
    A[np.diag_indices_from(A)] += ridge * n
    return np.linalg.solve(A, b)[F:].reshape(theta.shape)


def condition_params(condition):
    """Parameters of a condition (features, "step_size", "lam", "ridge"): best of the hyperparameter optimization of
    NAC if available (and USE_HPO_PARAMS), else the features of the best ones of PPO, else DEFAULT_PARAMS. The control
    "proj" takes the parameters of HAG."""
    kind = condition_kind(condition)
    if USE_HPO_PARAMS:
        from hag.rl.nac import hpo  # (avoids a circular import)
        from hag.rl.ppo import hpo as ppo_hpo
        for best in (hpo.best_params(kind), ppo_hpo.best_params(kind)):
            if best is not None:
                return {key: value for key, value in best.items() if key != "learning_rate"}
    return DEFAULT_PARAMS[kind]


def run(condition, seed, episodes, params=None, total_timesteps=TOTAL_TIMESTEPS, save_curves=True, check=None,
        units=UNITS):
    """NAC on the features of a condition, fitted on the pretraining episodes, and return the results of the run.
    params: parameters of the features, "step_size", "lam" and "ridge" (default: condition_params; STEP_SIZE, LAM and
    RIDGE if missing). check: function called with the feature pipeline before the training. units: reservoir size."""
    start = time.time()
    params = condition_params(condition) if params is None else params
    nac = dict(dict(step_size=STEP_SIZE, lam=LAM, ridge=RIDGE),
               **{key: params[key] for key in NAC_PARAMS if key in params})
    feature_params = {key: value for key, value in params.items() if key not in NAC_PARAMS + ("learning_rate",)}
    pipeline = make_pipeline(condition, episodes, units, seed, feature_params)
    if check is not None:
        check(pipeline)
    features = Features(pipeline, episodes)
    env = make_env(ENV_ID)
    rng = np.random.default_rng(seed)
    theta = np.zeros((env.action_space.n, len(features.mean) + 1))
    steps, update, train_returns, curve, best = 0, 0, [], [], (-np.inf, None, -1)
    next_selection = 0
    while steps < total_timesteps:
        batch, returns = play(env, features, theta, STEPS_PER_UPDATE, rng, seed=1_000_000 * seed + 10_000 * update)
        train_returns += returns
        steps += sum(len(episode[0]) for episode in batch)
        w = natural_gradient(batch, theta, GAMMA, nac["lam"], nac["ridge"])
        theta = theta + nac["step_size"] * w / (np.linalg.norm(w) + 1e-12)
        update += 1
        if steps >= next_selection or steps >= total_timesteps:
            _, selection = play(env, features, theta, N_SELECTION_EPISODES, rng, seed=500_000 + 100 * len(curve),
                                greedy=True)
            curve.append(float(np.mean(selection)))
            if curve[-1] > best[0]:
                best = (curve[-1], theta.copy(), steps)
            next_selection += total_timesteps / N_SELECTIONS
    _, evaluation = play(env, features, best[1], N_EVAL_EPISODES, rng, seed=10_000 + seed, greedy=True)

    if save_curves:
        curves = RL_RESULTS / "rl_curves"
        curves.mkdir(parents=True, exist_ok=True)
        np.savez(curves / f"nac_{RESULTS_NAME}_{condition}_seed{seed}.npz", returns=train_returns, selection=curve)
    W = None if pipeline.reservoir is None else pipeline.reservoir["W"]
    train = np.asarray(train_returns)
    return {"env": ENV_ID, "condition": condition, "seed": seed, "n_features": pipeline.n_features,
            "connections_per_neuron": np.nan if W is None else np.count_nonzero(W) / len(W),
            "eval_return_mean": float(np.mean(evaluation)), "eval_return_std": float(np.std(evaluation)),
            "train_return_mean": float(train.mean()),
            "train_return_last10%": float(train[-max(1, len(train) // 10):].mean()),
            "greedy_return_mean": float(np.mean(curve)), "best_timestep": best[2], "selection_curve": curve,
            "updates": update, "time_s": round(time.time() - start), "params": dict(feature_params, **nac)}


def main(conditions):
    output = RL_RESULTS / f"nac_{RESULTS_NAME}.csv"
    rows = []
    with ProcessPoolExecutor(N_WORKERS, mp_context=get_context("spawn")) as pool:
        episodes = dict(zip(SEEDS, pool.map(pretraining_episodes, SEEDS)))
        futures = [pool.submit(run, condition, seed, episodes[seed]) for seed in SEEDS for condition in conditions]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(f"[nac] {ENV_ID} {row['condition']:16s} seed {row['seed']}: eval {row['eval_return_mean']:7.3f} ± "
                  f"{row['eval_return_std']:5.3f} | train mean {row['train_return_mean']:7.3f} | {row['updates']} "
                  f"updates | {row['time_s']}s", flush=True)
            output.parent.mkdir(parents=True, exist_ok=True)
            rows.sort(key=lambda row: (row["seed"], conditions.index(row["condition"])))
            pd.DataFrame(rows).to_csv(output, index=False)
    results = pd.DataFrame(rows)
    metrics = ["eval_return_mean", "greedy_return_mean", "train_return_mean"]
    print(results.groupby("condition")[metrics].agg(["mean", "std"]).round(3).to_string())
    print(f"Results saved to {output}")


if __name__ == "__main__":
    main(sys.argv[1:] or CONDITIONS)
