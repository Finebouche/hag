"""
OpenAI evolution strategy (Salimans et al., 2017, "Evolution strategies as a scalable alternative to reinforcement
learning") on the linear readout of the features (see hag.rl.population), on the benchmark and conditions of
hag.rl.experiment (features fitted on its pretraining episodes): the readout is searched directly on the returns of
episodes, without value function.

Each generation evaluates N_PAIRS antithetic pairs of perturbations of the readout, theta +- SIGMA * epsilon (one
episode each, all on the same episode: same reset seed), and moves theta along the gradient estimated from their ranks
(fitness shaping), with Adam (learning rate "learning_rate", searched) and an L2 penalty L2, until TOTAL_TIMESTEPS
environment steps. theta starts at zero. theta is evaluated N_SELECTIONS times during the training
("selection_curve"), the best one on N_EVAL_EPISODES episodes (other seeds).
Parameters of a condition (condition_params): best ones of the hyperparameter optimization of OpenAI-ES
(hag.rl.openai_es.hpo: features and learning rate), else the features of the best ones of PPO (hag.rl.ppo.hpo) with
LEARNING_RATE, else the default ones (hag.rl.experiment.DEFAULT_PARAMS).

Results: <RL_RESULTS>/openai_es_<name>.csv (columns of hag.rl.ppo.train and of the evolution strategies).
Run from the repository root:  HAG_RL_ENV=<benchmark> python -m hag.rl.openai_es.train [condition ...]
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
from hag.rl.population import Features, n_outputs, run_strategy
from hag.rl.pretrain import make_pipeline
from hag.rl.utils import N_WORKERS, RL_RESULTS

# =============================== PARAMETERS ===============================
TOTAL_TIMESTEPS = BENCHMARK.train_timesteps
N_PAIRS = 20                # antithetic pairs per generation (population of 2 * N_PAIRS)
SIGMA = 0.05                # standard deviation of the perturbations
LEARNING_RATE = 0.01        # of Adam (default, searched by hag.rl.openai_es.hpo)
L2 = 0.005                  # L2 penalty of the readout
SEARCHED = ("learning_rate",)
USE_HPO_PARAMS = True
N_SELECTIONS = 20
N_SELECTION_EPISODES = 10
N_EVAL_EPISODES = 50
# ===========================================================================


class OpenAIES:
    """OpenAI evolution strategy maximizing a fitness (ask: candidates (popsize, d), tell: their fitnesses)."""

    def __init__(self, d: int, learning_rate: float, rng: np.random.Generator, sigma: float = SIGMA,
                 n_pairs: int = N_PAIRS, l2: float = L2):
        self.d, self.learning_rate, self.rng = d, learning_rate, rng
        self.sigma, self.n_pairs, self.l2 = sigma, n_pairs, l2
        self.popsize = 2 * n_pairs
        self.mean = np.zeros(d)
        self.m, self.v, self.t = np.zeros(d), np.zeros(d), 0  # Adam

    def ask(self) -> np.ndarray:
        self.epsilon = self.rng.standard_normal((self.n_pairs, self.d))
        return np.concatenate([self.mean + self.sigma * self.epsilon, self.mean - self.sigma * self.epsilon])

    def tell(self, candidates: np.ndarray, fitnesses: np.ndarray):
        ranks = np.empty(len(fitnesses))
        ranks[np.argsort(fitnesses)] = np.arange(len(fitnesses))
        ranks = ranks / (len(fitnesses) - 1) - 0.5  # centered ranks in [-0.5, 0.5]
        gradient = (ranks[:self.n_pairs] - ranks[self.n_pairs:]) @ self.epsilon / (2 * self.n_pairs * self.sigma)
        gradient -= self.l2 * self.mean
        self.t += 1
        self.m = 0.9 * self.m + 0.1 * gradient
        self.v = 0.999 * self.v + 0.001 * gradient ** 2
        m_hat, v_hat = self.m / (1 - 0.9 ** self.t), self.v / (1 - 0.999 ** self.t)
        self.mean = self.mean + self.learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)  # (ascent)


def condition_params(condition):
    """Parameters of a condition (features and learning rate): best of the hyperparameter optimization of OpenAI-ES if
    available (and USE_HPO_PARAMS), else the features of the best ones of PPO, else DEFAULT_PARAMS. The control "proj"
    takes the parameters of HAG."""
    kind = condition_kind(condition)
    if USE_HPO_PARAMS:
        from hag.rl.openai_es import hpo  # (avoids a circular import)
        from hag.rl.ppo import hpo as ppo_hpo
        best = hpo.best_params(kind)
        if best is not None:
            return best
        best = ppo_hpo.best_params(kind)
        if best is not None:
            return {key: value for key, value in best.items() if key != "learning_rate"}
    return DEFAULT_PARAMS[kind]


def run(condition, seed, episodes, params=None, total_timesteps=TOTAL_TIMESTEPS, save_curves=True, check=None,
        units=UNITS):
    """OpenAI-ES on the readout of the features of a condition, fitted on the pretraining episodes, and return the
    results of the run. params: parameters of the features and "learning_rate" (default: condition_params;
    LEARNING_RATE if missing). check: function called with the feature pipeline before the training. units: reservoir
    size."""
    start = time.time()
    params = condition_params(condition) if params is None else params
    learning_rate = params.get("learning_rate", LEARNING_RATE)
    feature_params = {key: value for key, value in params.items() if key not in SEARCHED}
    pipeline = make_pipeline(condition, episodes, units, seed, feature_params)
    if check is not None:
        check(pipeline)
    features = Features(pipeline, episodes)
    d = n_outputs(make_env(ENV_ID).action_space) * features.n_features
    strategy = OpenAIES(d, learning_rate, np.random.default_rng(seed))
    row = run_strategy(strategy, ENV_ID, features, seed, total_timesteps, N_SELECTIONS, N_SELECTION_EPISODES,
                       N_EVAL_EPISODES)
    if save_curves:
        curves = RL_RESULTS / "rl_curves"
        curves.mkdir(parents=True, exist_ok=True)
        np.savez(curves / f"openai_es_{RESULTS_NAME}_{condition}_seed{seed}.npz", selection=row["selection_curve"])
    W = None if pipeline.reservoir is None else pipeline.reservoir["W"]
    return dict({"env": ENV_ID, "condition": condition, "seed": seed, "n_features": pipeline.n_features,
                 "connections_per_neuron": np.nan if W is None else np.count_nonzero(W) / len(W)}, **row,
                time_s=round(time.time() - start), params=dict(feature_params, learning_rate=learning_rate))


def main(conditions):
    output = RL_RESULTS / f"openai_es_{RESULTS_NAME}.csv"
    rows = []
    with ProcessPoolExecutor(N_WORKERS, mp_context=get_context("spawn")) as pool:
        episodes = dict(zip(SEEDS, pool.map(pretraining_episodes, SEEDS)))
        futures = [pool.submit(run, condition, seed, episodes[seed]) for seed in SEEDS for condition in conditions]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(f"[openai_es] {ENV_ID} {row['condition']:16s} seed {row['seed']}: eval "
                  f"{row['eval_return_mean']:7.3f} ± {row['eval_return_std']:5.3f} | train mean "
                  f"{row['train_return_mean']:7.3f} | {row['generations']} generations | {row['time_s']}s", flush=True)
            output.parent.mkdir(parents=True, exist_ok=True)
            rows.sort(key=lambda row: (row["seed"], conditions.index(row["condition"])))
            pd.DataFrame(rows).to_csv(output, index=False)
    results = pd.DataFrame(rows)
    metrics = ["eval_return_mean", "greedy_return_mean", "train_return_mean"]
    print(results.groupby("condition")[metrics].agg(["mean", "std"]).round(3).to_string())
    print(f"Results saved to {output}")


if __name__ == "__main__":
    main(sys.argv[1:] or CONDITIONS)
