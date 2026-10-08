"""
CMA-ES (Hansen & Ostermeier, 2001) on the linear readout of the features (see hag.rl.population), on the benchmark and
conditions of hag.rl.experiment (features fitted on its pretraining episodes): the readout is searched directly on the
returns of episodes, without gradient nor value function, as hag.rl.openai_es.

Separable CMA-ES (sep-CMA-ES, Ros & Hansen, 2008, "A simple modification in CMA-ES achieving linear time and space
complexity"): diagonal covariance matrix, with the learning rates of the covariance scaled by (d + 2) / 3, for the
thousands of parameters of the readouts (d = n_outputs * (F + 1)). Default population size 4 + floor(3 ln d), initial
mean zero, initial step size "sigma0" (searched). Each readout is evaluated on one episode, all the readouts of a
generation on the same episode (same reset seed), until TOTAL_TIMESTEPS environment steps. The mean of the search
distribution is evaluated N_SELECTIONS times during the training ("selection_curve"), the best one on N_EVAL_EPISODES
episodes (other seeds).
Parameters of a condition (condition_params): best ones of the hyperparameter optimization of CMA-ES (hag.rl.cmaes.hpo:
features and sigma0), else the features of the best ones of PPO (hag.rl.ppo.hpo) with SIGMA0, else the default ones
(hag.rl.experiment.DEFAULT_PARAMS).

Results: <RL_RESULTS>/cmaes_<name>.csv (columns of hag.rl.ppo.train and of the evolution strategies).
Run from the repository root:  HAG_RL_ENV=<benchmark> python -m hag.rl.cmaes.train [condition ...]
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
SIGMA0 = 0.1                # initial step size (default, searched by hag.rl.cmaes.hpo)
SEARCHED = ("sigma0",)
USE_HPO_PARAMS = True
N_SELECTIONS = 20
N_SELECTION_EPISODES = 10
N_EVAL_EPISODES = 50
# ===========================================================================


class SepCMAES:
    """Separable CMA-ES maximizing a fitness (ask: candidates (popsize, d), tell: their fitnesses)."""

    def __init__(self, d: int, sigma0: float, rng: np.random.Generator, popsize: int = None):
        self.d, self.sigma, self.rng = d, sigma0, rng
        self.popsize = popsize or 4 + int(3 * np.log(d))
        mu = self.popsize // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        self.weights = weights / weights.sum()
        self.mu_eff = 1 / np.sum(self.weights ** 2)
        self.c_sigma = (self.mu_eff + 2) / (d + self.mu_eff + 5)
        self.d_sigma = 1 + 2 * max(0.0, np.sqrt((self.mu_eff - 1) / (d + 1)) - 1) + self.c_sigma
        self.c_c = (4 + self.mu_eff / d) / (d + 4 + 2 * self.mu_eff / d)
        c_1 = 2 / ((d + 1.3) ** 2 + self.mu_eff)
        c_mu = min(1 - c_1, 2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((d + 2) ** 2 + self.mu_eff))
        # separable: faster learning of the diagonal (Ros & Hansen, 2008)
        self.c_1, self.c_mu = c_1 * (d + 2) / 3, c_mu * (d + 2) / 3
        if self.c_1 + self.c_mu > 1:
            self.c_1, self.c_mu = self.c_1 / (self.c_1 + self.c_mu), self.c_mu / (self.c_1 + self.c_mu)
        self.chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))  # E||N(0, I)||
        self.mean = np.zeros(d)
        self.C = np.ones(d)  # diagonal of the covariance
        self.p_sigma, self.p_c = np.zeros(d), np.zeros(d)
        self.generation = 0

    def ask(self) -> np.ndarray:
        self.y = self.rng.standard_normal((self.popsize, self.d)) * np.sqrt(self.C)
        return self.mean + self.sigma * self.y

    def tell(self, candidates: np.ndarray, fitnesses: np.ndarray):
        y = self.y[np.argsort(-np.asarray(fitnesses))[:len(self.weights)]]  # best first
        y_w = self.weights @ y
        self.mean = self.mean + self.sigma * y_w
        self.p_sigma = ((1 - self.c_sigma) * self.p_sigma
                        + np.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff) * y_w / np.sqrt(self.C))
        norm = np.linalg.norm(self.p_sigma)
        self.generation += 1
        threshold = (1.4 + 2 / (self.d + 1)) * self.chi_n
        h_sigma = norm / np.sqrt(1 - (1 - self.c_sigma) ** (2 * self.generation)) < threshold
        self.p_c = (1 - self.c_c) * self.p_c + h_sigma * np.sqrt(self.c_c * (2 - self.c_c) * self.mu_eff) * y_w
        self.C = ((1 - self.c_1 - self.c_mu) * self.C
                  + self.c_1 * (self.p_c ** 2 + (1 - h_sigma) * self.c_c * (2 - self.c_c) * self.C)
                  + self.c_mu * (self.weights @ y ** 2))
        self.sigma *= np.exp(self.c_sigma / self.d_sigma * (norm / self.chi_n - 1))


def condition_params(condition):
    """Parameters of a condition (features and sigma0): best of the hyperparameter optimization of CMA-ES if available
    (and USE_HPO_PARAMS), else the features of the best ones of PPO, else DEFAULT_PARAMS. The control "proj" takes the
    parameters of HAG."""
    kind = condition_kind(condition)
    if USE_HPO_PARAMS:
        from hag.rl.cmaes import hpo  # (avoids a circular import)
        from hag.rl.ppo import hpo as ppo_hpo
        for best in (hpo.best_params(kind), ppo_hpo.best_params(kind)):
            if best is not None:
                return {key: value for key, value in best.items() if key != "learning_rate"}
    return DEFAULT_PARAMS[kind]


def run(condition, seed, episodes, params=None, total_timesteps=TOTAL_TIMESTEPS, save_curves=True, check=None,
        units=UNITS):
    """CMA-ES on the readout of the features of a condition, fitted on the pretraining episodes, and return the results
    of the run. params: parameters of the features and "sigma0" (default: condition_params; SIGMA0 if missing). check:
    function called with the feature pipeline before the training. units: reservoir size."""
    start = time.time()
    params = condition_params(condition) if params is None else params
    sigma0 = params.get("sigma0", SIGMA0)
    feature_params = {key: value for key, value in params.items() if key not in SEARCHED + ("learning_rate",)}
    pipeline = make_pipeline(condition, episodes, units, seed, feature_params)
    if check is not None:
        check(pipeline)
    features = Features(pipeline, episodes)
    d = n_outputs(make_env(ENV_ID).action_space) * features.n_features
    strategy = SepCMAES(d, sigma0, np.random.default_rng(seed))
    row = run_strategy(strategy, ENV_ID, features, seed, total_timesteps, N_SELECTIONS, N_SELECTION_EPISODES,
                       N_EVAL_EPISODES)
    if save_curves:
        curves = RL_RESULTS / "rl_curves"
        curves.mkdir(parents=True, exist_ok=True)
        np.savez(curves / f"cmaes_{RESULTS_NAME}_{condition}_seed{seed}.npz", selection=row["selection_curve"])
    W = None if pipeline.reservoir is None else pipeline.reservoir["W"]
    return dict({"env": ENV_ID, "condition": condition, "seed": seed, "n_features": pipeline.n_features,
                 "connections_per_neuron": np.nan if W is None else np.count_nonzero(W) / len(W)}, **row,
                time_s=round(time.time() - start), params=dict(feature_params, sigma0=sigma0))


def main(conditions):
    output = RL_RESULTS / f"cmaes_{RESULTS_NAME}.csv"
    rows = []
    with ProcessPoolExecutor(N_WORKERS, mp_context=get_context("spawn")) as pool:
        episodes = dict(zip(SEEDS, pool.map(pretraining_episodes, SEEDS)))
        futures = [pool.submit(run, condition, seed, episodes[seed]) for seed in SEEDS for condition in conditions]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(f"[cmaes] {ENV_ID} {row['condition']:16s} seed {row['seed']}: eval {row['eval_return_mean']:7.3f} ± "
                  f"{row['eval_return_std']:5.3f} | train mean {row['train_return_mean']:7.3f} | {row['generations']} "
                  f"generations | {row['time_s']}s", flush=True)
            output.parent.mkdir(parents=True, exist_ok=True)
            rows.sort(key=lambda row: (row["seed"], conditions.index(row["condition"])))
            pd.DataFrame(rows).to_csv(output, index=False)
    results = pd.DataFrame(rows)
    metrics = ["eval_return_mean", "greedy_return_mean", "train_return_mean"]
    print(results.groupby("condition")[metrics].agg(["mean", "std"]).round(3).to_string())
    print(f"Results saved to {output}")


if __name__ == "__main__":
    main(sys.argv[1:] or CONDITIONS)
