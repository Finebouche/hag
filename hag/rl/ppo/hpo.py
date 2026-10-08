"""
Hyperparameter optimization of PPO (hag.rl.ppo.train): learning rate of PPO and parameters of the features, in the
studies of hag.rl.search (search space of the features, parallel run, pruning of HAG's trials without recurrence).
Objective (maximized): mean return over all the training episodes of PPO (sample efficiency), averaged over the seeds
of the search, with a training budget of TOTAL_TIMESTEPS (budget and number of trials: those of the benchmark, see
hag.rl.envs.BENCHMARKS). Trials are pruned (median rule) after each seed. The policy is never evaluated on its
evaluation episodes during the search.

Studies: <RL_RESULTS>/rl_hpo_<name>.sqlite3, name being hag.rl.experiment.RESULTS_NAME (study names:
hag.rl.search.STUDIES).
Run from the repository root:  HAG_RL_ENV=<benchmark> python -m hag.rl.ppo.hpo [study ...]  (all the studies by
default)
"""
import sys

import numpy as np
import optuna

from hag.rl import search
from hag.rl.experiment import BENCHMARK, RESULTS_NAME
from hag.rl.ppo import train

# =============================== PARAMETERS ===============================
N_TRIALS = BENCHMARK.n_trials              # trials per study
TOTAL_TIMESTEPS = BENCHMARK.hpo_timesteps  # PPO budget per seed
DB_PREFIX = "rl_hpo"
DB_NAME = f"{DB_PREFIX}_{RESULTS_NAME}"         # database of the studies
# ===========================================================================


def suggest_params(trial: optuna.Trial, study_name: str) -> dict:
    """Parameters of a trial of a study: learning rate of PPO ("learning_rate") and features (see
    hag.rl.search.suggest_features)."""
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    return dict(learning_rate=learning_rate, **search.suggest_features(trial, study_name))


def trial_params(study_name: str, values: dict) -> dict:
    """Parameters of a trial from its values (e.g. the best one)."""
    return suggest_params(optuna.trial.FixedTrial(values), study_name)


def best_params(kind: str, db_name: str = DB_NAME, root=None):
    """Best parameters of the studies of a kind in the database db_name (default: that of the benchmark) of the folder
    root (default: RL_RESULTS), see hag.rl.search.best_params, or None."""
    return search.best_params(db_name, kind, suggest_params, root)


def objective(trial: optuna.Trial, study_name: str, episodes: dict) -> float:
    params = suggest_params(trial, study_name)
    check = search.connections_check(trial) if search.study_kind(study_name) == "hag" else None
    returns, connections = [], []
    for k, seed in enumerate(range(search.SEED, search.SEED + search.N_SEEDS)):
        row = train.run(search.study_condition(study_name), seed, episodes[seed], params=params,
                        total_timesteps=TOTAL_TIMESTEPS, evaluate=False, save_curves=False, check=check)
        returns.append(row["train_return_mean"])
        connections.append(row["connections_per_neuron"])
        trial.set_user_attr("connections_per_neuron", float(np.mean(connections)))
        trial.report(float(np.mean(returns)), k)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return float(np.mean(returns))


if __name__ == "__main__":
    search.optimize(DB_NAME, sys.argv[1:] or search.STUDIES, objective, N_TRIALS)
