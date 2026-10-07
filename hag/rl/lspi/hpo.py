"""
Hyperparameter optimization of LSPI (hag.rl.lspi.train): regularization ("ridge") and discount ("gamma") of LSTD-Q and
parameters of the features, in the studies of hag.rl.search (search space of the features, parallel run, pruning of
HAG's trials without recurrence). Benchmarks with discrete actions only.
Objective (maximized): mean return of the greedy policies of the iterations of LSPI ("greedy_return_mean": sample
efficiency of the learned policies), averaged over the seeds of the search (number of trials: that of the benchmark,
see hag.rl.envs.BENCHMARKS; budget: N_ITERATIONS * STEPS_PER_ITERATION of hag.rl.lspi.train). Trials are pruned
(median rule) after each seed. The final evaluation episodes are never used during the search.

Studies: <RL_RESULTS>/lspi_hpo_<name>.sqlite3, name being hag.rl.experiment.RESULTS_NAME (study names:
hag.rl.search.STUDIES).
Run from the repository root:  HAG_RL_ENV=<benchmark> python -m hag.rl.lspi.hpo [study ...]  (all the studies by
default)
"""
import sys

import numpy as np
import optuna
from gymnasium import spaces

from hag.rl import search
from hag.rl.envs import make_env
from hag.rl.experiment import BENCHMARK, ENV_ID, RESULTS_NAME
from hag.rl.lspi import train

# =============================== PARAMETERS ===============================
N_TRIALS = BENCHMARK.n_trials              # trials per study
DB_NAME = f"lspi_hpo_{RESULTS_NAME}"       # database of the studies
# ===========================================================================


def suggest_params(trial: optuna.Trial, study_name: str) -> dict:
    """Parameters of a trial of a study: regularization and discount of LSTD-Q ("ridge", "gamma") and features (see
    hag.rl.search.suggest_features)."""
    lspi = dict(ridge=trial.suggest_float("ridge", 1e-6, 1.0, log=True),
                gamma=trial.suggest_float("gamma", 0.9, 0.999))
    return dict(lspi, **search.suggest_features(trial, study_name))


def trial_params(study_name: str, values: dict) -> dict:
    """Parameters of a trial from its values (e.g. the best one)."""
    return suggest_params(optuna.trial.FixedTrial(values), study_name)


def best_params(kind: str):
    """Best parameters of the studies of a kind (see hag.rl.search.best_params), or None."""
    return search.best_params(DB_NAME, kind, suggest_params)


def objective(trial: optuna.Trial, study_name: str, episodes: dict) -> float:
    params = suggest_params(trial, study_name)
    check = search.connections_check(trial) if search.study_kind(study_name) == "hag" else None
    returns, connections = [], []
    for k, seed in enumerate(range(search.SEED, search.SEED + search.N_SEEDS)):
        row = train.run(search.study_condition(study_name), seed, episodes[seed], params=params, save_curves=False,
                        check=check)
        returns.append(row["greedy_return_mean"])
        connections.append(row["connections_per_neuron"])
        trial.set_user_attr("connections_per_neuron", float(np.mean(connections)))
        trial.report(float(np.mean(returns)), k)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return float(np.mean(returns))


if __name__ == "__main__":
    if not isinstance(make_env(ENV_ID).action_space, spaces.Discrete):
        raise ValueError(f"LSPI needs discrete actions: {ENV_ID} has {make_env(ENV_ID).action_space}")
    search.optimize(DB_NAME, sys.argv[1:] or search.STUDIES, objective, N_TRIALS)
