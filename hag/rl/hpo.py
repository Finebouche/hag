"""
Hyperparameter optimization (Optuna, multivariate TPE sampler) of the conditions of hag.rl.train: learning rate of
PPO and parameters of the features, one study per kind of features and, for the reservoirs, per decomposition of the
filter bank (DECOMPOSITIONS: "ema" or "legendre"):
  - "obs": scaled observations (learning rate only),
  - "filterbank": filter bank alone (decomposition searched),
  - "esn_<decomposition>": random reservoir (spectral radius, connectivity),
  - "hag_mean_<decomposition>", "hag_variance_<decomposition>": HAG with mean or variance homeostasis, target (and
    spread of the mean) in the ranges of HAG's studies on the classification datasets (hag/hpo/hpo_esn.py), spread of
    the variance below the target (else HAG cannot grow connections). HAG must use its
    recurrence: a trial whose reservoir has less than MIN_HAG_CONNECTIONS connections per neuron is pruned before the
    training of PPO,
all of them but "obs" with the number of filters and the slowest time scale of the filter bank, the reservoirs also
with the input scaling (down to the low values of HAG's studies), the bias (distribution and scaling) and the leak rate.
Objective (maximized): mean return over all the training episodes of PPO (sample efficiency), averaged over N_SEEDS
seeds, with a training budget of TOTAL_TIMESTEPS (budget and number of trials: those of the benchmark, see
hag.rl.envs.BENCHMARKS). Trials are pruned (median rule) after each seed. The policy is never evaluated on its
evaluation episodes during the search. The number of connections per neuron of the reservoirs is saved as attribute of
the trials ("connections_per_neuron").
The studies run at the same time, the N_WORKERS parallel processes being shared between them: the workers of a study
share it (TPE samplers of different seeds, with constant liar: the parameters of the running trials are avoided).

The pretraining episodes are those of hag.rl.train (see PRETRAIN_POLICY).
Studies: outputs/rl_results/rl_hpo_<name>.sqlite3, name being hag.rl.train.RESULTS_NAME (study names: STUDIES).
Run from the repository root:  HAG_RL_ENV=<benchmark> python -m hag.rl.hpo [study ...]  (all the studies by default)
"""
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
import optuna

from hag.rl.utils import N_WORKERS, RL_RESULTS
from hag.rl import train

# =============================== PARAMETERS ===============================
N_TRIALS = train.BENCHMARK.n_trials              # trials per study
N_SEEDS = 3                 # seeds per trial (the pretraining episodes and PPO of seed k are the same for all trials)
TOTAL_TIMESTEPS = train.BENCHMARK.hpo_timesteps  # PPO budget per seed
MIN_HAG_CONNECTIONS = 2     # minimal number of connections per neuron of HAG's reservoirs (else the trial is pruned)
SEED = 2026                 # seeds of the search: SEED, SEED + 1, ... (different from the seeds of hag.rl.train)
DECOMPOSITIONS = ("ema", "legendre")  # decompositions of the filter bank (one study per reservoir and decomposition)
STUDIES = ("obs", "filterbank") + tuple(f"{kind}_{decomposition}" for kind in ("esn", "hag_mean", "hag_variance")
                                  for decomposition in DECOMPOSITIONS)
# ===========================================================================

warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)  # (multivariate TPE)


def storage_path(name: str):
    return RL_RESULTS / f"rl_hpo_{name}.sqlite3"


def storage(name: str) -> optuna.storages.RDBStorage:
    # timeout: the workers write to the same database
    return optuna.storages.RDBStorage(f"sqlite:///{storage_path(name)}",
                                      engine_kwargs={"connect_args": {"timeout": 60}})


def suggest_params(trial: optuna.Trial, study_name: str) -> dict:
    """Parameters of a trial of a study (see STUDIES): learning rate of PPO ("learning_rate"), filter bank (see
    hag.rl.features.make_filter_bank) and reservoir (see hag.rl.pretrain.build_reservoir)."""
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    if study_name == "obs":
        return dict(learning_rate=learning_rate)
    if study_name == "filterbank":
        decomposition = trial.suggest_categorical("decomposition", list(DECOMPOSITIONS))
    else:
        study_name, decomposition = study_name.rsplit("_", 1)
    params = dict(learning_rate=learning_rate, decomposition=decomposition,
                  n_filters=trial.suggest_categorical("n_filters", [4, 6, 8, 12]),
                  slowest=trial.suggest_float("slowest", 0.005, 0.1, log=True))
    if study_name == "filterbank":
        return params
    params.update(input_scaling=trial.suggest_float("input_scaling", 0.005, 1.0, log=True),
                  bias_dist=trial.suggest_categorical("bias_dist", ["foldnorm", "uniform"]),
                  bias_scaling=trial.suggest_float("bias_scaling", 0.0, 0.5),
                  lr=trial.suggest_float("lr", 0.1, 1.0))
    if study_name == "esn":
        params.update(sr=trial.suggest_float("sr", 0.1, 1.5),
                      rc_connectivity=trial.suggest_float("rc_connectivity", 0.01, 0.5, log=True))
        return params
    min_window = trial.suggest_int("min_window", 3, 20)
    max_partners = trial.suggest_categorical("max_partners", [5, 10, 20, 50, None])  # None: no limit
    params.update(weight_increment=trial.suggest_float("weight_increment", 0.001, 0.5, log=True),
                  min_window=min_window, max_window=trial.suggest_int("max_window", min_window, 60),
                  use_full_instance=trial.suggest_categorical("use_full_instance", [True, False]),
                  max_partners=np.inf if max_partners is None else max_partners)
    # target and spread: ranges of HAG's studies (hag/hpo/hpo_esn.py)
    if study_name == "hag_mean":
        params.update(homeostasis="mean", target=trial.suggest_float("target_rate", 0.5, 1, step=0.01),
                      spread=trial.suggest_float("rate_spread", 0.01, 0.4, step=0.005))
    else:
        # spread: a fraction of the target (HAG's studies: 0.001 to 0.05, often above the target), so that HAG can grow
        # connections: it adds one to the neurons whose standard deviation is below target - spread
        target = trial.suggest_float("variance_target", 0.001, 0.02, step=0.001)
        params.update(homeostasis="variance", target=target,
                      spread=target * trial.suggest_float("variance_spread_ratio", 0.05, 0.9),
                      intrinsic_saturation=trial.suggest_float("intrinsic_saturation", 0.8, 0.98),
                      intrinsic_coef=trial.suggest_float("intrinsic_coef", 0.8, 0.98))
    return params


def trial_params(study_name: str, values: dict) -> dict:
    """Parameters of the features from the parameters of a trial (e.g. the best one)."""
    return suggest_params(optuna.trial.FixedTrial(values), study_name)


def best_params(kind: str):
    """Best parameters of the studies of a kind ("obs", "filterbank", "esn": best of the "esn_*" studies, or "hag": best
    of the "hag_mean_*" and "hag_variance_*" studies), or None if there is no completed study."""
    name = train.RESULTS_NAME
    if not storage_path(name).exists():  # (load_study would create an empty database)
        return None
    names = [study for study in STUDIES if study.split("_")[0] == kind]
    best = None
    for study_name in names:
        try:
            study = optuna.load_study(study_name=study_name, storage=storage(name))
            if best is None or study.best_value > best[0]:
                best = (study.best_value, study_name, study.best_params)
        except (KeyError, ValueError):  # study missing or without completed trial
            continue
    return None if best is None else trial_params(best[1], best[2])


def objective(trial: optuna.Trial, study_name: str, episodes: list) -> float:
    params = suggest_params(trial, study_name)
    kind = study_name.split("_")[0]
    condition = {"obs": "obs", "filterbank": "filterbank", "esn": "filterbank+esn", "hag": "filterbank+hag"}[kind]
    returns, connections = [], []

    def check(pipeline):
        """Prune the trial if its HAG reservoir does not use its recurrence (before the training of PPO)."""
        W = pipeline.reservoir["W"]
        trial.set_user_attr("connections_per_neuron", float(np.count_nonzero(W) / len(W)))
        if np.count_nonzero(W) / len(W) < MIN_HAG_CONNECTIONS:
            trial.set_user_attr("pruned_reason", "too few connections")
            raise optuna.TrialPruned()

    for k, seed in enumerate(range(SEED, SEED + N_SEEDS)):
        row = train.run(condition, seed, episodes[seed], params=params, total_timesteps=TOTAL_TIMESTEPS,
                        evaluate=False, save_curves=False, check=check if kind == "hag" else None)
        returns.append(row["train_return_mean"])
        connections.append(row["connections_per_neuron"])
        trial.set_user_attr("connections_per_neuron", float(np.mean(connections)))
        trial.report(float(np.mean(returns)), k)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return float(np.mean(returns))


def worker(name: str, n_trials: int, worker: int, episodes: dict):
    """Run n_trials trials of the study name (in a worker process; worker: its index, seed of its sampler)."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.load_study(study_name=name, storage=storage(train.RESULTS_NAME),
                              sampler=optuna.samplers.TPESampler(seed=SEED + worker, constant_liar=True,
                                                                 multivariate=True),
                              pruner=optuna.pruners.MedianPruner(n_startup_trials=10))

    def log(study, trial):
        value = (f"{trial.value:6.1f}" if trial.value is not None
                 else f"pruned ({trial.user_attrs.get('pruned_reason', 'median rule')})")
        try:
            best = f"{study.best_value:6.1f}"
        except ValueError:  # no completed trial yet
            best = "-"
        print(f"[hpo] {train.RESULTS_NAME} {name} trial {trial.number:3d}: {value} | best {best}", flush=True)

    # catch: a failing trial is recorded as failed, the study goes on
    study.optimize(lambda trial: objective(trial, name, episodes), n_trials=n_trials, callbacks=[log],
                   catch=(ValueError,))


def main(study_names):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    RL_RESULTS.mkdir(parents=True, exist_ok=True)
    for name in study_names:
        if name not in STUDIES:
            raise ValueError(f"Unknown study {name!r}: {STUDIES}")
    seeds = range(SEED, SEED + N_SEEDS)
    # workers per study: the studies run at the same time
    n_workers = max(1, N_WORKERS // len(study_names))
    with ProcessPoolExecutor(N_WORKERS, mp_context=get_context("spawn")) as pool:
        episodes = dict(zip(seeds, pool.map(train.pretraining_episodes, seeds)))
        futures = []
        for name in study_names:
            study = optuna.create_study(study_name=name, storage=storage(train.RESULTS_NAME), direction="maximize",
                                        load_if_exists=True)
            remaining = max(N_TRIALS - len([t for t in study.trials if t.state.is_finished()]), 0)
            # the remaining trials, shared between the workers of the study
            n_trials = [remaining // n_workers + (k < remaining % n_workers) for k in range(n_workers)]
            futures += [pool.submit(worker, name, n, k, episodes) for k, n in enumerate(n_trials) if n > 0]
        for future in futures:
            future.result()
    for name in study_names:
        study = optuna.load_study(study_name=name, storage=storage(train.RESULTS_NAME))
        try:
            print(f"[hpo] {name}: best mean training return {study.best_value:.1f} with {study.best_params}")
        except ValueError:  # no completed trial
            print(f"[hpo] {name}: no completed trial")


if __name__ == "__main__":
    main(sys.argv[1:] or STUDIES)
