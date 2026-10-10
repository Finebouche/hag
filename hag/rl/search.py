"""
Hyperparameter searches of the reinforcement learning methods (hag.rl.ppo.hpo, hag.rl.lspi.hpo): search space of the
features (shared by the methods), Optuna studies (multivariate TPE sampler, median pruner) and their parallel run.

Studies (STUDIES): one per kind of features and, for the reservoirs, per decomposition of the filter bank
(DECOMPOSITIONS: "ema" or "legendre"):
  - "obs": scaled observations (only the parameters of the method),
  - "filterbank": filter bank alone (decomposition searched),
  - "esn_<decomposition>": random reservoir (spectral radius, connectivity),
  - "hag_mean_<decomposition>", "hag_variance_<decomposition>": HAG with mean or variance homeostasis, target (and
    spread of the mean) in the ranges of HAG's studies on the classification datasets (hag/hpo/hpo_esn.py), spread of
    the variance below the target (else HAG cannot grow connections), without limit of incoming connections
    (MAX_PARTNERS) and with plasticity steps on windows of random lengths (USE_FULL_INSTANCE: the episodes of RL are
    not instances as the samples of the classification tasks). HAG must use its recurrence: a trial whose
    reservoir has less than MIN_HAG_CONNECTIONS connections per neuron is pruned before the training,
  - "<hybrid>_mean", "<hybrid>_variance": the hybrids of an ESN and HAG (hag.rl.pretrain.HYBRIDS) with mean or
    variance homeostasis, searching the parameters of the ESN and of HAG together (decomposition searched; the HAG half
    of "esn_hag" has the same inputs as the ESN half). Without completed study, the parameters of a hybrid are the
    best ones of the ESN and of HAG (see best_params),
all of them but "obs" with the number of filters and the slowest time scale of the filter bank, the reservoirs also
with the input scaling (down to the low values of HAG's studies) and the bias (distribution and scaling), without leak
(LEAK_RATE).
The number of connections per neuron of the reservoirs is saved as attribute of the trials ("connections_per_neuron").
The studies run at the same time, the N_WORKERS parallel processes being shared between them: the workers of a study
share it (TPE samplers of different seeds, with constant liar: the parameters of the running trials are avoided).
The trials are evaluated on the seeds SEED, ..., SEED + N_SEEDS - 1 (different from the seeds of the final training):
the pretraining episodes of seed k are the same for all the trials.
"""
import warnings
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
import optuna

from hag.rl import experiment
from hag.rl.pretrain import HYBRIDS, INPUT_PARAMS
from hag.rl.utils import N_WORKERS, RL_RESULTS

# =============================== PARAMETERS ===============================
N_SEEDS = 3                 # seeds per trial
SEED = 2026                 # seeds of the search: SEED, SEED + 1, ...
MIN_HAG_CONNECTIONS = 2     # minimal number of connections per neuron of HAG's reservoirs (else the trial is pruned)
DECOMPOSITIONS = ("ema", "legendre")  # decompositions of the filter bank (one study per reservoir and decomposition)
MAX_PARTNERS = np.inf       # maximal number of incoming connections of HAG's neurons (not searched: no limit)
USE_FULL_INSTANCE = False   # HAG's plasticity steps on windows of random lengths, not one per episode (not searched)
LEAK_RATE = 1.0             # leak rate of the reservoirs (not searched: no leak)
STUDIES = (("obs", "filterbank") + tuple(f"{kind}_{decomposition}" for kind in ("esn", "hag_mean", "hag_variance")
                                   for decomposition in DECOMPOSITIONS)
           + tuple(f"{hybrid}_{homeostasis}" for hybrid in HYBRIDS for homeostasis in ("mean", "variance")))
# ===========================================================================

warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)  # (multivariate TPE)


def storage_path(db_name: str, root=None, study_name: str = None):
    """Path of the database of the study study_name of db_name, in the folder root (default: RL_RESULTS):
    <db_name>_<study_name>.sqlite3, one database per study (the jobs of different studies, on different nodes, never
    write to the same file), or <db_name>.sqlite3, the database of all the studies of the earlier experiments, if it
    exists and the former does not (and if study_name is None)."""
    root = RL_RESULTS if root is None else root
    shared = root / f"{db_name}.sqlite3"
    if study_name is None:
        return shared
    own = root / f"{db_name}_{study_name}.sqlite3"
    return own if own.exists() or not shared.exists() else shared


def storage(db_name: str, root=None, study_name: str = None) -> optuna.storages.RDBStorage:
    """Storage of the study study_name of db_name (see storage_path)."""
    # timeout: the workers write to the same database
    return optuna.storages.RDBStorage(f"sqlite:///{storage_path(db_name, root, study_name)}",
                                      engine_kwargs={"connect_args": {"timeout": 60}})


def study_kind(study_name: str) -> str:
    """Kind of features of a study: "obs", "filterbank", "esn", "hag" or a hybrid (hag.rl.pretrain.HYBRIDS)."""
    for hybrid in HYBRIDS:
        if study_name.startswith(f"{hybrid}_"):
            return hybrid
    return study_name.split("_")[0]


def study_condition(study_name: str) -> str:
    """Condition (see hag.rl.experiment.CONDITIONS and hag.rl.sweep.HYBRID_CONDITIONS) of a study."""
    kind = study_kind(study_name)
    return kind if kind in ("obs", "filterbank") else f"filterbank+{kind}"


def suggest_features(trial: optuna.Trial, study_name: str) -> dict:
    """Parameters of the features of a trial of a study (see STUDIES): filter bank (see
    hag.rl.features.make_filter_bank) and reservoir (see hag.rl.pretrain.build_reservoir; the parameters of HAG of the
    hybrids are in "hag")."""
    kind = study_kind(study_name)
    if kind == "obs":
        return {}
    if kind == "filterbank" or kind in HYBRIDS:
        decomposition = trial.suggest_categorical("decomposition", list(DECOMPOSITIONS))
    else:
        decomposition = study_name.rsplit("_", 1)[1]
    params = dict(decomposition=decomposition, n_filters=trial.suggest_categorical("n_filters", [4, 6, 8, 12]),
                  slowest=trial.suggest_float("slowest", 0.005, 0.1, log=True))
    if kind == "filterbank":
        return params
    params.update(input_scaling=trial.suggest_float("input_scaling", 0.005, 1.0, log=True),
                  bias_dist=trial.suggest_categorical("bias_dist", ["foldnorm", "uniform"]),
                  bias_scaling=trial.suggest_float("bias_scaling", 0.0, 0.5),
                  lr=LEAK_RATE)
    if kind != "hag":  # (ESN and hybrids)
        params.update(sr=trial.suggest_float("sr", 0.1, 1.5),
                      rc_connectivity=trial.suggest_float("rc_connectivity", 0.01, 0.5, log=True))
    if kind == "esn":
        return params
    homeostasis = study_name.split("_")[1] if kind == "hag" else study_name[len(kind) + 1:]
    hag = suggest_hag(trial, homeostasis)
    if kind == "hag":
        return dict(params, **hag)
    return dict(params, hag=dict(hag, **{key: params[key] for key in INPUT_PARAMS}))


def suggest_hag(trial: optuna.Trial, homeostasis: str) -> dict:
    """Parameters of the plasticity of HAG with the homeostasis "mean" or "variance"."""
    min_window = trial.suggest_int("min_window", 3, 20)
    params = dict(weight_increment=trial.suggest_float("weight_increment", 0.001, 0.5, log=True),
                  min_window=min_window, max_window=trial.suggest_int("max_window", min_window, 60),
                  use_full_instance=USE_FULL_INSTANCE, max_partners=MAX_PARTNERS)
    # target and spread: ranges of HAG's studies (hag/hpo/hpo_esn.py)
    if homeostasis == "mean":
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


def connections_check(trial: optuna.Trial):
    """Function pruning the trial if the HAG reservoir of a feature pipeline does not use its recurrence (called before
    the training)."""

    def check(pipeline):
        W = pipeline.reservoir["W"]
        trial.set_user_attr("connections_per_neuron", float(np.count_nonzero(W) / len(W)))
        if np.count_nonzero(W) / len(W) < MIN_HAG_CONNECTIONS:
            trial.set_user_attr("pruned_reason", "too few connections")
            raise optuna.TrialPruned()

    return check


def best_params(db_name: str, kind: str, suggest, root=None):
    """Best parameters (suggest(trial, study_name): search space of the method) of the studies of a kind ("obs",
    "filterbank", "esn": best of the "esn_*" studies, "hag": best of the "hag_mean_*" and "hag_variance_*" studies, or a
    hybrid: best of its "<hybrid>_mean" and "<hybrid>_variance" studies) of the database db_name (in the folder root,
    default: RL_RESULTS), or None if there is no completed study. Hybrid without completed study: the best parameters
    of the ESN, and those of HAG in "hag"."""
    best = None
    for study_name in [study for study in STUDIES if study_kind(study) == kind]:
        if not storage_path(db_name, root, study_name).exists():  # (load_study would create an empty database)
            continue
        try:
            study = optuna.load_study(study_name=study_name, storage=storage(db_name, root, study_name))
            if best is None or study.best_value > best[0]:
                best = (study.best_value, study_name, study.best_params)
        except (KeyError, ValueError):  # study missing or without completed trial
            continue
    if best is None and kind in HYBRIDS:
        esn, hag = best_params(db_name, "esn", suggest, root), best_params(db_name, "hag", suggest, root)
        return None if esn is None or hag is None else dict(esn, hag=hag)
    return None if best is None else suggest(optuna.trial.FixedTrial(best[2]), best[1])


def worker(db_name: str, name: str, n_trials: int, worker: int, objective, episodes: dict):
    """Run n_trials trials of the study name (objective(trial, study_name, episodes); in a worker process; worker: its
    index, seed of its sampler)."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.load_study(study_name=name, storage=storage(db_name, study_name=name),
                              sampler=optuna.samplers.TPESampler(seed=SEED + worker, constant_liar=True,
                                                                 multivariate=True),
                              pruner=optuna.pruners.MedianPruner(n_startup_trials=10))

    def log(study, trial):
        value = (f"{trial.value:7.3f}" if trial.value is not None
                 else f"pruned ({trial.user_attrs.get('pruned_reason', 'median rule')})")
        try:
            best = f"{study.best_value:7.3f}"
        except ValueError:  # no completed trial yet
            best = "-"
        print(f"[hpo] {db_name} {name} trial {trial.number:3d}: {value} | best {best}", flush=True)

    # catch: a failing trial is recorded as failed, the study goes on
    study.optimize(lambda trial: objective(trial, name, episodes), n_trials=n_trials, callbacks=[log],
                   catch=(ValueError,))


def optimize(db_name: str, study_names, objective, n_trials: int):
    """Run the studies study_names (n_trials trials each, the finished ones included) of the database db_name, with
    objective(trial, study_name, episodes) (a module-level function: it is sent to the worker processes)."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    RL_RESULTS.mkdir(parents=True, exist_ok=True)
    for name in study_names:
        if name not in STUDIES:
            raise ValueError(f"Unknown study {name!r}: {STUDIES}")
    seeds = range(SEED, SEED + N_SEEDS)
    # workers per study: the studies run at the same time
    n_workers = max(1, N_WORKERS // len(study_names))
    with ProcessPoolExecutor(N_WORKERS, mp_context=get_context("spawn")) as pool:
        episodes = dict(zip(seeds, pool.map(experiment.pretraining_episodes, seeds)))
        futures = []
        for name in study_names:
            study = optuna.create_study(study_name=name, storage=storage(db_name, study_name=name),
                                        direction="maximize", load_if_exists=True)
            remaining = max(n_trials - len([t for t in study.trials if t.state.is_finished()]), 0)
            # the remaining trials, shared between the workers of the study
            counts = [remaining // n_workers + (k < remaining % n_workers) for k in range(n_workers)]
            futures += [pool.submit(worker, db_name, name, n, k, objective, episodes)
                        for k, n in enumerate(counts) if n > 0]
        for future in futures:
            future.result()
    for name in study_names:
        study = optuna.load_study(study_name=name, storage=storage(db_name, study_name=name))
        try:
            print(f"[hpo] {db_name} {name}: best objective {study.best_value:.3f} with {study.best_params}")
        except ValueError:  # no completed trial
            print(f"[hpo] {db_name} {name}: no completed trial")
