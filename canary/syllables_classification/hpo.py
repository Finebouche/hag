"""
Hyperparameter optimization of the syllable classification (canary.syllables_classification.classification), on the train
songs only (the test songs are used once, by canary.syllables_classification.classification).

Objective: out-of-fold frame accuracy of the syllable classification, in a 3-fold cross-validation by song on the
train songs (canary.syllables_classification.classification.choose_ridge: in each fold, the reservoirs are built, HAG learning
on phrases of the fold's training songs, and the ridge readout is trained for every coefficient of RIDGES; the ridge
coefficient is chosen on the out-of-fold frames, stored in the trial (user attribute "ridge") for the final evaluation).
The ridge coefficient is thus not a hyperparameter of the search, for all the models.

Studies: "syllables_<model>_<size>" (see canary.models), "syllables_gains_<model>_<size>" with
HAG_CANARY_INPUT_GAINS=1 (gain of each group of input features in the search, see objective_function) and
"syllables_gains_leak_<model>_<size>" with HAG_CANARY_LEAK=1 too (leak rate of each reservoir in the search, instead of
1); one SQLite database each in STUDIES_DIR (HAG_CANARY_STUDIES, default outputs/canary/studies). The joint
optimization of a hybrid "<hag>+<esn>" of SIZE neurons starts with (enqueued trial) the best parameters of the studies
"<hag>_<SIZE/2>" and "<esn>_<SIZE/2>".

Usage: python -m canary.syllables_classification.hpo <model> <size>   (N_TRIALS trials)
  workers: HAG_CANARY_WORKERS processes (default 1) optimizing the same study.
"""
import os
import sys

import optuna
from joblib import Parallel, delayed
from optuna.samplers import TPESampler
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState

from canary.common import load_dataset
from canary.hpo import N_TRIALS, stop_after_failures, storage, study_exists
from canary.models import ESN_MODELS, HAG_MODELS, PURE_MODELS, parts

# gain of each group of input features (MFCC, deltas, delta-deltas) in the search, studies "syllables_gains_*"
INPUT_GAINS = os.environ.get("HAG_CANARY_INPUT_GAINS") == "1"
LEAK = os.environ.get("HAG_CANARY_LEAK") == "1"
# prefix of the studies and of the results of the syllable classification
STUDY_PREFIX = "syllables_" + "gains_" * INPUT_GAINS + "leak_" * LEAK


def study_name_of(model, size):
    return f"{STUDY_PREFIX}{model}_{size}"


def load_task_dataset():
    """(songs, train songs, frame, classes of the syllable classification)."""
    from canary.syllables_classification.classification import add_frame_labels
    songs, train_songs, _, frame = load_dataset()
    return songs, train_songs, frame, add_frame_labels(songs, frame)


def objective_function(model, size, songs, train_songs, frame, classes):
    from canary.syllables_classification.classification import choose_ridge
    length = max(int(end / frame) + 1 - int(start / frame) for w in train_songs for start, end, _ in songs[w]["phrases"])

    def objective(trial):
        # parameters of each part of the model (prefix "<part>:" in a hybrid), same ranges as hag.hpo.hpo_esn. With
        # INPUT_GAINS, a gain for each group of input features: the MFCC (input_scaling) and, relative to them, their
        # deltas (delta_gain) and delta-deltas (delta2_gain), as canapy (per-group input scaling of its ESN). With LEAK,
        # the leak rate of each reservoir (1 otherwise; canapy: 0.1), used by the HAG learning too.
        params = {}
        for part, p in parts(model):
            params[p + "input_scaling"] = trial.suggest_float(p + "input_scaling", 0.01, 0.2, step=0.005)
            if INPUT_GAINS:
                params[p + "delta_gain"] = trial.suggest_float(p + "delta_gain", 0.01, 100, log=True)
                params[p + "delta2_gain"] = trial.suggest_float(p + "delta2_gain", 0.01, 100, log=True)
            if LEAK:
                params[p + "leaky_rate"] = trial.suggest_float(p + "leaky_rate", 0.01, 1, log=True)
            params[p + "bias_scaling"] = trial.suggest_float(p + "bias_scaling", 0, 0.2, step=0.005)
            if part in HAG_MODELS:
                params[p + "weight_increment"] = trial.suggest_float(p + "weight_increment", 0.001, 0.1, step=0.001)
                params[p + "use_full_instance"] = trial.suggest_categorical(p + "use_full_instance", [True, False])
                params[p + "min_increment"] = trial.suggest_int(p + "min_increment", 3, length)
                params[p + "max_increment"] = trial.suggest_int(p + "max_increment", params[p + "min_increment"],
                                                                length * 5)
                if part == "mean_hag":
                    params[p + "target_rate"] = trial.suggest_float(p + "target_rate", 0.5, 1, step=0.01)
                    params[p + "rate_spread"] = trial.suggest_float(p + "rate_spread", 0.01, 0.4, step=0.005)
                else:
                    params[p + "variance_target"] = trial.suggest_float(p + "variance_target", 0.001, 0.02, step=0.001)
                    params[p + "variance_spread"] = trial.suggest_float(p + "variance_spread", 0.001, 0.05, step=0.002)
                    params[p + "intrinsic_saturation"] = trial.suggest_float(p + "intrinsic_saturation", 0.8, 0.98,
                                                                             step=0.02)
                    params[p + "intrinsic_coef"] = trial.suggest_float(p + "intrinsic_coef", 0.8, 0.98, step=0.02)
            elif part in ESN_MODELS:
                params[p + "connectivity"] = trial.suggest_float(p + "connectivity", 0, 1)
                params[p + "spectral_radius"] = trial.suggest_float(p + "spectral_radius", 0.4, 1.6, step=0.01)
            else:
                raise ValueError(f"Unknown model {part}")
        ridge, accuracy = choose_ridge(model, params, size, songs, train_songs, frame, classes)
        trial.set_user_attr("ridge", ridge)
        return accuracy
    return objective


def initial_trial(model, size):
    """Best parameters of the parts of a hybrid, each optimized alone with SIZE / 2 neurons (prefixed)."""
    params = {}
    for part, prefix in parts(model):
        name = study_name_of(part, size // 2)
        if not study_exists(name):
            raise FileNotFoundError(f"No study {name}: the parts of the hybrid {model} are optimized first")
        study = optuna.load_study(study_name=name, storage=storage(name))
        params.update({prefix + key: value for key, value in study.best_trial.params.items()})
    return params


def worker(model, size, worker_index):
    songs, train_songs, frame, classes = load_task_dataset()
    study_name = study_name_of(model, size)
    study = optuna.load_study(study_name=study_name, storage=storage(study_name), sampler=TPESampler(constant_liar=True, seed=worker_index))
    study.optimize(objective_function(model, size, songs, train_songs, frame, classes), gc_after_trial=True, catch=(Exception,),
                   callbacks=[MaxTrialsCallback(N_TRIALS, states=(TrialState.COMPLETE,)), stop_after_failures()])


if __name__ == "__main__":
    model, size = sys.argv[1], int(sys.argv[2])
    n_workers = int(os.environ.get("HAG_CANARY_WORKERS", 1))
    for part, _ in parts(model):
        if part not in PURE_MODELS:
            raise ValueError(f"Unknown model {part}, expected one of {PURE_MODELS} or hybrids <hag>+<esn>")

    study_name = study_name_of(model, size)
    study = optuna.create_study(study_name=study_name, storage=storage(study_name), direction="maximize", load_if_exists=True)

    if "+" in model and not study.trials:
        # the starting trial is run here, alone: several workers would take the enqueued trial at the same time (the
        # second one then fails when it reports it: "Cannot tell a COMPLETE trial")
        study.enqueue_trial(initial_trial(model, size))
        songs, train_songs, frame, classes = load_task_dataset()
        study.optimize(objective_function(model, size, songs, train_songs, frame, classes), n_trials=1)
        del songs

    done = len([t for t in study.trials if t.state == TrialState.COMPLETE])
    print(f"{study_name}: {done} / {N_TRIALS} trials done, {n_workers} workers", flush=True)
    if done < N_TRIALS:
        Parallel(n_jobs=n_workers)(delayed(worker)(model, size, i) for i in range(n_workers))

    study = optuna.load_study(study_name=study_name, storage=storage(study_name))
    best = study.best_trial
    print(f"best trial {best.number}: frame accuracy {best.value:.4f}, ridge {best.user_attrs['ridge']:.0e}, "
          f"{best.params}")
