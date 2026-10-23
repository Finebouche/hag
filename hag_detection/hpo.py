"""
Hyperparameter optimization of the canary boundary detection, on the train songs only (the test songs are used once,
by hag_detection.evaluate).

Objective: F1 of the boundaries, out of fold, in a 3-fold cross-validation by song on the train songs. In each fold,
the reservoirs are built (HAG learning on phrases of the fold's training songs), the ridge readout is trained on the
fold's training songs for every coefficient of RIDGES, and outputs the boundaries of the fold's validation songs; the
ridge coefficient and the peak threshold are then chosen on the out-of-fold outputs of all the train songs, and the
F1 is the objective. They are stored in the trial (user attributes "ridge", "threshold") for the final evaluation.
The ridge coefficient is thus not a hyperparameter of the search, for all the models.

Studies: "<model>_<size>" (see hag_detection.models), one SQLite database each in STUDIES_DIR
(HAG_DETECTION_STUDIES, default outputs/detection/studies). The joint optimization of a hybrid "<hag>+<esn>" of SIZE
neurons starts with (enqueued trial) the best parameters of the studies "<hag>_<SIZE/2>" and "<esn>_<SIZE/2>".

With HAG_DETECTION_TASK=syllables, the objective is instead the out-of-fold frame accuracy of the syllable
classification (hag_detection.syllables.classification.choose_ridge: same folds, ridge coefficient chosen on the out-of-fold frames),
in the studies "syllables_<model>_<size>" ("syllables_gains_<model>_<size>" with HAG_DETECTION_INPUT_GAINS=1: gain of
each group of input features in the search, hag_detection.models.suggest; "syllables_gains_leak_<model>_<size>" with
HAG_DETECTION_LEAK=1 too: leak rate of each reservoir in the search, instead of 1).

With HAG_DETECTION_TASK=next_phrase, the objective is the out-of-fold accuracy of the prediction of the next phrase
(hag_detection.next_phrase.prediction.choose_ridge: same folds), in the studies "next_phrase_<model>_<size>" (and the same
"gains_" and "leak_" variants), "next_phrase_calls_*" with HAG_DETECTION_CALLS=1 (the calls are phrases too) and
"next_phrase_reset_*" with HAG_DETECTION_RESET=1 (the reservoirs hear the current phrase only, from a zero state).

Usage: python -m hag_detection.hpo <model> <size>   (N_TRIALS trials)
  workers: HAG_DETECTION_WORKERS processes (default 1) optimizing the same study.
"""
import os
import sys
from pathlib import Path

import numpy as np
import optuna
from joblib import Parallel, delayed
from optuna.samplers import TPESampler
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
from sklearn.model_selection import KFold

from hag_detection.common import OUTPUTS, RIDGES, THRESHOLDS, load_dataset, pretrain_phrases, scores
from hag_detection.models import PURE_MODELS, build, normal_equations, parts, readout_outputs, ridge_readout, suggest

STUDIES_DIR = Path(os.environ.get("HAG_DETECTION_STUDIES", OUTPUTS / "studies"))
N_FOLDS = 3
FOLD_SEED = 0
CV_SEED = 1000  # matrices of fold k drawn with np.random.seed(CV_SEED + k), the same for all the trials
N_TRIALS = 400
TASK = os.environ.get("HAG_DETECTION_TASK", "boundaries")  # "boundaries", "syllables" or "next_phrase"
# gain of each group of input features (MFCC, deltas, delta-deltas) in the search, studies "<task>_gains_*"
INPUT_GAINS = TASK != "boundaries" and os.environ.get("HAG_DETECTION_INPUT_GAINS") == "1"
LEAK = TASK != "boundaries" and os.environ.get("HAG_DETECTION_LEAK") == "1"
# next-phrase prediction: calls among the phrases, reservoirs reset at each phrase (hag_detection.next_phrase.prediction)
CALLS = TASK == "next_phrase" and os.environ.get("HAG_DETECTION_CALLS") == "1"
RESET = TASK == "next_phrase" and os.environ.get("HAG_DETECTION_RESET") == "1"
# prefix of the studies and of the results of the syllable classification and of the next-phrase prediction
STUDY_PREFIX = (f"{TASK}_" + "calls_" * CALLS + "reset_" * RESET + "gains_" * INPUT_GAINS + "leak_" * LEAK
                if TASK != "boundaries" else "")


def study_name_of(model, size):
    return f"{STUDY_PREFIX}{model}_{size}"


def load_task_dataset():
    """(songs, train songs, frame, classes): classes of the syllable classification or of the next-phrase prediction,
    None for the boundaries."""
    songs, train_songs, _, frame = load_dataset()
    classes = None
    if TASK == "syllables":
        from hag_detection.syllables.classification import add_frame_labels
        classes = add_frame_labels(songs, frame)
    elif TASK == "next_phrase":
        from hag_detection.next_phrase.prediction import prepare
        classes = prepare(songs, train_songs, frame)
    return songs, train_songs, frame, classes


def storage(study_name):
    STUDIES_DIR.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{STUDIES_DIR / study_name}.sqlite3"


def study_exists(study_name):
    """Whether the database of the study exists (opening a missing SQLite database creates it, empty)."""
    return (STUDIES_DIR / f"{study_name}.sqlite3").exists()


def cross_validate(model, params, size, songs, train_songs, frame):
    """(out-of-fold F1, ridge, threshold) of the model on the train songs, see the module docstring."""
    folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED).split(train_songs)
    outputs = {}  # (ridge index, wave) -> readout output
    for k, (fit_index, val_index) in enumerate(folds):
        fit = [train_songs[i] for i in fit_index]
        val = [train_songs[i] for i in val_index]
        np.random.seed(CV_SEED + k)
        reservoirs = build(model, params, size, pretrain_phrases(songs, fit, frame))
        statistics = normal_equations(reservoirs, songs, fit)
        Wouts = [ridge_readout(statistics, ridge) for ridge in RIDGES]
        for w, output in readout_outputs(reservoirs, songs, val, Wouts).items():
            for r in range(len(RIDGES)):
                outputs[r, w] = output[:, r]

    best = (-1.0, None, None)
    for r, ridge in enumerate(RIDGES):
        song_outputs = [outputs[r, w] for w in train_songs]
        for threshold in THRESHOLDS:
            f1 = scores(song_outputs, songs, train_songs, threshold, frame)["f1"]
            if f1 > best[0]:
                best = (f1, float(ridge), float(threshold))
    return best


def objective_function(model, size, songs, train_songs, frame, classes=None):
    length = max(int(end / frame) + 1 - int(start / frame) for w in train_songs for start, end, _ in songs[w]["phrases"])

    def objective(trial):
        params = suggest(trial, model, length, input_gains=INPUT_GAINS, leak=LEAK)
        if TASK == "syllables":
            from hag_detection.syllables.classification import choose_ridge
            ridge, accuracy = choose_ridge(model, params, size, songs, train_songs, frame, classes)
            trial.set_user_attr("ridge", ridge)
            return accuracy
        if TASK == "next_phrase":
            from hag_detection.next_phrase.prediction import choose_ridge
            ridge, accuracy = choose_ridge(model, params, size, songs, train_songs, frame, classes)
            trial.set_user_attr("ridge", ridge)
            return accuracy
        f1, ridge, threshold = cross_validate(model, params, size, songs, train_songs, frame)
        trial.set_user_attr("ridge", ridge)
        trial.set_user_attr("threshold", threshold)
        return f1
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


def stop_after_failures(max_failures=10):
    """Callback stopping the optimization after max_failures failed trials in a row (systematic error)."""
    def callback(study, trial):
        last = [t.state for t in study.get_trials(deepcopy=False)][-max_failures:]
        if len(last) == max_failures and all(state == TrialState.FAIL for state in last):
            study.stop()
    return callback


def worker(model, size, worker_index):
    songs, train_songs, frame, classes = load_task_dataset()
    study_name = study_name_of(model, size)
    study = optuna.load_study(study_name=study_name, storage=storage(study_name), sampler=TPESampler(constant_liar=True, seed=worker_index))
    study.optimize(objective_function(model, size, songs, train_songs, frame, classes), gc_after_trial=True, catch=(Exception,),
                   callbacks=[MaxTrialsCallback(N_TRIALS, states=(TrialState.COMPLETE,)), stop_after_failures()])


if __name__ == "__main__":
    model, size = sys.argv[1], int(sys.argv[2])
    n_workers = int(os.environ.get("HAG_DETECTION_WORKERS", 1))
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
    threshold = best.user_attrs.get("threshold")
    print(f"best trial {best.number}: {dict(syllables='frame accuracy', next_phrase='accuracy').get(TASK, 'F1')} {best.value:.4f}, ridge "
          f"{best.user_attrs['ridge']:.0e}" + (f", threshold {threshold:.2f}" if threshold is not None else "")
          + f", {best.params}")
