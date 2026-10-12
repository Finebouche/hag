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

Usage: python -m hag_detection.hpo <model> <size> [n_trials] [max_songs]
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
from hag_detection.models import PURE_MODELS, build, parts, states, suggest

STUDIES_DIR = Path(os.environ.get("HAG_DETECTION_STUDIES", OUTPUTS / "studies"))
N_FOLDS = 3
FOLD_SEED = 0
CV_SEED = 1000  # matrices of fold k drawn with np.random.seed(CV_SEED + k), the same for all the trials
N_TRIALS = 400


def storage(study_name):
    STUDIES_DIR.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{STUDIES_DIR / study_name}.sqlite3"


def max_phrase_length(songs, train_songs, frame):
    return max(int(end / frame) + 1 - int(start / frame) for w in train_songs for start, end, _ in songs[w]["phrases"])


def cross_validate(model, params, size, songs, train_songs, frame):
    """(out-of-fold F1, ridge, threshold) of the model on the train songs, see the module docstring."""
    folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED).split(train_songs)
    outputs = {}  # (ridge index, wave) -> readout output
    for k, (fit_index, val_index) in enumerate(folds):
        fit = [train_songs[i] for i in fit_index]
        val = [train_songs[i] for i in val_index]
        np.random.seed(CV_SEED + k)
        reservoirs = build(model, params, size, pretrain_phrases(songs, fit, frame))
        fit_states = states(reservoirs, [songs[w]["mfcc"] for w in fit])
        val_states = states(reservoirs, [songs[w]["mfcc"] for w in val])
        dim = fit_states[0].shape[1] + 1
        XtX, XtY = np.zeros((dim, dim)), np.zeros(dim)
        for x, w in zip(fit_states, fit):
            x = np.hstack([x, np.ones((len(x), 1), dtype=np.float32)]).astype(np.float64)
            XtX += x.T @ x
            XtY += x.T @ songs[w]["target"]
        for r, ridge in enumerate(RIDGES):
            Wout = np.linalg.solve(XtX + ridge * np.eye(dim), XtY)
            for x, w in zip(val_states, val):
                outputs[r, w] = x @ Wout[:-1] + Wout[-1]
        del fit_states, val_states

    best = (-1.0, None, None)
    for r, ridge in enumerate(RIDGES):
        song_outputs = [outputs[r, w] for w in train_songs]
        for threshold in THRESHOLDS:
            f1 = scores(song_outputs, songs, train_songs, threshold, frame)["f1"]
            if f1 > best[0]:
                best = (f1, float(ridge), float(threshold))
    return best


def objective_function(model, size, songs, train_songs, frame):
    length = max_phrase_length(songs, train_songs, frame)

    def objective(trial):
        params = suggest(trial, model, length)
        f1, ridge, threshold = cross_validate(model, params, size, songs, train_songs, frame)
        trial.set_user_attr("ridge", ridge)
        trial.set_user_attr("threshold", threshold)
        return f1
    return objective


def initial_trial(model, size):
    """Best parameters of the parts of a hybrid, each optimized alone with SIZE / 2 neurons (prefixed)."""
    params = {}
    for part, prefix in parts(model):
        study = optuna.load_study(study_name=f"{part}_{size // 2}", storage=storage(f"{part}_{size // 2}"))
        params.update({prefix + key: value for key, value in study.best_trial.params.items()})
    return params


def stop_after_failures(max_failures=10):
    """Callback stopping the optimization after max_failures failed trials in a row (systematic error)."""
    def callback(study, trial):
        last = [t.state for t in study.get_trials(deepcopy=False)][-max_failures:]
        if len(last) == max_failures and all(state == TrialState.FAIL for state in last):
            study.stop()
    return callback


def worker(model, size, n_trials, max_songs, worker_index):
    songs, train_songs, _, frame = load_dataset(max_songs)
    study_name = f"{model}_{size}"
    study = optuna.load_study(study_name=study_name, storage=storage(study_name),
                              sampler=TPESampler(constant_liar=True, seed=worker_index))
    study.optimize(objective_function(model, size, songs, train_songs, frame), gc_after_trial=True, catch=(Exception,),
                   callbacks=[MaxTrialsCallback(n_trials, states=(TrialState.COMPLETE,)), stop_after_failures()])


def main():
    model, size = sys.argv[1], int(sys.argv[2])
    n_trials = int(sys.argv[3]) if len(sys.argv) > 3 else N_TRIALS
    max_songs = int(sys.argv[4]) if len(sys.argv) > 4 else None
    n_workers = int(os.environ.get("HAG_DETECTION_WORKERS", 1))
    for part, _ in parts(model):
        if part not in PURE_MODELS:
            raise ValueError(f"Unknown model {part}, expected one of {PURE_MODELS} or hybrids <hag>+<esn>")

    study_name = f"{model}_{size}"
    study = optuna.create_study(study_name=study_name, storage=storage(study_name), direction="maximize",
                                load_if_exists=True)
    if "+" in model and not study.trials:
        study.enqueue_trial(initial_trial(model, size))
    done = len([t for t in study.trials if t.state == TrialState.COMPLETE])
    print(f"{study_name}: {done} / {n_trials} trials done, {n_workers} workers", flush=True)
    if done < n_trials:
        Parallel(n_jobs=n_workers)(delayed(worker)(model, size, n_trials, max_songs, i) for i in range(n_workers))

    study = optuna.load_study(study_name=study_name, storage=storage(study_name))
    best = study.best_trial
    print(f"best trial {best.number}: F1 {best.value:.4f}, ridge {best.user_attrs['ridge']:.0e}, "
          f"threshold {best.user_attrs['threshold']:.2f}, {best.params}")


if __name__ == "__main__":
    main()
