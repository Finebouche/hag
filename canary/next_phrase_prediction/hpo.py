"""
Hyperparameter optimization of the canary boundary detection, on the train songs only (the test songs are used once,
by canary.next_phrase_prediction.evaluate).

Objective: F1 of the boundaries, out of fold, in a 3-fold cross-validation by song on the train songs. In each fold,
the reservoirs are built (HAG learning on phrases of the fold's training songs), the ridge readout is trained on the
fold's training songs for every coefficient of RIDGES, and outputs the boundaries of the fold's validation songs; the
ridge coefficient and the peak threshold are then chosen on the out-of-fold outputs of all the train songs, and the
F1 is the objective. They are stored in the trial (user attributes "ridge", "threshold") for the final evaluation.
The ridge coefficient is thus not a hyperparameter of the search, for all the models.

Studies: "<model>_<size>" (see canary.models), one SQLite database each in STUDIES_DIR
(HAG_CANARY_STUDIES, default outputs/canary/studies). The joint optimization of a hybrid "<hag>+<esn>" of SIZE
neurons starts with (enqueued trial) the best parameters of the studies "<hag>_<SIZE/2>" and "<esn>_<SIZE/2>".

With HAG_CANARY_TASK=next_phrase, the objective is the out-of-fold accuracy of the prediction of the next phrase
(canary.next_phrase_prediction.prediction.choose_ridge: same folds, ridge coefficient chosen on the out-of-fold points), in the
studies "next_phrase_<model>_<size>": "next_phrase_gains_*" with HAG_CANARY_INPUT_GAINS=1 (gain of each group of input
features in the search, see objective_function), "next_phrase_leak_*" with HAG_CANARY_LEAK=1 (leak rate of each
reservoir in the search, instead of 1), "next_phrase_calls_*" with HAG_CANARY_CALLS=1 (the calls are phrases too) and
"next_phrase_reset_*" with HAG_CANARY_RESET=1 (the reservoirs hear the current phrase only, from a zero state) and
"next_phrase_nodeltas_*" with HAG_CANARY_DELTAS=0 (the 13 MFCC only, without their deltas and delta-deltas).
(The syllable classification has its own package, canary.syllables_classification.)

Usage: python -m canary.next_phrase_prediction.hpo <model> <size>   (N_TRIALS trials)
  workers: HAG_CANARY_WORKERS processes (default 1) optimizing the same study.
"""
import os
import sys

import numpy as np
import optuna
from joblib import Parallel, delayed
from optuna.samplers import TPESampler
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
from sklearn.model_selection import KFold

from canary.common import RIDGES, pretrain_phrases
from canary.hpo import CV_SEED, FOLD_SEED, N_FOLDS, N_TRIALS, stop_after_failures, storage, study_exists
from canary.models import ESN_MODELS, HAG_MODELS, PURE_MODELS, build, normal_equations, parts, readout_outputs, ridge_readout
from canary.next_phrase_prediction.boundaries import THRESHOLDS, load_dataset, scores

TASK = os.environ.get("HAG_CANARY_TASK", "boundaries")  # "boundaries" or "next_phrase"
# gain of each group of input features (MFCC, deltas, delta-deltas) in the search, studies "<task>_gains_*"
INPUT_GAINS = TASK != "boundaries" and os.environ.get("HAG_CANARY_INPUT_GAINS") == "1"
LEAK = TASK != "boundaries" and os.environ.get("HAG_CANARY_LEAK") == "1"
# next-phrase prediction: calls among the phrases, reservoirs reset at each phrase (canary.next_phrase_prediction.prediction)
CALLS = TASK == "next_phrase" and os.environ.get("HAG_CANARY_CALLS") == "1"
RESET = TASK == "next_phrase" and os.environ.get("HAG_CANARY_RESET") == "1"
# next-phrase prediction without the deltas and delta-deltas of the MFCC in the input features
NO_DELTAS = TASK == "next_phrase" and os.environ.get("HAG_CANARY_DELTAS") == "0"
if NO_DELTAS and INPUT_GAINS:
    raise ValueError("HAG_CANARY_DELTAS=0: no deltas to weight (HAG_CANARY_INPUT_GAINS=1)")
# prefix of the studies and of the results of the next-phrase prediction
STUDY_PREFIX = (f"{TASK}_" + "calls_" * CALLS + "reset_" * RESET + "nodeltas_" * NO_DELTAS + "gains_" * INPUT_GAINS
                + "leak_" * LEAK if TASK != "boundaries" else "")


def study_name_of(model, size):
    return f"{STUDY_PREFIX}{model}_{size}"


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
        if TASK == "next_phrase":
            from canary.next_phrase_prediction.prediction import choose_ridge
            ridge, accuracy = choose_ridge(model, params, size, songs, train_songs, frame, classes)
            trial.set_user_attr("ridge", ridge)
            return accuracy
        f1, ridge, threshold = cross_validate(model, params, size, songs, train_songs, frame)
        trial.set_user_attr("ridge", ridge)
        trial.set_user_attr("threshold", threshold)
        return f1
    return objective


def load_task_dataset():
    """(songs, train songs, frame, classes of the next phrase (None for the boundaries))."""
    songs, train_songs, _, frame = load_dataset()
    classes = None
    if TASK == "next_phrase":
        from canary.next_phrase_prediction.prediction import prepare
        classes = prepare(songs, train_songs, frame)
    return songs, train_songs, frame, classes


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
        # second one then fails when it reports it: "Cannot tell a COMPLETE trial"). Its parameters: the best ones of
        # the parts of the hybrid, each optimized alone with SIZE / 2 neurons (prefixed)
        params = {}
        for part, prefix in parts(model):
            name = study_name_of(part, size // 2)
            if not study_exists(name):
                raise FileNotFoundError(f"No study {name}: the parts of the hybrid {model} are optimized first")
            params.update({prefix + key: value for key, value in
                           optuna.load_study(study_name=name, storage=storage(name)).best_trial.params.items()})
        study.enqueue_trial(params)
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
    print(f"best trial {best.number}: {'accuracy' if TASK == 'next_phrase' else 'F1'} {best.value:.4f}, ridge "
          f"{best.user_attrs['ridge']:.0e}" + (f", threshold {threshold:.2f}" if threshold is not None else "")
          + f", {best.params}")
