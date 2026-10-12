"""
Phrase boundary detection in canary songs (M1-2016-spring, "Marron1").

A canary phrase is a repetition of one syllable type, so a phrase change is a change of syllable type. A ridge readout
is trained on the states of a reservoir, frame by frame on whole songs, to output a bump (Gaussian, BUMP_STD) at each
phrase end followed by a phrase of another label; boundaries are the peaks of its output. Calls, SIL and TRASH are
ignored, and consecutive segments of the same label are one phrase (annotation splits, not transitions).

Conditions: MFCC alone (no reservoir), and reservoirs built with the best hyperparameters of the canary next-phrase
studies (legacy_studies/tpe_canary_db.sqlite3; unaffected by the MFCC fix): random_ee, var_hag, mean_hag, and
mean_hag_proj (HAG's matrices without the HAG learning). HAG learns on phrases of the train songs.

The ridge coefficient and the peak threshold are chosen on validation songs (taken from the train songs), then the
readout is trained again on all train songs. Score: F1 of the boundaries matched one to one within TOLERANCE, plus the
recall of the boundaries with and without silence between the two phrases.

Usage: python -m hag_detection.canary_boundaries [conditions] [nb_seeds] [max_songs] [network_size]
network_size: number of neurons of the reservoirs instead of the study's (500); results in canary_boundaries_<size>.csv
"""
import sys

import numpy as np
import optuna
import pandas as pd

from hag.hpo.utility import LEGACY_STUDIES
from hag.performances.esn_model_evaluation import run_reservoir
from hag.performances.evaluation_esn import build_reservoir_from_study
from hag_detection.common import (OUTPUTS, RIDGES, THRESHOLDS, boundary_target, compute_features, load_songs,
                                  pretrain_phrases, ridge_fit, ridge_run, scale, scores)

STUDIES_DB = f"sqlite:///{LEGACY_STUDIES / 'tpe_canary_db.sqlite3'}"
CONDITIONS = ["mfcc", "random_ee", "var_hag", "mean_hag", "mean_hag_proj"]
OUTPUT = OUTPUTS / "canary_boundaries.csv"


def evaluate(condition, songs, train_songs, test_songs, frame, seed, network_size=None):
    np.random.seed(seed)
    if condition == "mfcc":
        states = {w: songs[w]["mfcc"] for w in songs}
    else:
        function_name = condition.removesuffix("_proj")
        study = optuna.load_study(study_name=f"{function_name}_Canary_normal_multi", storage=STUDIES_DB)
        hyperparams = dict(study.best_trial.params)
        if network_size is not None:
            hyperparams["network_size"] = network_size
        reservoir, _ = build_reservoir_from_study(function_name, hyperparams, pretrain_phrases(songs, train_songs, frame),
                                                  True, plasticity=not condition.endswith("_proj"))
        waves = list(songs)
        states = dict(zip(waves, run_reservoir(reservoir, [songs[w]["mfcc"] for w in waves], reset=True)))
    targets = {w: boundary_target(songs[w], frame) for w in songs}

    # ridge and threshold on validation songs
    rng = np.random.default_rng(seed)
    validation = set(rng.choice(train_songs, max(1, len(train_songs) // 5), replace=False))
    fit_songs = [w for w in train_songs if w not in validation]
    validation = sorted(validation)
    best = (-1, None, None)
    for ridge in RIDGES:
        Wout = ridge_fit([states[w] for w in fit_songs], [targets[w] for w in fit_songs], ridge)
        outputs = ridge_run([states[w] for w in validation], Wout)
        for threshold in THRESHOLDS:
            f1 = scores(outputs, songs, validation, threshold, frame)["f1"]
            if f1 > best[0]:
                best = (f1, ridge, threshold)
    _, ridge, threshold = best

    Wout = ridge_fit([states[w] for w in train_songs], [targets[w] for w in train_songs], ridge)
    result = scores(ridge_run([states[w] for w in test_songs], Wout), songs, test_songs, threshold, frame)
    size = None if condition == "mfcc" else hyperparams["network_size"]
    return dict(condition=condition, seed=seed, network_size=size, ridge=ridge, threshold=threshold, validation_f1=best[0], **result)


def main():
    conditions = sys.argv[1].split(",") if len(sys.argv) > 1 else CONDITIONS
    nb_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    max_songs = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] != "all" else None
    network_size = int(sys.argv[4]) if len(sys.argv) > 4 else None
    output = OUTPUT if network_size is None else OUTPUT.with_stem(f"{OUTPUT.stem}_{network_size}")

    songs, train_songs, test_songs = load_songs(max_songs)
    frame = compute_features(songs)
    scale(songs, train_songs)
    print(f"{len(train_songs)} train / {len(test_songs)} test songs, frame = {frame * 1000:.1f} ms, "
          f"{sum(len(songs[w]['boundaries']) for w in test_songs)} test boundaries")

    results = []
    for condition in conditions:
        for seed in range(nb_seeds):
            result = evaluate(condition, songs, train_songs, test_songs, frame, seed, network_size)
            print({k: round(v, 3) if isinstance(v, float) else v for k, v in result.items()}, flush=True)
            results.append(result)

    results = pd.DataFrame(results)
    print(results.groupby("condition")[["f1", "precision", "recall", "recall_silence", "recall_no_silence"]]
          .agg(["mean", "std"]).round(3))
    if max_songs is None:
        output.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output, index=False)
        print("Saved to", output)


if __name__ == "__main__":
    main()
