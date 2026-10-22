"""
Syllable classification of canary songs, frame by frame, as canapy (https://github.com/birds-canopy/canapy, Trouvain &
Hinaut): the readout predicts the label of every frame (the 27 syllable types, call, TRASH and SIL), and the score is
canapy's frame accuracy: the share of the frames of the recordings (silences included) whose raw prediction (most
likely class, no post-processing) is the annotated label. A frame k is labeled with the annotated segment that covers
it, frames from floor(onset / hop) to floor(offset / hop) (canapy's as_frame_comparison), SIL elsewhere.

Same songs, MFCC (44.1 kHz) and reservoirs as the boundary detection (hag_detection.evaluate.conditions: the best
parameters of the detection studies, no leak and no spectral radius of canapy's ESN). Only the readout changes: a ridge
regression on the one-hot labels (centered statistics, hag_detection.models.normal_equations), its coefficient chosen
by the same 3-fold cross-validation by song on the train songs (out-of-fold frame accuracy), then trained on all the
train songs and scored on the test songs for each seed (paired comparisons as hag_detection.evaluate).

Results: outputs/detection/syllables.csv and syllables_comparisons.csv; with HAG_DETECTION_TASK=syllables, the reservoirs
of the syllable studies (hag_detection.hpo optimized on the frame accuracy) and syllables_hpo.csv,
syllables_hpo_comparisons.csv.
Usage:
  python -m hag_detection.syllables [sizes] [nb_seeds] [max_songs]   all the conditions (default 250,500,1000, 10 seeds)
  python -m hag_detection.syllables run <condition> <size> [nb_seeds]   one condition (one job of slurm/submit_syllables.sh),
                                                                        results in outputs/detection/syllables_parts
  python -m hag_detection.syllables merge                               results of the parts and the comparisons
  python -m hag_detection.syllables list <sizes>                        "<condition> <size>" of all the conditions
"""
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from hag.datasets.load_categorical_forecasting import download_canary_if_needed
from hag_detection.common import OUTPUTS, RIDGES, load_dataset, pretrain_phrases
from hag_detection.evaluate import SIZES, compare, conditions
from hag_detection.hpo import CV_SEED, FOLD_SEED, N_FOLDS, STUDY_PREFIX, TASK
from hag_detection.models import build, normal_equations, ridge_readout, state_chunks

SILENCE = "SIL"


def add_frame_labels(songs, frame):
    """songs[w]["labels"]: label of each frame (all the annotated labels, SIL between segments); returns the classes."""
    annotations_dir, _ = download_canary_if_needed()
    annotations = pd.concat([pd.read_csv(f, index_col=0) for f in sorted(annotations_dir.rglob("*.csv"))])
    for wave, segments in annotations[annotations["wave"].isin(list(songs))].groupby("wave"):
        n_frames = len(songs[wave]["mfcc"])
        labels = np.full(n_frames, SILENCE, dtype=object)
        for start, end, label in segments.sort_values("start")[["start", "end", "syll"]].itertuples(index=False):
            labels[int(np.floor(start / frame)):min(int(np.floor(end / frame)), n_frames)] = label
        songs[wave]["labels"] = labels
    classes = np.array(sorted({label for song in songs.values() for label in song["labels"]}))
    for song in songs.values():
        song["Y"] = (song["labels"][:, None] == classes[None, :]).astype(np.float64)
    return classes


def correct_frames(reservoirs, songs, waves, Wouts, classes):
    """(correct frames per readout of Wouts, correct non-silent frames per readout, frames, non-silent frames)."""
    stacked = np.concatenate(Wouts, axis=1)  # (units + 1, readouts * classes)
    n_classes = len(classes)
    correct, correct_sound, total, total_sound = np.zeros(len(Wouts)), np.zeros(len(Wouts)), 0, 0
    for part, part_states in state_chunks(reservoirs, songs, waves):
        for x, w in zip(part_states, part):
            outputs = x.astype(np.float64) @ stacked[:-1] + stacked[-1]
            predicted = classes[outputs.reshape(len(x), len(Wouts), n_classes).argmax(axis=2)]  # (T, readouts)
            labels = songs[w]["labels"][:, None]
            sound = labels[:, 0] != SILENCE
            correct += (predicted == labels).sum(axis=0)
            correct_sound += (predicted[sound] == labels[sound]).sum(axis=0)
            total += len(x)
            total_sound += int(sound.sum())
    return correct, correct_sound, total, total_sound


def choose_ridge(model, params, size, songs, train_songs, frame, classes):
    """Ridge coefficient of the best out-of-fold frame accuracy, in a 3-fold cross-validation by song."""
    correct = np.zeros(len(RIDGES))
    total = 0
    for k, (fit_index, val_index) in enumerate(KFold(N_FOLDS, shuffle=True, random_state=FOLD_SEED).split(train_songs)):
        np.random.seed(CV_SEED + k)
        fit = [train_songs[i] for i in fit_index]
        reservoirs = build(model, params, size, pretrain_phrases(songs, fit, frame))
        statistics = normal_equations(reservoirs, songs, fit, target="Y")
        fold_correct, _, fold_total, _ = correct_frames(reservoirs, songs, [train_songs[i] for i in val_index],
                                                        [ridge_readout(statistics, ridge) for ridge in RIDGES], classes)
        correct += fold_correct
        total += fold_total
    best = int(np.argmax(correct))
    return float(RIDGES[best]), correct[best] / total


def evaluate(model, params, size, ridge, songs, train_songs, test_songs, frame, classes, seed):
    np.random.seed(seed)
    reservoirs = build(model, params, size, pretrain_phrases(songs, train_songs, frame))
    Wout = ridge_readout(normal_equations(reservoirs, songs, train_songs, target="Y"), ridge)
    correct, correct_sound, total, total_sound = correct_frames(reservoirs, songs, test_songs, [Wout], classes)
    return dict(frame_accuracy=correct[0] / total, frame_accuracy_without_silence=correct_sound[0] / total_sound)


# reservoirs of the syllable studies (with the input gains, the leak rate) or of the boundary studies
NAME = f"{STUDY_PREFIX}hpo" if TASK == "syllables" else "syllables"
PARTS = OUTPUTS / f"{NAME}_parts"


def all_conditions(sizes):
    """[(condition, model, params, size)], mfcc first (size 0)."""
    todo = [("mfcc", "mfcc", {}, 0)]
    for size in sizes:
        todo += [(condition, model, params, size) for condition, model, params, _, _ in conditions(size)]
    return todo


def run(todo, nb_seeds, max_songs=None):
    songs, train_songs, test_songs, frame = load_dataset(max_songs)
    classes = add_frame_labels(songs, frame)
    print(f"{len(train_songs)} train / {len(test_songs)} test songs, {len(classes)} classes, "
          f"{sum(len(songs[w]['labels']) for w in test_songs)} test frames", flush=True)
    rows = []
    for condition, model, params, size in todo:
        ridge, cv_accuracy = choose_ridge(model, params, size, songs, train_songs, frame, classes)
        for seed in range(nb_seeds):
            result = evaluate(model, params, size, ridge, songs, train_songs, test_songs, frame, classes, seed)
            rows.append(dict(condition=condition, network_size=size, seed=seed, ridge=ridge,
                             cv_frame_accuracy=cv_accuracy, **result))
        accuracy = [row["frame_accuracy"] for row in rows[-nb_seeds:]]
        print(f"{condition} ({size}): frame accuracy {100 * np.mean(accuracy):.2f} ± {100 * np.std(accuracy):.2f} % "
              f"(ridge {ridge:.0e}, cross-validation {100 * cv_accuracy:.2f} %)", flush=True)
    return pd.DataFrame(rows)


def save(results):
    comparisons = compare(results, metric="frame_accuracy")
    pd.set_option("display.width", 200)
    print(comparisons.round(4).to_string(index=False))
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUTS / f"{NAME}.csv", index=False)
    comparisons.to_csv(OUTPUTS / f"{NAME}_comparisons.csv", index=False)
    print("Saved to", OUTPUTS / f"{NAME}.csv", "and", OUTPUTS / f"{NAME}_comparisons.csv")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "list":
        for condition, _, _, size in all_conditions([int(s) for s in sys.argv[2].split(",")]):
            print(condition, size)
    elif command == "run":
        condition, size = sys.argv[2], int(sys.argv[3])
        nb_seeds = int(sys.argv[4]) if len(sys.argv) > 4 else 10
        todo = [c for c in all_conditions([size] if size else []) if c[0] == condition and c[3] == size]
        if not todo:
            raise ValueError(f"Unknown condition {condition} ({size})")
        PARTS.mkdir(parents=True, exist_ok=True)
        run(todo, nb_seeds).to_csv(PARTS / f"{condition}_{size}.csv", index=False)
    elif command == "merge":
        parts = sorted(PARTS.glob("*.csv"))
        print(f"{len(parts)} conditions")
        save(pd.concat([pd.read_csv(part) for part in parts], ignore_index=True))
    else:
        sizes = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else SIZES
        nb_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        max_songs = int(sys.argv[3]) if len(sys.argv) > 3 else None
        results = run(all_conditions(sizes), nb_seeds, max_songs)
        if max_songs is None:
            save(results)
        else:
            print(compare(results, metric="frame_accuracy").round(4).to_string(index=False))

