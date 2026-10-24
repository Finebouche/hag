"""
Prediction of the next phrase of canary songs from their audio, in the configuration of Maxime Canonne's next-phrase
prediction (ESN reading the phrase labels of the song, with START and END), but with the MFCC of the song as input:
the reservoir runs over the whole song (no reset between the phrases) and, at the end of each phrase, the readout predicts
the label of the next one (END after the last phrase); before the first phrase (Maxime's START), it predicts the first
one. Phrases as in canary.common.load_songs: SIL, TRASH and call ignored, repeated labels merged; with
HAG_CANARY_CALLS=1, the calls are phrases too (SIL and TRASH ignored). With HAG_CANARY_RESET=1, the reservoirs hear
the current phrase only, from a zero state (as the Canary task of the main branch), instead of the whole song: the
state of a point is the last one of a run on the frames of the phrase that ends there (the zero state before the first
phrase); the HAG learning is the same (on phrases). With HAG_CANARY_DELTAS=0, the input is the 13 MFCC only, without
their deltas and delta-deltas (same prediction points).

No input after the prediction point: the MFCC are computed without the floor of the dB at 80 dB below the maximum of
the whole song (librosa's default, which depends on the future of each frame: canary.common.load_dataset(top_db=None)),
the MFCC frames are centered (half a window, 11.6 ms, of audio after their time)
and their deltas use 4 frames on each side, so the state used to predict what follows the time t is the one of the
last frame whose window and deltas end before t (DELAY earlier). The deltas are recomputed with mode="nearest" instead
of the "wrap" of the cached features, which joins the end of each song to its beginning.

Readout: ridge regression on the one-hot next label at the prediction points, its coefficient chosen by the 3-fold
cross-validation by song on the train songs (out-of-fold accuracy), then trained on all the
train songs and scored on the test songs for each seed. Metrics as Maxime's: accuracy and macro F1 of all the points,
without the last one of each song ("skip_last": its target is END), and of the phrases only ("no_end": points whose
target is a phrase, argmax over the phrases).
References: n-grams of the true labels of the previous phrases (canary.next_phrase_prediction.prediction baselines): fixed order n
(the n previous labels, START before the first phrase; unseen context: the most frequent next label) and adaptive (the
longest context seen in the train songs, up to MAX_ORDER).

Results: outputs/canary/<HAG_CANARY_TASK=next_phrase prefix>hpo.csv (e.g. next_phrase_leak_hpo.csv) and
<...>_comparisons.csv, references in next_phrase_ngrams.csv (next_phrase_calls_ngrams.csv with the calls).
Usage (HAG_CANARY_TASK=next_phrase, with the variables of the studies, e.g. HAG_CANARY_LEAK=1):
  python -m canary.next_phrase_prediction.prediction [sizes] [nb_seeds]   all the conditions (default 500,1000, 10 seeds)
  python -m canary.next_phrase_prediction.prediction run <condition> <size> [nb_seeds]   one condition (slurm/submit_next_phrase.sh),
                                                                          results in outputs/canary/<name>_parts
  python -m canary.next_phrase_prediction.prediction merge                               results of the parts and the comparisons
  python -m canary.next_phrase_prediction.prediction list <sizes>                        "<condition> <size>" of all the conditions
  python -m canary.next_phrase_prediction.prediction baselines                           the n-grams
"""
import sys
from collections import Counter

import numpy as np
import pandas as pd
from librosa.feature import delta
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

from canary.common import (OUTPUTS, RIDGES, SAMPLING_RATE, annotations_of_songs, load_dataset, pretrain_phrases,
                           song_phrases)
from canary.evaluate import compare
from canary.hpo import CV_SEED, FOLD_SEED, N_FOLDS
from canary.models import CHUNK, build, ridge_readout, state_chunks, states
from canary.next_phrase_prediction.evaluate import conditions
from canary.next_phrase_prediction.hpo import CALLS, NO_DELTAS, RESET, STUDY_PREFIX, TASK
from hag.datasets.load_categorical_forecasting import canary_mfcc_config

SIZES = [500, 1000]
START, END = "START", "END"
DELTA_WIDTH = 9  # frames of the deltas (librosa's default), DELTA_WIDTH // 2 after each frame
N_STATIC = 13  # MFCC before their deltas and delta-deltas in the features
MAX_ORDER = 10
# audio after the time of a frame in its features: half a window, and the frames of the deltas
DELAY = canary_mfcc_config(SAMPLING_RATE)["win_length"] / 2 / SAMPLING_RATE


def prepare(songs, train_songs, frame):
    """
    Deltas recomputed without wrapping (scaled on the train songs as the others; removed with NO_DELTAS, the
    prediction points staying the same) and, for each song, its phrases
    songs[w]["next_phrases"] (with the calls if CALLS) and the prediction points: songs[w]["points"] (frame of each
    point, -1 for the initial zero state), songs[w]["starts"] (first frame of the phrase ending at each point, -1 for
    the point before the first phrase) and songs[w]["next"] (label of the next phrase, END after the last one).
    Returns the classes (the phrase labels and END).
    """
    annotations = annotations_of_songs() if CALLS else None
    for song in songs.values():
        static = song["mfcc"][:, :N_STATIC]
        deltas = [delta(static.T, width=DELTA_WIDTH, order=order, mode="nearest").T for order in (1, 2)]
        song["mfcc"] = np.hstack([static, *deltas])
    train = np.concatenate([songs[w]["mfcc"][:, N_STATIC:] for w in train_songs])
    low, high = train.min(axis=0), train.max(axis=0)
    delay = DELAY + DELTA_WIDTH // 2 * frame
    for song in songs.values():
        song["mfcc"][:, N_STATIC:] = (song["mfcc"][:, N_STATIC:] - low) / np.where(high > low, high - low, 1)
        if NO_DELTAS:
            song["mfcc"] = np.ascontiguousarray(song["mfcc"][:, :N_STATIC])
    for wave, song in songs.items():
        phrases = song_phrases(annotations, wave, ("SIL", "TRASH")) if CALLS else [tuple(p) for p in song["phrases"]]
        song["next_phrases"] = phrases
        labels = [label for _, _, label in phrases]
        times = [phrases[0][0]] + [end for _, end, _ in phrases]  # before the 1st, after each phrase
        song["points"] = np.array([min(int(np.floor((t - delay) / frame)), len(song["mfcc"]) - 1) for t in times])
        song["points"] = np.maximum(song["points"], -1)
        song["starts"] = np.array([-1] + [int(np.floor(start / frame)) for start, _, _ in phrases])
        song["next"] = np.array(labels + [END], dtype=object)
    return np.array(sorted({label for song in songs.values() for label in song["next"]} - {END}) + [END], dtype=object)


def point_states(reservoirs, songs, waves):
    """
    {wave: states of the reservoirs (inputs without reservoir) at the prediction points, (points, units)}. With the
    reservoirs reset at each phrase (RESET): last state of a run from a zero state on the frames of the phrase ending
    at the point, the zero state for the point before the first phrase and for a phrase shorter than the delay of the
    points (no frame before its point).
    """
    result = {}
    if not RESET:
        for part, part_states in state_chunks(reservoirs, songs, waves):
            for x, w in zip(part_states, part):
                x = np.vstack([np.zeros((1, x.shape[1]), dtype=np.float64), np.asarray(x, dtype=np.float64)])
                result[w] = x[songs[w]["points"] + 1]  # row 0: the initial zero state (point -1)
        return result
    for i in range(0, len(waves), CHUNK):
        part = waves[i:i + CHUNK]
        segments, where = [], []
        for w in part:
            for j, (start, point) in enumerate(zip(songs[w]["starts"], songs[w]["points"])):
                if 0 <= start <= point:
                    segments.append(songs[w]["mfcc"][start:point + 1])
                    where.append((w, j))
        runs = states(reservoirs, segments)
        for w in part:
            result[w] = np.zeros((len(songs[w]["points"]), runs[0].shape[1]))
        for (w, j), run in zip(where, runs):
            result[w][j] = run[-1]
    return result


def statistics(states, songs, waves, classes):
    """Centered statistics of the ridge readout (canary.models.ridge_readout) on the one-hot next labels."""
    X = np.vstack([states[w] for w in waves])
    Y = np.vstack([songs[w]["next"][:, None] == classes[None, :] for w in waves]).astype(np.float64)
    x_mean, y_mean = X.mean(axis=0), Y.mean(axis=0)
    Xc = X - x_mean
    return Xc.T @ Xc, Xc.T @ (Y - y_mean), x_mean, y_mean


def outputs(states, waves, Wout):
    return np.vstack([states[w] @ Wout[:-1] + Wout[-1] for w in waves])


def choose_ridge(model, params, size, songs, train_songs, frame, classes):
    """Ridge coefficient of the best out-of-fold accuracy, in a 3-fold cross-validation by song (the folds of
    canary.next_phrase_prediction.hpo), and this accuracy."""
    correct = np.zeros(len(RIDGES))
    total = 0
    for k, (fit_index, val_index) in enumerate(KFold(N_FOLDS, shuffle=True, random_state=FOLD_SEED).split(train_songs)):
        np.random.seed(CV_SEED + k)
        fit, val = [train_songs[i] for i in fit_index], [train_songs[i] for i in val_index]
        reservoirs = build(model, params, size, pretrain_phrases(songs, fit, frame))
        states = point_states(reservoirs, songs, fit + val)
        stats = statistics(states, songs, fit, classes)
        true = np.concatenate([songs[w]["next"] for w in val])
        for r, ridge in enumerate(RIDGES):
            predicted = classes[outputs(states, val, ridge_readout(stats, ridge)).argmax(axis=1)]
            correct[r] += (predicted == true).sum()
        total += len(true)
    best = int(np.argmax(correct))
    return float(RIDGES[best]), correct[best] / total


def metrics(scores, true, classes):
    """Maxime's metrics of the scores (points, classes) of the next labels true (points,)."""
    predicted = classes[scores.argmax(axis=1)]
    phrase = true != END
    no_end = classes[:-1][scores[:, :-1].argmax(axis=1)]  # (END is the last class)

    def f1(t, p):
        return f1_score(t.astype(str), p.astype(str), average="macro", zero_division=0)
    return dict(accuracy=np.mean(predicted == true), f1_macro=f1(true, predicted),
                accuracy_skip_last=np.mean(predicted[phrase] == true[phrase]),
                f1_macro_skip_last=f1(true[phrase], predicted[phrase]),
                accuracy_no_end=np.mean(no_end[phrase] == true[phrase]), f1_macro_no_end=f1(true[phrase], no_end[phrase]),
                end_recall=np.mean(predicted[~phrase] == END))


def evaluate(model, params, size, ridge, songs, train_songs, test_songs, frame, classes, seed):
    np.random.seed(seed)
    reservoirs = build(model, params, size, pretrain_phrases(songs, train_songs, frame))
    states = point_states(reservoirs, songs, train_songs + test_songs)
    Wout = ridge_readout(statistics(states, songs, train_songs, classes), ridge)
    return metrics(outputs(states, test_songs, Wout), np.concatenate([songs[w]["next"] for w in test_songs]), classes)


# ------------------------------------------------- n-grams --------------------------------------------------------
def baselines():
    """For each order (None: adaptive), metrics of the scores (frequencies of the next labels) of the n-gram at the
    test points, and the share of the points whose context was not seen in the train songs (marginal frequencies
    then)."""
    songs, train_songs, test_songs, frame = load_dataset(top_db=None)
    classes = prepare(songs, train_songs, frame)
    true = np.concatenate([songs[w]["next"] for w in test_songs])
    index = {label: i for i, label in enumerate(classes)}

    def contexts(song, n):
        history = [START] * n + [label for _, _, label in song["next_phrases"]]
        return [tuple(history[j:j + n]) for j in range(len(song["next"]))]

    rows = []
    for order in [*range(0, MAX_ORDER + 1), None]:
        orders = list(range(MAX_ORDER, -1, -1)) if order is None else sorted({order, 0}, reverse=True)
        counts = {n: {} for n in orders}
        for w in train_songs:
            for n in orders:
                for context, label in zip(contexts(songs[w], n), songs[w]["next"]):
                    counts[n].setdefault(context, Counter())[label] += 1
        scores, unseen = [], 0
        for w in test_songs:
            by_order = {n: contexts(songs[w], n) for n in orders}
            for j in range(len(songs[w]["next"])):
                n = next(n for n in orders if by_order[n][j] in counts[n])
                unseen += n == 0 and (order is None or order > 0)
                row = np.zeros(len(classes))
                for label, count in counts[n][by_order[n][j]].items():
                    row[index[label]] = count
                scores.append(row / row.sum())
        rows.append(dict(condition="ngram_adaptive" if order is None else f"ngram_{order}", order=order,
                         unseen_context=unseen / len(scores), **metrics(np.array(scores), true, classes)))
    results = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(results.round(4).to_string(index=False))
    file_name = OUTPUTS / f"next_phrase_{'calls_' * CALLS}ngrams.csv"
    results.to_csv(file_name, index=False)
    print("Saved to", file_name)


# ------------------------------------------------ reservoirs ------------------------------------------------------
NAME = f"{STUDY_PREFIX}hpo"
PARTS = OUTPUTS / f"{NAME}_parts"


def all_conditions(sizes):
    """[(condition, model, params, ridge, size)], mfcc first (size 0); ridge None when chosen by cross-validation."""
    todo = [("mfcc", "mfcc", {}, None, 0)]
    for size in sizes:
        todo += [(condition, model, params, ridge, size) for condition, model, params, ridge, _ in conditions(size)]
    return todo


def run(todo, nb_seeds):
    songs, train_songs, test_songs, frame = load_dataset(top_db=None)
    classes = prepare(songs, train_songs, frame)
    print(f"{len(train_songs)} train / {len(test_songs)} test songs, {len(classes)} classes, "
          f"{sum(len(songs[w]['next']) for w in test_songs)} test points", flush=True)
    rows = []
    for condition, model, params, ridge, size in todo:
        cv_accuracy = np.nan
        if ridge is None:
            ridge, cv_accuracy = choose_ridge(model, params, size, songs, train_songs, frame, classes)
        for seed in range(nb_seeds):
            result = evaluate(model, params, size, ridge, songs, train_songs, test_songs, frame, classes, seed)
            rows.append(dict(condition=condition, network_size=size, seed=seed, ridge=ridge, cv_accuracy=cv_accuracy,
                             **result))
        accuracy = [row["accuracy"] for row in rows[-nb_seeds:]]
        print(f"{condition} ({size}): accuracy {100 * np.mean(accuracy):.2f} ± {100 * np.std(accuracy):.2f} % "
              f"(ridge {ridge:.0e})", flush=True)
    return pd.DataFrame(rows)


def save(results):
    comparisons = compare(results, metric="accuracy")
    pd.set_option("display.width", 200)
    print(comparisons.round(4).to_string(index=False))
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUTS / f"{NAME}.csv", index=False)
    comparisons.to_csv(OUTPUTS / f"{NAME}_comparisons.csv", index=False)
    print("Saved to", OUTPUTS / f"{NAME}.csv", "and", OUTPUTS / f"{NAME}_comparisons.csv")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "baselines":
        baselines()
    elif TASK != "next_phrase":
        raise RuntimeError("HAG_CANARY_TASK=next_phrase is needed (names of the studies)")
    elif command == "list":
        for condition, _, _, _, size in all_conditions([int(s) for s in sys.argv[2].split(",")]):
            print(condition, size)
    elif command == "run":
        condition, size = sys.argv[2], int(sys.argv[3])
        nb_seeds = int(sys.argv[4]) if len(sys.argv) > 4 else 10
        todo = [c for c in all_conditions([size] if size else []) if c[0] == condition and c[4] == size]
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
        save(run(all_conditions(sizes), nb_seeds))

