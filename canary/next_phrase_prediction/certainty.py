"""
Certainty of the prediction of the next phrase (canary.next_phrase_prediction.prediction) along the current phrase: its
readout is trained at the prediction points of the train songs, then applied to every frame of each phrase of the test
songs up to its prediction point (no frame hearing the next phrase), the true class being the next phrase (END after
the last one); output of its unit, margin and correctness along the phrase (canary.certainty). With
HAG_CANARY_RESET=1, the frames of a phrase are those of a run of the reservoirs from a zero state on the phrase.

Reservoirs: best parameters of the next-phrase studies (HAG_CANARY_TASK=next_phrase and the variables of the
studies, e.g. HAG_CANARY_LEAK=1, HAG_CANARY_RESET=1, HAG_CANARY_CALLS=1), ridge of the best trial of the study.
Results: outputs/canary/<prefix of the studies>certainty_<size>.csv, ..._phrases.csv and .png (e.g.
next_phrase_leak_certainty_1000.csv).

Usage: python -m canary.next_phrase_prediction.certainty [size] [conditions] [nb_seeds]
  (default 1000, mean_hag,var_hag,random_ee,random_ei, 3 seeds)
"""
import sys

import numpy as np

from canary.certainty import MIN_FRAMES, curve, evaluate_conditions, save, summarize
from canary.common import load_dataset, pretrain_phrases
from canary.models import CHUNK, build, ridge_readout, state_chunks, states
from canary.next_phrase_prediction.evaluate import conditions
from canary.next_phrase_prediction.hpo import RESET, STUDY_PREFIX, TASK
from canary.next_phrase_prediction.prediction import choose_ridge, point_states, prepare, statistics


def phrase_frames(songs, waves, frame):
    """[(wave, j, first frame, end frame)] of each phrase j of the songs: from its onset to its prediction point
    (canary.next_phrase_prediction.prediction), the phrases of fewer than MIN_FRAMES frames skipped."""
    result = []
    for w in waves:
        for j, (start, _, _) in enumerate(songs[w]["next_phrases"]):
            f0, f1 = int(np.floor(start / frame)), songs[w]["points"][j + 1] + 1
            if f1 - f0 >= MIN_FRAMES:
                result.append((w, j, f0, f1))
    return result


def next_phrase_curves(reservoirs, Wout, songs, test_songs, frame, classes):
    """Curves of the readout of the next phrase on the frames of the phrases of the test songs (phrase_frames)."""
    index = {label: k for k, label in enumerate(classes)}
    curves = []
    for i in range(0, len(test_songs), CHUNK):
        part = test_songs[i:i + CHUNK]
        phrases = phrase_frames(songs, part, frame)
        if RESET:  # runs of the reservoirs on each phrase, from a zero state
            runs = states(reservoirs, [songs[w]["mfcc"][f0:f1] for w, _, f0, f1 in phrases])
            outputs = [run.astype(np.float64) @ Wout[:-1] + Wout[-1] for run in runs]
        else:
            song_outputs = {w: x.astype(np.float64) @ Wout[:-1] + Wout[-1]
                            for waves, xs in state_chunks(reservoirs, songs, part) for w, x in zip(waves, xs)}
            outputs = [song_outputs[w][f0:f1] for w, _, f0, f1 in phrases]
        for (w, j, _, _), out in zip(phrases, outputs):
            label = songs[w]["next"][j + 1]
            curves.append(curve(out, index[label], label, frame))
    return curves


def run(condition, model, params, size, ridge, songs, train_songs, test_songs, frame, classes, seed):
    np.random.seed(seed)
    reservoirs = build(model, params, size, pretrain_phrases(songs, train_songs, frame))
    Wout = ridge_readout(statistics(point_states(reservoirs, songs, train_songs), songs, train_songs, classes), ridge)
    return summarize(next_phrase_curves(reservoirs, Wout, songs, test_songs, frame, classes), condition, seed)


if __name__ == "__main__":
    if TASK != "next_phrase":
        raise RuntimeError("HAG_CANARY_TASK=next_phrase is needed (names of the studies)")
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    wanted = sys.argv[2].split(",") if len(sys.argv) > 2 else ["mean_hag", "var_hag", "random_ee", "random_ei"]
    nb_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    songs, train_songs, test_songs, frame = load_dataset()
    classes = prepare(songs, train_songs, frame)
    bins, phrases = evaluate_conditions(conditions, size, wanted, nb_seeds, songs, train_songs, test_songs, frame, classes,
                                        choose_ridge, run)
    save(bins, phrases, f"{STUDY_PREFIX}certainty", size, "Certainty of the next phrase along the phrases",
         ("Output of the next-phrase unit", "Accuracy of the next phrase"))
