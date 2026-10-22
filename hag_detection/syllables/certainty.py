"""
Does the certainty of the syllable classification rise along a phrase? After training the readout of the syllable
classification (hag_detection.syllables.classification), on every phrase of the test songs (merged annotated segments
of one syllable type, SIL, TRASH and calls excluded: songs[w]["phrases"]), frame by frame, the output of the unit of the
true class, its margin and the correctness (canapy's frame accuracy), along the phrase (hag_detection.certainty).

Reservoirs: best parameters of the studies of hag_detection.evaluate.conditions (syllable studies with
HAG_DETECTION_TASK=syllables, boundary studies otherwise), ridge of the best trial of the study (or chosen by the
cross-validation of hag_detection.syllables.classification). Results: outputs/detection/certainty_<size>.csv (bins),
certainty_<size>_phrases.csv (first / last third per condition and seed) and certainty_<size>.png.

Usage: python -m hag_detection.syllables.certainty [size] [conditions] [nb_seeds] [max_songs]
  (default 1000, mean_hag,var_hag,random_ee,random_ee_dense, 3 seeds)
"""
import sys

import numpy as np

from hag_detection.certainty import MIN_FRAMES, curve, evaluate_conditions, save, summarize
from hag_detection.common import load_dataset, pretrain_phrases
from hag_detection.models import build, normal_equations, ridge_readout, state_chunks
from hag_detection.syllables.classification import add_frame_labels, choose_ridge


def phrase_curves(output, labels_index, phrases, frame):
    """Curves (hag_detection.certainty.curve) of the readout output (frames, classes) of a song on its phrases."""
    curves = []
    for start, end, label in phrases:
        f0, f1 = int(np.floor(start / frame)), min(int(np.floor(end / frame)), len(output))
        if f1 - f0 >= MIN_FRAMES and label in labels_index:
            curves.append(curve(output[f0:f1], labels_index[label], label, frame))
    return curves


def run(condition, model, params, size, ridge, songs, train_songs, test_songs, frame, classes, seed):
    np.random.seed(seed)
    reservoirs = build(model, params, size, pretrain_phrases(songs, train_songs, frame))
    Wout = ridge_readout(normal_equations(reservoirs, songs, train_songs, target="Y"), ridge)
    labels_index = {label: k for k, label in enumerate(classes)}
    curves = []
    for part, part_states in state_chunks(reservoirs, songs, test_songs):
        for x, w in zip(part_states, part):
            output = x.astype(np.float64) @ Wout[:-1] + Wout[-1]
            curves += phrase_curves(output, labels_index, songs[w]["phrases"], frame)
    return summarize(curves, condition, seed)


if __name__ == "__main__":
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    wanted = sys.argv[2].split(",") if len(sys.argv) > 2 else ["mean_hag", "var_hag", "random_ee", "random_ee_dense"]
    nb_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    max_songs = int(sys.argv[4]) if len(sys.argv) > 4 else None

    songs, train_songs, test_songs, frame = load_dataset(max_songs)
    classes = add_frame_labels(songs, frame)
    bins, phrases = evaluate_conditions(size, wanted, nb_seeds, songs, train_songs, test_songs, frame, classes,
                                        choose_ridge, run)
    save(bins, phrases, "certainty", size, max_songs is not None, "Certainty along the phrases",
         ("Output of the true-class unit", "Frame accuracy"))
