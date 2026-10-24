"""
Does the certainty of the syllable classification rise along a phrase? After training the readout of the syllable
classification (canary.syllables_classification.classification), on every phrase of the test songs (merged annotated segments
of one syllable type, SIL, TRASH and calls excluded: songs[w]["phrases"]), frame by frame, the output of the unit of the
true class, its margin and the correctness (canapy's frame accuracy), along the phrase
(canary.certainty).

Reservoirs: best parameters of the syllable studies (canary.syllables_classification.evaluate.conditions, with the variables
of the studies, e.g. HAG_CANARY_INPUT_GAINS=1, HAG_CANARY_LEAK=1), ridge of the best trial of the study (or chosen by the
cross-validation of canary.syllables_classification.classification). Results: outputs/canary/certainty_<size>.csv (bins),
certainty_<size>_phrases.csv (first / last third per condition and seed) and certainty_<size>.png.

Usage: python -m canary.syllables_classification.certainty [size] [conditions] [nb_seeds]
  (default 1000, mean_hag,var_hag,random_ee,random_ee_dense, 3 seeds)
"""
import sys

import numpy as np

from canary.certainty import MIN_FRAMES, curve, evaluate_conditions, save, summarize
from canary.common import load_dataset, pretrain_phrases
from canary.models import build, normal_equations, ridge_readout, state_chunks
from canary.syllables_classification.classification import add_frame_labels, choose_ridge
from canary.syllables_classification.evaluate import conditions


def run(condition, model, params, size, ridge, songs, train_songs, test_songs, frame, classes, seed):
    np.random.seed(seed)
    reservoirs = build(model, params, size, pretrain_phrases(songs, train_songs, frame))
    Wout = ridge_readout(normal_equations(reservoirs, songs, train_songs, target="Y"), ridge)
    labels_index = {label: k for k, label in enumerate(classes)}
    curves = []
    for part, part_states in state_chunks(reservoirs, songs, test_songs):
        for x, w in zip(part_states, part):
            output = x.astype(np.float64) @ Wout[:-1] + Wout[-1]
            for start, end, label in songs[w]["phrases"]:
                f0, f1 = int(np.floor(start / frame)), min(int(np.floor(end / frame)), len(output))
                if f1 - f0 >= MIN_FRAMES and label in labels_index:
                    curves.append(curve(output[f0:f1], labels_index[label], label, frame))
    return summarize(curves, condition, seed)


if __name__ == "__main__":
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    wanted = sys.argv[2].split(",") if len(sys.argv) > 2 else ["mean_hag", "var_hag", "random_ee", "random_ee_dense"]
    nb_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    songs, train_songs, test_songs, frame = load_dataset()
    classes = add_frame_labels(songs, frame)
    bins, phrases = evaluate_conditions(conditions, size, wanted, nb_seeds, songs, train_songs, test_songs, frame, classes,
                                        choose_ridge, run)
    save(bins, phrases, "certainty", size, "Certainty along the phrases",
         ("Output of the true-class unit", "Frame accuracy"))
