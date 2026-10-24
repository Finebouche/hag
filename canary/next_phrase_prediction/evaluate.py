"""
Final evaluation of the canary boundary detection on the test songs (used only here), after canary.next_phrase_prediction.hpo.

Conditions, for each SIZE (number of neurons in all):
  - "mfcc" (no reservoir, once);
  - the pure models (canary.models.PURE_MODELS) with the best parameters of their study "<model>_<SIZE>";
  - "<hag>_proj": the parameters of "<hag>_<SIZE>" without the HAG learning;
  - "<hag>+<esn>": hybrid of the best parts optimized alone with SIZE / 2 neurons, without further optimization;
  - "<hag>+<esn>_joint": the best parameters of the joint optimization "<hag>+<esn>_<SIZE>".
The ridge coefficient and the peak threshold are those of the best trial of the study, or, for the conditions without a
study (mfcc, proj, hybrids without joint optimization), chosen with the same cross-validation on the train songs
(canary.next_phrase_prediction.hpo.cross_validate). Missing studies are skipped.

For each seed (the same seeds for all the conditions, i.e. paired), the reservoirs are drawn with np.random.seed(seed)
and learn (HAG) on phrases of all the train songs, the readout is trained on all the train songs and the boundaries of
the test songs are scored. Then paired comparisons by seed: mean difference of F1, its bootstrap 95 % confidence
interval and the p-value of the Wilcoxon signed-rank test.

Results: outputs/canary/evaluation.csv (one row per condition, size and seed) and comparisons.csv.
Usage: python -m canary.next_phrase_prediction.evaluate [sizes] [nb_seeds]   (default 250,500,1000 and 10 seeds)
"""
import sys

import numpy as np
import pandas as pd

from canary.common import OUTPUTS, pretrain_phrases
from canary.evaluate import compare, conditions as task_conditions
from canary.models import build, normal_equations, readout_outputs, ridge_readout
from canary.next_phrase_prediction.boundaries import load_dataset, scores
from canary.next_phrase_prediction.hpo import cross_validate, study_name_of

SIZES = [250, 500, 1000]


def conditions(size):
    """canary.evaluate.conditions of the studies of the task (canary.next_phrase_prediction.hpo.study_name_of)."""
    return task_conditions(size, study_name_of)


def evaluate(model, params, size, ridge, threshold, songs, train_songs, test_songs, frame, seed):
    np.random.seed(seed)
    reservoirs = build(model, params, size, pretrain_phrases(songs, train_songs, frame))
    Wout = ridge_readout(normal_equations(reservoirs, songs, train_songs), ridge)
    test_outputs = readout_outputs(reservoirs, songs, test_songs, [Wout])
    outputs = [test_outputs[w][:, 0] for w in test_songs]
    return scores(outputs, songs, test_songs, threshold, frame)


if __name__ == "__main__":
    sizes = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else SIZES
    nb_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    songs, train_songs, test_songs, frame = load_dataset()
    print(f"{len(train_songs)} train / {len(test_songs)} test songs, "
          f"{sum(len(songs[w]['boundaries']) for w in test_songs)} test boundaries", flush=True)

    todo = [("mfcc", "mfcc", {}, None, None, 0)]
    for size in sizes:
        todo += [(*condition, size) for condition in conditions(size)]

    rows = []
    for condition, model, params, ridge, threshold, size in todo:
        if ridge is None:
            _, ridge, threshold = cross_validate(model, params, size, songs, train_songs, frame)
        for seed in range(nb_seeds):
            result = evaluate(model, params, size, ridge, threshold, songs, train_songs, test_songs, frame, seed)
            rows.append(dict(condition=condition, network_size=size, seed=seed, ridge=ridge, threshold=threshold,
                             **result))
        f1 = [row["f1"] for row in rows[-nb_seeds:]]
        print(f"{condition} ({size}): F1 {np.mean(f1):.3f} ± {np.std(f1):.3f} (ridge {ridge:.0e}, threshold "
              f"{threshold:.2f})", flush=True)

    results = pd.DataFrame(rows)
    comparisons = compare(results, metric="f1")
    pd.set_option("display.width", 200)
    print(comparisons.round(4).to_string(index=False))
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUTS / "evaluation.csv", index=False)
    comparisons.to_csv(OUTPUTS / "comparisons.csv", index=False)
    print("Saved to", OUTPUTS / "evaluation.csv", "and", OUTPUTS / "comparisons.csv")

