"""
Final evaluation of the canary boundary detection on the test songs (used only here), after hag_detection.hpo.

Conditions, for each SIZE (number of neurons in all):
  - "mfcc" (no reservoir, once);
  - the pure models (hag_detection.models.PURE_MODELS) with the best parameters of their study "<model>_<SIZE>";
  - "<hag>_proj": the parameters of "<hag>_<SIZE>" without the HAG learning;
  - "<hag>+<esn>": hybrid of the best parts optimized alone with SIZE / 2 neurons, without further optimization;
  - "<hag>+<esn>_joint": the best parameters of the joint optimization "<hag>+<esn>_<SIZE>".
The ridge coefficient and the peak threshold are those of the best trial of the study, or, for the conditions without a
study (mfcc, proj, hybrids without joint optimization), chosen with the same cross-validation on the train songs
(hag_detection.hpo.cross_validate). Missing studies are skipped.

For each seed (the same seeds for all the conditions, i.e. paired), the reservoirs are drawn with np.random.seed(seed)
and learn (HAG) on phrases of all the train songs, the readout is trained on all the train songs and the boundaries of
the test songs are scored. Then paired comparisons by seed: mean difference of F1, its bootstrap 95 % confidence
interval and the p-value of the Wilcoxon signed-rank test.

Results: outputs/detection/evaluation.csv (one row per condition, size and seed) and comparisons.csv.
Usage: python -m hag_detection.evaluate [sizes] [nb_seeds] [max_songs]   (default 250,500,1000 and 10 seeds)
"""
import sys

import numpy as np
import optuna
import pandas as pd
from scipy.stats import wilcoxon

from hag_detection.common import OUTPUTS, load_dataset, pretrain_phrases, scores
from hag_detection.hpo import cross_validate, storage, study_name_of
from hag_detection.models import (ESN_MODELS, HAG_MODELS, PURE_MODELS, build, normal_equations, parts, readout_outputs,
                                  ridge_readout)

SIZES = [250, 500, 1000]
REFERENCE = "random_ee"


def best_trial(study_name):
    try:
        return optuna.load_study(study_name=study_name, storage=storage(study_name)).best_trial
    except (KeyError, ValueError):
        return None


def conditions(size):
    """[(condition, model, params, ridge, threshold)], ridge and threshold None when chosen by cross-validation."""
    result = []

    def from_study(condition, model, study_name):
        trial = best_trial(study_name)
        if trial is None:
            print(f"  (no study {study_name}: {condition} skipped)")
        else:
            result.append((condition, model, trial.params, trial.user_attrs["ridge"], trial.user_attrs.get("threshold")))

    for model in PURE_MODELS:
        from_study(model, model, study_name_of(model, size))
    for hag in HAG_MODELS:
        trial = best_trial(study_name_of(hag, size))
        if trial is not None:
            result.append((f"{hag}_proj", f"{hag}_proj", trial.params, None, None))
    for hag in HAG_MODELS:
        for esn in ESN_MODELS:
            hybrid = f"{hag}+{esn}"
            halves = [best_trial(study_name_of(part, size // 2)) for part, _ in parts(hybrid)]
            if all(trial is not None for trial in halves):
                params = {prefix + key: value for (_, prefix), trial in zip(parts(hybrid), halves)
                          for key, value in trial.params.items()}
                result.append((hybrid, hybrid, params, None, None))
            from_study(f"{hybrid}_joint", hybrid, study_name_of(hybrid, size))
    return result


def evaluate(model, params, size, ridge, threshold, songs, train_songs, test_songs, frame, seed):
    np.random.seed(seed)
    reservoirs = build(model, params, size, pretrain_phrases(songs, train_songs, frame))
    Wout = ridge_readout(normal_equations(reservoirs, songs, train_songs), ridge)
    test_outputs = readout_outputs(reservoirs, songs, test_songs, [Wout])
    outputs = [test_outputs[w][:, 0] for w in test_songs]
    return scores(outputs, songs, test_songs, threshold, frame)


def compare(results, nb_bootstrap=10000, metric="f1"):
    """Paired comparisons by seed of the metric (condition A - condition B), for each size."""
    rows = []
    rng = np.random.default_rng(0)
    mfcc = results[results.condition == "mfcc"]
    for size, by_size in results[results.condition != "mfcc"].groupby("network_size"):
        by_size = pd.concat([by_size, mfcc])
        f1 = by_size.pivot_table(index="seed", columns="condition", values=metric)
        pairs = [(c, REFERENCE) for c in f1.columns if c != REFERENCE]
        pairs += [(hag, f"{hag}_proj") for hag in HAG_MODELS]
        for hag in HAG_MODELS:
            for esn in ESN_MODELS:
                hybrid = f"{hag}+{esn}"
                pairs += [(hybrid, hag), (hybrid, esn), (f"{hybrid}_joint", hybrid), (f"{hybrid}_joint", hag),
                          (f"{hybrid}_joint", esn)]
        for a, b in dict.fromkeys(pairs):
            if a not in f1.columns or b not in f1.columns:
                continue
            diff = (f1[a] - f1[b]).dropna().to_numpy()
            if len(diff) == 0:
                continue
            means = rng.choice(diff, size=(nb_bootstrap, len(diff))).mean(axis=1)
            p_value = wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
            rows.append(dict(network_size=size, a=a, b=b, nb_seeds=len(diff), mean_diff=diff.mean(),
                             ci_low=np.percentile(means, 2.5), ci_high=np.percentile(means, 97.5),
                             wilcoxon_p=p_value, a_better=int((diff > 0).sum())))
    return pd.DataFrame(rows)


def main():
    sizes = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else SIZES
    nb_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    max_songs = int(sys.argv[3]) if len(sys.argv) > 3 else None

    songs, train_songs, test_songs, frame = load_dataset(max_songs)
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
    comparisons = compare(results)
    pd.set_option("display.width", 200)
    print(comparisons.round(4).to_string(index=False))
    if max_songs is None:
        OUTPUTS.mkdir(parents=True, exist_ok=True)
        results.to_csv(OUTPUTS / "evaluation.csv", index=False)
        comparisons.to_csv(OUTPUTS / "comparisons.csv", index=False)
        print("Saved to", OUTPUTS / "evaluation.csv", "and", OUTPUTS / "comparisons.csv")


if __name__ == "__main__":
    main()
