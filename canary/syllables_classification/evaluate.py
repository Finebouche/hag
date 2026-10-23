"""
Conditions of the syllable classification (syllables_classification.classification) and paired comparisons of their
results. Conditions, for each SIZE (number of neurons in all):
  - "mfcc" (no reservoir, once, added by syllables_classification.classification);
  - the pure models (syllables_classification.models.PURE_MODELS) with the best parameters of their study
    "<prefix><model>_<SIZE>" (syllables_classification.hpo.STUDY_PREFIX);
  - "<hag>_proj": the parameters of "<hag>_<SIZE>" without the HAG learning;
  - "<hag>+<esn>": hybrid of the best parts optimized alone with SIZE / 2 neurons, without further optimization;
  - "<hag>+<esn>_joint": the best parameters of the joint optimization "<hag>+<esn>_<SIZE>".
The ridge coefficient is the one of the best trial of the study, or None for the conditions without a study (chosen
by the cross-validation of syllables_classification.classification). Missing studies are skipped.
"""
import numpy as np
import optuna
import pandas as pd
from scipy.stats import wilcoxon

from syllables_classification.hpo import storage, study_exists, study_name_of
from syllables_classification.models import ESN_MODELS, HAG_MODELS, PURE_MODELS, parts

SIZES = [250, 500, 1000]
REFERENCE = "random_ee"


def best_trial(study_name):
    if not study_exists(study_name):  # (without creating an empty database)
        return None
    try:
        return optuna.load_study(study_name=study_name, storage=storage(study_name)).best_trial
    except (KeyError, ValueError):
        return None


def conditions(size):
    """[(condition, model, params, ridge, threshold)], ridge None when chosen by cross-validation, threshold None (no
    threshold in the classification)."""
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


def compare(results, nb_bootstrap=10000, metric="frame_accuracy"):
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
