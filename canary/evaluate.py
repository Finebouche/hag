"""
Shared parts of the evaluations of the canary tasks: conditions, for each SIZE (number of neurons in all),
  - "mfcc" (no reservoir, once, added by the task);
  - the pure models (canary.models.PURE_MODELS) with the best parameters of their study "<model>_<SIZE>" (named by the
    task, e.g. "next_phrase_leak_<model>_<SIZE>");
  - "<hag>_proj": the parameters of "<hag>_<SIZE>" without the HAG learning;
  - "<hag>+<esn>": hybrid of the best parts optimized alone with SIZE / 2 neurons, without further optimization;
  - "<hag>+<esn>_joint": the best parameters of the joint optimization "<hag>+<esn>_<SIZE>"
(missing studies skipped), and the paired comparisons by seed of their results: mean difference of a metric, its
bootstrap 95 % confidence interval and the p-value of the Wilcoxon signed-rank test.
"""
import numpy as np
import optuna
import pandas as pd
from scipy.stats import wilcoxon

from canary.hpo import storage, study_exists
from canary.models import ESN_MODELS, HAG_MODELS, PURE_MODELS, parts

REFERENCE = "random_ee"


def best_trial(study_name):
    if not study_exists(study_name):  # (without creating an empty database)
        return None
    try:
        return optuna.load_study(study_name=study_name, storage=storage(study_name)).best_trial
    except (KeyError, ValueError):
        return None


def conditions(size, study_name_of):
    """[(condition, model, params, ridge, threshold)] of size neurons, the studies being named by study_name_of(model,
    size) (the task's): ridge and threshold None when chosen by cross-validation, threshold None without threshold."""
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


def compare(results, metric, nb_bootstrap=10000):
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
