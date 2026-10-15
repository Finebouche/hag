"""
Best hyperparameters and best cross-validation scores of the corrected studies (the "Export best parameters" section of
2.test_analysis.ipynb): for each function, outputs/hag/best_hyperparameters/best_hyperparameters_<function>.csv (one row
per dataset, the parameters of the best trial), and best_cv_scores.csv (one row per dataset and function). The studies
not found (e.g. the LSTM and GRU of a dataset not optimized yet) are skipped.

Usage: python -m hag.analysis.best_hyperparameters [datasets] [functions]
"""
import sys

import pandas as pd

from hag.hpo.utility import PROJECT_ROOT, retrieve_best_model

DATASETS = ["JapaneseVowels", "CatsDogs", "FSDD", "SpokenArabicDigits", "SPEECHCOMMANDS"]
FUNCTIONS = ["random_ee", "random_ei", "ip_correct", "anti-oja", "ip-anti-oja", "mean_hag", "var_hag", "lstm_last", "gru"]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "hag" / "best_hyperparameters"


def smart_format(x, ndigits=5):
    """Scientific notation below 1e-3, else rounded to ndigits decimals (non-floats unchanged)."""
    if not isinstance(x, float) or x == 0.0:
        return x
    if abs(x) < 1e-3:
        return f"{x:.{ndigits}e}"
    return round(x, ndigits)


def main():
    datasets = sys.argv[1].split(",") if len(sys.argv) > 1 else DATASETS
    functions = sys.argv[2].split(",") if len(sys.argv) > 2 else FUNCTIONS
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cv_scores = []
    for function_name in functions:
        rows = []
        for dataset_name in datasets:
            try:
                study = retrieve_best_model(function_name, dataset_name, True, verbosity=0)
                best = study.best_trial
            except (FileNotFoundError, KeyError, ValueError) as error:
                print(f"{dataset_name} / {function_name}: skipped ({type(error).__name__})")
                continue
            rows.append({"dataset": dataset_name, "function_name": function_name,
                         **{key: smart_format(value) for key, value in best.params.items()}})
            cv_scores.append({"dataset": dataset_name, "function": function_name, "best_score": best.value,
                              "nb_trials": len(study.trials)})
        if rows:
            pd.DataFrame(rows).to_csv(OUTPUT_DIR / f"best_hyperparameters_{function_name}.csv", index=False)
    pd.DataFrame(cv_scores).to_csv(OUTPUT_DIR / "best_cv_scores.csv", index=False)
    print(f"Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
