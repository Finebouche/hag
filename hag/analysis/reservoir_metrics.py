"""
Richness and separability of the reservoirs built with the best hyperparameters of the corrected studies (the "Richness"
and "Separability" sections of 2.test_analysis.ipynb), on the test set of each dataset:
  - richness (hag.performances.evaluation_esn.evaluate_dataset_on_test with record_metrics): spectral radius of W,
    Pearson correlation, CEV and distance correlation of the states on the test inputs;
  - separability (hag.metrics.separability): inter- and intra-class distances and their ratio, silhouette,
    Davies-Bouldin and Calinski-Harabasz indices of the last states of the test sequences, for the reservoirs of
    hag.performances.evaluation_esn.build_reservoir_from_study (those of the test evaluation).
nb_trials reservoirs per dataset and function (mean and standard deviation over them), numpy's global RNG seeded with
SEED before each of the two.

One row per dataset and function in outputs/hag/metrics.csv (the rows computed replace the previous ones of the same
dataset and function, saved after each function), and the same rounded to 5 decimals in metrics_rounded.csv.
HAG_METRICS_JOBS (environment variable): processes of the distance correlation (slurm/analysis.sbatch: the cores of the
job).

Usage: python -m hag.analysis.reservoir_metrics [datasets] [functions] [nb_trials]
  e.g. python -m hag.analysis.reservoir_metrics JapaneseVowels,CatsDogs,FSDD
"""
import sys

import numpy as np
import pandas as pd

from hag.datasets.pipeline import prepare_data
from hag.hpo.utility import PROJECT_ROOT, retrieve_best_model
from hag.metrics.separability import calinski_harabasz, davies_bouldin, inter_intra_class_distance, silhouette
from hag.performances.esn_model_evaluation import run_reservoir
from hag.performances.evaluation_esn import build_reservoir_from_study, evaluate_dataset_on_test

DATASETS = ["JapaneseVowels", "CatsDogs", "FSDD"]
FUNCTIONS = ["random_ee", "random_ei", "ip_correct", "anti-oja", "ip-anti-oja", "mean_hag", "var_hag"]
NB_TRIALS = 4
SEED = 0
OUTPUT = PROJECT_ROOT / "outputs" / "hag" / "metrics.csv"
OUTPUT_ROUNDED = PROJECT_ROOT / "outputs" / "hag" / "metrics_rounded.csv"


def richness(study, dataset_name, function_name, data, nb_trials):
    pretrain_data, train_data, test_data, Y_train, Y_test, _, is_instances_classification = data
    spectral_radii, pearsons, CEVs, dcors = evaluate_dataset_on_test(
        study, dataset_name, function_name, pretrain_data, train_data, test_data, Y_train, Y_test,
        is_instances_classification, nb_trials=nb_trials, record_metrics=True, seed=SEED)
    return {"spectral_radius": spectral_radii, "pearson": pearsons, "CEVD": CEVs, "dcor": dcors}


def separability(study, function_name, data, nb_trials):
    pretrain_data, _, test_data, _, Y_test, _, is_instances_classification = data
    hyperparams = dict(study.best_trial.params)
    labels = np.argmax(Y_test, axis=1) if np.ndim(Y_test) == 2 else np.asarray(Y_test)
    values = {key: [] for key in ["inter_dists", "intra_dists", "separability_ratios", "silhouette", "davies_bouldin",
                                  "calinski_harabasz"]}
    np.random.seed(SEED)
    for _ in range(nb_trials):
        reservoir, _ = build_reservoir_from_study(function_name, hyperparams, pretrain_data,
                                                  is_instances_classification)
        final_states = np.array([states[-1] for states in run_reservoir(reservoir, list(test_data), reset=True)])
        inter, intra, ratio = inter_intra_class_distance(final_states, labels)
        values["inter_dists"].append(inter)
        values["intra_dists"].append(intra)
        values["separability_ratios"].append(ratio)
        values["silhouette"].append(silhouette(final_states, labels))
        values["davies_bouldin"].append(davies_bouldin(final_states, labels))
        values["calinski_harabasz"].append(calinski_harabasz(final_states, labels))
    return values


def main():
    datasets = sys.argv[1].split(",") if len(sys.argv) > 1 else DATASETS
    functions = sys.argv[2].split(",") if len(sys.argv) > 2 else FUNCTIONS
    nb_trials = int(sys.argv[3]) if len(sys.argv) > 3 else NB_TRIALS

    for dataset_name in datasets:
        np.random.seed(SEED)  # pretraining instances drawn by prepare_data
        data = prepare_data(dataset_name, "mfcc", visualize=False)
        is_multivariate = data[5]
        for function_name in functions:
            print(f"\n=== {dataset_name} / {function_name}", flush=True)
            study = retrieve_best_model(function_name, dataset_name, is_multivariate, verbosity=0)
            metrics = {**richness(study, dataset_name, function_name, data, nb_trials),
                       **separability(study, function_name, data, nb_trials)}
            row = {"dataset": dataset_name, "function_name": function_name}
            for key, values in metrics.items():
                row[f"{key}_mean"], row[f"{key}_std"] = float(np.mean(values)), float(np.std(values))
            print(row, flush=True)
            save([row])
    print(f"Results saved to {OUTPUT} and {OUTPUT_ROUNDED}")


def save(rows):
    """Add the rows to OUTPUT (and OUTPUT_ROUNDED), replacing those of the same dataset and function."""
    new = pd.DataFrame(rows).set_index(["dataset", "function_name"])
    if OUTPUT.exists():
        previous = pd.read_csv(OUTPUT).set_index(["dataset", "function_name"])
        new = pd.concat([previous.drop(index=new.index, errors="ignore"), new])
    new = new.reset_index().sort_values(["dataset", "function_name"])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(OUTPUT, index=False)
    new.round(5).to_csv(OUTPUT_ROUNDED, index=False)


if __name__ == "__main__":
    main()
