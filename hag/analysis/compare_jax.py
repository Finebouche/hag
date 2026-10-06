"""
Fit time of the JAX version of HAGReservoir (hag.models.jax_hag_reservoir) and of reservoirpy's NumPy HAGReservoir, on
the classification datasets, for mean_hag (FUNCTIONS) with the best hyperparameters of HAG's studies and the same
initial matrices. One fit per new instance, NB_TRIALS instances (seeds SEED + k) per dataset and function, times in
seconds: first fit and mean of the next ones. The JAX fit is compiled at each call (about 0.2 to 0.5 s).
The results are written to outputs/analysis_results/jax_hag_fit_time.csv.

(A previous version compared this compilation at each fit with a compilation cache shared by the instances, jitted
functions at module level: 0.2 to 0.45 s saved per fit, -7 to -47 % on the small datasets, nothing on SPEECHCOMMANDS,
identical W. Results in outputs/analysis_results/jax_hag_compilation.csv.)

Run from the repository root:  python -m hag.analysis.compare_jax [dataset ...]
"""
import sys
import time

import numpy as np
import pandas as pd
from numpy import random

from hag.analysis.compare_hag_implementations import initial_matrices
from hag.analysis.utils import ANALYSIS_RESULTS
from hag.datasets.pipeline import prepare_data
from hag.hpo.utility import hag_reservoir_from_hyperparameters, retrieve_best_model

# =============================== PARAMETERS ===============================
DATASETS = ["JapaneseVowels", "CatsDogs", "FSDD", "SpokenArabicDigits", "SPEECHCOMMANDS"]
FUNCTIONS = ["mean_hag"]
SEED = 923984      # instance k uses the seed SEED + k (initial matrices and HAG's random choices)
NB_TRIALS = 5      # instances (one fit each) per dataset, function and implementation
DATA_SEED = 0      # draw of the pretraining instances among the train set
OUTPUT = ANALYSIS_RESULTS / "jax_hag_fit_time.csv"
# ===========================================================================


def fit_time(hp, function, pretrain, seed, jax):
    """Fit time of a new instance of the JAX (jax) or NumPy HAGReservoir."""
    W, Win, bias = initial_matrices(hp, pretrain[0].shape[1], seed)
    node = hag_reservoir_from_hyperparameters(hp, function, input_dim=pretrain[0].shape[1], W=W, Win=Win, bias=bias,
                                              seed=seed, jax=jax)
    start = time.time()
    np.asarray(node.fit(list(pretrain)).W)  # (np.asarray waits for the end of the computation)
    return time.time() - start


def compare(dataset, function, pretrain, is_multivariate):
    hp = dict(retrieve_best_model(function, dataset, is_multivariate, verbosity=0).best_trial.params)
    row = {"dataset": dataset, "function": function, "timesteps": sum(len(x) for x in pretrain)}
    for name, jax in (("jax", True), ("numpy", False)):
        times = [fit_time(hp, function, pretrain, SEED + k, jax) for k in range(NB_TRIALS)]
        row.update({f"{name}_first_s": times[0], f"{name}_next_s": np.mean(times[1:])})
    row["speedup"] = row["numpy_next_s"] / row["jax_next_s"]
    return row


def main(datasets):
    rows = []
    for dataset in datasets:
        random.seed(DATA_SEED)  # prepare_data draws the pretraining instances with numpy's global RNG
        pretrain, _, _, _, _, is_multivariate, _ = prepare_data(dataset, "mfcc")
        for function in FUNCTIONS:
            row = compare(dataset, function, pretrain, is_multivariate)
            rows.append(row)
            print(f"[jax] {dataset:20s} {function:9s} {row['timesteps']} steps | fit: jax {row['jax_first_s']:.2f}s then "
                  f"{row['jax_next_s']:.2f}s, numpy {row['numpy_first_s']:.2f}s then {row['numpy_next_s']:.2f}s | "
                  f"speedup {row['speedup']:.1f}x", flush=True)
            # saved after each comparison, so that an interrupted run keeps its results
            OUTPUT.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(OUTPUT, index=False)
    print(f"Results saved to {OUTPUT}")


if __name__ == "__main__":
    main(sys.argv[1:] or DATASETS)
