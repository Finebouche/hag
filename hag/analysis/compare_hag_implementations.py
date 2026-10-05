"""
Compare HAG's reference implementation (hag.hag.hag.run_algorithm) with reservoirpy's HAGReservoir node, for mean_hag
and var_hag with the best hyperparameters of HAG's studies, on the classification datasets:
  1. weights: same initial matrices and same random draws (the node's random choices taken from numpy's global RNG,
     as run_algorithm) -> identical W ("max_W_diff" = 0),
  2. test accuracy: run_algorithm (random choices of numpy's global RNG) vs the node used as in reservoirpy (random
     choices from its seed), same initial matrices and readout -> same accuracy up to the randomness of HAG's choices.
The results are written to outputs/analysis_results/hag_node_vs_run_algorithm.csv.

Run from the repository root:  python -m hag.analysis.compare_hag_implementations [dataset ...]
"""
import math
import sys
import time

import numpy as np
import pandas as pd
from numpy import random
from scipy import stats

from hag.analysis.utils import ANALYSIS_RESULTS
from hag.datasets.pipeline import prepare_data
from hag.hag.hag import run_algorithm
from hag.hpo.utility import hag_reservoir_from_hyperparameters, retrieve_best_model
from hag.models.activation_functions import tanh
from hag.models.reservoir import init_matrices
from hag.performances.esn_model_evaluation import (compute_score, init_readout, init_reservoir,
                                                   predict_model_for_classification, train_model_for_classification)

# =============================== PARAMETERS ===============================
DATASETS = ["JapaneseVowels", "CatsDogs", "FSDD", "SpokenArabicDigits", "SPEECHCOMMANDS"]
FUNCTIONS = ["mean_hag", "var_hag"]
SEED = 923984      # trial k uses the seed SEED + k (initial matrices and HAG's random choices)
NB_TRIALS = 8      # reservoirs evaluated on the test set
N_W_CHECK = 3      # draws for the exact comparison of the weights
DATA_SEED = 0      # draw of the pretraining instances among the train set
OUTPUT = ANALYSIS_RESULTS / "hag_node_vs_run_algorithm.csv"
# ===========================================================================


def initial_matrices(hp, input_dim, seed):
    """HAG's initial matrices (init_matrices), as in its evaluation."""
    K = math.ceil(hp['network_size'] / input_dim)
    Win, W, bias = init_matrices(input_dim * K, 1, hp['connectivity'], K, w_distribution=stats.uniform(loc=-1, scale=2),
                                 seed=seed)
    return W, Win * hp['input_scaling'], bias * hp['bias_scaling']


def W_run_algorithm(hp, function, pretrain, seed):
    """W learned by the reference implementation, random choices of numpy's global RNG seeded with seed."""
    W, Win, bias = initial_matrices(hp, pretrain[0].shape[1], seed)
    if function == "mean_hag":
        target, spread, extra = hp['target_rate'], hp['rate_spread'], {}
    else:
        target, spread = hp['variance_target'], hp['variance_spread']
        extra = dict(intrinsic_saturation=hp['intrinsic_saturation'], intrinsic_coef=hp['intrinsic_coef'])
    random.seed(seed)
    W, _ = run_algorithm(W, Win, bias, hp['leaky_rate'], tanh, pretrain, hp['weight_increment'], target, spread,
                         function, multiple_instances=True, min_increment=hp['min_increment'],
                         max_increment=hp['max_increment'], use_full_instance=hp['use_full_instance'],
                         max_partners=np.inf, method="pearson", n_jobs=1, progress_bar=False, **extra)
    return np.asarray(W, dtype=float), Win, bias


def W_node(hp, function, pretrain, seed, global_rng=False):
    """W learned by reservoirpy's HAGReservoir with the same initial matrices: random choices from the node's seed, or
    (global_rng) from numpy's global RNG seeded with seed, as run_algorithm."""
    W, Win, bias = initial_matrices(hp, pretrain[0].shape[1], seed)
    node = hag_reservoir_from_hyperparameters(hp, function, input_dim=pretrain[0].shape[1], W=W, Win=Win, bias=bias,
                                              seed=seed)
    if global_rng:
        # verification only: replace the node's generator of random choices by numpy's global RNG
        node.initialize(list(pretrain))
        node._plasticity_rng = random.mtrand._rand
        random.seed(seed)
    return node.fit(list(pretrain)).W, Win, bias


def test_accuracy(W, Win, bias, hp, train, test, Y_train, Y_test):
    """Test accuracy with the readout of HAG's evaluation (last state of each sequence, ridge)."""
    reservoir = init_reservoir(W, Win, bias, hp['leaky_rate'], tanh)
    readout = init_readout(ridge_coef=10 ** hp['ridge'])
    train_model_for_classification(reservoir, readout, train, Y_train, mode="sequence-to-vector")
    Y_pred = predict_model_for_classification(reservoir, readout, test, mode="sequence-to-vector")
    return compute_score(Y_pred, Y_test, True)


def compare(dataset, function, data):
    pretrain, train, test, Y_train, Y_test, is_multivariate, _ = data
    hp = dict(retrieve_best_model(function, dataset, is_multivariate, verbosity=0).best_trial.params)
    # 1. same random draws -> same weights
    w_diff = max(np.abs(W_run_algorithm(hp, function, pretrain, SEED + k)[0]
                        - W_node(hp, function, pretrain, SEED + k, global_rng=True)[0]).max() for k in range(N_W_CHECK))
    row = {"dataset": dataset, "function": function, "max_W_diff": w_diff}

    # 2. test accuracy: reference implementation vs node used as in reservoirpy
    for name, learn in (("run_algorithm", lambda k: W_run_algorithm(hp, function, pretrain, SEED + k)),
                        ("HAGReservoir", lambda k: W_node(hp, function, pretrain, SEED + k))):
        start, scores = time.time(), []
        for k in range(NB_TRIALS):
            W, Win, bias = learn(k)
            scores.append(test_accuracy(W, Win, bias, hp, train, test, Y_train, Y_test))
        row.update({f"{name}_mean_%": 100 * np.mean(scores), f"{name}_std_%": 100 * np.std(scores),
                    f"{name}_time_s": round(time.time() - start)})
    return row


def main(datasets):
    rows = []
    for dataset in datasets:
        random.seed(DATA_SEED)  # prepare_data draws the pretraining instances with numpy's global RNG
        data = prepare_data(dataset, "mfcc")
        for function in FUNCTIONS:
            row = compare(dataset, function, data)
            rows.append(row)
            print(f"[compare] {dataset:20s} {function:9s} max|W diff| {row['max_W_diff']:.1e} | test run_algorithm "
                  f"{row['run_algorithm_mean_%']:.2f}% ± {row['run_algorithm_std_%']:.2f} ({row['run_algorithm_time_s']}s) | "
                  f"HAGReservoir {row['HAGReservoir_mean_%']:.2f}% ± {row['HAGReservoir_std_%']:.2f} "
                  f"({row['HAGReservoir_time_s']}s)", flush=True)
            # saved after each comparison, so that an interrupted run keeps its results
            OUTPUT.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(OUTPUT, index=False)

    results = pd.DataFrame(rows)
    print(f"Results saved to {OUTPUT}")
    if not results["max_W_diff"].eq(0).all():
        print("WARNING: the implementations do not give the same weights")


if __name__ == "__main__":
    main(sys.argv[1:] or DATASETS)
