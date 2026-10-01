"""
Compare HAG's two implementations on the classification datasets:
  - the base algorithm, hag.hag.hag.run_algorithm,
  - the reservoirpy node, hag.models.hag_reservoir.HAGReservoir.

For each dataset and function (mean_hag, var_hag), with the best hyperparameters of HAG's studies:
  1. weights: same initial matrices and same random draws (numpy's global RNG) -> the W learned by both
     implementations must be identical (max |W diff| = 0),
  2. test: hag.analysis.commons.evaluate_dataset_on_test with the same seed, once with run_algorithm and once with
     use_hag_node=True (HAG trained, then run, by HAGReservoir) -> identical test scores.
The results are written to outputs/test_results/hag_node_vs_run_algorithm.csv.

Run from the repository root:  python -m hag.performances.compare_hag_implementations [dataset ...]
"""
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numpy import random
from scipy import stats

from hag.analysis.commons import activation_function, evaluate_dataset_on_test, load_data, study_db_dir
from hag.hag.hag import run_algorithm
from hag.hpo.utility import hag_reservoir_from_hyperparameters, retrieve_best_model
from hag.models.reservoir import init_matrices

# =============================== PARAMETERS ===============================
DATASETS = ["JapaneseVowels", "CatsDogs", "FSDD", "SpokenArabicDigits", "SPEECHCOMMANDS"]
FUNCTIONS = ["mean_hag", "var_hag"]
SEED = 923984      # seed of evaluate_dataset_on_test (the same for both implementations)
NB_TRIALS = 8      # reservoirs evaluated on the test set
N_W_CHECK = 3      # draws for the exact comparison of the weights
OUTPUT = Path("outputs/test_results/hag_node_vs_run_algorithm.csv")
# ===========================================================================


def best_hyperparameters(function, dataset, is_multivariate):
    study = retrieve_best_model(function, dataset, is_multivariate, prefix="tpe", db_dir=study_db_dir(dataset),
                                verbosity=0)
    hp = dict(study.best_trial.params)
    if 'variance_target' not in hp and 'min_variance' in hp:
        hp['variance_target'] = hp['min_variance']
    return study, hp


def initial_matrices(hp, input_dim):
    """HAG's initial matrices, as in evaluate_dataset_on_test (seed drawn from numpy's global RNG)."""
    K = math.ceil(hp['network_size'] / input_dim)
    Win, W, bias = init_matrices(input_dim * K, 1, hp['connectivity'], K, w_distribution=stats.uniform(loc=-1, scale=2),
                                 seed=random.randint(0, 1000))
    return W, Win * hp['input_scaling'], bias * hp['bias_scaling']


def W_run_algorithm(hp, function, pretrain, seed):
    random.seed(seed)
    W, Win, bias = initial_matrices(hp, pretrain[0].shape[1])
    if function == "mean_hag":
        target, spread, extra = hp['target_rate'], hp['rate_spread'], {}
    else:
        target, spread = hp['variance_target'], hp['variance_spread']
        extra = dict(intrinsic_saturation=hp['intrinsic_saturation'], intrinsic_coef=hp['intrinsic_coef'])
    W, _ = run_algorithm(W, Win, bias, hp['leaky_rate'], activation_function, pretrain, hp['weight_increment'], target,
                         spread, function, multiple_instances=True, min_increment=hp['min_increment'],
                         max_increment=hp['max_increment'], use_full_instance=hp['use_full_instance'],
                         max_partners=np.inf, method="pearson", n_jobs=1, progress_bar=False, **extra)
    return W


def W_hag_node(hp, function, pretrain, seed):
    random.seed(seed)
    W, Win, bias = initial_matrices(hp, pretrain[0].shape[1])
    # rng=random.mtrand._rand: numpy's global RNG, so the same draws as run_algorithm
    node = hag_reservoir_from_hyperparameters(hp, function, input_dim=pretrain[0].shape[1], W=W, Win=Win, bias=bias,
                                              rng=random.mtrand._rand)
    return node.fit(list(pretrain)).W


def compare(dataset, function, data):
    pretrain, train, test, Y_train, Y_test, is_multivariate, is_classif = data
    study, hp = best_hyperparameters(function, dataset, is_multivariate)

    # 1. weights: max difference between the two implementations, same draws
    w_diff = max(np.abs(W_run_algorithm(hp, function, pretrain, s) - W_hag_node(hp, function, pretrain, s)).max()
                 for s in range(N_W_CHECK))

    # 2. test, same seed
    scores, times = {}, {}
    for name, use_hag_node in (("run_algorithm", False), ("HAGReservoir", True)):
        start = time.time()
        scores[name] = evaluate_dataset_on_test(study, dataset, function, pretrain, train, test, Y_train, Y_test,
                                                is_classif, nb_trials=NB_TRIALS, seed=SEED, use_hag_node=use_hag_node)
        times[name] = time.time() - start

    row = {"dataset": dataset, "function": function, "max_W_diff": w_diff,
           "identical_scores": bool(np.array_equal(scores["run_algorithm"], scores["HAGReservoir"]))}
    for name in scores:
        row.update({f"{name}_mean_%": 100 * np.mean(scores[name]), f"{name}_std_%": 100 * np.std(scores[name]),
                    f"{name}_time_s": round(times[name])})
    return row


def main(datasets):
    rows = []
    for dataset in datasets:
        data = load_data(dataset, "mfcc")
        for function in FUNCTIONS:
            row = compare(dataset, function, data)
            rows.append(row)
            print(f"[compare] {dataset:20s} {function:9s} max|W diff| {row['max_W_diff']:.1e} | test run_algorithm "
                  f"{row['run_algorithm_mean_%']:.3f}% ± {row['run_algorithm_std_%']:.3f} "
                  f"({row['run_algorithm_time_s']}s) | HAGReservoir {row['HAGReservoir_mean_%']:.3f}% ± "
                  f"{row['HAGReservoir_std_%']:.3f} ({row['HAGReservoir_time_s']}s) | identical scores: "
                  f"{row['identical_scores']}", flush=True)
            # saved after each comparison, so that an interrupted run keeps its results
            OUTPUT.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(OUTPUT, index=False)

    results = pd.DataFrame(rows)
    print(results.round(3).to_string(index=False))
    print(f"Results saved to {OUTPUT}")
    if not (results["max_W_diff"].eq(0).all() and results["identical_scores"].all()):
        print("WARNING: the two implementations differ")


if __name__ == "__main__":
    main(sys.argv[1:] or DATASETS)
