"""
Does the precise structure of HAG's connexions between blocks matter, or only their statistics?

HAG's recurrent matrices are mostly block diagonal (one block of K neurons per input feature), with a few sparse
connexions between blocks. For each dataset and function (mean_hag, var_hag), with the best hyperparameters of HAG's
studies, each trial:
  1. trains HAG (base algorithm, hag.hag.hag.run_algorithm) as in hag.analysis.commons.evaluate_dataset_on_test,
  2. fits, for each off-diagonal block, a trio (distribution, connectivity, scaling) reproducing its statistics:
       - connectivity: fraction of nonzero weights of the block,
       - distribution and scaling: among DISTRIBUTIONS (one scale parameter each, fitted on the absolute values of the
         nonzero weights), the one closest to their empirical distribution (Kolmogorov-Smirnov distance),
  3. regenerates the off-diagonal blocks at random with these statistics (same number of connexions, at random
     positions, weights drawn from the fitted distribution; same sign proportion), the diagonal blocks being kept,
  4. evaluates on the test set (same readout as evaluate_dataset_on_test): HAG's matrix ("hag"), the resampled one
     ("resampled") and, as a control, HAG's matrix without its off-diagonal blocks ("no_off_diagonal").
The results are written to outputs/analysis_results/hag_off_diagonal_resampling.csv.

Run from the repository root:  python -m hag.analysis.off_diagonal_resampling [dataset ...]
"""
import math
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats

from hag.analysis.commons import activation_function, load_data
from hag.hag.hag import run_algorithm
from hag.hpo.utility import ANALYSIS_RESULTS, retrieve_best_model
from hag.models.reservoir import init_matrices
from hag.performances.esn_model_evaluation import (compute_score, init_readout, init_reservoir,
                                                   predict_model_for_classification, train_model_for_classification)

# =============================== PARAMETERS ===============================
DATASETS = ["JapaneseVowels", "CatsDogs", "FSDD", "SpokenArabicDigits", "SPEECHCOMMANDS"]
FUNCTIONS = ["mean_hag", "var_hag"]
CONDITIONS = ["hag", "resampled", "no_off_diagonal"]
SEED = 923984      # trial k uses the seed SEED + k (initial matrices, HAG's random choices, resampling)
NB_TRIALS = 8
DATA_SEED = 0      # draw of the pretraining instances among the train set
OUTPUT = ANALYSIS_RESULTS / "hag_off_diagonal_resampling.csv"
# ===========================================================================

# candidate distributions of the absolute values of the weights: (fitted scale, cdf, sampler)
DISTRIBUTIONS = {
    "constant": (lambda x: np.median(x), lambda x, s: (x >= s).astype(float), lambda rng, s, n: np.full(n, s)),
    "uniform": (lambda x: np.max(x), lambda x, s: np.clip(x / s, 0, 1), lambda rng, s, n: rng.uniform(0, s, n)),
    "exponential": (lambda x: np.mean(x), lambda x, s: 1 - np.exp(-x / s), lambda rng, s, n: rng.exponential(s, n)),
    "halfnormal": (lambda x: np.sqrt(np.mean(x ** 2)), lambda x, s: stats.halfnorm.cdf(x, scale=s),
                   lambda rng, s, n: np.abs(rng.normal(0, s, n))),
}


def ks_distance(x, cdf):
    """Kolmogorov-Smirnov distance between the empirical distribution of x and a cdf, at the values of x and just
    before them (HAG's weights are multiples of weight_increment: many ties, and the constant cdf is a step)."""
    u = np.unique(x)
    left = np.nextafter(u, -np.inf)
    empirical, empirical_left = np.searchsorted(np.sort(x), u, side="right") / x.size, np.mean(x[:, None] < u, axis=0)
    return max(np.max(np.abs(empirical - cdf(u))), np.max(np.abs(empirical_left - cdf(left))))


def fit_block(block):
    """(distribution, connectivity, scaling, fraction of negative weights) of a block."""
    weights = block[block != 0]
    connectivity = weights.size / block.size
    if weights.size == 0:
        return {"distribution": None, "connectivity": 0.0, "scaling": 0.0, "negative": 0.0}
    x = np.abs(weights)
    fits = {name: fit(x) for name, (fit, _, _) in DISTRIBUTIONS.items()}
    best = min(DISTRIBUTIONS, key=lambda name: ks_distance(x, lambda v: DISTRIBUTIONS[name][1](v, fits[name])))
    return {"distribution": best, "connectivity": connectivity, "scaling": fits[best],
            "negative": float(np.mean(weights < 0))}


def blocks_of(W, K):
    """(row block, column block) -> slices of the off-diagonal blocks of W (K neurons per block)."""
    n = W.shape[0] // K
    return {(i, j): (slice(i * K, (i + 1) * K), slice(j * K, (j + 1) * K)) for i in range(n) for j in range(n) if i != j}


def resample_off_diagonal(W, K, rng):
    """W with its off-diagonal blocks regenerated at random with their fitted statistics, and the fitted trios."""
    W_new, trios = W.copy(), []
    for rows, cols in blocks_of(W, K).values():
        trio = fit_block(W[rows, cols])
        trios.append(trio)
        block = np.zeros((K, K))
        n = int(round(trio["connectivity"] * K * K))
        if n > 0:
            positions = rng.choice(K * K, size=n, replace=False)
            values = DISTRIBUTIONS[trio["distribution"]][2](rng, trio["scaling"], n)
            block.flat[positions] = values * np.where(rng.random(n) < trio["negative"], -1, 1)
        W_new[rows, cols] = block
    return W_new, trios


def without_off_diagonal(W, K):
    W_new = W.copy()
    for rows, cols in blocks_of(W, K).values():
        W_new[rows, cols] = 0
    return W_new


def train_hag(hp, function, pretrain, seed):
    """HAG's base algorithm, as in evaluate_dataset_on_test (initial matrices and random choices from seed)."""
    input_dim = pretrain[0].shape[1]
    K = math.ceil(hp['network_size'] / input_dim)
    Win, W, bias = init_matrices(input_dim * K, 1, hp['connectivity'], K, w_distribution=stats.uniform(loc=-1, scale=2),
                                 seed=seed)
    Win, bias = Win * hp['input_scaling'], bias * hp['bias_scaling']
    if function == "mean_hag":
        target, spread, extra = hp['target_rate'], hp['rate_spread'], {}
    else:
        target, spread = hp['variance_target'], hp['variance_spread']
        extra = dict(intrinsic_saturation=hp['intrinsic_saturation'], intrinsic_coef=hp['intrinsic_coef'])
    np.random.seed(seed)
    W, _ = run_algorithm(W, Win, bias, hp['leaky_rate'], activation_function, pretrain, hp['weight_increment'], target,
                         spread, function, multiple_instances=True, min_increment=hp['min_increment'],
                         max_increment=hp['max_increment'], use_full_instance=hp['use_full_instance'],
                         max_partners=np.inf, method="pearson", n_jobs=1, progress_bar=False, **extra)
    return np.asarray(W, dtype=float), Win, bias, K


def test_accuracy(W, Win, bias, hp, train, test, Y_train, Y_test):
    """Test accuracy with the readout of evaluate_dataset_on_test (last state of each sequence, ridge)."""
    reservoir = init_reservoir(W, Win, bias, hp['leaky_rate'], activation_function)
    readout = init_readout(ridge_coef=10 ** hp['ridge'])
    train_model_for_classification(reservoir, readout, train, Y_train, mode="sequence-to-vector")
    Y_pred = predict_model_for_classification(reservoir, readout, test, mode="sequence-to-vector")
    return compute_score(Y_pred, Y_test, True)


def spectral_radius(W):
    return float(np.max(np.abs(np.linalg.eigvals(W))))


def run(dataset, function, data):
    pretrain, train, test, Y_train, Y_test, is_multivariate, _ = data
    hp = dict(retrieve_best_model(function, dataset, is_multivariate, verbosity=0).best_trial.params)
    scores = {c: [] for c in CONDITIONS}
    radii = {c: [] for c in CONDITIONS}
    trios, start = [], time.time()
    for k in range(NB_TRIALS):
        W, Win, bias, K = train_hag(hp, function, pretrain, SEED + k)
        W_resampled, block_trios = resample_off_diagonal(W, K, np.random.default_rng(SEED + k))
        trios += block_trios
        matrices = {"hag": W, "resampled": W_resampled, "no_off_diagonal": without_off_diagonal(W, K)}
        for condition in CONDITIONS:
            scores[condition].append(test_accuracy(matrices[condition], Win, bias, hp, train, test, Y_train, Y_test))
            radii[condition].append(spectral_radius(matrices[condition]))

    off = [t for t in trios if t["distribution"] is not None]
    row = {"dataset": dataset, "function": function, "units": int(W.shape[0]), "block_size": K,
           "off_diagonal_connectivity": float(np.mean([t["connectivity"] for t in trios])),
           "nonempty_off_diagonal_blocks_%": 100 * len(off) / len(trios),
           "distributions": dict(Counter(t["distribution"] for t in off)),
           "time_s": round(time.time() - start)}
    for c in CONDITIONS:
        row.update({f"{c}_mean_%": 100 * np.mean(scores[c]), f"{c}_std_%": 100 * np.std(scores[c]),
                    f"{c}_spectral_radius": float(np.mean(radii[c]))})
    return row


def main(datasets):
    rows = []
    for dataset in datasets:
        np.random.seed(DATA_SEED)  # load_data draws the pretraining instances with numpy's global RNG
        data = load_data(dataset, "mfcc")
        for function in FUNCTIONS:
            row = run(dataset, function, data)
            rows.append(row)
            print(f"[off-diagonal] {dataset:20s} {function:9s} off-diag. connectivity {row['off_diagonal_connectivity']:.4f} "
                  f"({row['nonempty_off_diagonal_blocks_%']:.0f}% of blocks nonempty, {row['distributions']}) | "
                  + " | ".join(f"{c} {row[f'{c}_mean_%']:.2f}% ± {row[f'{c}_std_%']:.2f} (sr {row[f'{c}_spectral_radius']:.2f})"
                               for c in CONDITIONS) + f" | {row['time_s']}s", flush=True)
            # saved after each run, so that an interrupted study keeps its results
            OUTPUT.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(OUTPUT, index=False)
    print(f"Results saved to {OUTPUT}")


if __name__ == "__main__":
    main(sys.argv[1:] or DATASETS)
