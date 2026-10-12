"""
NumPy and JAX implementations of the plastic reservoirs of HAG's evaluation: same results, and the speedup of JAX.

HAG: compare HAG's reference implementation (hag.hag.hag.run_algorithm) with reservoirpy's HAGReservoir node and its JAX
version (hag.models.jax_hag_reservoir), for mean_hag and var_hag with the best hyperparameters of HAG's studies, on the
classification datasets:
  1. weights: same initial matrices and same random draws (the node's random choices taken from numpy's global RNG,
     as run_algorithm) -> identical W ("max_W_diff" = 0),
  2. test accuracy: run_algorithm (random choices of numpy's global RNG) vs the nodes used as in reservoirpy
     (random choices from their seed), same initial matrices and readout -> same accuracy up to the randomness of HAG's
     choices. The JAX node draws its random choices with Jax's generator: it can only be compared statistically.
  3. fit time of each implementation (W learning only, without the readout): first fit of a new instance and mean of
     the next ones (the JAX fit is compiled at each call, about 0.2 to 0.5 s), and the speedup of the JAX node over the
     NumPy node ("jax_speedup").
The results are written to outputs/hag/analysis_results/hag_node_vs_run_algorithm.csv.
(Earlier results of this file: jax_hag_fit_time.csv, fit times of the two nodes; jax_hag_compilation.csv, compilation at
each fit vs a compilation cache shared by the instances: 0.2 to 0.45 s saved per fit, -7 to -47 % on the small
datasets, nothing on SPEECHCOMMANDS, identical W.)

Also the local plasticity reservoirs, anti-oja (reservoirpy's LocalPlasticityReservoir) and ip-anti-oja
(hag.models.IPLocalPlasticityReservoir), with the best hyperparameters of their studies: their NumPy learning
(reservoir.fit) vs the JAX learner used by HAG's evaluation (hag.performances.esn_model_evaluation.fit_reservoir), same
matrices, same pretraining data (concatenated, warmup of 100 steps) and same post-synaptic states (rule_states "0.4",
the only ones of the NumPy LocalPlasticityReservoir; HAG's evaluation uses "0.3", JAX only). Both are deterministic:
same learned parameters (max_W_diff, max_ab_diff for the IP gains and biases), test accuracy and learning time, for
NB_TRIALS_LOCAL draws of the matrices (the NumPy learning can take hours on the larger datasets).
The results are written to outputs/hag/analysis_results/local_plasticity_numpy_vs_jax.csv.

Run from the repository root:
  python -m hag.analysis.compare_jax [dataset ...] [--functions mean_hag,var_hag,anti-oja,ip-anti-oja]
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
from hag.performances.esn_model_evaluation import (compute_score, fit_reservoir, init_ip_local_rule_reservoir,
                                                   init_local_rule_reservoir, init_readout, init_reservoir,
                                                   predict_model_for_classification, train_model_for_classification)
from scipy import sparse

# =============================== PARAMETERS ===============================
DATASETS = ["JapaneseVowels", "CatsDogs", "FSDD", "SpokenArabicDigits", "SPEECHCOMMANDS"]
FUNCTIONS = ["mean_hag", "var_hag", "anti-oja", "ip-anti-oja"]
HAG_FUNCTIONS = ["mean_hag", "var_hag"]
LOCAL_FUNCTIONS = ["anti-oja", "ip-anti-oja"]
SEED = 923984      # trial k uses the seed SEED + k (initial matrices and HAG's random choices)
NB_TRIALS = 8      # reservoirs evaluated on the test set
N_W_CHECK = 3      # draws for the exact comparison of the weights
DATA_SEED = 0      # draw of the pretraining instances among the train set
NB_TRIALS_LOCAL = 3  # draws of the matrices for the local plasticity reservoirs
LOCAL_WARMUP = 100   # as in HAG's evaluation
OUTPUT = ANALYSIS_RESULTS / "hag_node_vs_run_algorithm.csv"
OUTPUT_LOCAL = ANALYSIS_RESULTS / "local_plasticity_numpy_vs_jax.csv"
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


def W_node(hp, function, pretrain, seed, global_rng=False, jax=False):
    """W learned by reservoirpy's HAGReservoir (jax: its JAX version) with the same initial matrices: random choices
    from the node's seed, or (global_rng, NumPy node only) from numpy's global RNG seeded with seed, as run_algorithm."""
    W, Win, bias = initial_matrices(hp, pretrain[0].shape[1], seed)
    node = hag_reservoir_from_hyperparameters(hp, function, input_dim=pretrain[0].shape[1], W=W, Win=Win, bias=bias,
                                              seed=seed, jax=jax)
    if global_rng:
        # verification only: replace the node's generator of random choices by numpy's global RNG
        node.initialize(list(pretrain))
        node._plasticity_rng = random.mtrand._rand
        random.seed(seed)
    return np.asarray(node.fit(list(pretrain)).W, dtype=float), Win, bias


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

    # 2. test accuracy: reference implementation vs nodes used as in reservoirpy
    for name, learn in (("run_algorithm", lambda k: W_run_algorithm(hp, function, pretrain, SEED + k)),
                        ("HAGReservoir", lambda k: W_node(hp, function, pretrain, SEED + k)),
                        ("JaxHAGReservoir", lambda k: W_node(hp, function, pretrain, SEED + k, jax=True))):
        fit_times, scores = [], []
        for k in range(NB_TRIALS):
            start = time.time()
            W, Win, bias = learn(k)  # (W as a NumPy array: the JAX computation is finished)
            fit_times.append(time.time() - start)
            scores.append(test_accuracy(W, Win, bias, hp, train, test, Y_train, Y_test))
        row.update({f"{name}_mean_%": 100 * np.mean(scores), f"{name}_std_%": 100 * np.std(scores),
                    f"{name}_first_fit_s": fit_times[0], f"{name}_fit_s": np.mean(fit_times[1:])})
    row["jax_speedup"] = row["HAGReservoir_fit_s"] / row["JaxHAGReservoir_fit_s"]
    return row


def local_rule_reservoir(hp, function, input_dim, seed):
    """anti-oja or ip-anti-oja reservoir with the initial matrices of HAG's evaluation (build_reservoir_from_study)."""
    K = math.ceil(hp['network_size'] / input_dim)
    Win, W, bias = init_matrices(input_dim * K, 1, hp['connectivity'], K, w_distribution=stats.uniform(loc=-1, scale=2),
                                 seed=seed)
    if hp['connectivity'] > 0:
        eigen = sparse.linalg.eigs(W, k=1, which="LM", maxiter=W.shape[0] * 20, tol=0.1, return_eigenvectors=False,
                                   v0=np.ones(W.shape[0]))
        W = W * hp['spectral_radius'] / max(abs(eigen))
    Win, bias = Win * hp['input_scaling'], bias * hp['bias_scaling']
    common = dict(local_rule="anti-oja", eta=hp['oja_eta'], synapse_normalization=False, bcm_theta=None,
                  leaking_rate=hp['leaky_rate'])
    if function == "anti-oja":
        return init_local_rule_reservoir(W, Win, bias, activation_function=tanh, **common)
    return init_ip_local_rule_reservoir(W, Win, bias, mu=hp['mu'], sigma=hp['sigma'], learning_rate=hp['learning_rate'],
                                        **common)


def learn_local(hp, function, pretrain, seed, jax):
    """Reservoir after its local learning, NumPy (reservoir.fit) or JAX (fit_reservoir), and the learning time."""
    reservoir = local_rule_reservoir(hp, function, pretrain[0].shape[1], seed)
    x = np.concatenate(pretrain).astype(float)
    start = time.time()
    if jax:
        fit_reservoir(reservoir, x, warmup=LOCAL_WARMUP, rule_states="0.4")
    else:
        if function == "ip-anti-oja":
            reservoir.rule_states = "0.4"
        reservoir.fit(x, warmup=LOCAL_WARMUP)
    return reservoir, time.time() - start


def classification_accuracy(reservoir, hp, train, test, Y_train, Y_test):
    readout = init_readout(ridge_coef=10 ** hp['ridge'])
    train_model_for_classification(reservoir, readout, train, Y_train, mode="sequence-to-vector")
    Y_pred = predict_model_for_classification(reservoir, readout, test, mode="sequence-to-vector")
    return compute_score(Y_pred, Y_test, True)


def compare_local(dataset, function, data):
    pretrain, train, test, Y_train, Y_test, is_multivariate, _ = data
    hp = dict(retrieve_best_model(function, dataset, is_multivariate, verbosity=0).best_trial.params)
    row = {"dataset": dataset, "function": function, "max_W_diff": 0.0, "max_ab_diff": np.nan}
    scores = {"NumPy": [], "JAX": []}
    times = {"NumPy": 0.0, "JAX": 0.0}
    for k in range(NB_TRIALS_LOCAL):
        learned = {}
        for name, jax in (("NumPy", False), ("JAX", True)):
            learned[name], duration = learn_local(hp, function, pretrain, SEED + k, jax)
            times[name] += duration
            scores[name].append(classification_accuracy(learned[name], hp, train, test, Y_train, Y_test))
        row["max_W_diff"] = max(row["max_W_diff"], np.abs(np.asarray(learned["NumPy"].W, dtype=float)
                                                         - np.asarray(learned["JAX"].W, dtype=float)).max())
        if function == "ip-anti-oja":
            ab_diff = max(np.abs(np.ravel(getattr(learned["NumPy"], name)) - np.ravel(getattr(learned["JAX"], name))).max()
                          for name in ("a", "b"))
            row["max_ab_diff"] = ab_diff if np.isnan(row["max_ab_diff"]) else max(row["max_ab_diff"], ab_diff)
    for name in ("NumPy", "JAX"):
        row.update({f"{name}_mean_%": 100 * np.mean(scores[name]), f"{name}_std_%": 100 * np.std(scores[name]),
                    f"{name}_fit_time_s": times[name] / NB_TRIALS_LOCAL})
    row["speedup"] = row["NumPy_fit_time_s"] / row["JAX_fit_time_s"]
    return row


def main(datasets, functions):
    rows, local_rows = [], []
    for dataset in datasets:
        random.seed(DATA_SEED)  # prepare_data draws the pretraining instances with numpy's global RNG
        data = prepare_data(dataset, "mfcc")
        for function in functions:
            if function in LOCAL_FUNCTIONS:
                try:
                    row = compare_local(dataset, function, data)
                except KeyError:
                    print(f"[compare] {dataset} {function}: no study, skipped", flush=True)
                    continue
                local_rows.append(row)
                ab = "" if np.isnan(row["max_ab_diff"]) else f" max|a, b diff| {row['max_ab_diff']:.1e}"
                print(f"[compare] {dataset:20s} {function:11s} max|W diff| {row['max_W_diff']:.1e}{ab} | test NumPy "
                      f"{row['NumPy_mean_%']:.2f}% ± {row['NumPy_std_%']:.2f} (fit {row['NumPy_fit_time_s']:.1f}s) | JAX "
                      f"{row['JAX_mean_%']:.2f}% ± {row['JAX_std_%']:.2f} (fit {row['JAX_fit_time_s']:.1f}s) | "
                      f"speedup x{row['speedup']:.0f}", flush=True)
                OUTPUT_LOCAL.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(local_rows).to_csv(OUTPUT_LOCAL, index=False)
                continue
            row = compare(dataset, function, data)
            rows.append(row)
            print(f"[compare] {dataset:20s} {function:9s} max|W diff| {row['max_W_diff']:.1e} | test (fit) "
                  + " | ".join(f"{name} {row[name + '_mean_%']:.2f}% ± {row[name + '_std_%']:.2f} "
                               f"({row[name + '_first_fit_s']:.2f}s then {row[name + '_fit_s']:.2f}s)"
                               for name in ("run_algorithm", "HAGReservoir", "JaxHAGReservoir"))
                  + f" | JAX speedup x{row['jax_speedup']:.1f}", flush=True)
            # saved after each comparison, so that an interrupted run keeps its results
            OUTPUT.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(OUTPUT, index=False)

    if rows:
        print(f"Results saved to {OUTPUT}")
        if not pd.DataFrame(rows)["max_W_diff"].eq(0).all():
            print("WARNING: the HAG implementations do not give the same weights")
    if local_rows:
        print(f"Results saved to {OUTPUT_LOCAL}")


if __name__ == "__main__":
    args = sys.argv[1:]
    functions = FUNCTIONS
    if "--functions" in args:
        i = args.index("--functions")
        functions = args[i + 1].split(",")
        args = args[:i] + args[i + 2:]
    main(args or DATASETS, functions)
