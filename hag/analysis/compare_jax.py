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
     the next ones (the JAX fit is compiled at each call, about 0.2 to 0.5 s), and the speedup of each implementation
     over run_algorithm.
(Earlier results of this file: jax_hag_fit_time.csv, fit times of the two nodes; jax_hag_compilation.csv, compilation at
each fit vs a compilation cache shared by the instances: 0.2 to 0.45 s saved per fit, -7 to -47 % on the small
datasets, nothing on SPEECHCOMMANDS, identical W.)

Also the local plasticity reservoirs, anti-oja (reservoirpy's LocalPlasticityReservoir) and ip-anti-oja
(hag.models.IPLocalPlasticityReservoir), with the best hyperparameters of their studies: their NumPy learning
(reservoir.fit) vs the JAX learner used by HAG's evaluation (hag.performances.esn_model_evaluation.fit_reservoir), same
matrices, same pretraining data (concatenated, warmup of 100 steps) and same post-synaptic states (RULE_STATES:
ip-anti-oja with "0.3", as HAG's evaluation and its optimization; anti-oja with "0.4", the only ones of the NumPy
LocalPlasticityReservoir, HAG's evaluation using "0.3" in JAX only). With other states than those of its optimization,
a reservoir can become chaotic (e.g. ip-anti-oja on CatsDogs with "0.4": IP gains up to 1.4, chance accuracy), and the
rounding differences between NumPy and JAX then grow. Even with "0.3", ip-anti-oja on CatsDogs goes through a chaotic
phase (gains up to about 2 after 15000 steps, then back to about 0): the difference grows steadily from the rounding
(1e-15 after 500 steps, 4e-11 after 15000) to about 1e-3 on W and 1e-2 on a, b at the end, with the same accuracy.
Both are deterministic:
same learned parameters (max_W_diff, max_ab_diff for the IP gains and biases), test accuracy and learning time, for
NB_TRIALS_LOCAL draws of the matrices (the NumPy learning can take hours on the larger datasets).

All the results are added to outputs/hag/analysis_results/numpy_vs_jax.csv (the rows of a dataset and function compared
again replace their previous ones, the others are kept), one row per dataset, function and
implementation ("implementation"; "backend" numpy or jax), compared with the reference implementation of its function
("reference": run_algorithm for HAG, the NumPy learning for the local rules): max_W_diff (and max_ab_diff, IP gains and
biases) with the same draws (empty when the random choices differ: JaxHAGReservoir), test accuracy (test_mean_%,
test_std_%), fit time (first_fit_s: first fit of a new instance, with the JAX compilation; fit_s: mean of the next
ones) and speedup = fit_s of the reference / fit_s.

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
from hag.models.reservoir import init_matrices, scale_spectral_radius
from hag.performances.esn_model_evaluation import (compute_score, fit_reservoir, init_ip_local_rule_reservoir,
                                                   init_local_rule_reservoir, init_readout, init_reservoir,
                                                   predict_model_for_classification, train_model_for_classification)
from scipy import sparse

# =============================== PARAMETERS ===============================
DATASETS = ["JapaneseVowels", "CatsDogs", "FSDD", "SpokenArabicDigits", "SPEECHCOMMANDS"]  # "JapaneseVowels", "CatsDogs", "FSDD", "SpokenArabicDigits", "SPEECHCOMMANDS"
FUNCTIONS = ["mean_hag", "var_hag", "anti-oja", "ip-anti-oja"]
HAG_FUNCTIONS = ["mean_hag", "var_hag"]
LOCAL_FUNCTIONS = ["anti-oja", "ip-anti-oja"]
SEED = 923984      # trial k uses the seed SEED + k (initial matrices and HAG's random choices)
NB_TRIALS = 8      # reservoirs evaluated on the test set
N_W_CHECK = 3      # draws for the exact comparison of the weights
DATA_SEED = 0      # draw of the pretraining instances among the train set
NB_TRIALS_LOCAL = 3  # draws of the matrices for the local plasticity reservoirs
LOCAL_WARMUP = 100   # as in HAG's evaluation
RULE_STATES = {"anti-oja": "0.4", "ip-anti-oja": "0.3"}  # post-synaptic states of the local rules, see the docstring
OUTPUT = ANALYSIS_RESULTS / "numpy_vs_jax.csv"
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


def fit_times_summary(times):
    """(first fit, mean of the next fits) in seconds."""
    return times[0], float(np.mean(times[1:])) if len(times) > 1 else times[0]


def compare(dataset, function, data):
    """Rows of the three HAG implementations: run_algorithm (reference), HAGReservoir and JaxHAGReservoir."""
    pretrain, train, test, Y_train, Y_test, is_multivariate, _ = data
    hp = dict(retrieve_best_model(function, dataset, is_multivariate, verbosity=0).best_trial.params)
    # 1. same random draws -> same weights (run_algorithm vs HAGReservoir; the JAX node has its own random choices)
    w_diff = max(np.abs(W_run_algorithm(hp, function, pretrain, SEED + k)[0]
                        - W_node(hp, function, pretrain, SEED + k, global_rng=True)[0]).max() for k in range(N_W_CHECK))

    # 2. and 3. test accuracy and fit time: reference implementation vs nodes used as in reservoirpy
    rows = []
    for name, backend, diff, learn in (
            ("run_algorithm", "numpy", np.nan, lambda k: W_run_algorithm(hp, function, pretrain, SEED + k)),
            ("HAGReservoir", "numpy", w_diff, lambda k: W_node(hp, function, pretrain, SEED + k)),
            ("JaxHAGReservoir", "jax", np.nan, lambda k: W_node(hp, function, pretrain, SEED + k, jax=True))):
        fit_times, scores = [], []
        for k in range(NB_TRIALS):
            start = time.time()
            W, Win, bias = learn(k)  # (W as a NumPy array: the JAX computation is finished)
            fit_times.append(time.time() - start)
            scores.append(test_accuracy(W, Win, bias, hp, train, test, Y_train, Y_test))
        first, fit = fit_times_summary(fit_times)
        rows.append({"dataset": dataset, "function": function, "implementation": name, "backend": backend,
                     "reference": "run_algorithm", "rule_states": "", "max_W_diff": diff, "max_ab_diff": np.nan,
                     "test_mean_%": 100 * np.mean(scores), "test_std_%": 100 * np.std(scores),
                     "first_fit_s": first, "fit_s": fit})
    for row in rows:
        row["speedup"] = rows[0]["fit_s"] / row["fit_s"]
    return rows


def local_rule_reservoir(hp, function, input_dim, seed):
    """anti-oja or ip-anti-oja reservoir with the initial matrices of HAG's evaluation (build_reservoir_from_study)."""
    K = math.ceil(hp['network_size'] / input_dim)
    Win, W, bias = init_matrices(input_dim * K, 1, hp['connectivity'], K, w_distribution=stats.uniform(loc=-1, scale=2),
                                 seed=seed)
    if hp['connectivity'] > 0:
        W = scale_spectral_radius(W, hp['spectral_radius'])
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
        fit_reservoir(reservoir, x, warmup=LOCAL_WARMUP, rule_states=RULE_STATES[function])
    else:
        if function == "ip-anti-oja":
            reservoir.rule_states = RULE_STATES[function]
        reservoir.fit(x, warmup=LOCAL_WARMUP)
    return reservoir, time.time() - start


def classification_accuracy(reservoir, hp, train, test, Y_train, Y_test):
    readout = init_readout(ridge_coef=10 ** hp['ridge'])
    train_model_for_classification(reservoir, readout, train, Y_train, mode="sequence-to-vector")
    Y_pred = predict_model_for_classification(reservoir, readout, test, mode="sequence-to-vector")
    return compute_score(Y_pred, Y_test, True)


def compare_local(dataset, function, data):
    """Rows of the NumPy (reference) and JAX learnings of a local plasticity reservoir."""
    pretrain, train, test, Y_train, Y_test, is_multivariate, _ = data
    hp = dict(retrieve_best_model(function, dataset, is_multivariate, verbosity=0).best_trial.params)
    scores, times = {"NumPy": [], "JAX": []}, {"NumPy": [], "JAX": []}
    w_diff, ab_diff = 0.0, np.nan
    for k in range(NB_TRIALS_LOCAL):
        learned = {}
        for name, jax in (("NumPy", False), ("JAX", True)):
            learned[name], duration = learn_local(hp, function, pretrain, SEED + k, jax)
            times[name].append(duration)
            scores[name].append(classification_accuracy(learned[name], hp, train, test, Y_train, Y_test))
        w_diff = max(w_diff, np.abs(np.asarray(learned["NumPy"].W, dtype=float)
                                    - np.asarray(learned["JAX"].W, dtype=float)).max())
        if function == "ip-anti-oja":
            diff = max(np.abs(np.ravel(getattr(learned["NumPy"], name)) - np.ravel(getattr(learned["JAX"], name))).max()
                       for name in ("a", "b"))
            ab_diff = diff if np.isnan(ab_diff) else max(ab_diff, diff)
    rows = []
    for name, backend in (("NumPy", "numpy"), ("JAX", "jax")):
        first, fit = fit_times_summary(times[name])
        reference = name == "NumPy"
        rows.append({"dataset": dataset, "function": function, "implementation": name, "backend": backend,
                     "reference": "NumPy", "rule_states": RULE_STATES[function],
                     "max_W_diff": np.nan if reference else w_diff, "max_ab_diff": np.nan if reference else ab_diff,
                     "test_mean_%": 100 * np.mean(scores[name]), "test_std_%": 100 * np.std(scores[name]),
                     "first_fit_s": first, "fit_s": fit})
    for row in rows:
        row["speedup"] = rows[0]["fit_s"] / row["fit_s"]
    return rows


def save(new_rows):
    """Add the rows to OUTPUT, replacing the previous rows of the same dataset and function."""
    new = pd.DataFrame(new_rows)
    if OUTPUT.exists():
        old = pd.read_csv(OUTPUT)
        replaced = old.set_index(["dataset", "function"]).index.isin(new.set_index(["dataset", "function"]).index)
        new = pd.concat([old[~replaced], new], ignore_index=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(OUTPUT, index=False)


def main(datasets, functions):
    rows = []
    for dataset in datasets:
        random.seed(DATA_SEED)  # prepare_data draws the pretraining instances with numpy's global RNG
        data = prepare_data(dataset, "mfcc")
        for function in functions:
            try:
                new_rows = (compare_local if function in LOCAL_FUNCTIONS else compare)(dataset, function, data)
            except (KeyError, FileNotFoundError):
                print(f"[compare] {dataset} {function}: no study, skipped", flush=True)
                continue
            rows += new_rows
            for row in new_rows:
                diff = "" if np.isnan(row["max_W_diff"]) else f" | max|W diff| {row['max_W_diff']:.1e}"
                diff += "" if np.isnan(row["max_ab_diff"]) else f" max|a, b diff| {row['max_ab_diff']:.1e}"
                print(f"[compare] {dataset:18s} {function:11s} {row['implementation']:15s} test "
                      f"{row['test_mean_%']:.2f}% ± {row['test_std_%']:.2f} | fit {row['first_fit_s']:.2f}s then "
                      f"{row['fit_s']:.2f}s (x{row['speedup']:.1f} vs {row['reference']}){diff}", flush=True)
            # saved after each comparison, so that an interrupted run keeps its results
            save(new_rows)

    print(f"Results saved to {OUTPUT}")
    hag = pd.DataFrame([row for row in rows if row["implementation"] == "HAGReservoir"])
    if len(hag) and not hag["max_W_diff"].eq(0).all():
        print("WARNING: run_algorithm and HAGReservoir do not give the same weights")


if __name__ == "__main__":
    args = sys.argv[1:]
    functions = FUNCTIONS
    if "--functions" in args:
        i = args.index("--functions")
        functions = args[i + 1].split(",")
        args = args[:i] + args[i + 2:]
    main(args or DATASETS, functions)
