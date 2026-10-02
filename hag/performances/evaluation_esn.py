"""
Test evaluation of the reservoirs of HAG's study (random ESNs, IP, anti-Oja, IP + anti-Oja, HAG and its variants):
reservoirs built with the best hyperparameters of a study, trained on the train set, scored on the test set.
"""
import math

import numpy as np
from numpy import random
from scipy import sparse, stats

from hag.hag.hag import run_algorithm
from hag.metrics.richness import distance_correlation, pearson, spectral_radius, squared_uncoupled_dynamics_alternative
from hag.models.activation_functions import tanh
from hag.models.reservoir import init_matrices
from hag.performances.esn_model_evaluation import (compute_score, fit_reservoir, init_ip_local_rule_reservoir,
                                                   init_ip_reservoir, init_local_rule_reservoir, init_readout,
                                                   init_reservoir, predict_model_for_classification, run_reservoir,
                                                   train_model_for_classification, train_model_for_prediction)

N_JOBS = 1


def evaluate_dataset_on_test(study, dataset_name, function_name, pretrain_data, train_data, test_data, Y_train, Y_test,
                             is_instances_classification, nb_trials=8, record_metrics=False,
                             random_projection_experiment=False, seed=None):
    """
    Test scores of nb_trials reservoirs of function_name built with the best hyperparameters of study (or, with
    record_metrics, their spectral radius, Pearson correlation, CEV and distance correlation on the test inputs).
    seed: seeds numpy's global RNG, so that two calls draw the same matrices and the same HAG random choices.
    """
    if seed is not None:
        random.seed(seed)
    hyperparams = {param_name: param_value for param_name, param_value in study.best_trial.params.items()}
    print(hyperparams)
    leaky_rate = 1
    input_connectivity = 1

    # score for prediction
    if dataset_name == "Sunspot":
        start_step = 30
        end_step = 500
    else:
        start_step = 500
        end_step = 1500
    SLICE_RANGE = slice(start_step, end_step)

    if not is_instances_classification:
        hyperparams['use_full_instance'] = False

    RIDGE_COEF = 10**hyperparams['ridge']

    scores = []
    if record_metrics:
        spectral_radii = []
        pearson_correlations = []
        CEVs = []
        dcors = []
    for i in range(nb_trials):
        common_index = 1
        if is_instances_classification:
            common_size = pretrain_data[0].shape[common_index]
        else:
            common_size = pretrain_data.shape[common_index]

        # We want the size of the models to be at least network_size
        K = math.ceil(hyperparams['network_size'] / common_size)
        n = common_size * K

        use_block = function_name in ["diag_ee", "diag_ei"]

        # UNSUPERVISED PRETRAINING
        if function_name in ["random_ee", "diag_ee"]:
            Win, W, bias = init_matrices(n, input_connectivity, hyperparams['connectivity'],  K, w_distribution=stats.uniform(loc=0, scale=1), use_block=use_block, seed=random.randint(0, 1000), random_projection_experiment=random_projection_experiment)
        else:
            Win, W, bias = init_matrices(n, input_connectivity, hyperparams['connectivity'],  K, w_distribution=stats.uniform(loc=-1, scale=2), use_block=use_block, seed=random.randint(0, 1000), random_projection_experiment=random_projection_experiment)
        bias *= hyperparams['bias_scaling']
        Win *= hyperparams['input_scaling']

        if function_name == "mean_hag":
            W, (_, _, _) = run_algorithm(W, Win, bias, hyperparams['leaky_rate'], tanh, pretrain_data,
                                         hyperparams['weight_increment'], hyperparams['target_rate'], hyperparams['rate_spread'], "mean_hag",
                                         multiple_instances=is_instances_classification,
                                         min_increment=hyperparams['min_increment'], max_increment=hyperparams['max_increment'], use_full_instance=hyperparams['use_full_instance'],
                                         max_partners=np.inf, method="pearson", n_jobs=N_JOBS)
        elif function_name == "var_hag":
            W, (_, _, _) = run_algorithm(W, Win, bias, hyperparams['leaky_rate'], tanh, pretrain_data,
                                         hyperparams['weight_increment'], hyperparams['variance_target'], hyperparams['variance_spread'], "var_hag",
                                         multiple_instances=is_instances_classification,
                                         min_increment=hyperparams['min_increment'], max_increment=hyperparams['max_increment'], use_full_instance=hyperparams['use_full_instance'],
                                         max_partners=np.inf, method="pearson",
                                         intrinsic_saturation=hyperparams['intrinsic_saturation'], intrinsic_coef=hyperparams['intrinsic_coef'],
                                         n_jobs=N_JOBS)
        elif function_name == "short-hag":
            W, (_, _, _) = run_algorithm(W, Win, bias, hyperparams['leaky_rate'], tanh, pretrain_data,
                                         hyperparams['weight_increment'], hyperparams['target_rate'], hyperparams['rate_spread'], "mean_hag",
                                         multiple_instances=is_instances_classification,
                                         min_increment=1, max_increment=1, use_full_instance=False,
                                         max_partners=np.inf, method="hebbian", n_jobs=N_JOBS)
        elif function_name == "hsp":
            W, (_, _, _) = run_algorithm(W, Win, bias, hyperparams['leaky_rate'], tanh, pretrain_data,
                                         hyperparams['weight_increment'], hyperparams['target_rate'], hyperparams['rate_spread'], "mean_hag",
                                         multiple_instances=is_instances_classification,
                                         min_increment=100, max_increment=100, use_full_instance=False,
                                         max_partners=np.inf, method="random", n_jobs=N_JOBS)
        elif function_name in ["random_ee", "random_ei", "diag_ee", "diag_ei", "ip_correct", "anti-oja", "ip-anti-oja"]:
            eigen = sparse.linalg.eigs(W, k=1, which="LM", maxiter=W.shape[0] * 20, tol=0.1, return_eigenvectors=False, v0=np.ones(W.shape[0]))
            W *= hyperparams['spectral_radius'] / max(abs(eigen))
        else:
            raise ValueError(f"Invalid function: {function_name}")

        # unsupervised local rules
        if is_instances_classification:
            unsupervised_pretrain = np.concatenate(pretrain_data).astype(float)
        else:
            unsupervised_pretrain = pretrain_data.astype(float)
        if function_name == "ip_correct":
            reservoir = init_ip_reservoir(W, Win, bias, mu=hyperparams['mu'], sigma=hyperparams['sigma'], learning_rate=hyperparams['learning_rate'],
                                          leaking_rate=hyperparams['leaky_rate'])
            fit_reservoir(reservoir, unsupervised_pretrain, warmup=100)
        elif function_name == "anti-oja":
            reservoir = init_local_rule_reservoir(W, Win, bias, local_rule="anti-oja", eta=hyperparams['oja_eta'],
                                                  synapse_normalization=False, bcm_theta=None,
                                                  leaking_rate=hyperparams['leaky_rate'], activation_function=tanh)
            fit_reservoir(reservoir, unsupervised_pretrain, warmup=100)
        elif function_name == "ip-anti-oja":
            reservoir = init_ip_local_rule_reservoir(W, Win, bias, local_rule="anti-oja", eta=hyperparams['oja_eta'],
                                                     synapse_normalization=False, bcm_theta=None,
                                                     mu=hyperparams['mu'], sigma=hyperparams['sigma'], learning_rate=hyperparams['learning_rate'],
                                                     leaking_rate=hyperparams['leaky_rate'])
            fit_reservoir(reservoir, unsupervised_pretrain, warmup=100)
        else:
            reservoir = init_reservoir(W, Win, bias, leaky_rate, tanh)
        readout = init_readout(ridge_coef=RIDGE_COEF)

        # TRAINING and EVALUATION
        if record_metrics:
            inputs = np.concatenate(test_data, axis=0) if is_instances_classification else test_data
            states_history_multi = run_reservoir(reservoir, inputs, reset=False)

            sr = spectral_radius(W)
            pearson_correlation, _ = pearson(states_history_multi, num_windows=1, size_window=len(states_history_multi), step_size=1, show_progress=False)
            CEV = squared_uncoupled_dynamics_alternative(states_history_multi, num_windows=1, size_window=len(states_history_multi), step_size=1, show_progress=True)
            dcor = distance_correlation(states_history_multi, num_windows=1, size_window=len(states_history_multi), step_size=1, show_progress=True, method="auto", nb_jobs=N_JOBS)

            spectral_radii.append(sr)
            pearson_correlations.append(pearson_correlation[0])
            CEVs.append(CEV[0])
            dcors.append(dcor[0])
        else:
            if is_instances_classification:
                mode = "sequence-to-vector"
                train_model_for_classification(reservoir, readout, train_data, Y_train, mode=mode)
                Y_pred = predict_model_for_classification(reservoir, readout, test_data, mode=mode)
                score = compute_score(Y_pred, Y_test, is_instances_classification)
            else:
                esn = train_model_for_prediction(reservoir, readout, train_data, Y_train, warmup=start_step, n_jobs=N_JOBS)
                Y_pred = esn.run(test_data, reset=False)
                score = compute_score(Y_pred[SLICE_RANGE], Y_test[SLICE_RANGE], is_instances_classification)

            scores.append(score)

    if record_metrics:
        return spectral_radii, pearson_correlations, CEVs, dcors

    return scores
