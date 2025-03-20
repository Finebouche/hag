import numpy as np
import math
from scipy import sparse
import os
import pandas as pd
from scipy import stats
from numpy import random

SEED = 923984

from reservoir.activation_functions import tanh
activation_function = lambda x : tanh(x)

# Preprocessing
from datasets.multivariate_generation import generate_multivariate_dataset, extract_peak_frequencies
from sklearn.preprocessing import MinMaxScaler
from datasets.preprocessing import scale_data
from datasets.load_data import load_data as load_dataset

# Evaluating
from performances.esn_model_evaluation import init_reservoir, init_ip_reservoir, init_local_rule_reservoir, init_ip_local_rule_reservoir
from analysis.richness import spectral_radius, pearson, squared_uncoupled_dynamics_alternative, distance_correlation
from reservoir.reservoir import init_matrices
from connexion_generation.hag import run_algorithm
from performances.utility import retrieve_best_model

nb_jobs = 10

from seaborn import color_palette

# -- Define color palettes for each group --
blues = color_palette("Blues", 5)      # shades of blue
oranges = color_palette("Oranges", 2)  # shades of orange
greens = color_palette("Greens", 2)    # shades of green

# -- Map each function to its color --
function_colors = {
    'E-ESN':                greens[0],
    'ESN':                  greens[1],
    'IP':                   blues[0],
    'Anti-Oja':             blues[1],
    'Anti-Oja-Fast':        blues[2],
    'IP +\nAnti-Oja':       blues[3],
    'IP +\nAnti-Oja\nFast': blues[4],
    'mean HAG':             oranges[0],
    'variance HAG':         oranges[1],
}

# If you want a specific order for the bars, you can enforce it:
functions_order = [
    'E-ESN',
    'ESN',
    'IP',
    'Anti-Oja',
#    'Anti-Oja-Fast',
    'IP +\nAnti-Oja',
 #   'IP +\nAnti-Oja\nFast',
    'mean HAG',
    'variance HAG',
]

function_mapping = {
    'ip-anti-oja_fast': 'IP +\nAnti-Oja',
#    'ip-anti-oja':      'IP +\nAnti-Oja',
    'ip_correct':       'IP',
    'anti-oja_fast':    'Anti-Oja',
#    'anti-oja':         'Anti-Oja',
    'desp':             'variance HAG',
    'hadsp':            'mean HAG',
    'random_ei':        'ESN',
    'random_ee':        'E-ESN',
}

def load_data(dataset_name, step_ahead=5):
    (is_instances_classification, is_multivariate, sampling_rate,
     X_train_raw, X_test_raw, Y_train_raw, Y_test,
     use_spectral_representation, spectral_representation,
     groups) = load_dataset(dataset_name, step_ahead, visualize=False)
    del Y_train_raw
    del Y_test

    WINDOW_LENGTH = 10
    freq_train_data = X_train_raw
    flat_train_data = np.concatenate(freq_train_data, axis=0) if is_instances_classification else freq_train_data
    extract_peak_frequencies(flat_train_data, sampling_rate, smooth=True,
                             window_length=WINDOW_LENGTH, threshold=1e-5,
                             nperseg=1024, visualize=False)

    if is_multivariate:
        X_train_band, X_test_band = X_train_raw, X_test_raw
        del X_train_raw, X_test_raw
        X_val_band = None
    else:
        X_test, X_train = X_test_raw, X_train_raw
        X_val, X_val_band = None, None
        del X_train_raw, X_test_raw

    # PREPROCESSING
    freq_train_data = X_train_band if is_multivariate else X_train
    flat_train_data = np.concatenate(freq_train_data, axis=0) if is_instances_classification else freq_train_data
    peak_freqs = extract_peak_frequencies(flat_train_data, sampling_rate, smooth=True, window_length=WINDOW_LENGTH,
                                          threshold=1e-5, nperseg=1024, visualize=False)

    if not is_multivariate:
        X_train_band = generate_multivariate_dataset(
            X_train, sampling_rate, is_instances_classification, peak_freqs, spectral_representation, hop=100
        )

        X_test_band = generate_multivariate_dataset(
            X_test, sampling_rate, is_instances_classification, peak_freqs, spectral_representation, hop=100
        )
    elif not use_spectral_representation:
        X_train_band = generate_multivariate_dataset(
            X_train_band, sampling_rate, is_instances_classification, peak_freqs, spectral_representation, hop=100
        )
        X_test_band = generate_multivariate_dataset(
            X_test_band, sampling_rate, is_instances_classification, peak_freqs, spectral_representation, hop=100
        )
    else:
        print("Data is already spectral and multivariate, nothing to do")

    scaler_multi = MinMaxScaler(feature_range=(0, 1))
    X_train_band, X_val_band, X_test_band = scale_data(X_train_band, X_val_band, X_test_band, scaler_multi, is_instances_classification)

    if not is_multivariate:
        scaler_x_uni = MinMaxScaler(feature_range=(0, 1))
        X_train, X_val, X_test = scale_data(X_train, X_val, X_test, scaler_x_uni, is_instances_classification)

    # Define the number of instances you want to select
    x_size = len(X_train_band) if is_multivariate else len(X_train)
    num_samples_for_pretrain = 500 if x_size >= 500 else x_size
    if is_instances_classification:
        indices = np.random.choice(x_size, num_samples_for_pretrain, replace=False)
    else:
        indices = range(x_size)

    X_pretrain_band = np.array(X_train_band, dtype=object)[indices]

    return X_pretrain_band, X_test_band, is_multivariate, is_instances_classification


def evaluate_dataset_on_test(study, function_name, pretrain_data, test_data, is_instances_classification, nb_trials=8):
    # Collect all hyperparameters in a dictionary
    hyperparams = {param_name: param_value for param_name, param_value in study.best_trial.params.items()}
    print(hyperparams)
    leaky_rate = 1
    input_connectivity = 1

    if 'variance_target' not in hyperparams and 'min_variance' in hyperparams:
        hyperparams['variance_target'] = hyperparams['min_variance']
    if not is_instances_classification:
        hyperparams['use_full_instance'] = False


    if function_name in ["hadsp", "desp"]:
        MAX_TIME_INCREMENT = hyperparams['time_increment'] + hyperparams[
            'time_increment_span']  # int(max_window_size) or None or TIME_INCREMENT

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

        # We want the size of the reservoir to be at least network_size
        K = math.ceil(hyperparams['network_size'] / common_size)
        n = common_size * K

        # UNSUPERVISED PRETRAINING
        if function_name == "random_ee":
            Win, W, bias = init_matrices(n, input_connectivity, hyperparams['connectivity'], K,
                                         w_distribution=stats.uniform(loc=0, scale=1), seed=random.randint(0, 1000))
        else:
            Win, W, bias = init_matrices(n, input_connectivity, hyperparams['connectivity'], K,
                                         w_distribution=stats.uniform(loc=-1, scale=2), seed=random.randint(0, 1000))
        bias *= hyperparams['bias_scaling']
        Win *= hyperparams['input_scaling']

        if function_name == "hadsp":
            W, (_, _, _) = run_algorithm(W, Win, bias, hyperparams['leaky_rate'], activation_function, pretrain_data,
                                         hyperparams['time_increment'], hyperparams['weight_increment'],
                                         hyperparams['target_rate'], hyperparams['rate_spread'], function_name,
                                         is_instance=is_instances_classification,
                                         use_full_instance=hyperparams['use_full_instance'],
                                         max_increment=MAX_TIME_INCREMENT, max_partners=hyperparams['max_partners'],
                                         method="pearson", n_jobs=nb_jobs)
        elif function_name == "desp":
            W, (_, _, _) = run_algorithm(W, Win, bias, hyperparams['leaky_rate'], activation_function, pretrain_data,
                                         hyperparams['time_increment'], hyperparams['weight_increment'],
                                         hyperparams['variance_target'], hyperparams['variance_spread'], function_name,
                                         is_instance=is_instances_classification,
                                         use_full_instance=hyperparams['use_full_instance'],
                                         max_increment=MAX_TIME_INCREMENT, max_partners=hyperparams['max_partners'],
                                         method="pearson",
                                         intrinsic_saturation=hyperparams['intrinsic_saturation'],
                                         intrinsic_coef=hyperparams['intrinsic_coef'], n_jobs=nb_jobs)
        elif function_name in ["random_ee", "random_ei", "ip_correct", "anti-oja", "anti-oja_fast", "ip-anti-oja",
                               "ip-anti-oja_fast"]:
            eigen = sparse.linalg.eigs(W, k=1, which="LM", maxiter=W.shape[0] * 20, tol=0.1, return_eigenvectors=False)
            W *= hyperparams['spectral_radius'] / max(abs(eigen))
        else:
            raise ValueError(f"Invalid function: {function_name}")

        # unsupervised local rules
        if is_instances_classification:
            unsupervised_pretrain = np.concatenate(pretrain_data).astype(float)
        else:
            unsupervised_pretrain = pretrain_data.astype(float)
        if function_name == "ip_correct":
            reservoir = init_ip_reservoir(W, Win, bias, mu=hyperparams['mu'], sigma=hyperparams['sigma'],
                                          learning_rate=hyperparams['learning_rate'],
                                          leaking_rate=hyperparams['leaky_rate'],
                                          activation_function=activation_function
                                          )
            _ = reservoir.fit(unsupervised_pretrain, warmup=100)
        elif function_name == "anti-oja":
            reservoir = init_local_rule_reservoir(W, Win, bias, local_rule="anti-oja", eta=hyperparams['oja_eta'],
                                                  synapse_normalization=True, bcm_theta=None,
                                                  leaking_rate=hyperparams['leaky_rate'],
                                                  activation_function=activation_function,
                                                  )
            _ = reservoir.fit(unsupervised_pretrain, warmup=100)
        elif function_name == "anti-oja_fast":
            reservoir = init_local_rule_reservoir(W, Win, bias, local_rule="anti-oja", eta=hyperparams['oja_eta'],
                                                  synapse_normalization=False, bcm_theta=None,
                                                  leaking_rate=hyperparams['leaky_rate'],
                                                  activation_function=activation_function,
                                                  )
            _ = reservoir.fit(unsupervised_pretrain, warmup=100)
        elif function_name == "ip-anti-oja":
            reservoir = init_ip_local_rule_reservoir(W, Win, bias, local_rule="anti-oja", eta=hyperparams['oja_eta'],
                                                     synapse_normalization=True, bcm_theta=None,
                                                     mu=hyperparams['mu'], sigma=hyperparams['sigma'],
                                                     learning_rate=hyperparams['learning_rate'],
                                                     leaking_rate=hyperparams['leaky_rate'],
                                                     activation_function=activation_function,
                                                     )
            _ = reservoir.fit(unsupervised_pretrain, warmup=100)
        elif function_name == "ip-anti-oja_fast":
            reservoir = init_ip_local_rule_reservoir(W, Win, bias, local_rule="anti-oja", eta=hyperparams['oja_eta'],
                                                     synapse_normalization=False, bcm_theta=None,
                                                     mu=hyperparams['mu'], sigma=hyperparams['sigma'],
                                                     learning_rate=hyperparams['learning_rate'],
                                                     leaking_rate=hyperparams['leaky_rate'],
                                                     activation_function=activation_function,
                                                     )
            _ = reservoir.fit(unsupervised_pretrain, warmup=100)
        else:
            reservoir = init_reservoir(W, Win, bias, leaky_rate, activation_function)

        inputs = np.concatenate(test_data, axis=0) if is_instances_classification else test_data
        states_history_multi = reservoir.run(inputs)

        sr = spectral_radius(W)
        pearson_correlation, _ = pearson(states_history_multi, num_windows=1, size_window=len(states_history_multi),
                                         step_size=1, show_progress=False)
        CEV = squared_uncoupled_dynamics_alternative(states_history_multi, num_windows=1,
                                                size_window=len(states_history_multi), step_size=1,
                                                show_progress=True)
        dcor = distance_correlation(states_history_multi, num_windows=1, size_window=len(states_history_multi),
                                    step_size=1, show_progress=True, method="mergesort", nb_jobs=nb_jobs)

        spectral_radii.append(sr)
        pearson_correlations.append(pearson_correlation[0])
        CEVs.append(CEV[0])
        dcors.append(dcor[0])

    return spectral_radii, pearson_correlations, CEVs, dcors



# Create an empty DataFrame to store the results
columns = [
    "dataset",
    "function_name",
    "spectral_radius_mean",
    "spectral_radius_std",
    "pearson_mean",
    "pearson_std",
    "CEV_mean",
    "CEV_std",
    "dcor_mean",
    "dcor_std",
]

# List of datasets (extract from filenames)
dataset = "SPEECHCOMMANDS"

new_results = pd.DataFrame(columns=columns)
print(dataset)
pretrain_data, test_data, is_multivariate, is_instances_classification = load_data(dataset)
for function_name in ["random_ee", "random_ei", "ip_correct", "anti-oja_fast"]:  # "random_ee", "random_ei", "ip_correct", "anti-oja_fast",  "ip-anti-oja_fast", "hadsp", "desp"
    # Get the best trial from the study
    print(function_name)
    study = retrieve_best_model(function_name, dataset, is_multivariate, variate_type="multi", data_type="normal")

    SRs, pearsons, CEVs, dcors = evaluate_dataset_on_test(
        study,
        function_name,
        pretrain_data,
        test_data[:500],
        is_instances_classification,
        nb_trials=4,
    )
    # Create a new DataFrame row
    new_row = pd.DataFrame({
        "dataset": [dataset],
        "function_name": [function_name],
        "spectral_radius_mean": [np.mean(SRs)],
        "spectral_radius_std": [np.std(SRs)],
        "pearson_mean": [np.mean(pearsons)],
        "pearson_std": [np.std(pearsons)],
        "CEV_mean": [np.mean(CEVs)],
        "CEV_std": [np.std(CEVs)],
        "dcor_mean": [np.mean(dcors)],
        "dcor_std": [np.std(dcors)],
    })

    # Concatenate the new row to the results DataFrame
    new_results = pd.concat([new_results, new_row], ignore_index=True)

# Display the DataFrame
print(new_results)
file_name = "metrics.csv"

# Load the existing CSV
if os.path.exists(file_name) and os.path.getsize(file_name) > 0:
    try:
        previous_results = pd.read_csv(file_name)
    except pd.errors.EmptyDataError:
        print(f"{file_name} is empty. Initializing with default columns.")
        previous_results = pd.DataFrame(columns=columns)
        previous_results.to_csv(file_name, index=False)
else:
    print(f"{file_name} does not exist or is empty. Creating a new file.")
    previous_results = pd.DataFrame(columns=columns)
    previous_results.to_csv(file_name, index=False)

tots_results = pd.concat([new_results, previous_results], axis=0)

tots_results.to_csv(file_name, index=False)

print(f"Results saved to {file_name}")
