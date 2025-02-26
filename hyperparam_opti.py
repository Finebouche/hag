import numpy as np
from scipy import sparse, stats
from numpy import random
from joblib import Parallel, delayed
import math

SEED = 923984

from datasets.load_data import load_data
from reservoir.activation_functions import tanh, heaviside, sigmoid

step_ahead=5
dataset_name = "SPEECHCOMMANDS" # can be "CatsDogs", "FSDD", "JapaneseVowels", "SPEECHCOMMANDS", "SpokenArabicDigits", "MackeyGlass", "Sunspot", "Lorenz", "Henon", "NARMA"

# score for prediction
if dataset_name == "Sunspot":
    start_step = 30
    end_step = 500
else:
    start_step = 500
    end_step = 1500
SLICE_RANGE = slice(start_step, end_step)

print(f"Loading {dataset_name}")

(is_instances_classification, is_multivariate, sampling_rate,
 X_train_raw, X_test_raw, Y_train_raw, Y_test,
 use_spectral_representation, spectral_representation,
 groups) = load_data(dataset_name, step_ahead, visualize=False)

# the activation function chosen for the rest of the experiment
activation_function = lambda x : tanh(x)

# Cross validation
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, StratifiedGroupKFold
from datasets.preprocessing import flexible_indexing

# Preprocessing
from datasets.multivariate_generation import generate_multivariate_dataset, extract_peak_frequencies
from sklearn.preprocessing import MinMaxScaler
from datasets.preprocessing import scale_data, add_noise, duplicate_data

# Define noise parameter
noise_std = 0.001

nb_splits = 3
if is_instances_classification:
    if groups is None:
        splits = (StratifiedKFold(n_splits=nb_splits, shuffle=True, random_state=SEED)
                  .split(X_train_raw, np.argmax(Y_train_raw, axis=1)))
    else:
        splits = (StratifiedGroupKFold(n_splits=nb_splits, shuffle=True, random_state=SEED)
                  .split(X_train_raw, np.argmax(Y_train_raw, axis=1), groups))
else:  # prediction
    splits = TimeSeriesSplit(n_splits=nb_splits).split(X_train_raw)

data_type = "normal"  # "normal" ou "noisy"

X_pretrain = []
X_pretrain_noisy = []
X_train = []
X_train_noisy = []
X_val = []
X_val_noisy = []
X_pretrain_band = []
X_pretrain_band_noisy = []
X_train_band = []
X_train_band_noisy = []
X_val_band = []
X_val_band_noisy = []

Y_train = []
Y_val = []

WINDOW_LENGTH = 10
print("Extracting frequencies")
freq_train_data = X_train_raw
flat_train_data = np.concatenate(freq_train_data, axis=0) if is_instances_classification else freq_train_data
extract_peak_frequencies(flat_train_data, sampling_rate, smooth=False, threshold=1e-5, nperseg=1024, visualize=True)

print("Preprocessing")
for i, (train_index, val_index) in enumerate(splits):
    x_train = flexible_indexing(X_train_raw, train_index)
    x_val = flexible_indexing(X_train_raw, val_index)
    Y_train.append(flexible_indexing(Y_train_raw, train_index))
    Y_val.append(flexible_indexing(Y_train_raw, val_index))

    if is_multivariate:
        x_train_band, x_val_band = x_train, x_val
        del x_train, x_val

    # PREPROCESSING
    freq_train_data = x_train_band if is_multivariate else x_train
    flat_train_data = np.concatenate(freq_train_data, axis=0) if is_instances_classification else freq_train_data
    peak_freqs = extract_peak_frequencies(flat_train_data, sampling_rate, smooth=True, window_length=WINDOW_LENGTH,
                                          threshold=1e-5, nperseg=1024, visualize=False)

    # if it has use_spectral_representation, then it is multivariate
    if use_spectral_representation == True:
        if is_multivariate == False:
            raise ValueError("Cannot use spectral representation if it's not multivariate !")

    if not is_multivariate:
        x_train_band = generate_multivariate_dataset(
            x_train, sampling_rate, is_instances_classification, peak_freqs, spectral_representation, hop=100
        )
        x_val_band = generate_multivariate_dataset(
            x_val, sampling_rate, is_instances_classification, peak_freqs, spectral_representation, hop=100
        )
    elif is_multivariate and not use_spectral_representation:
        x_train_band = generate_multivariate_dataset(
            x_train_band, sampling_rate, is_instances_classification, peak_freqs, spectral_representation, hop=100
        )
        x_val_band = generate_multivariate_dataset(
            x_val_band, sampling_rate, is_instances_classification, peak_freqs, spectral_representation, hop=100
        )
    else:
        print("Data is already spectral, nothing to do")

    if not is_multivariate:
        scaler_x_uni = MinMaxScaler(feature_range=(0, 1))
        x_train, x_val, _ = scale_data(x_train, x_val, None, scaler_x_uni, is_instances_classification)
        X_train.append(x_train)
        X_val.append(x_val)

    scaler_multi = MinMaxScaler(feature_range=(0, 1))
    x_train_band, x_val_band, _ = scale_data(x_train_band, x_val_band, None, scaler_multi, is_instances_classification)
    X_train_band.append(x_train_band)
    X_val_band.append(x_val_band)

    # NOISE
    if data_type == "noisy":
        if is_instances_classification:
            # UNI
            if not is_multivariate:
                x_train_noisy = [add_noise(instance, noise_std) for instance in x_train]
                X_train_noisy.append([add_noise(instance, noise_std) for instance in x_train])
                X_val_noisy.append([add_noise(instance, noise_std) for instance in x_val])

            # MULTI
            x_train_band_noisy = [add_noise(instance, noise_std) for instance in x_train_band]
            X_train_band_noisy.append(x_train_band_noisy)
            X_val_band_noisy.append([add_noise(instance, noise_std) for instance in x_val_band])

        else:  # if prediction
            # UNI
            if not is_multivariate:
                x_train_noisy = add_noise(x_train, noise_std)
                X_train_noisy.append(x_train_noisy)
                X_val_noisy.append(add_noise(x_val, noise_std))

            # MULTI
            x_train_band_noisy = add_noise(x_train_band, noise_std)
            X_train_band_noisy.append(x_train_band_noisy)
            X_val_band_noisy.append(add_noise(x_val_band, noise_std))

    # Define the number of instances you want to select
    x_size = len(x_train_band) if is_multivariate else len(x_train)
    num_samples_for_pretrain = 500 if x_size >= 500 else x_size
    if is_instances_classification:
        indices = np.random.choice(x_size, num_samples_for_pretrain, replace=False)
    else:
        indices = range(x_size)

    if data_type == "noisy":
        # Defining pretrain
        if not is_multivariate:
            X_pretrain_noisy.append(np.array(x_train_noisy, dtype=object)[indices].flatten())
        X_pretrain_band_noisy.append(np.array(x_train_band_noisy, dtype=object)[indices])

    # Defining pretrain
    if not is_multivariate:
        X_pretrain.append(np.array(x_train, dtype=object)[indices].flatten())
    X_pretrain_band.append(np.array(x_train_band, dtype=object)[indices])

from scipy.special import comb

def find_optimal_order(dim, delay, s, max_network_size, max_order_size=5):
    """
    Finds the optimal order N and the total number of variables such that
    total_variables is less than or equal to max_network_size.
    """
    delay_eff = (delay - 1) // (s + 1)
    N = 0
    total_variables = 0

    while True:
        N += 1
        # Ensure N does not exceed max_order_size
        if N > max_order_size:
            N -= 1
            break

        # Calculate the new combination incrementally
        new_comb = comb(dim * delay_eff + N - 1, N, exact=True)
        total_variables += new_comb

        # Check if the total exceeds the maximum network size
        if total_variables > max_network_size:
            # Subtract the last addition and decrement N
            total_variables -= new_comb
            N -= 1
            break

    return N, total_variables


# Pretraining
from reservoir.reservoir import init_matrices
from connexion_generation.hag import run_algorithm


# Evaluating
from performances.esn_model_evaluation import train_model_for_classification, predict_model_for_classification, \
    compute_score, init_readout
from performances.esn_model_evaluation import train_model_for_prediction, init_reservoir, init_ip_reservoir, \
    init_local_rule_reservoir, init_ip_local_rule_reservoir, init_readout

RESERVOIR_SIZE = 500

nb_jobs_per_trial = 10
variate_type = "multi"  # "multi" ou "uni"
if variate_type == "uni" and is_multivariate:
    raise ValueError(f"Invalid variable type: {variate_type}")

#function_name = "ip-anti-oja"  # "random", "random_ei", "desp", "hadsp", "ip_correct", "anti-oja", "ip-anti-oja"
for function_name in ["ip-anti-oja"]:

    def objective(trial):
        # Suggest values for the parameters you want to optimize
        # COMMON
        ridge = trial.suggest_int('ridge', -12, 1)
        RIDGE_COEF = 10 ** ridge

        network_size = trial.suggest_int('network_size', RESERVOIR_SIZE, RESERVOIR_SIZE)
        input_scaling = trial.suggest_float('input_scaling', 0.01, 0.2, step=0.005)
        bias_scaling = trial.suggest_float('bias_scaling', 0, 0.2, step=0.005)
        leaky_rate = trial.suggest_float('leaky_rate', 1, 1)
        input_connectivity = trial.suggest_float('input_connectivity', 1, 1)

        min_window_size = sampling_rate / np.max(np.hstack(peak_freqs))
        max_window_size = sampling_rate / np.min(np.hstack(peak_freqs))

        # HADSP
        if function_name == "hadsp":
            target_rate = trial.suggest_float('target_rate', 0.5, 1, step=0.01)
            rate_spread = trial.suggest_float('rate_spread', 0.01, 0.4, step=0.005)
            method = trial.suggest_categorical("method", ["random", "pearson"])
        # DESP
        elif function_name == "desp":
            variance_target = trial.suggest_float('variance_target', 0.001, 0.02, step=0.001)
            variance_spread = trial.suggest_float('variance_spread', 0.001, 0.05, step=0.002)
            intrinsic_saturation = trial.suggest_float('intrinsic_saturation', 0.8, 0.98, step=0.02)
            intrinsic_coef = trial.suggest_float('intrinsic_coef', 0.8, 0.98, step=0.02)
            method = trial.suggest_categorical("method", ["pearson"])
        elif function_name in ["random", "random_ei", "ip_correct", "anti-oja", "ip-anti-oja"]:
            connectivity = trial.suggest_float('connectivity', 0, 1)
            sr = trial.suggest_float('spectral_radius', 0.4, 1.6, step=0.01)
        else:
            raise ValueError(f"Invalid function name: {function_name}")

        if function_name in ["ip_correct", "ip-anti-oja"] :
            mu = trial.suggest_float('mu', 0, 1)
            sigma = trial.suggest_float('sigma', 0, 1)
            learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-1, log=True)
        if function_name in ["anti-oja", "ip-anti-oja"]:
            # We often use a log-uniform distribution for learning rates:
            oja_eta = trial.suggest_float('oja_eta', 1e-5, 1e-1, log=True)


        if function_name in ["hadsp", "desp"]:
            connectivity = trial.suggest_float('connectivity', 0, 0)
            weight_increment = trial.suggest_float('weight_increment', 0.001, 0.1, step=0.001)
            max_partners = trial.suggest_int('max_partners', 10, 20)
            if is_instances_classification:
                use_full_instance = trial.suggest_categorical('use_full_instance', [True, False])
            else:
                use_full_instance = False
            TIME_INCREMENT = trial.suggest_int('time_increment', int(min_window_size + 1),
                                               100)  # int(min_window_size+1) or int(max_window_size)
            max_increment_span = int(max_window_size) if int(max_window_size) - 100 < 0 else int(max_window_size) - 100
            time_increment_span = trial.suggest_int('time_increment_span', 0, max_increment_span)
            MAX_TIME_INCREMENT = TIME_INCREMENT + time_increment_span  # int(max_window_size) or None or TIME_INCREMENT

        # CROSS-VALIDATION METHODS
        total_score = 0
        for i in range(nb_splits):
            common_index = 1
            if is_instances_classification:
                common_size = X_train_band[i][0].shape[common_index]
            else:
                common_size = X_train_band[i].shape[common_index]

            # We want the size of the reservoir to be at least network_size
            K = math.ceil(network_size / common_size)
            n = common_size * K

            pretrain_data = X_pretrain_band[i]
            train_data = X_train_band[i]  # X_train_band_noisy_duplicated or X_train_band_duplicated
            val_data = X_val_band_noisy[i] if data_type == "noisy" else X_val_band[i]

            # INITIALISATION AND UNSUPERVISED PRETRAINING
            if function_name == "random_ei":
                Win, W, bias = init_matrices(n, input_connectivity, connectivity, K, w_distribution=stats.uniform(-1, 1),
                                             seed=random.randint(0, 1000))
            else:
                Win, W, bias = init_matrices(n, input_connectivity, connectivity, K, seed=random.randint(0, 1000))
            bias *= bias_scaling
            Win *= input_scaling

            if function_name == "hadsp":
                W, (_, _, _) = run_algorithm(W, Win, bias, leaky_rate, activation_function, pretrain_data, TIME_INCREMENT,
                                             weight_increment,
                                             target_rate, rate_spread, function_name,
                                             is_instance=is_instances_classification, use_full_instance=use_full_instance,
                                             max_increment=MAX_TIME_INCREMENT, max_partners=max_partners, method=method,
                                             n_jobs=nb_jobs_per_trial)
            elif function_name == "desp":
                W, (_, _, _) = run_algorithm(W, Win, bias, leaky_rate, activation_function, pretrain_data, TIME_INCREMENT,
                                             weight_increment,
                                             variance_target, variance_spread, function_name,
                                             is_instance=is_instances_classification, use_full_instance=use_full_instance,
                                             max_increment=MAX_TIME_INCREMENT, max_partners=max_partners, method=method,
                                             intrinsic_saturation=intrinsic_saturation, intrinsic_coef=intrinsic_coef,
                                             n_jobs=nb_jobs_per_trial)
            elif function_name in ["random", "random_ei", "ip_correct", "anti-oja", "ip-anti-oja"]:
                eigen = sparse.linalg.eigs(W, k=1, which="LM", maxiter=W.shape[0] * 20, tol=0.1, return_eigenvectors=False)
                W *= sr / max(abs(eigen))
            else:
                raise ValueError(f"Invalid function: {function_name}")

            # unsupervised local rules
            if is_instances_classification:
                unsupervised_pretrain = np.concatenate(pretrain_data).astype(float)
            else:
                unsupervised_pretrain = pretrain_data.astype(float)
            if function_name == "ip_correct":
                reservoir = init_ip_reservoir(W, Win, bias, mu=mu, sigma=sigma, learning_rate=learning_rate,
                                              leaking_rate=leaky_rate, activation_function=activation_function
                                              )
                _ = reservoir.fit(unsupervised_pretrain, warmup=100)
            elif function_name == "anti_oja":
                reservoir = init_local_rule_reservoir(W, Win, bias, local_rule="anti-oja", eta=oja_eta,
                                                       synapse_normalization=True, bcm_theta=None,
                                                       leaking_rate=leaky_rate, activation_function=activation_function,
                                                       )
                _ = reservoir.fit(unsupervised_pretrain, warmup=100)
            elif function_name == "ip-anti-oja":
                reservoir = init_ip_local_rule_reservoir(W, Win, bias, local_rule="anti-oja", eta=oja_eta,
                                                          synapse_normalization=True, bcm_theta=None,
                                                          mu=mu, sigma=sigma, learning_rate=learning_rate,
                                                          leaking_rate=leaky_rate, activation_function=activation_function,
                                                          )
                _ = reservoir.fit(unsupervised_pretrain, warmup=100)
            else:
                reservoir = init_reservoir(W, Win, bias, leaky_rate, activation_function)
            readout = init_readout(ridge_coef=RIDGE_COEF)

            # TRAINING and EVALUATION
            if is_instances_classification:
                mode = "sequence-to-vector"
                train_model_for_classification(reservoir, readout, train_data, Y_train[i], n_jobs=nb_jobs_per_trial, mode=mode)

                Y_pred = predict_model_for_classification(reservoir, readout, val_data, n_jobs=nb_jobs_per_trial, mode=mode)
                score = compute_score(Y_pred, Y_val[i], is_instances_classification)
            else:
                esn = train_model_for_prediction(reservoir, readout, train_data, Y_train[i], warmup=start_step)

                Y_pred = esn.run(val_data, reset=False)
                score = compute_score(Y_pred[SLICE_RANGE], Y_val[i][SLICE_RANGE], is_instances_classification)

            total_score += score

        average_score = total_score / nb_splits  # Average the score

        return average_score

    import optuna
    from optuna.samplers import TPESampler
    from performances.utility import camel_to_snake

    url= "sqlite:///tpe_" + camel_to_snake(dataset_name) + "_db.sqlite3"
    print(url)

    study_name = function_name + "_" + dataset_name + "_" + data_type + "_" + variate_type
    print(study_name)

    direction = "maximize" if is_instances_classification else "minimize"

    storage = optuna.storages.RDBStorage(
        url=url,
        engine_kwargs={"pool_size": 20, "connect_args": {"timeout": 10}},
    )
    sampler = TPESampler()

    def optimize_study(n_trials):
        study = optuna.create_study(storage=storage, sampler=sampler, study_name=study_name, direction=direction, load_if_exists=True)
        study.optimize(objective, n_trials=n_trials)

    N_TRIALS = 300
    n_parallel_studies = 1
    trials_per_process = N_TRIALS // n_parallel_studies

    # Use joblib to parallelize the optimization
    # Parallel(n_jobs=n_parallel_studies)(
    #     delayed(optimize_study)(trials_per_process) for _ in range(n_parallel_studies)
    # )
    #
    study = optuna.create_study(storage=storage, sampler=sampler, study_name=study_name, direction=direction, load_if_exists=True)
    study.optimize(objective, n_trials=N_TRIALS)