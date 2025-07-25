from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
from scipy import sparse, stats
from numpy import random
from matplotlib import pyplot as plt

SEED = 923984

# load dataset using torchaudio
from torchaudio.datasets import VoxCeleb1Identification, SPEECHCOMMANDS
from torch.utils.data import ConcatDataset
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

print("Loading Speechcommands")

dataset_name = "SPEECHCOMMANDS"

dataset_train = SPEECHCOMMANDS(root="datasets/", download=True, subset="training")
dataset_val = SPEECHCOMMANDS(root="datasets/", download=True, subset="validation")

sampling_rate = dataset_train[0][1]

print("Concatenating datasets")

dataset = ConcatDataset([dataset_train, dataset_val])

def process_sample(sample):
    X = sample[0][0].numpy().reshape(-1, 1)
    Y = sample[2]
    group = sample[3]
    return X, Y, group

results = Parallel(n_jobs=-1)(delayed(process_sample)(sample) for sample in tqdm(dataset))
X_train_raw, Y_train_raw, groups = zip(*results)
X_train_raw = list(X_train_raw)
Y_train_raw = list(Y_train_raw)
groups = list(groups)

is_multivariate = False
use_spectral_representation = False
is_instances_classification = True
spectral_representation = "mfcc"

le = LabelEncoder()
Y_train_raw = le.fit_transform(Y_train_raw).reshape(-1, 1)

# One-hot encode the labels
ohe = OneHotEncoder(sparse_output=False)
Y_train_raw = ohe.fit_transform(Y_train_raw.reshape(-1, 1))


from reservoir.activation_functions import tanh, heaviside, sigmoid

# the activation function choosen for the rest of the experiment
# activation_function = lambda x : sigmoid(2*(x-0.5))tanh(x)
activation_function = lambda x : tanh(x)

plt.plot(np.linspace(0, 3, 100), activation_function(np.linspace(0, 3, 100)))
plt.grid()

import math

# Cross validation
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, StratifiedGroupKFold
from datasets.preprocessing import flexible_indexing

#Preprocessing
from datasets.multivariate_generation import generate_multivariate_dataset
from sklearn.preprocessing import MinMaxScaler
from datasets.preprocessing import scale_data
from datasets.preprocessing import add_noise

# Define noise parameter
noise_std = 0.001


nb_splits=3
if is_instances_classification:
    if groups is None:
        splits = StratifiedKFold(n_splits=nb_splits, shuffle=True, random_state=SEED).split(X_train_raw, np.argmax(Y_train_raw, axis=1))
    else:
        splits = StratifiedGroupKFold(n_splits=nb_splits, shuffle=True, random_state=SEED).split(X_train_raw, np.argmax(Y_train_raw, axis=1), groups)
else: #prediction
    splits = TimeSeriesSplit(n_splits=nb_splits).split(X_train_raw)

data_type = "normal" # "normal" ou "noisy"

X_pretrain = []
X_pretrain_noisy  = []
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

for i, (train_index, val_index) in enumerate(splits):
    x_train = flexible_indexing(X_train_raw, train_index)
    x_val = flexible_indexing(X_train_raw, val_index)
    Y_train.append(flexible_indexing(Y_train_raw, train_index))
    Y_val.append(flexible_indexing(Y_train_raw, val_index))
    # SPLITS
    if is_multivariate:
        x_train_band, x_val_band = x_train, x_val
        del x_train, x_val

    # PREPROCESSING
    if use_spectral_representation == True:
        if is_multivariate==False:
            raise ValueError("Cannot use spectral representation if it's not multivariate !")

    hop = 50 if is_instances_classification else 1
    win_length = edge_cut = 100
    if not is_multivariate:
        x_train_band = generate_multivariate_dataset(
            x_train, is_instances_classification, spectral_representation, hop=hop, win_length=win_length
        )
        x_val_band = generate_multivariate_dataset(
            x_val, is_instances_classification, spectral_representation, hop=hop, win_length=win_length
        )
    elif is_multivariate and not use_spectral_representation:
        x_train_band = generate_multivariate_dataset(
            x_train_band, is_instances_classification, spectral_representation, hop=hop, win_length=win_length
        )
        x_val_band = generate_multivariate_dataset(
            x_val_band, is_instances_classification, spectral_representation, hop=hop, win_length=win_length
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
                x_train_noisy=[add_noise(instance, noise_std) for instance in x_train]
                X_train_noisy.append([add_noise(instance, noise_std) for instance in x_train])
                X_val_noisy.append([add_noise(instance, noise_std) for instance in x_val])

            # MULTI
            x_train_band_noisy=[add_noise(instance, noise_std) for instance in x_train_band]
            X_train_band_noisy.append(x_train_band_noisy)
            X_val_band_noisy.append([add_noise(instance, noise_std) for instance in x_val_band])

        else:  #if prediction
            # UNI
            if not is_multivariate:
                x_train_noisy=add_noise(x_train, noise_std)
                X_train_noisy.append(x_train_noisy)
                X_val_noisy.append(add_noise(x_val, noise_std))

            # MULTI
            x_train_band_noisy=add_noise(x_train_band, noise_std)
            X_train_band_noisy.append(x_train_band_noisy)
            X_val_band_noisy.append(add_noise(x_val_band, noise_std))

    # Define the number of instances you want to select
    if is_instances_classification:
        num_samples_for_pretrain = 500 if len(x_train_band) >= 500 else len(x_train_band)
        indices = np.random.choice(len(x_train_band), num_samples_for_pretrain, replace=False)
    else:
        indices = range(len(x_train_band))

    # Defining pretrain
    if data_type == "noisy":
        if not is_multivariate:
            X_pretrain_noisy.append(np.array(x_train_noisy, dtype=object)[indices].flatten())
        X_pretrain_band_noisy.append(np.array(x_train_band_noisy, dtype=object)[indices])

    if not is_multivariate:
        X_pretrain.append(np.array(x_train, dtype=object)[indices].flatten())
    X_pretrain_band.append(np.array(x_train_band, dtype=object)[indices])

if is_instances_classification:
    max_time_increment_possible = max(len(instance) for fold in X_train_band for instance in fold)
else:
    max_time_increment_possible = 500

#Pretraining
from reservoir.reservoir import init_matrices
from connexion_generation.hag import run_algorithm
from scipy import sparse

# Evaluating
from performances.esn_model_evaluation import train_model_for_classification, predict_model_for_classification, compute_score
from performances.esn_model_evaluation import (train_model_for_prediction, init_reservoir, init_ip_reservoir,
                                               init_local_rule_reservoir, init_ip_local_rule_reservoir, init_readout)


# score for prediction
start_step = 30
end_step = 500
SLICE_RANGE = slice(start_step, end_step)
RESERVOIR_SIZE = 500

nb_jobs_per_trial = 10
function_name = "ip-anti-oja_fast" # "random_ee", "random_ei", "desp", "hadsp", "ip_correct", "anti-oja_fast", "ip-anti-oja_fast"
variate_type = "multi"  # "multi" ou "uni"
if variate_type == "uni" and is_multivariate:
    raise ValueError(f"Invalid variable type: {variate_type}")


def objective(trial):
    # Suggest values for the parameters you want to optimize
    # COMMON
    ridge = trial.suggest_int('ridge', -13, 1)
    RIDGE_COEF = 10 ** ridge

    network_size = trial.suggest_int('network_size', RESERVOIR_SIZE, RESERVOIR_SIZE)
    input_scaling = trial.suggest_float('input_scaling', 0.01, 0.2, step=0.005)
    bias_scaling = trial.suggest_float('bias_scaling', 0, 0.2, step=0.005)
    leaky_rate = trial.suggest_float('leaky_rate', 1, 1)
    input_connectivity = trial.suggest_float('input_connectivity', 1, 1)

    # HADSP
    if function_name == "hadsp":
        target_rate = trial.suggest_float('target_rate', 0.5, 1, step=0.01)
        rate_spread = trial.suggest_float('rate_spread', 0.01, 0.4, step=0.005)
        method = "pearson"
    # DESP
    elif function_name == "desp":
        variance_target = trial.suggest_float('variance_target', 0.001, 0.02, step=0.001)
        variance_spread = trial.suggest_float('variance_spread', 0.001, 0.05, step=0.002)
        intrinsic_saturation = trial.suggest_float('intrinsic_saturation', 0.8, 0.98, step=0.02)
        intrinsic_coef = trial.suggest_float('intrinsic_coef', 0.8, 0.98, step=0.02)
        method = "pearson"
    elif function_name in ["random_ee", "random_ei", "ip_correct", "anti-oja_fast", "ip-anti-oja_fast"]:
        connectivity = trial.suggest_float('connectivity', 0, 1)
        sr = trial.suggest_float('spectral_radius', 0.4, 1.6, step=0.01)
    else:
        raise ValueError(f"Invalid function name: {function_name}")

    if function_name in ["ip_correct", "ip-anti-oja_fast"]:
        mu = trial.suggest_float('mu', 0, 1)
        sigma = trial.suggest_float('sigma', 0, 1)
        learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-1, log=True)
    if function_name in ["anti-oja_fast", "ip-anti-oja_fast"]:
        # We often use a log-uniform distribution for learning rates:
        oja_eta = trial.suggest_float('oja_eta', 1e-8, 1e-3, log=True)

    if function_name in ["hadsp", "desp"]:
        connectivity = trial.suggest_float('connectivity', 0, 0)
        weight_increment = trial.suggest_float('weight_increment', 0.001, 0.1, step=0.001)
        max_partners = np.inf  # trial.suggest_int('max_partners', 10, 20)
        if is_instances_classification:
            use_full_instance = trial.suggest_categorical('use_full_instance', [True, False])
        else:
            use_full_instance = False
        min_increment = trial.suggest_int('min_increment', 3, max_time_increment_possible)
        max_increment = trial.suggest_int('max_increment', min_increment, max_time_increment_possible * 5)

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

        if function_name in ["diag_ee", "diag_ei"]:
            use_block = True
        else:
            use_block = False

        # INITIALISATION AND UNSUPERVISED PRETRAINING
        if function_name in ["random_ee", "diag_ee"]:
            Win, W, bias = init_matrices(n, input_connectivity, connectivity, K,
                                         w_distribution=stats.uniform(loc=0, scale=1),
                                         use_block=use_block, seed=random.randint(0, 1000))
        else:
            Win, W, bias = init_matrices(n, input_connectivity, connectivity, K,
                                         w_distribution=stats.uniform(loc=-1, scale=2),
                                         use_block=use_block, seed=random.randint(0, 1000))
        bias *= bias_scaling
        Win *= input_scaling

        if function_name == "hadsp":
            W, (_, _, _) = run_algorithm(W, Win, bias, leaky_rate, activation_function, pretrain_data,
                                         weight_increment, target_rate, rate_spread, function_name,
                                         multiple_instances=is_instances_classification,
                                         min_increment=min_increment, max_increment=max_increment,
                                         use_full_instance=use_full_instance,
                                         max_partners=max_partners, method=method,
                                         n_jobs=nb_jobs_per_trial)
        elif function_name == "desp":
            W, (_, _, _) = run_algorithm(W, Win, bias, leaky_rate, activation_function, pretrain_data,
                                         weight_increment, variance_target, variance_spread, function_name,
                                         multiple_instances=is_instances_classification,
                                         min_increment=min_increment, max_increment=max_increment,
                                         use_full_instance=use_full_instance,
                                         max_partners=max_partners, method=method,
                                         intrinsic_saturation=intrinsic_saturation, intrinsic_coef=intrinsic_coef,
                                         n_jobs=nb_jobs_per_trial)
        elif function_name in ["random_ee", "random_ei", "ip_correct", "anti-oja_fast", "ip-anti-oja_fast"]:
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
        elif function_name == "anti-oja_fast":
            reservoir = init_local_rule_reservoir(W, Win, bias, local_rule="anti-oja", eta=oja_eta,
                                                  synapse_normalization=False, bcm_theta=None,
                                                    leaking_rate=leaky_rate, activation_function=activation_function,
                                                  )
            _ = reservoir.fit(unsupervised_pretrain, warmup=100)
        elif function_name == "ip-anti-oja_fast":
            reservoir = init_ip_local_rule_reservoir(W, Win, bias, local_rule="anti-oja", eta=oja_eta,
                                                     synapse_normalization=False, bcm_theta=None,
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
            train_model_for_classification(reservoir, readout, train_data, Y_train[i], n_jobs=nb_jobs_per_trial,
                                           mode=mode)

            Y_pred = predict_model_for_classification(reservoir, readout, val_data, n_jobs=nb_jobs_per_trial, mode=mode)
            score = compute_score(Y_pred, Y_val[i], is_instances_classification)
        else:
            esn = train_model_for_prediction(reservoir, readout, train_data, Y_train[i])

            Y_pred = esn.run(val_data, reset=False)
            score = compute_score(Y_pred, Y_val[i], is_instances_classification)

        total_score += score

    average_score = total_score / nb_splits  # Average the score

    return average_score


import optuna
from optuna.samplers import TPESampler
import re

print("Start optuna")

def camel_to_snake(name):
    str1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', str1).lower()
url= "sqlite:///new_tpe_" + camel_to_snake(dataset_name) + "_db.sqlite3"
print(url)

storage = optuna.storages.RDBStorage(
    url=url,
    engine_kwargs={"pool_size": 20, "connect_args": {"timeout": 10}},
)
study_name = function_name + "_" + dataset_name + "_" + data_type + "_" + variate_type
print(study_name)

direction = "maximize" if is_instances_classification else "minimize"
sampler = TPESampler()

N_TRIALS = 400
# n_parallel_studies = 10
# study = optuna.create_study(storage=storage, sampler=sampler, study_name=study_name, direction=direction, load_if_exists=True)
# completed_trials = len([trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE])
# trials_per_process = (N_TRIALS - completed_trials) // n_parallel_studies
#
# # Use joblib to parallelize the optimization
# def optimize_study(n_trials_per_process):
#     study = optuna.create_study(storage=storage, sampler=sampler, study_name=study_name, direction=direction,
#                                 load_if_exists=True)
#     study.optimize(objective, n_trials=n_trials_per_process - completed_trials)
#
#
# Parallel(n_jobs=n_parallel_studies)(
#     delayed(optimize_study)(trials_per_process) for _ in range(n_parallel_studies)
# )

study = optuna.create_study(storage=storage, sampler=sampler, study_name=study_name, direction=direction, load_if_exists=True)
completed_trials = len([trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE])
while completed_trials < N_TRIALS:
    # get the number of trials already done that have been completed
    completed_trials = len([trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE])
    print(f"Completed trials: {completed_trials}/{N_TRIALS}")
    try:
        study.optimize(objective, n_trials=N_TRIALS-completed_trials)
    except Exception as e:
        print(f"Error during optimization: {e}, retrying...")
