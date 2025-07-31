import numpy as np
from scipy import sparse, stats
from numpy import random
from joblib import Parallel, delayed
import math

SEED = 923984

from datasets.load_data import load_data
from models.activation_functions import tanh

# the activation function chosen for the rest of the experiment
activation_function = lambda x : tanh(x)

# Cross validation
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, StratifiedGroupKFold
from datasets.preprocessing import flexible_indexing

# Preprocessing
from datasets.multivariate_generation import generate_multivariate_dataset
from sklearn.preprocessing import MinMaxScaler
from datasets.preprocessing import scale_data, add_noise


if __name__ == '__main__':

    step_ahead=5
    # can be  "JapaneseVowels", "CatsDogs", "FSDD", "SpokenArabicDigits", "SPEECHCOMMANDS", "MackeyGlass", "Sunspot_daily", "Lorenz", "Henon", "NARMA"
    datasets = ["CatsDogs", "JapaneseVowels", "FSDD"]
    for dataset_name in datasets:
        # score for prediction
        start_step = 500
        end_step = 1500
        SLICE_RANGE = slice(start_step, end_step)

        print(f"Loading {dataset_name}")

        (is_instances_classification, is_multivariate, sampling_rate,
         X_train_raw, X_test_raw, Y_train_raw, Y_test,
         use_spectral_representation, spectral_representation,
         groups) = load_data(dataset_name, step_ahead, visualize=False)

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
        print("Preprocessing")
        for i, (train_index, val_index) in enumerate(splits):
            x_train = flexible_indexing(X_train_raw, train_index)
            x_val = flexible_indexing(X_train_raw, val_index)

            y_train = flexible_indexing(Y_train_raw, train_index)
            y_val = flexible_indexing(Y_train_raw, val_index)

            if is_multivariate:
                x_train_band, x_val_band = x_train, x_val
                del x_train, x_val

            # PREPROCESSING
            # if it has use_spectral_representation, then it is multivariate
            if use_spectral_representation == True:
                if is_multivariate == False:
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

            if not is_instances_classification:
                x_train_band = x_train_band[edge_cut:-edge_cut]
                x_val_band = x_val_band[edge_cut:-edge_cut]

                y_train = y_train[edge_cut:-edge_cut]
                y_val = y_val[edge_cut:-edge_cut]

            Y_train.append(y_train)
            Y_val.append(y_val)

            # NORMALIZATION
            if not is_multivariate:
                scaler_x_uni = MinMaxScaler(feature_range=(0, 1))
                x_train, x_val, _ = scale_data(x_train, x_val, None, scaler_x_uni, is_instances_classification)
                X_train.append(x_train)
                X_val.append(x_val)

            scaler_multi = MinMaxScaler(feature_range=(0, 1))
            x_train_band, x_val_band, _ = scale_data(x_train_band, x_val_band, None, scaler_multi, is_instances_classification)
            X_train_band.append(x_train_band)
            X_val_band.append(x_val_band)

            # OPTIONAL NOISE
            if data_type == "noisy":
                if is_instances_classification:
                    # uni
                    if not is_multivariate:
                        x_train_noisy = [add_noise(instance, noise_std) for instance in x_train]
                        X_train_noisy.append([add_noise(instance, noise_std) for instance in x_train])
                        X_val_noisy.append([add_noise(instance, noise_std) for instance in x_val])

                    # multi
                    x_train_band_noisy = [add_noise(instance, noise_std) for instance in x_train_band]
                    X_train_band_noisy.append(x_train_band_noisy)
                    X_val_band_noisy.append([add_noise(instance, noise_std) for instance in x_val_band])

                else:  # if prediction
                    # uni
                    if not is_multivariate:
                        x_train_noisy = add_noise(x_train, noise_std)
                        X_train_noisy.append(x_train_noisy)
                        X_val_noisy.append(add_noise(x_val, noise_std))

                    # multi
                    x_train_band_noisy = add_noise(x_train_band, noise_std)
                    X_train_band_noisy.append(x_train_band_noisy)
                    X_val_band_noisy.append(add_noise(x_val_band, noise_std))

            # PRETRAINING SET
            if is_instances_classification:
                num_samples_for_pretrain = 500 if len(x_train_band) >= 500 else len(x_train_band)
                indices = np.random.choice(len(x_train_band), num_samples_for_pretrain, replace=False)
            else:
                indices = range(len(x_train_band))

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

        # Pretraining
        from models.reservoir import init_matrices
        from hag.hag import run_algorithm

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

        # "random_ee", "random_ei", "diag_ee", "diag_ei", "desp", "hadsp", "ip_correct", "anti-oja_fast", "ip-anti-oja_fast", "lstm"
        for function_name in ["mean_hag_marked"]:
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

                # HADSP
                if function_name in ("hadsp", "mean_hag_marked"):
                    target_rate = trial.suggest_float('target_rate', 0.5, 1, step=0.01)
                    rate_spread = trial.suggest_float('rate_spread', 0.01, 0.4, step=0.005)
                    method = "pearson"
                # DESP
                elif function_name in ("desp", "var_hag_marked"):
                    variance_target = trial.suggest_float('variance_target', 0.001, 0.02, step=0.001)
                    variance_spread = trial.suggest_float('variance_spread', 0.001, 0.05, step=0.002)
                    intrinsic_saturation = trial.suggest_float('intrinsic_saturation', 0.8, 0.98, step=0.02)
                    intrinsic_coef = trial.suggest_float('intrinsic_coef', 0.8, 0.98, step=0.02)
                    method = "pearson"
                elif function_name in ["random_ee", "random_ei", "diag_ee", "diag_ei", "ip_correct", "anti-oja_fast", "ip-anti-oja_fast"]:
                    connectivity = trial.suggest_float('connectivity', 0, 1)
                    sr = trial.suggest_float('spectral_radius', 0.4, 1.6, step=0.01)
                else:
                    raise ValueError(f"Invalid function name: {function_name}")

                if function_name in ["ip_correct", "ip-anti-oja", "ip-anti-oja_fast"] :
                    mu = trial.suggest_float('mu', 0, 1)
                    sigma = trial.suggest_float('sigma', 0, 1)
                    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-1, log=True)
                if function_name in ["anti-oja", "anti-oja_fast", "ip-anti-oja", "ip-anti-oja_fast"]:
                    # We often use a log-uniform distribution for learning rates:
                    oja_eta = trial.suggest_float('oja_eta', 1e-8, 1e-3, log=True)


                if function_name in ["hadsp", "desp", "mean_hag_marked", "var_hag_marked"]:
                    connectivity = trial.suggest_float('connectivity', 0, 0)
                    weight_increment = trial.suggest_float('weight_increment', 0.001, 0.1, step=0.001)
                    max_partners = np.inf # trial.suggest_int('max_partners', 10, 20)
                    if is_instances_classification:
                        use_full_instance = trial.suggest_categorical('use_full_instance', [True, False])
                    else:
                        use_full_instance = False
                    min_increment = trial.suggest_int('min_increment', 3, max_time_increment_possible)
                    max_increment = trial.suggest_int('max_increment', min_increment, max_time_increment_possible*5)

                # CROSS-VALIDATION METHODS
                total_score = 0
                for i in range(nb_splits):
                    common_index = 1
                    if is_instances_classification:
                        common_size = X_train_band[i][0].shape[common_index]
                    else:
                        common_size = X_train_band[i].shape[common_index]

                    # We want the size of the models to be at least network_size
                    # K is the number of time a single input is repeated to the models
                    K = math.ceil(network_size / common_size)
                    n = common_size * K

                    pretrain_data = X_pretrain_band[i]
                    train_data = X_train_band[i]
                    val_data = X_val_band_noisy[i] if data_type == "noisy" else X_val_band[i]

                    if function_name in ["diag_ee", "diag_ei"]:
                        use_block = True
                    else:
                        use_block = False

                    # INITIALISATION AND UNSUPERVISED PRETRAINING
                    if function_name in ["random_ee", "diag_ee"]:
                        Win, W, bias = init_matrices(n, input_connectivity, connectivity, K, w_distribution=stats.uniform(loc=0, scale=1),
                                                     use_block=use_block, seed=random.randint(0, 1000))
                    else:
                        Win, W, bias = init_matrices(n, input_connectivity, connectivity, K, w_distribution=stats.uniform(loc=-1, scale=2),
                                                     use_block=use_block, seed=random.randint(0, 1000))
                    bias *= bias_scaling
                    Win *= input_scaling

                    if function_name in ("hadsp", "mean_hag_marked"):
                        W, (_, _, _) = run_algorithm(W, Win, bias, leaky_rate, activation_function, pretrain_data,
                                                     weight_increment, target_rate, rate_spread, function_name,
                                                     multiple_instances=is_instances_classification,
                                                     min_increment = min_increment, max_increment=max_increment, use_full_instance=use_full_instance,
                                                     max_partners=max_partners, method=method,
                                                     n_jobs=nb_jobs_per_trial)
                    elif function_name in ("desp", "var_hag_marked"):
                        W, (_, _, _) = run_algorithm(W, Win, bias, leaky_rate, activation_function, pretrain_data,
                                                     weight_increment, variance_target, variance_spread, function_name,
                                                     multiple_instances=is_instances_classification,
                                                     min_increment = min_increment, max_increment=max_increment, use_full_instance=use_full_instance,
                                                     max_partners=max_partners, method=method,
                                                     intrinsic_saturation=intrinsic_saturation, intrinsic_coef=intrinsic_coef,
                                                     n_jobs=nb_jobs_per_trial)
                    elif function_name in ["random_ee", "random_ei", "diag_ee", "diag_ei", "ip_correct", "anti-oja_fast", "ip-anti-oja_fast"]:
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
                        train_model_for_classification(reservoir, readout, train_data, Y_train[i], n_jobs=nb_jobs_per_trial, mode=mode)

                        Y_pred = predict_model_for_classification(reservoir, readout, val_data, n_jobs=nb_jobs_per_trial, mode=mode)
                        score = compute_score(Y_pred, Y_val[i], is_instances_classification)
                    else:
                        esn = train_model_for_prediction(reservoir, readout, train_data, Y_train[i], warmup=start_step, n_jobs=nb_jobs_per_trial)

                        Y_pred = esn.run(val_data, reset=False)
                        score = compute_score(Y_pred[SLICE_RANGE], Y_val[i][SLICE_RANGE], is_instances_classification)

                    total_score += score

                average_score = total_score / nb_splits  # Average the score

                return average_score


            import optuna
            from optuna.samplers import TPESampler, CmaEsSampler
            import re

            def camel_to_snake(name):
                str1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
                return re.sub('([a-z0-9])([A-Z])', r'\1_\2', str1).lower()

            print("Start optuna")

            sampler = TPESampler()
            sampler_name = "cmaes" if isinstance(sampler, CmaEsSampler) else "tpe"
            url = f"sqlite:///new_{sampler_name}_{camel_to_snake(dataset_name)}_db.sqlite3"
            storage = optuna.storages.RDBStorage(url=url, engine_kwargs={"pool_size": 20, "connect_args": {"timeout": 10}})
            print(url)
            study_name = function_name + "_" + dataset_name + "_" + data_type + "_" + variate_type
            print(study_name)
            direction = "maximize" if is_instances_classification else "minimize"

            N_TRIALS = 400
            study = optuna.create_study(storage=storage, sampler=sampler, study_name=study_name, direction=direction, load_if_exists=True)
            completed_trials = len([trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE])

            # Parallelized
            n_parallel_studies = 10
            trials_per_process = (N_TRIALS - completed_trials) // n_parallel_studies
            # Use joblib to parallelize the optimization
            def optimize_study(n_trials_per_process):
                study = optuna.create_study(storage=storage, sampler=sampler, study_name=study_name, direction=direction, load_if_exists=True)
                study.optimize(objective, n_trials=n_trials_per_process - completed_trials)
            Parallel(n_jobs=n_parallel_studies)(
                delayed(optimize_study)(trials_per_process) for _ in range(n_parallel_studies)
            )

            # Not Parallelized
            # while completed_trials < N_TRIALS:
            #     # get the number of trials already done that have been completed
            #     completed_trials = len([trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE])
            #     print(f"Completed trials: {completed_trials}/{N_TRIALS}")
            #     try:
            #         study.optimize(objective, n_trials=N_TRIALS-completed_trials)
            #     except Exception as e:
            #         print(f"Error during optimization: {e}, retrying...")
