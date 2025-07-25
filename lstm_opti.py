import numpy as np
import torch

SEED = 923984
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")

from datasets.load_data import load_data
from reservoir.activation_functions import tanh

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

    print("Starting LSTM optimization...")

    step_ahead=5
    # can be "JapaneseVowels", "CatsDogs", "FSDD", "SpokenArabicDigits", "SPEECHCOMMANDS", "MackeyGlass", "Sunspot_daily", "Lorenz", "Henon", "NARMA"
    datasets = ["MackeyGlass"]
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

        # Evaluating
        from reservoir.lstm import (LSTMModel, SequenceDataset, train, evaluate, pad_collate, BucketBatchSampler,
                                    PrecomputedForecastDataset, make_sliding_windows)
        import torch.nn as nn
        from torch.utils.data import DataLoader

        RESERVOIR_SIZE = 500
        avg_length = np.mean([len(x) for fold in X_train_band for x in fold])
        # min an max lengths
        min_length = min([len(x) for fold in X_train_band for x in fold])
        max_length = max([len(x) for fold in X_train_band for x in fold])
        print(f"Average length: {avg_length}, Min length: {min_length}, Max length: {max_length}")

        nb_jobs_per_trial = 10
        variate_type = "multi"  # "multi" ou "uni"
        if variate_type == "uni" and is_multivariate:
            raise ValueError(f"Invalid variable type: {variate_type}")

        # "random_ee", "random_ei", "diag_ee", "diag_ei", "desp", "hadsp", "ip_correct", "anti-oja_fast", "ip-anti-oja_fast", "lstm"
        for function_name in ["lstm_last"]:

            def objective(trial):
                # 1) HYPERPARAMETERS TO OPTIMIZE
                hidden_size = trial.suggest_int('hidden_size', 320, 512, step=32)
                num_layers = trial.suggest_int('num_layers', 1, 1)
                dropout = trial.suggest_float('dropout', 0.0, 0.5)
                bidirectional = trial.suggest_categorical('bidirectional', [False, True])
                learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
                batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512])  # [8, 16, 32, 64]
                epochs = trial.suggest_int('epochs', 5, 20)

                task_type = 'classification' if is_instances_classification else 'regression'
                criterion = torch.nn.CrossEntropyLoss() if task_type == 'classification'else torch.nn.MSELoss()

                # 2) CROSS‐VALIDATION LOOP
                total_metric = 0.0
                for fold_idx in range(nb_splits):
                    # prepare PyTorch datasets/loaders
                    X_tr = X_train_band[fold_idx]  # shape: (n_samples, seq_len, feat_dim)
                    y_tr = Y_train[fold_idx]  # one-hot or reg targets
                    X_va = X_val_band[fold_idx]
                    y_va = Y_val[fold_idx]

                    if task_type == 'classification':
                        # variable‐length audio → use SequenceDataset + padding
                        train_ds = SequenceDataset(X_tr, y_tr)
                        val_ds = SequenceDataset(X_va, y_va)

                        # bucket/pad as before
                        train_lengths = [len(x) for x in X_tr]
                        if len(set(train_lengths)) > 1:
                            sampler = BucketBatchSampler(train_lengths, batch_size=batch_size, bucket_size=batch_size * 20, shuffle=True)
                            train_loader = DataLoader(train_ds, batch_sampler=sampler, collate_fn=pad_collate)
                        else:
                            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)

                        val_lengths = [len(x) for x in X_va]
                        if len(set(val_lengths)) > 1:
                            sampler = BucketBatchSampler(val_lengths, batch_size=batch_size, bucket_size=batch_size * 20, shuffle=False)
                            val_loader = DataLoader(val_ds, batch_sampler=sampler, collate_fn=pad_collate)
                        else:
                            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=pad_collate)

                    else:
                        WINDOW = 100
                        X_tr_win, y_tr_tgt = make_sliding_windows(X_tr, y=y_tr, window=WINDOW)
                        X_va_win, y_va_tgt = make_sliding_windows(X_va, y=y_va, window=WINDOW)

                        train_ds = PrecomputedForecastDataset(X_tr_win, y_tr_tgt)
                        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

                        val_ds = PrecomputedForecastDataset(X_va_win, y_va_tgt)
                        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

                    # instantiate model + optimizer + loss
                    # get one sample
                    sample_x, sample_y = train_ds[0]
                    # sample_x has shape (seq_len, D_in)
                    input_size = sample_x.shape[-1]
                    if task_type == 'classification':
                        # sample_y is one-hot vector
                        output_size = sample_y.shape[-1]
                    else:
                        # sample_y could be (D_out,) or scalar
                        output_size = sample_y.shape[-1] if sample_y.ndim > 0 else 1

                    model = LSTMModel(input_size=input_size,
                                    hidden_size=hidden_size,
                                    num_layers=num_layers,
                                    output_size=output_size,
                                    dropout=dropout,
                                    bidirectional=bidirectional,
                    ).to(DEVICE)
                    model = torch.compile(model)

                    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

                    # train for a few epochs
                    for epoch in range(epochs):
                        _ = train(model, train_loader, criterion, optimizer, task_type=task_type)

                    # evaluate
                    fold_metric = evaluate(model, val_loader, task_type=task_type)
                    total_metric += fold_metric

                # 3) RETURN MEAN METRIC (to maximize accuracy or minimize MSE)
                average_metric = total_metric / nb_splits
                return average_metric


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
            uri = (
                f"{url}"
                "?cache=shared"  # allow multiple connections to share page cache
                "&journal_mode=WAL"  # enable write-ahead log
            )
            storage = optuna.storages.RDBStorage(
                url=uri,
                heartbeat_interval=1.0,   # optional
            )
            print(url)
            study_name = function_name + "_" + dataset_name + "_" + data_type + "_" + variate_type
            print(study_name)
            direction = "maximize" if is_instances_classification else "minimize"

            N_TRIALS = 400
            study = optuna.create_study(storage=storage, sampler=sampler, study_name=study_name, direction=direction, load_if_exists=True)
            completed_trials = len([trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE])

            # Parallelized
            n_parallel_studies = 10
            # trials_per_process = (N_TRIALS - completed_trials) // n_parallel_studies
            # # Use joblib to parallelize the optimization
            # def optimize_study(n_trials_per_process):
            #     study = optuna.create_study(storage=storage, sampler=sampler, study_name=study_name, direction=direction, load_if_exists=True)
            #     study.optimize(objective, n_trials=n_trials_per_process - completed_trials)
            # Parallel(n_jobs=n_parallel_studies)(
            #     delayed(optimize_study)(trials_per_process) for _ in range(n_parallel_studies)
            # )

            # Not Parallelized
            while completed_trials < N_TRIALS:
                # get the number of trials already done that have been completed
                completed_trials = len([trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE])
                print(f"Completed trials: {completed_trials}/{N_TRIALS}")
                try:
                    study.optimize(objective, n_trials=N_TRIALS-completed_trials)
                except Exception as e:
                    print(f"Error during optimization: {e}, retrying...")
