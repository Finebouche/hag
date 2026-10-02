"""
Test evaluation of the recurrent networks of HAG's study (LSTM, GRU, RNN, RNN initialized by HAG) trained with
PyTorch, with the best hyperparameters of a study. Separate from test_evaluation so that only it imports torch.
"""
import math

import numpy as np
import torch
from numpy import random
from scipy import stats
from torch.utils.data import DataLoader
from tqdm import tqdm

from hag.hag.hag import run_algorithm
from hag.hpo.utility import retrieve_best_model
from hag.models.reservoir import init_matrices
from hag.models.rnn import (
    LSTMModel, RNNModel, GRUModel,
    SequenceDataset, PrecomputedForecastDataset, make_sliding_windows,
    pad_collate, BucketBatchSampler,
    train as lstm_train,
    evaluate as lstm_evaluate,
)
from hag.performances.esn_model_evaluation import (init_readout, init_reservoir, train_model_for_classification,
                                                   train_model_for_prediction)

# device setup
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

rnn_nb_jobs = 8


def evaluate_dataset_on_test_rnn(
        study,
        dataset_name,
        function_name,  # e.g. "lstm_last", "rnn", "rnn-mean_hag"
        pretrain_data,
        X_train,  # list of train sequences or array
        X_test,  # list of test sequences or array
        Y_train,  # labels or targets for train
        Y_test,  # labels or targets for test
        is_instances_classification,
        nb_trials=8,
        record_metrics=False
):
    # 1) best hyperparameters for LSTM/RNN
    hp = study.best_trial.params.copy()
    batch_size = hp.pop("batch_size")
    epochs = hp.pop("epochs")
    lr = hp.pop("learning_rate")
    nlayers = hp.pop("num_layers")
    dropout = hp.pop("dropout")

    if function_name in ["lstm", "rnn", "lstm_last", "gru"]:
        hidden = hp.pop("hidden_size")
        bidir = hp.pop("bidirectional")

    task_type = "classification" if is_instances_classification else "regression"
    criterion = torch.nn.CrossEntropyLoss() if task_type == "classification" else torch.nn.MSELoss()

    # for regression forecast slicing
    if not is_instances_classification:
        SLICE_RANGE = slice(500, 1500) if dataset_name != "Sunspot" else slice(30, 500)

    all_scores = []

    for seed in tqdm(range(nb_trials), desc="Seeds", unit="seed"):
        torch.manual_seed(seed)

        # — build PyTorch Dataset & DataLoader —
        if is_instances_classification:
            train_ds = SequenceDataset(X_train, Y_train)
            train_lens = [len(x) for x in X_train]
            if len(set(train_lens)) > 1:
                sampler = BucketBatchSampler(train_lens, batch_size=batch_size, bucket_size=batch_size * 20, shuffle=True)
                train_loader = DataLoader(train_ds, batch_sampler=sampler, collate_fn=pad_collate)
            else:
                train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)

            test_ds = SequenceDataset(X_test, Y_test)
            test_lens = [len(x) for x in X_test]
            if len(set(test_lens)) > 1:
                sampler = BucketBatchSampler(test_lens, batch_size=batch_size, bucket_size=batch_size * 20, shuffle=False)
                test_loader = DataLoader(test_ds, batch_sampler=sampler, collate_fn=pad_collate)
            else:
                test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=pad_collate)

        else:
            WINDOW = 100
            X_tr_win, y_tr_tgt = make_sliding_windows(X_train, y=Y_train, window=WINDOW)
            X_test_win, y_test_tgt = make_sliding_windows(X_test, y=Y_test, window=WINDOW)

            train_ds = PrecomputedForecastDataset(X_tr_win, y_tr_tgt)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

            test_ds = PrecomputedForecastDataset(X_test_win, y_test_tgt)
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        # infer dims
        sample_x, sample_y = train_ds[0]
        D_in = sample_x.shape[-1]
        if task_type == "classification":
            D_out = sample_y.shape[-1]
        else:
            D_out = sample_y.shape[-1] if sample_y.ndim > 0 else 1

        # instantiate models
        if function_name == "lstm_last":
            model = LSTMModel(
                input_size=D_in,
                hidden_size=hidden,
                num_layers=nlayers,
                output_size=D_out,
                dropout=dropout,
                bidirectional=bidir
            ).to(DEVICE)

        elif function_name == "gru":
            model = GRUModel(
                input_size=D_in,
                hidden_size=hidden,
                num_layers=nlayers,
                output_size=D_out,
                dropout=dropout,
                bidirectional=bidir
            ).to(DEVICE)

        elif function_name == "rnn":
            model = RNNModel(
                input_size=D_in,
                hidden_size=hidden,
                num_layers=nlayers,
                output_size=D_out,
                dropout=dropout,
                bidirectional=bidir
            ).to(DEVICE)

        elif function_name == "rnn-mean_hag":
            # HAG-based reservoir initialization
            # 1) Retrieve best HAG hyperparameters
            hag_study = retrieve_best_model("mean_hag", dataset_name, False, variate_type="multi", data_type="normal")
            hyper = {k: v for k, v in hag_study.best_trial.params.items()}
            hyper['use_full_instance'] = not is_instances_classification

            # 2) Build reservoir matrices
            input_connectivity = 1
            common_size = X_train[0].shape[1] if is_instances_classification else X_train.shape[1]
            K = math.ceil(hyper['network_size'] / common_size)
            n = common_size * K
            Win, W, bias = init_matrices(n, input_connectivity, hyper['connectivity'], K,
                                         w_distribution=stats.uniform(loc=-1, scale=2), seed=random.randint(0, 1000))
            bias *= hyper['bias_scaling']
            Win *= hyper['input_scaling']

            # 3) Adapt weights via HAG
            activation_function = np.tanh
            fold_idx = seed
            X_pre = pretrain_data
            W, _ = run_algorithm(
                W, Win, bias,
                hyper['leaky_rate'], activation_function,
                X_pre, hyper['weight_increment'],
                hyper['target_rate'], hyper['rate_spread'],
                "mean_hag", multiple_instances=is_instances_classification,
                min_increment=hyper['min_increment'], max_increment=hyper['max_increment'],
                use_full_instance=hyper['use_full_instance'], max_partners=np.inf,
                method="pearson", n_jobs=1
            )

            # 4) Readout training
            reservoir = init_reservoir(W, Win, bias, hyper['leaky_rate'], activation_function)
            RIDGE_COEF = 10 ** hyper['ridge']
            readout = init_readout(ridge_coef=RIDGE_COEF)
            start_step = SLICE_RANGE.start if not is_instances_classification else None
            if is_instances_classification:
                train_model_for_classification(reservoir, readout, X_train, Y_train, mode="sequence-to-vector")
            else:
                _ = train_model_for_prediction(reservoir, readout, X_train, Y_train, warmup=start_step, n_jobs=rnn_nb_jobs)
            Wout = readout.Wout
            bias_out = readout.bias.reshape(-1)

            # 5) Instantiate PyTorch RNNModel and overwrite weights
            model = RNNModel(
                input_size=D_in,
                hidden_size=n,
                num_layers=1,
                output_size=D_out,
                dropout=dropout,
                bidirectional=False
            ).to(DEVICE)
            with torch.no_grad():
                model.rnn.weight_ih_l0.copy_(torch.tensor(Win, dtype=torch.float32, device=DEVICE))
                model.rnn.weight_hh_l0.copy_(torch.tensor(W, dtype=torch.float32, device=DEVICE))
                model.rnn.bias_ih_l0.zero_()
                model.rnn.bias_hh_l0.copy_(torch.tensor(bias, dtype=torch.float32, device=DEVICE))
                model.fc.weight.copy_(torch.tensor(Wout.T, dtype=torch.float32, device=DEVICE))
                model.fc.bias.copy_(torch.tensor(bias_out, dtype=torch.float32, device=DEVICE))
        else:
            raise ValueError(f"Unknown function_name: {function_name}")

        # Train
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        model = torch.compile(model)
        # train for a few epochs
        for _ in range(epochs):
            _ = lstm_train(model, train_loader, criterion, optimizer, task_type=task_type)

        # evaluate
        metric = lstm_evaluate(model, test_loader, task_type=task_type)
        all_scores.append(metric)

    if record_metrics:
        raise NotImplementedError("Hidden-state metrics for LSTM not yet supported.")
    return all_scores
