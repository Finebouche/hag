from hag.models.activation_functions import tanh

activation_function = lambda x : tanh(x)

######
#
# Common visualisation definitions
#
######

from seaborn import color_palette

# -- Define color palettes for each group --
blues = color_palette("Blues", 5)      # shades of blue
oranges = color_palette("Oranges", 2)  # shades of orange
greens = color_palette("Greens", 2)    # shades of green
reds = color_palette("Reds", 2)    # shades of red
greys = color_palette("Greys", 3)    # shades of greys
purples = color_palette("Purples", 3)
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
    'LSTM':                 greys[0],
#    'RNN':                  greys[1],
    'GRU':                  greys[2],
#    'RNN-HAG':              greys[2],
#    'diag EE':             reds[0],
#    'diag EI':             reds[1],
#    'HSP':                 purples[0],
#    'short HAG':           purples[2],
}

# If you want a specific order for the bars, you can enforce it:
functions_order = [
    'E-ESN',
    'ESN',
    'IP',
    'Anti-Oja',
    'IP +\nAnti-Oja',
    'mean HAG',
    'variance HAG',
#    'RNN',
#    'RNN-HAG',
    'LSTM',
    'GRU',
#    'diag EE',
#    'diag EI',
#    'HSP',
#    'short HAG'
]

function_mapping = {
    'random_ee':        'E-ESN',
    'random_ei':        'ESN',
    'ip_correct':       'IP',
    'anti-oja_fast':    'Anti-Oja',
    'ip-anti-oja_fast': 'IP +\nAnti-Oja',
    'hadsp':            'mean HAG',
    'desp':             'variance HAG',
    'lstm_last':        'LSTM',
#    'rnn':              'RNN',
    'gru':              'GRU',
#    "hsp":              "HSP",
#    "short-hag":        "short HAG",
#    'rnn-mean_hag':     'RNN-HAG',
#    'diag_ei':        'diag EI',
#    'diag_ee':        'diag EE',
}

dataset_label_map = {"JapaneseVowels": "Japanese Vowels", "CatsDogs": "Cats vs Dogs", "FSDD": "FSDD"}


######
#
# DATALOADING
#
######
from hag.datasets.spectral_decomposition import generate_multivariate_dataset
from sklearn.preprocessing import MinMaxScaler
from hag.datasets.preprocessing import scale_data
from hag.datasets.load_data import load_data as load_dataset
from hag.datasets.peak_centered_decomposition import process_instance_func, extract_peak_frequencies
import numpy as np

def load_data(dataset_name, spectral_representation, data_type="normal", noise_std=0.001, step_ahead=5, visualize=False):
    # check if data_type is valid
    if data_type not in ["normal", "noisy"]:
        raise ValueError(f"Invalid data_type: {data_type}. Must be 'normal' or 'noisy'.")

    (is_instances_classification, is_multivariate, sampling_rate,
     X_train_raw, X_test_raw, Y_train_raw, Y_test,
     use_spectral_representation, groups) = load_dataset(dataset_name, step_ahead, visualize=False)

    if is_multivariate:
        X_train_band, X_test_band = X_train_raw, X_test_raw
    else:
        X_test, X_train = X_test_raw, X_train_raw
    X_val_band, X_val = None, None
    del X_train_raw, X_test_raw
    Y_train = Y_train_raw
    del Y_train_raw

    # PREPROCESSING
    hop = 50 if is_instances_classification else 1
    win_length = edge_cut = 100
    if is_multivariate and use_spectral_representation:
        print("Data is already spectral, nothing to do")
    else:
        # choose the right source tensors
        base_train, base_test = (X_train_band, X_test_band) if is_multivariate else (X_train, X_test)

        if spectral_representation in ["stft", "mfcc"]:
            X_train_band = generate_multivariate_dataset(
                base_train, is_instances_classification, spectral_representation, hop=hop, win_length=win_length
            )
            X_test_band = generate_multivariate_dataset(
                base_test, is_instances_classification, spectral_representation, hop=hop, win_length=win_length
            )
        elif spectral_representation == "custom":

            peaks = extract_peak_frequencies(
                input_data=base_train,
                is_instances_classification=is_instances_classification,
                sampling_rate=sampling_rate,
                threshold=1e-5,
                smooth=True,
                window_length=10,
                nperseg=1024,
                visualize=True,
            )
            X_train_band = process_instance_func(base_train, is_instances_classification, sampling_rate, peaks)
            X_test_band = process_instance_func(base_test, is_instances_classification, sampling_rate, peaks)
        elif spectral_representation == "none":
            X_train_band = base_train
            X_test_band = base_test

    # We cut the edges to remove the edges effects
    if not is_instances_classification:
        X_train_band = X_train_band[edge_cut:-edge_cut]
        X_test_band = X_test_band[edge_cut:-edge_cut]

        Y_train = Y_train[edge_cut:-edge_cut]
        Y_test = Y_test[edge_cut:-edge_cut]

    # NORMALIZATION
    scaler_multi = MinMaxScaler(feature_range=(0, 1))
    X_train_band, X_val_band, X_test_band = scale_data(X_train_band, X_val_band, X_test_band, scaler_multi,
                                                       is_instances_classification)

    # PRETRAINING SET
    # Define the number of instances you want to select
    if is_instances_classification:
        num_samples_for_pretrain = 500 if len(X_train_band) >= 500 else len(X_train_band)
        indices = np.random.choice(len(X_train_band), num_samples_for_pretrain, replace=False)
    else:
        indices = range(len(X_train_band))

    X_pretrain_band = np.array(X_train_band, dtype=object)[indices]

    return X_pretrain_band, X_train_band, X_test_band, Y_train, Y_test, is_multivariate, is_instances_classification


######
#
# Evaluation function
#
######

# Evaluating
from hag.performances.esn_model_evaluation import train_model_for_classification, predict_model_for_classification
from hag.performances.esn_model_evaluation import train_model_for_prediction, init_reservoir, init_ip_reservoir, init_local_rule_reservoir, init_ip_local_rule_reservoir, init_readout
from hag.metrics.richness import spectral_radius, pearson, squared_uncoupled_dynamics_alternative, distance_correlation


nb_jobs = 10
def evaluate_dataset_on_test(study, dataset_name, function_name, pretrain_data, train_data, test_data, Y_train, Y_test, is_instances_classification, nb_trials = 8, record_metrics=False, random_projection_experiment=False):
    # Collect all hyperparameters in a dictionary
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

    if 'variance_target' not in hyperparams and 'min_variance' in hyperparams:
        hyperparams['variance_target'] = hyperparams['min_variance']
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

        if function_name in ["diag_ee", "diag_ei"]:
            use_block = True
        else:
            use_block = False

        # UNSUPERVISED PRETRAINING
        if function_name == "random_ee":
            Win, W, bias = init_matrices(n, input_connectivity, hyperparams['connectivity'],  K, w_distribution=stats.uniform(loc=0, scale=1), use_block=use_block, seed=random.randint(0, 1000), random_projection_experiment=random_projection_experiment)
        else:
            Win, W, bias = init_matrices(n, input_connectivity, hyperparams['connectivity'],  K, w_distribution=stats.uniform(loc=-1, scale=2), use_block=use_block, seed=random.randint(0, 1000), random_projection_experiment=random_projection_experiment)
        bias *= hyperparams['bias_scaling']
        Win *= hyperparams['input_scaling']

        if function_name in ("hadsp", "mean_hag"):
            W, (_, _, _) = run_algorithm(W, Win, bias, hyperparams['leaky_rate'], activation_function, pretrain_data,
                                     hyperparams['weight_increment'], hyperparams['target_rate'], hyperparams['rate_spread'], "mean_hag",
                                     multiple_instances=is_instances_classification,
                                     min_increment = hyperparams['min_increment'], max_increment=hyperparams['max_increment'], use_full_instance=hyperparams['use_full_instance'],
                                     max_partners=np.inf, method="pearson", n_jobs=nb_jobs)
        elif function_name in ("desp", "var_hag"):
            W, (_, _, _) = run_algorithm(W, Win, bias, hyperparams['leaky_rate'], activation_function, pretrain_data,
                                         hyperparams['weight_increment'], hyperparams['variance_target'], hyperparams['variance_spread'], "var_hag",
                                         multiple_instances=is_instances_classification,
                                         min_increment = hyperparams['min_increment'], max_increment=hyperparams['max_increment'], use_full_instance = hyperparams['use_full_instance'],
                                         max_partners=np.inf, method = "pearson",
                                         intrinsic_saturation=hyperparams['intrinsic_saturation'], intrinsic_coef=hyperparams['intrinsic_coef'],
                                         n_jobs = nb_jobs)
        elif function_name == "short-hag":
            W, (_, _, _) = run_algorithm(W, Win, bias, hyperparams['leaky_rate'], activation_function, pretrain_data,
                                         hyperparams['weight_increment'], hyperparams['target_rate'], hyperparams['rate_spread'], "mean_hag",
                                         multiple_instances=is_instances_classification,
                                         min_increment = 100, max_increment=100, use_full_instance = False,
                                         max_partners=np.inf, method="pearson", n_jobs=nb_jobs)
        elif function_name == "hsp":
            W, (_, _, _) = run_algorithm(W, Win, bias, hyperparams['leaky_rate'], activation_function, pretrain_data,
                                         hyperparams['weight_increment'], hyperparams['target_rate'], hyperparams['rate_spread'],"mean_hag",
                                         multiple_instances=is_instances_classification,
                                         min_increment=1, max_increment=1, use_full_instance=False,
                                         max_partners=np.inf, method="random", n_jobs=nb_jobs)
        elif function_name in ["random_ee", "random_ei", "diag_ee", "diag_ei", "ip_correct", "anti-oja_fast", "ip-anti-oja_fast"]:
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
            reservoir = init_ip_reservoir(W, Win, bias, mu=hyperparams['mu'], sigma=hyperparams['sigma'], learning_rate=hyperparams['learning_rate'],
                                          leaking_rate=hyperparams['leaky_rate'], activation_function=activation_function
                                          )
            _ = reservoir.fit(unsupervised_pretrain, warmup=100)
        elif function_name == "anti-oja_fast":
            reservoir = init_local_rule_reservoir(W, Win, bias, local_rule="anti-oja", eta=hyperparams['oja_eta'],
                                                  synapse_normalization=False, bcm_theta=None,
                                                  leaking_rate=hyperparams['leaky_rate'], activation_function=activation_function,
                                                  )
            _ = reservoir.fit(unsupervised_pretrain, warmup=100)
        elif function_name == "ip-anti-oja_fast":
            reservoir = init_ip_local_rule_reservoir(W, Win, bias, local_rule="anti-oja", eta=hyperparams['oja_eta'],
                                                      synapse_normalization=False, bcm_theta=None,
                                                      mu=hyperparams['mu'], sigma=hyperparams['sigma'], learning_rate=hyperparams['learning_rate'],
                                                      leaking_rate=hyperparams['leaky_rate'], activation_function=activation_function,
                                                      )
            _ = reservoir.fit(unsupervised_pretrain, warmup=100)
        else:
            reservoir = init_reservoir(W, Win, bias, leaky_rate, activation_function)
        readout = init_readout(ridge_coef=RIDGE_COEF)


        # TRAINING and EVALUATION
        if record_metrics:
            inputs = np.concatenate(test_data, axis=0) if is_instances_classification else test_data
            states_history_multi = reservoir.run(inputs)

            sr = spectral_radius(W)
            pearson_correlation, _ = pearson(states_history_multi, num_windows=1, size_window=len(states_history_multi), step_size = 1, show_progress=False)
            CEV = squared_uncoupled_dynamics_alternative(states_history_multi, num_windows=1, size_window=len(states_history_multi), step_size = 1, show_progress=True)
            dcor = distance_correlation(states_history_multi, num_windows=1, size_window=len(states_history_multi), step_size = 1, show_progress=True, method="auto", nb_jobs=nb_jobs)

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
                esn = train_model_for_prediction(reservoir, readout, train_data, Y_train, warmup=start_step, n_jobs = nb_jobs)

                Y_pred =  esn.run(test_data, reset=False)
                score = compute_score(Y_pred[SLICE_RANGE], Y_test[SLICE_RANGE], is_instances_classification)

            scores.append(score)

    if record_metrics:
        return spectral_radii, pearson_correlations, CEVs, dcors

    return scores


import torch
from torch.utils.data import DataLoader

from hag.performances.esn_model_evaluation import compute_score
from hag.models.rnn import (
    LSTMModel, RNNModel, GRUModel,
    SequenceDataset, PrecomputedForecastDataset, make_sliding_windows,
    pad_collate, BucketBatchSampler,
    train as lstm_train,
    evaluate as lstm_evaluate,
)

# new imports for HAG branch
from hag.hpo.utility import retrieve_best_model
from hag.hag.hag import run_algorithm
from hag.models.reservoir import init_matrices
import math
from scipy import sparse, stats
from numpy import random
from tqdm import tqdm

# device setup
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

nb_jobs = 8


def evaluate_dataset_on_test_rnn(
        study,
        dataset_name,
        function_name,  # e.g. "lstm_last", "rnn", "rnn-mean_hag"
        pretrain_data,  # list/array of pretraining bands for HAG
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

    for seed in tqdm.tqdm(range(nb_trials), desc="Seeds", unit="seed"):
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
            hag_study = retrieve_best_model("hadsp", dataset_name, False, variate_type="multi", data_type="normal")
            hyper = {k: v for k, v in hag_study.best_trial.params.items()}
            if 'variance_target' not in hyper and 'min_variance' in hyper:
                hyper['variance_target'] = hyper.pop('min_variance')
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
                function_name, multiple_instances=is_instances_classification,
                min_increment=hyper['min_increment'], max_increment=hyper['max_increment'],
                use_full_instance=hyper['use_full_instance'], max_partners=np.inf,
                method="pearson", n_jobs=nb_jobs
            )

            # 4) Readout training
            from hag.performances.esn_model_evaluation import train_model_for_classification, train_model_for_prediction, \
                init_readout, init_reservoir
            reservoir = init_reservoir(W, Win, bias, hyper['leaky_rate'], activation_function)
            RIDGE_COEF = 10 ** hyper['ridge']
            readout = init_readout(ridge_coef=RIDGE_COEF)
            start_step = SLICE_RANGE.start if not is_instances_classification else None
            if is_instances_classification:
                train_model_for_classification(reservoir, readout, X_train, Y_train, n_jobs=1, mode="sequence-to-vector")
            else:
                _ = train_model_for_prediction(reservoir, readout, X_train, Y_train, warmup=start_step, n_jobs=nb_jobs)
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
