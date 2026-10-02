"""
Data preparation of HAG's evaluations: raw dataset (hag.datasets.load_data), spectral representation, MinMax scaling
and pretraining instances.
"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from hag.datasets.load_categorical_forecasting import canary_mfcc_config
from hag.datasets.load_data import load_data as load_dataset
from hag.datasets.peak_centered_decomposition import extract_peak_frequencies, process_instance_func
from hag.datasets.preprocessing import scale_data
from hag.datasets.spectral_decomposition import generate_multivariate_dataset


def prepare_data(dataset_name, spectral_representation, data_type="normal", noise_std=0.001, step_ahead=5,
                 visualize=False):
    """
    (pretrain, train, test, Y_train, Y_test, is_multivariate, is_instances_classification) of a dataset:
    spectral representation ("stft", "mfcc", "custom" or "none") of univariate data, edges cut for prediction tasks,
    MinMax scaling (0, 1) fitted on the train set, and pretraining instances (up to 500 train sequences drawn with
    numpy's global RNG for classification, the whole train series for prediction).
    """
    if data_type not in ["normal", "noisy"]:
        raise ValueError(f"Invalid data_type: {data_type}. Must be 'normal' or 'noisy'.")

    (is_instances_classification, is_multivariate, sampling_rate,
     X_train_raw, X_test_raw, Y_train_raw, Y_test,
     use_spectral_representation, groups) = load_dataset(dataset_name, step_ahead, visualize=False)

    if is_multivariate:
        X_train_band, X_test_band = X_train_raw, X_test_raw
    else:
        X_test, X_train = X_test_raw, X_train_raw
    X_val_band = None
    Y_train = Y_train_raw

    # PREPROCESSING
    hop = 50 if is_instances_classification else 1
    win_length = edge_cut = 100
    # dataset-specific MFCC, as in hpo_esn.py (None -> default hop / win_length)
    mfcc_config = canary_mfcc_config(sampling_rate) if dataset_name == "Canary" else None
    if is_multivariate and use_spectral_representation:
        print("Data is already spectral, nothing to do")
    else:
        base_train, base_test = (X_train_band, X_test_band) if is_multivariate else (X_train, X_test)

        if spectral_representation in ["stft", "mfcc"]:
            X_train_band = generate_multivariate_dataset(
                base_train, is_instances_classification, spectral_representation, hop=hop, win_length=win_length,
                mfcc_config=mfcc_config
            )
            X_test_band = generate_multivariate_dataset(
                base_test, is_instances_classification, spectral_representation, hop=hop, win_length=win_length,
                mfcc_config=mfcc_config
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
        else:
            raise ValueError(f"Invalid spectral_representation: {spectral_representation}")

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
    if is_instances_classification:
        num_samples_for_pretrain = 500 if len(X_train_band) >= 500 else len(X_train_band)
        indices = np.random.choice(len(X_train_band), num_samples_for_pretrain, replace=False)
    else:
        indices = range(len(X_train_band))

    X_pretrain_band = np.array(X_train_band, dtype=object)[indices]

    return X_pretrain_band, X_train_band, X_test_band, Y_train, Y_test, is_multivariate, is_instances_classification
