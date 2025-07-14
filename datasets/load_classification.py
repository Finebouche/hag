import os
import glob
import zipfile

import urllib.request
import torchaudio

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from aeon.datasets import load_classification
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import GroupShuffleSplit

# load dataset using torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
from torch.utils.data import ConcatDataset


def process_audio(file_path: str):
    """
    Load a WAV via torchaudio, return a dict (label,speaker,audio,filename) plus its sampling rate.
    """
    filename = os.path.basename(file_path)
    parts   = filename.split('_')
    label, speaker = parts[0], parts[1]

    # torchaudio.load returns (waveform, sample_rate), waveform shape (channels, time)
    waveform, sampling_rate = torchaudio.load(file_path)
    # convert to mono if needed, shape (1, T)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    # transpose to (T, 1)
    audio_np = waveform.T.numpy()

    return {
        'label':        label,
        'speaker':      speaker,
        'audio':        audio_np,
        'audio/filename': filename,
    }, sampling_rate


def load_SPEECHCOMMANDS():
    dataset_train = SPEECHCOMMANDS(root="datasets/", download=True, subset="training")
    dataset_val = SPEECHCOMMANDS(root="datasets/", download=True, subset="validation")
    dataset_test = SPEECHCOMMANDS(root="datasets/", download=True, subset="testing")

    sampling_rate = dataset_train[0][1]

    dataset = ConcatDataset([dataset_train, dataset_val])

    X_train_raw = [sample[0][0].numpy().reshape(-1, 1) for sample in dataset]
    Y_train_raw = [sample[2] for sample in dataset]
    X_test_raw = [sample[0][0].numpy().reshape(-1, 1) for sample in dataset_test]
    Y_test = [sample[2] for sample in dataset_test]

    le = LabelEncoder()
    Y_train_raw = le.fit_transform(Y_train_raw).reshape(-1, 1)
    Y_test = le.transform(Y_test).reshape(-1, 1)

    # One-hot encode the labels
    ohe = OneHotEncoder(sparse_output=False)
    Y_train_raw = ohe.fit_transform(Y_train_raw.reshape(-1, 1))
    Y_test = ohe.transform(Y_test.reshape(-1, 1))

    groups = None

    return X_train_raw, Y_train_raw, X_test_raw, Y_test, sampling_rate, groups


def visualize_groups_distribution(groups):
    # Create unique lists of speakers
    unique_speakers = np.unique(groups)

    # Count number of samples for each speaker
    count = [np.sum(groups == speaker) for speaker in unique_speakers]

    fig, ax = plt.subplots(1, figsize=(6, 2))
    ax.bar(unique_speakers, count)
    ax.set_xlabel('Group')
    ax.set_ylabel('Count')
    plt.tight_layout()
    plt.show()



def load_FSDD_dataset(data_dir, test_split=1/3, validation_split=0.25, seed=None, visualize=False):
    # gather all .wav files
    audio_files = glob.glob(os.path.join(data_dir, '*.wav'))
    print("Number of audio files:", len(audio_files))

    features, labels, speakers = [], [], []
    sampling_rates = []

    # process all files
    for fp in audio_files:
        data, sr = process_audio(fp)
        features.append(data['audio'])
        labels.append(data['label'])
        speakers.append(data['speaker'])
        sampling_rates.append(sr)

    sampling_rate = int(np.mean(sampling_rates))
    print("Mean sampling rate:", sampling_rate)

    # numpy arrays
    X = np.array(features, dtype=object)
    Y = np.array(labels)
    groups = np.array(speakers)

    # label encode + one-hot
    le  = LabelEncoder()
    ohe = OneHotEncoder(sparse_output=False)
    y_enc = le.fit_transform(Y).reshape(-1, 1)
    Y_one = ohe.fit_transform(y_enc)

    # split into train/test by speaker groups
    gss = GroupShuffleSplit(n_splits=1, test_size=test_split, random_state=seed)
    train_idx, test_idx = next(gss.split(X, Y_one, groups))
    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y_one[train_idx], Y_one[test_idx]
    train_speakers, test_speakers = groups[train_idx], groups[test_idx]

    if visualize:
        visualize_groups_distribution(train_speakers)
        visualize_groups_distribution(test_speakers)

    return sampling_rate, X_train, X_test, Y_train, Y_test, train_speakers

def load_haart_dataset(train_path, test_path):
    # Charge HAART dataset from https://www.cs.ubc.ca/labs/spin/data/HAART%20DataSet.zip if it's not already done
    # download and unzip the dataset
    # https://www.cs.ubc.ca/labs/spin/data/
    os.makedirs('datasets/HAART', exist_ok=True)

    if not os.path.exists('datasets/HAART'):
        urllib.request.urlretrieve('https://www.cs.ubc.ca/labs/spin/data/HAART%20DataSet.zip',
                                   'datasets/HAART.zip')
        with zipfile.ZipFile('datasets/HAART.zip', 'r') as zip_ref:
            zip_ref.extractall('datasets/HAART')

        # delete zip
        os.remove('datasets/HAART.zip')

    df_train = pd.read_csv(train_path, header=0)
    df_test = pd.read_csv(test_path, header=0)

    # extract times series from the dataframe column 4 to the end (the first 4 columns are not time series)
    grouped = df_train.groupby(['ParticipantNo', ' "Substrate"', ' "Cover"', ' "Gesture"'])

    # This will create a dictionary where keys are the unique groupings and values are the multivariate time series data for each group
    X_train = []
    for name, group in grouped:
        X_train.append(group.iloc[:, 4:68].values.astype(np.float64))

    # extract labels use to group by for each time series
    Y_train = []
    for name, group in grouped:
        Y_train.append(name[-1])

    # do same for test set
    grouped = df_test.groupby(['ParticipantID', 'Substrate', 'Cover', 'Gesture'])

    X_test = []
    for name, group in grouped:
        X_test.append(group.iloc[:, 4:68].values.astype(np.float64))

    Y_test = []
    for name, group in grouped:
        Y_test.append(name[-1])

    # encode labels
    le = LabelEncoder()
    Y_train_encoded = le.fit_transform(Y_train)
    Y_test_encoded = le.transform(Y_test)

    # One-hot encode the labels
    ohe = OneHotEncoder(sparse_output=False)
    Y_train = ohe.fit_transform(Y_train_encoded.reshape(-1, 1))
    Y_test = ohe.transform(Y_test_encoded.reshape(-1, 1))

    # sampling rate is 54Hz
    sampling_rate = 54

    return sampling_rate, X_train, Y_train, X_test, Y_test


def load_aoen_dataset(dataset_name, seed=None):
    X_train_unprocessed, Y_train_raw, meta_data = load_classification(dataset_name, return_metadata=True,
                                                                      load_equal_length=False, split="train")
    X_test_unprocessed, Y_test_raw, meta_data = load_classification(dataset_name, return_metadata=True,
                                                                    load_equal_length=False, split="test")
    groups = None

    X_train_raw = []
    for x in X_train_unprocessed:
        X_train_raw.append(x.T)

    X_test_raw = []
    for x in X_test_unprocessed:
        X_test_raw.append(x.T)

    le = LabelEncoder()
    Y_train_raw = le.fit_transform(Y_train_raw).reshape(-1, 1)
    Y_test = le.transform(Y_test_raw).reshape(-1, 1)

    # One-hot encode the labels
    ohe = OneHotEncoder(sparse_output=False)
    Y_train_raw = ohe.fit_transform(Y_train_raw.reshape(-1, 1))
    Y_test = ohe.transform(Y_test.reshape(-1, 1))

    return X_train_raw, Y_train_raw, X_test_raw, Y_test, groups, meta_data


def load_dataset_classification(name, visualize=True, seed=None):
    if name == "SpokenArabicDigits" or name == "CatsDogs" or name == "LSST":
        X_train, Y_train, X_test, Y_test, groups, meta_data = load_aoen_dataset(name, seed)
        sampling_rate = 10000

        if X_train[0].shape[1] == 1:
            is_multivariate = False
            use_spectral_representation = False
        else:
            is_multivariate = True
            use_spectral_representation = True

        print(" Number of instances = ", len(X_train))
        print(" Shape of X = ", X_train[0].shape)
        print(" Shape of y = ", Y_train.shape)

        print(" Meta data = ", meta_data)
        print("Multivariate = ", is_multivariate)

        return use_spectral_representation, is_multivariate, sampling_rate, X_train, X_test, Y_train, Y_test, groups

    if name == "SPEECHCOMMANDS":
        is_multivariate = False
        use_spectral_representation = False
        X_train, Y_train, X_test, Y_test, sampling_rate, groups = load_SPEECHCOMMANDS()
        return use_spectral_representation, is_multivariate, sampling_rate, X_train, X_test, Y_train, Y_test, groups

    if name == "FSDD":
        sampling_rate, X_train, X_test, Y_train, Y_test, groups = load_FSDD_dataset(
            data_dir='datasets/fsdd/free-spoken-digit-dataset-master/recordings', seed=seed, visualize=visualize)

        is_multivariate = False
        use_spectral_representation = False
        return use_spectral_representation, is_multivariate, sampling_rate, X_train, X_test, Y_train, Y_test, groups

    if name == "HAART":
        sampling_rate, X_train_band, Y_train, X_test_band, Y_test = load_haart_dataset(
            train_path="datasets/HAART/training.csv", test_path="datasets/HAART/testWITHLABELS.csv")
        is_multivariate = True
        groups = None
        use_spectral_representation = False
        return use_spectral_representation, is_multivariate, sampling_rate, X_train_band, X_test_band, Y_train, Y_test, groups

    if name == "JapaneseVowels":
        from reservoirpy.datasets import japanese_vowels
        X_train_band, Y_train, X_test_band, Y_test = japanese_vowels()
        is_multivariate = True
        groups = None
        # Sampling rate : 10 kHz
        # Source : https://archive.ics.uci.edu/dataset/128/japanese+vowels
        sampling_rate = 10000
        # pretrain is the same as train
        Y_train = np.squeeze(np.array(Y_train), axis=1)
        Y_test = np.squeeze(np.array(Y_test), axis=1)
        use_spectral_representation = True
        return use_spectral_representation, is_multivariate, sampling_rate, X_train_band, X_test_band, Y_train, Y_test, groups

    else:
        raise ValueError(f"The dataset with name '{name}' is not loadable")
