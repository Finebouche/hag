import tensorflow as tf
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import os

from aeon.datasets import load_classification
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import GroupShuffleSplit

# load dataset using torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
from torch.utils.data import ConcatDataset


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


def process_audio(file_path):
    filename = tf.strings.split(file_path, '/')[-1]

    # Extract the label from the filename
    label = tf.strings.split(filename, '_')[0]
    speaker = tf.strings.split(filename, '_')[1]
    audio = tf.io.read_file(file_path)
    audio, sampling_rate = tf.audio.decode_wav(audio, desired_channels=1)

    return {
        'label': label,
        'audio': audio,
        'audio/filename': filename,
        'speaker': speaker,
    }, sampling_rate


def visualize_speaker_distribution(train_speakers, test_speakers):
    # Create unique lists of speakers in train and test
    unique_train_speakers = np.unique(train_speakers)
    unique_test_speakers = np.unique(test_speakers)

    # Count number of samples for each speaker in train and test
    train_counts = [np.sum(train_speakers == speaker) for speaker in unique_train_speakers]
    test_counts = [np.sum(test_speakers == speaker) for speaker in unique_test_speakers]

    # Create subplots
    fig, axs = plt.subplots(2)

    # Plot train speakers
    axs[0].bar(unique_train_speakers, train_counts)
    axs[0].set_title('Train Speakers Distribution')
    axs[0].set_xlabel('Speaker')
    axs[0].set_ylabel('Count')

    # Plot test speakers
    axs[1].bar(unique_test_speakers, test_counts)
    axs[1].set_title('Test Speakers Distribution')
    axs[1].set_xlabel('Speaker')
    axs[1].set_ylabel('Count')

    # Show the plots
    plt.tight_layout()
    plt.show()


def load_FSDD_dataset(data_dir, test_split=1 / 3, validation_split=0.25, seed=None, visualize=False):
    # Get the list of all audio files in the dataset directory
    audio_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.wav')]

    # Create a TensorFlow dataset from the audio files
    audio_files_dataset = tf.data.Dataset.from_tensor_slices(audio_files)

    # Print the number of audio files in the dataset
    print("Number of audio files:", len(audio_files))

    audio_files_dataset = audio_files_dataset.map(process_audio, num_parallel_calls=tf.data.AUTOTUNE)

    feature_dict = []
    sampling_rates = []

    for data, sampling_rate in tqdm(audio_files_dataset):
        feature_dict.append(data)
        sampling_rates.append(sampling_rate.numpy())

    # Calculate and print the mean sampling rate
    sampling_rate = np.mean(np.array(sampling_rates))
    print("Mean sampling rate:", sampling_rate)  # Should be 8000

    # List to store the features, labels and speakers
    features = []
    labels = []
    speakers = []

    # Iterate over the feature dictionary and store the features, labels and speakers
    for item in feature_dict:
        features.append(item['audio'].numpy())  # Use numpy() to convert the tensor to a numpy array
        labels.append(item['label'].numpy())  # Use numpy() to convert the tensor to a numpy array
        speakers.append(item['speaker'].numpy())  # Use numpy() to convert the tensor to a numpy array

    # Convert the features, labels and speakers to numpy arrays
    X = np.array(features, dtype=object)
    Y = np.array(labels)
    groups = np.array(speakers)

    # Encode the labels
    le = LabelEncoder()
    Y_encoded = le.fit_transform(Y)

    # One-hot encode the labels
    ohe = OneHotEncoder(sparse_output=False)
    Y_one_hot = ohe.fit_transform(Y_encoded.reshape(-1, 1))

    # Split the data into training and test sets
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_split, random_state=seed)
    train_val_idx, test_idx = next(gss_test.split(X, Y_one_hot, groups))  # Use the speaker groups for the split
    X_train, X_test = X[train_val_idx], X[test_idx]
    Y_train, Y_test = Y_one_hot[train_val_idx], Y_one_hot[test_idx]
    train_speakers, test_speakers = groups[train_val_idx], groups[test_idx]

    # Call the visualization function
    if visualize:
        visualize_speaker_distribution(train_speakers, test_speakers)

    return sampling_rate, X_train, X_test, Y_train, Y_test, train_speakers


def load_haart_dataset(train_path, test_path):
    # Charge HAART dataset from https://www.cs.ubc.ca/labs/spin/data/HAART%20DataSet.zip if it's not already done
    # download and unzip the dataset
    # https://www.cs.ubc.ca/labs/spin/data/
    import urllib.request
    import zipfile

    if not os.path.exists('datasets/HAART'):
        urllib.request.urlretrieve('https://www.cs.ubc.ca/labs/spin/data/HAART%20DataSet.zip',
                                   'datasets/HAART.zip')
        # unzip the dataset in "datasets/HAART" folder
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


def load_dataset_classification(name, seed=None):
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
        X_train, Y_train, X_test, Y_test, groups, sampling_rate = load_SPEECHCOMMANDS()
        return use_spectral_representation, is_multivariate, sampling_rate, X_train, X_test, Y_train, Y_test, groups

    if name == "FSDD":
        sampling_rate, X_train, X_test, Y_train, Y_test, groups = load_FSDD_dataset(
            data_dir='datasets/fsdd/free-spoken-digit-dataset-master/recordings', seed=seed, visualize=True)

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
        ValueError("The dataset with name {} is not loadable".format(name))
