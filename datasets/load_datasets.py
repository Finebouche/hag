import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import GroupShuffleSplit

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
    

def load_FSDD_dataset(data_dir, split=0.5, seed=49387, visualize=False):
        
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
    print("Mean sampling rate:", sampling_rate) # Should be 8000
    
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
    gss = GroupShuffleSplit(n_splits=1, test_size=split, random_state=seed)
    train_idx, test_idx = next(gss.split(X, Y_one_hot, groups))  # Use the speaker groups for the split
    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y_one_hot[train_idx], Y_one_hot[test_idx]
    train_speakers, test_speakers = groups[train_idx], groups[test_idx]

    # Call the visualization function
    if visualize:
        visualize_speaker_distribution(train_speakers, test_speakers)


    return sampling_rate, X_train, X_test, Y_train, Y_test 

    