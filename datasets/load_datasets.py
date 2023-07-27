import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

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


def load_FSDD_dataset(data_dir, split=0.5, seed=49387):
        
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
    
    # List to store the features and labels
    features = []
    labels = []
    
    # Iterate over the feature dictionary and store the features and labels
    for item in feature_dict:
        features.append(item['audio'].numpy())  # Use numpy() to convert the tensor to a numpy array
        labels.append(item['label'].numpy())  # Use numpy() to convert the tensor to a numpy array
    
    # Convert the features and labels to numpy arrays
    X = np.array(features, dtype=object)
    Y = np.array(labels)
    
    # Encode the labels
    le = LabelEncoder()
    Y_encoded = le.fit_transform(Y)
    
    # One-hot encode the labels
    ohe = OneHotEncoder(sparse_output=False)
    Y_one_hot = ohe.fit_transform(Y_encoded.reshape(-1, 1))
    
    # Split the data into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_one_hot, test_size=split, random_state=seed)  # Use 20% of the data for the test set
    return sampling_rate, X_train, X_test, Y_train, Y_test 
