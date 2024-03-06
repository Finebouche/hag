import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from tqdm import tqdm
from matplotlib.colors import to_rgb, to_rgba


def extract_peak_frequencies(input_data, sampling_rate, threshold, nperseg=1024, visualize=True):
    assert threshold < 1, "Threshold should be a fraction of the maximum power"
    print("Frequency limit: ", np.round(sampling_rate/2), "(Shannon sampling theorem)")
    filtered_peak_freqs = []
    max_power = 0
    max_frequency = 0
    for i in range(input_data.shape[1]):
        # Estimate power spectral density using Welch's method
        f, Pxx_den = signal.welch(input_data[:, i], sampling_rate, nperseg=nperseg)

        # Find the peaks in the power spectral density
        peak_indices, _ = signal.find_peaks(Pxx_den)
        peak_freqs = f[peak_indices]
        peak_powers = Pxx_den[peak_indices]

        # Define a power threshold and select the peaks based on that threshold
        power_threshold = threshold*np.max(Pxx_den[peak_indices])
        filtered_peak_freqs.append(peak_freqs[peak_powers > power_threshold])
        if visualize:
            lines = plt.semilogy(f, Pxx_den)
            line_color = to_rgb(lines[0].get_color())
            darkened_color = to_rgba([x * 0.7 for x in line_color])
            plt.plot(peak_freqs, Pxx_den[peak_indices], "o", color=darkened_color, markersize=4, label=i)
        # Calculate the maximum power peak and the maximum frequency
        if np.max(Pxx_den[peak_indices]) > max_power:
            max_power = np.max(Pxx_den[peak_indices])
        if np.max(peak_freqs[peak_powers > threshold * max_power * 1e-2]) > max_frequency:
            max_frequency = np.max(peak_freqs[peak_powers > (threshold * max_power * 1e-2)])

    if visualize:
        plt.ylim([threshold*max_power*1e-2, max_power*1e1])
        plt.xlim([0, max_frequency])
        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')
        # Add threshold lines
        plt.axhline(threshold*max_power, color='g', linestyle='--', label='P threshold')
        plt.axvline(sampling_rate/2, color='r', linestyle='--', label='f limit')
        plt.legend()  # Show legend with the threshold line
        plt.show()
    
        print("Filtered peak frequencies: ", filtered_peak_freqs)

    if input_data.shape[1] == 1:
        return filtered_peak_freqs[0]
    else:
        return np.array(filtered_peak_freqs)

from scipy.signal import butter, lfilter
from joblib import Parallel, delayed

def butterworth_filter(data, lowcut, highcut, fs, btype='band', order=2):
    b, a = butter(order, [lowcut/(fs/2), highcut/(fs/2)], btype=btype)
    return lfilter(b, a, data).flatten()


def generate_multivariate_dataset(filtered_peak_freqs, X_pretrain, X_train, X_test, sampling_rate, scaler,
                                  is_instances_classification, nb_jobs=1, verbosity=1):
    
    lowcut = np.concatenate(([filtered_peak_freqs[0]], (filtered_peak_freqs[:-1] + filtered_peak_freqs[1:]) / 2))
    highcut = np.concatenate(((filtered_peak_freqs[:-1] + filtered_peak_freqs[1:]) / 2, [filtered_peak_freqs[-1]]))
    
    def process_sample(x):
        return np.array(list(
            map(lambda f: butterworth_filter(x, lowcut[f], highcut[f], fs=sampling_rate), range(len(filtered_peak_freqs)))
        )).T

    # Pretrain data
    modulated_time_series = np.array(process_sample(X_pretrain.flatten())).T


    if is_instances_classification: # Multiple instances -> classification
        # Train
        X_train_band = Parallel(n_jobs=nb_jobs, verbose=verbosity)(delayed(process_sample)(x) for x in X_train)

        # Test
        X_test_band = Parallel(n_jobs=nb_jobs, verbose=verbosity)(delayed(process_sample)(x) for x in X_test)
    else:
        # Concatenate X_train and X_test for continuous processing
        concatenated_X = np.concatenate([X_train.flatten(), X_test.flatten()])
        processed_X = process_sample(concatenated_X)

        # Split the processed data back into train and test sets
        X_train_len = X_train.size  # Get the number of elements in X_train
        X_train_band = processed_X[:X_train_len]
        X_test_band = processed_X[X_train_len:]

    return modulated_time_series, X_train_band, X_test_band
