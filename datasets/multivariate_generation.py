import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def extract_peak_frequencies(input_data, sampling_rate, threshold, nperseg=1024, visualize=True):
    # Estimate power spectral density using Welch's method
    f, Pxx_den = signal.welch(input_data, sampling_rate, nperseg=nperseg)
    
    # Find the peaks in the power spectral density
    peak_indices, _ = signal.find_peaks(Pxx_den)
    peak_freqs = f[peak_indices]
    peak_powers = Pxx_den[peak_indices]
    
    # Define a power threshold and select the peaks based on that threshold
    # Calculate the maximum power peak
    max_power = np.max(Pxx_den[peak_indices])
    max_powerpeak_powers = Pxx_den[peak_indices]
    power_threshold = threshold*max_power
    filtered_peak_freqs = peak_freqs[peak_powers > power_threshold]

    if visualize: 
        # Plot the power spectral density
        plt.semilogy(f, Pxx_den)
        plt.ylim([power_threshold*1e-2, max_power*1e1])
        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')
        # Add threshold line
        plt.axhline(power_threshold, color='g', linestyle='--', label='Threshold')
        plt.legend()  # Show legend with the threshold line
        plt.plot(peak_freqs, Pxx_den[peak_indices], "ro")
        plt.show()
    
        print("Filtered peak frequencies: ", filtered_peak_freqs)

    return filtered_peak_freqs

from scipy.signal import butter, lfilter
from joblib import Parallel, delayed

def butterworth_filter(data, lowcut, highcut, fs, btype='band', order=2):
    b, a = butter(order, [lowcut/(fs/2), highcut/(fs/2)], btype=btype)
    return lfilter(b, a, data).flatten()


def generate_multivariate_dataset(filtered_peak_freqs, X_pretrain, X_train, X_test, sampling_rate, nb_jobs=1, verbosity=1):
    
    lowcut = np.concatenate(([filtered_peak_freqs[0]], (filtered_peak_freqs[:-1] + filtered_peak_freqs[1:]) / 2))
    highcut = np.concatenate(((filtered_peak_freqs[:-1] + filtered_peak_freqs[1:]) / 2, [filtered_peak_freqs[-1]]))
    
    def process_sample(x):
        return list(
            map(lambda f: butterworth_filter(x, lowcut[f], highcut[f], fs=sampling_rate), range(len(filtered_peak_freqs)))
        )
        
    # Pretrain data
    modulated_time_series = np.array(process_sample(X_pretrain.flatten()))

    if len(X_train[0].shape)>1:
        # Train
        X_train_band = np.array(Parallel(n_jobs=nb_jobs, verbose=verbosity)(delayed(process_sample)(x) for x in X_train), dtype=object)
        # Test
        X_test_band = np.array(Parallel(n_jobs=nb_jobs, verbose=verbosity)(delayed(process_sample)(x) for x in X_test), dtype=object)
    else: 
        X_train_band = np.array(process_sample(X_train.flatten()))
        X_test_band = np.array(process_sample(X_test.flatten()))

    return modulated_time_series, X_train_band, X_test_band
