import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.colors import to_rgb, to_rgba
from scipy.signal import butter, lfilter, cheby2
from joblib import Parallel, delayed


def extract_peak_frequencies(input_data, sampling_rate, threshold, nperseg=1024, visualize=True):
    assert threshold < 1, "Threshold should be a fraction of the maximum power"
    print("Frequency limit: ", np.round(sampling_rate / 2), "(Shannon sampling theorem)")
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
        relative_threshold = threshold * np.max(Pxx_den[peak_indices])
        filtered_peak_freqs.append(peak_freqs[peak_powers > relative_threshold])
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
        plt.ylim([threshold * max_power * 1e-2, max_power * 1e1])
        plt.xlim([0, max_frequency])
        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')
        # Add threshold lines
        plt.axhline(threshold * max_power, color='g', linestyle='--', label='P threshold')
        plt.axvline(sampling_rate / 2, color='r', linestyle='--', label='f limit')
        plt.legend()  # Show legend with the threshold line
        plt.show()

    if input_data.shape[1] == 1:
        return filtered_peak_freqs[0]
    else:
        return np.array(filtered_peak_freqs)


def filter(data, lowcut, highcut, fs, btype='band', order=4):
    b, a = butter(order, [lowcut / (fs / 2), highcut / (fs / 2)], btype=btype)
    # b, a = cheby2(order, 20,  [lowcut/(fs/2), highcut/(fs/2)], btype=btype)
    return lfilter(b, a, data).flatten()


def generate_multivariate_dataset(filtered_peak_freqs, X, sampling_rate, is_instances_classification, nb_jobs=1,
                                  verbosity=1):
    lowcut = np.concatenate(([filtered_peak_freqs[0]], (filtered_peak_freqs[:-1] + filtered_peak_freqs[1:]) / 2))
    highcut = np.concatenate(((filtered_peak_freqs[:-1] + filtered_peak_freqs[1:]) / 2, [filtered_peak_freqs[-1]]))

    def process_sample(x):
        return np.array(list(
            map(lambda f: filter(x, lowcut[f], highcut[f], fs=sampling_rate), range(len(filtered_peak_freqs)))
        )).T

    if is_instances_classification:  # Multiple instances -> classification
        X_band = Parallel(n_jobs=nb_jobs, verbose=verbosity)(delayed(process_sample)(x) for x in X)
        print("hello")
    else:
        # Concatenate X_train and X_test for continuous processing
        X_band = process_sample(X)

    return X_band


if __name__ == "__main__":
    # Generate a synthetic dataset
    fs = 1000  # Sampling frequency
    t = np.linspace(0, 1, fs)  # Time axis
    # Generate signals with two frequencies
    f1, f2 = 2, 15  # Frequencies to include in the signal
    X = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)
    print("Original shape: ", X.reshape(-1, 1).shape)

    # Define peak frequencies around which to filter
    filtered_peak_freqs = extract_peak_frequencies(X, fs, threshold=1e-5, nperseg=1024, visualize=False)
    print(filtered_peak_freqs)

    # Generate multivariate dataset
    X_band = generate_multivariate_dataset(filtered_peak_freqs, X.reshape(-1, 1), fs, is_instances_classification=False,
                                           nb_jobs=1,
                                           verbosity=0)

    print(X_band.shape)
    # Plot the original and filtered signals on the same graph
    # Plot the original signal
    plt.plot(t, X, label='Original signal', color='blue')

    # Plot the filtered signals
    # Note: If X_band is 2D, this assumes that each column is a filtered version of the original signal.
    for i, filtered_signal in enumerate(X_band.T):
        plt.plot(t, filtered_signal, label=f'Filtered signal {i + 1}', linestyle='--')

    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('Original and Filtered Signals')
    plt.legend()
    plt.show()
    print(X_band)
