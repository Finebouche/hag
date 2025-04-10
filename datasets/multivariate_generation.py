import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.colors import to_rgb, to_rgba
from joblib import Parallel, delayed
from librosa import stft
from librosa.feature import mfcc
from scipy.signal.windows import gaussian

def extract_peak_frequencies(input_data, sampling_rate, threshold, smooth=True, window_length=10, nperseg=1024, visualize=True):
    assert threshold < 1, "Threshold should be a fraction of the maximum power"
    if visualize:
        print("Frequency limit: ", np.round(sampling_rate / 2), "(Shannon sampling theorem)")
    filtered_peak_freqs = []
    max_power = 0
    max_frequency = 0
    for i in range(input_data.shape[1]):
        # Estimate power spectral density using Welch's method
        # https://dsp.stackexchange.com/questions/81640/trying-to-understand-the-nperseg-effect-of-welch-method
        f, Pxx_den = signal.welch(input_data[:, i], sampling_rate, nperseg=nperseg)

        # Smoothing the Power Spectral Density (Pxx_den) before peak detection helps in emphasizing
        # the more significant, broader peaks that are often of greater interest in signal processing tasks.
        if smooth:
            # Create a Gaussian window
            gaus_window = gaussian(window_length, std=7)
            gaus_window /= np.sum(gaus_window)
            # Apply the Gaussian filter
            Pxx_den = signal.convolve(Pxx_den, gaus_window, mode='same')

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
        return np.array(filtered_peak_freqs, dtype=object)


def generate_multivariate_dataset(X, is_instances_classification, spectral_representation=None, hop=50, win_length=100,
                                  nb_jobs=-1, verbosity=1):
    print("Using window length (nperseg):", win_length, "and hop:", hop)

    def compute_instance_spectrogram(x):
        g_std = 8  # standard deviation for Gaussian window in samples
        w = gaussian(win_length, std=g_std, sym=True)  # symmetric Gaussian window
        # w = hann(win_length, sym=True) # symmetric Hann window
        if spectral_representation == "stft":
            Sx = np.abs(stft(x, hop_length=hop, win_length=win_length, n_fft=win_length, window=w))
        elif spectral_representation == "mfcc":
            Sx = np.abs(mfcc(y = x, hop_length=hop, win_length=win_length, n_fft=win_length, window=w))

        if is_instances_classification:
            return np.hstack(Sx).T
        else:
            return Sx.T

    if is_instances_classification:  # classification -> Multiple instances
        X_band = Parallel(n_jobs=nb_jobs, verbose=verbosity)(delayed(compute_instance_spectrogram)(x.T) for x in X)
    else :
        for i in range(X.shape[1]):
            x_band = compute_instance_spectrogram(X[:,0])
            if i == 0:
                X_band = x_band
            else:
                X_band = np.hstack((X_band, x_band))

        # if dimension doesn't match the original signal, we need to remove 1
        if X_band.shape[0] == X.shape[0] + 1 :
            X_band = X_band[:-1, :]
            print("Dropped the last time frame to match expected shape.")

        print("X_band.shape", X_band.shape)

    return X_band


if __name__ == "__main__":

    # Generate a synthetic dataset
    fs = 1000  # Sampling frequency
    t = np.linspace(0, 2, 2 * fs)  # Time axis
    # Generate signals with two frequencies
    f1, f2 = 10, 30  # Frequencies to include in the signal
    X = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)

    # Define peak frequencies around which to filter
    filtered_peak_freqs = extract_peak_frequencies(X.reshape(-1, 1), fs, threshold=1e-5, nperseg=fs, visualize=True)
    print("filtered_peak_freqs", filtered_peak_freqs)

    # Generate multivariate dataset
    X_band = generate_multivariate_dataset(filtered_peak_freqs, is_instances_classification=False, nb_jobs=1, verbosity=0)

    plt.figure(figsize=(24, 6))
    # Plot the original and filtered signals on the same graph
    plt.plot(t, X, label='Original signal', color='blue')
    for i, filtered_signal in enumerate(X_band.T):
        plt.plot(t, filtered_signal, label=f'Filtered signal {i + 1}', linestyle='--')

    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('Original and Filtered Signals')
    plt.legend()
    plt.show()
