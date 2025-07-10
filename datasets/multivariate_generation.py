import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.colors import to_rgb, to_rgba
from joblib import Parallel, delayed
from librosa import stft
from librosa.feature import mfcc
from scipy.signal.windows import gaussian


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
