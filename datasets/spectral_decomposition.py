import numpy as np
from joblib import Parallel, delayed
from librosa import stft
from librosa.feature import mfcc
from typing import Sequence, Union
from scipy.signal.windows import gaussian

# ------------------------ Spectral dataset maker ----------------------
def generate_multivariate_dataset(
        X: Union[np.ndarray, Sequence[np.ndarray]],
        is_instances_classification: bool,
        spectral_representation: str = None,
        hop: int = 50,
        win_length: int = 100,
        nb_jobs: int = -1,
        verbosity: int = 1,
    ):
    print("Using window length (nperseg):", win_length, "and hop:", hop)
    # ---- Gaussian window (symmetric) with std in samples (computed once) --------------------------------------------
    g_std = 8.0 # standard deviation for Gaussian window in samples
    window = gaussian(win_length, std=g_std, sym=True)

    def compute_instance_spectrogram(x):
        if spectral_representation == "stft":
            Sx = np.abs(stft(x, hop_length=hop, win_length=win_length, n_fft=win_length, window=window))
        elif spectral_representation == "mfcc":
            Sx = np.abs(mfcc(y = x, hop_length=hop, win_length=win_length, n_fft=win_length, window=window))
        return np.hstack(Sx).T if is_instances_classification else Sx.T

    if is_instances_classification:  # classification -> Multiple instances
        X_band = Parallel(n_jobs=nb_jobs, verbose=verbosity)(delayed(compute_instance_spectrogram)(x.T) for x in X)
    else : # regression -> Single "instance"
        # multi-channel (single instance): concatenate per channel
        X = np.asarray(X)
        X_band = np.hstack([compute_instance_spectrogram(X[:, i]) for i in range(X.shape[1])])

        # if dimension doesn't match the original signal, we need to remove 1
        if X_band.shape[0] == X.shape[0] + 1 :
            X_band = X_band[:-1, :]
            print("Dropped the last time frame to match expected shape.")

        print("X_band.shape", X_band.shape)
    return X_band
