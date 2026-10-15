import warnings

import numpy as np
from joblib import Parallel, delayed
from librosa import stft
from librosa.feature import mfcc, delta
from typing import Sequence, Union
from scipy.signal.windows import gaussian

# default MFCC: 13 coefficients (spectral envelope) from 32 mel bands (the 51 frequency bins of a 100-sample window
# leave no mel band empty at 8 and 16 kHz, unlike librosa's default of 128 bands)
N_MFCC = 13
N_MELS = 32


# ------------------------ Spectral dataset maker ----------------------
def generate_multivariate_dataset(
        X: Union[np.ndarray, Sequence[np.ndarray]],
        is_instances_classification: bool,
        spectral_representation: str = None,
        hop: int = 50,
        win_length: int = 100,
        nb_jobs: int = -1,
        verbosity: int = 1,
        mfcc_config: dict = None,
        sampling_rate: float = None,
    ):
    # mfcc_config: dataset-specific librosa.feature.mfcc arguments (sr, n_mfcc, n_fft, win_length, hop_length, fmin,
    # fmax, lifter...) plus "deltas" (bool) to append delta and delta-delta features. Overrides hop and win_length.
    # sampling_rate: sampling rate of the signals (Hz), for the mel scale of the default MFCC (librosa assumes 22050 Hz
    # without it)
    if spectral_representation == "mfcc" and mfcc_config is not None:
        print("Using MFCC config:", mfcc_config)
    else:
        print("Using window length (nperseg):", win_length, "and hop:", hop)
    # ---- Gaussian window (symmetric) with std in samples (computed once) --------------------------------------------
    g_std = 8.0 # standard deviation for Gaussian window in samples
    window = gaussian(win_length, std=g_std, sym=True)

    def compute_instance_spectrogram(x):
        if spectral_representation == "stft":
            Sx = np.abs(stft(x, hop_length=hop, win_length=win_length, n_fft=win_length, window=window))
        elif spectral_representation == "mfcc" and mfcc_config is not None:
            config = dict(mfcc_config)
            add_deltas = config.pop("deltas", False)
            with warnings.catch_warnings():
                # instances shorter than n_fft are zero-padded by librosa, which is fine: don't warn for each of them
                warnings.filterwarnings("ignore", message="n_fft=.* is too large for input signal")
                Sx = mfcc(y=x, **config)
            if add_deltas:
                Sx = np.concatenate([Sx, delta(Sx, mode="wrap"), delta(Sx, order=2, mode="wrap")], axis=-2)
        elif spectral_representation == "mfcc":
            sr = {} if sampling_rate is None else {"sr": sampling_rate}
            Sx = np.abs(mfcc(y=x, hop_length=hop, win_length=win_length, n_fft=win_length, window=window,
                             n_mfcc=N_MFCC, n_mels=N_MELS, **sr))
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
