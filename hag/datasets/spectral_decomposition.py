import warnings

import numpy as np
from joblib import Parallel, delayed
from librosa import power_to_db, stft
from librosa.feature import delta, melspectrogram, mfcc
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
    # prediction: causal frames, frame t made of the win_length samples up to t (zero padding on the left), weighted by
    # the rising half of the Gaussian (largest weight on sample t). Centered frames (librosa's default) contain the
    # samples after t, among which the target (t + step_ahead).
    causal = not is_instances_classification
    if causal:
        if mfcc_config is not None:
            raise ValueError("No causal frames for a dataset-specific MFCC (mfcc_config): classification only")
        window = gaussian(2 * win_length - 1, std=g_std, sym=True)[:win_length]

    def frames_input(x):
        return np.pad(x, (win_length - 1, 0)) if causal else x

    def compute_instance_spectrogram(x):
        if spectral_representation == "stft":
            Sx = np.abs(stft(frames_input(x), hop_length=hop, win_length=win_length, n_fft=win_length, window=window,
                             center=not causal))
        elif spectral_representation == "mfcc" and mfcc_config is not None:
            config = dict(mfcc_config)
            add_deltas = config.pop("deltas", False)
            # floor of the dB at top_db below the maximum of the whole instance (librosa's default, 80), None: no floor
            # (the maximum depends on the whole instance, its future included)
            top_db = config.pop("top_db", 80.0)
            mfcc_args = {key: config.pop(key) for key in ("sr", "n_mfcc", "dct_type", "norm", "lifter") if key in config}
            with warnings.catch_warnings():
                # instances shorter than n_fft are zero-padded by librosa, which is fine: don't warn for each of them
                warnings.filterwarnings("ignore", message="n_fft=.* is too large for input signal")
                S = power_to_db(melspectrogram(y=x, sr=mfcc_args.get("sr", 22050), **config), top_db=top_db)
            Sx = mfcc(S=S, **mfcc_args)
            if add_deltas:
                Sx = np.concatenate([Sx, delta(Sx, mode="wrap"), delta(Sx, order=2, mode="wrap")], axis=-2)
        elif spectral_representation == "mfcc":
            sr = {} if sampling_rate is None else {"sr": sampling_rate}
            mel = dict(hop_length=hop, win_length=win_length, n_fft=win_length, window=window, n_mels=N_MELS, **sr)
            if causal:
                # no floor at 80 dB below the maximum of the whole series (librosa's default), which depends on its future
                S = power_to_db(melspectrogram(y=frames_input(x), center=False, **mel), top_db=None)
                Sx = np.abs(mfcc(S=S, n_mfcc=N_MFCC, **sr))
            else:
                Sx = np.abs(mfcc(y=x, n_mfcc=N_MFCC, **mel))
        return np.hstack(Sx).T if is_instances_classification else Sx.T

    if is_instances_classification:  # classification -> Multiple instances
        X_band = Parallel(n_jobs=nb_jobs, verbose=verbosity)(delayed(compute_instance_spectrogram)(x.T) for x in X)
    else : # regression -> Single "instance"
        # multi-channel (single instance): concatenate per channel
        X = np.asarray(X)
        X_band = np.hstack([compute_instance_spectrogram(X[:, i]) for i in range(X.shape[1])])
        if X_band.shape[0] != X.shape[0]:
            raise ValueError(f"{X_band.shape[0]} frames for {X.shape[0]} time steps (hop={hop}): one frame per step "
                             "expected")

        print("X_band.shape", X_band.shape)
    return X_band
