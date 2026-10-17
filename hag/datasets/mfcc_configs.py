"""
Dataset-specific MFCC (librosa.feature.mfcc arguments, plus "deltas" to append delta and delta-delta features), used
instead of the generic MFCC of hag.datasets.spectral_decomposition (window of 100 samples, hop of 50, 13 MFCC from
32 mel bands) by hag.datasets.pipeline.prepare_data and hag.hpo.hpo_esn. The datasets of raw audio (AUDIO_DATASETS)
all have the standard MFCC; the canary songs have the one of canapy.
"""
import math

from hag.datasets.load_categorical_forecasting import canary_mfcc_config

AUDIO_DATASETS = ("SPEECHCOMMANDS", "FSDD", "CatsDogs")


def standard_mfcc_config(sampling_rate):
    """
    Usual MFCC of speech recognition: 25 ms windows every 10 ms, FFT on the next power of 2, 40 mel bands from 20 Hz
    to the Nyquist frequency, 20 MFCC. (The generic MFCC uses 6 ms windows every 3 ms at 16 kHz, 51 frequencies,
    13 MFCC: too short windows and too few coefficients for these datasets.)
    """
    win_length = round(0.025 * sampling_rate)
    return dict(sr=sampling_rate, n_mfcc=20, n_mels=40, win_length=win_length, hop_length=round(0.010 * sampling_rate),
                n_fft=2 ** math.ceil(math.log2(win_length)), fmin=20, fmax=sampling_rate / 2, deltas=False)


def dataset_mfcc_config(dataset_name, sampling_rate):
    """MFCC configuration of the dataset, None for the generic MFCC."""
    if dataset_name == "Canary":
        return canary_mfcc_config(sampling_rate)
    if dataset_name in AUDIO_DATASETS:
        return standard_mfcc_config(sampling_rate)
    return None
