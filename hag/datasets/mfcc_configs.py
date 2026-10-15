"""
Dataset-specific MFCC (librosa.feature.mfcc arguments, plus "deltas" to append delta and delta-delta features), used
instead of the generic MFCC of hag.datasets.spectral_decomposition (window of 100 samples, hop of 50, 13 MFCC from
32 mel bands) by hag.datasets.pipeline.prepare_data and hag.hpo.hpo_esn.
"""
import math

from hag.datasets.load_categorical_forecasting import canary_mfcc_config


def speech_mfcc_config(sampling_rate):
    """
    Usual MFCC of speech recognition: 25 ms windows every 10 ms, FFT on the next power of 2, 40 mel bands from 20 Hz
    to the Nyquist frequency, 20 MFCC. (The generic MFCC uses 6 ms windows every 3 ms at 16 kHz, 51 frequencies,
    13 MFCC: on SPEECHCOMMANDS, var_hag loses 6.7 points of test accuracy with them compared with the earlier
    20 MFCC, hag.analysis.speechcommands_preprocessing.)
    """
    win_length = round(0.025 * sampling_rate)
    return dict(sr=sampling_rate, n_mfcc=20, n_mels=40, win_length=win_length, hop_length=round(0.010 * sampling_rate),
                n_fft=2 ** math.ceil(math.log2(win_length)), fmin=20, fmax=sampling_rate / 2, deltas=False)


def dataset_mfcc_config(dataset_name, sampling_rate):
    """MFCC configuration of the dataset, None for the generic MFCC."""
    if dataset_name == "Canary":
        return canary_mfcc_config(sampling_rate)
    if dataset_name == "SPEECHCOMMANDS":
        return speech_mfcc_config(sampling_rate)
    return None
