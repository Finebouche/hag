"""
Phrase boundaries of the canary songs (canary.next_phrase_prediction.evaluate, the boundary studies of
canary.next_phrase_prediction.hpo): boundary targets, peak picking and scores; the songs with their boundary targets.
"""
import numpy as np
from scipy.signal import find_peaks

from canary.common import load_dataset as load_songs_dataset

BUMP_STD = 0.025  # s
TOLERANCE = 0.05  # s
MIN_PEAK_DISTANCE = 0.1  # s
THRESHOLDS = np.linspace(0.05, 0.95, 19)


def scores(outputs, songs, waves, threshold, frame):
    """
    Scores of the boundaries detected (peaks of the outputs above threshold) in the songs. Predicted and true times are
    matched one to one within TOLERANCE, closest pairs first: when the nearest predictions of the true times within
    TOLERANCE are all different, each of them gets its nearest prediction (computed with searchsorted); otherwise the
    pairs are sorted by distance (slower).
    """
    nb_predicted, matched, silence = 0, [], []
    for output, wave in zip(outputs, waves):
        peaks, _ = find_peaks(output, height=threshold, distance=max(1, round(MIN_PEAK_DISTANCE / frame)))
        predicted = np.sort(peaks * frame)
        true = np.array([t for t, _ in songs[wave]["boundaries"]], dtype=float)
        nb_predicted += len(predicted)
        song_matched = np.zeros(len(true), dtype=bool)
        if len(predicted) and len(true):
            index = np.searchsorted(predicted, true)
            left, right = np.clip(index - 1, 0, len(predicted) - 1), np.clip(index, 0, len(predicted) - 1)
            left_distance, right_distance = np.abs(true - predicted[left]), np.abs(true - predicted[right])
            nearest = np.where(left_distance <= right_distance, left, right)
            song_matched = np.minimum(left_distance, right_distance) <= TOLERANCE
            if len(np.unique(nearest[song_matched])) < song_matched.sum():
                pairs = sorted((abs(p - t), i, j) for i, p in enumerate(predicted) for j, t in enumerate(true)
                               if abs(p - t) <= TOLERANCE)
                used_predicted, song_matched = set(), np.zeros(len(true), dtype=bool)
                for _, i, j in pairs:
                    if i not in used_predicted and not song_matched[j]:
                        used_predicted.add(i)
                        song_matched[j] = True
        matched.append(song_matched)
        silence.append(np.array([s for _, s in songs[wave]["boundaries"]], dtype=bool))
    matched, silence = np.concatenate(matched), np.concatenate(silence)
    precision = matched.sum() / max(nb_predicted, 1)
    recall = matched.mean() if len(matched) else 0.0
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return dict(f1=f1, precision=precision, recall=recall,
                recall_silence=matched[silence].mean() if silence.any() else np.nan,
                recall_no_silence=matched[~silence].mean() if (~silence).any() else np.nan)


def load_dataset():
    """canary.common.load_dataset, and the boundary target of each song: songs[wave]["target"]."""
    songs, train_songs, test_songs, frame = load_songs_dataset()
    for song in songs.values():
        times = np.arange(len(song["mfcc"])) * frame  # librosa frames are centered
        song["target"] = np.zeros(len(times))
        for time, _ in song["boundaries"]:
            song["target"] = np.maximum(song["target"], np.exp(-0.5 * ((times - time) / BUMP_STD) ** 2))
    return songs, train_songs, test_songs, frame
