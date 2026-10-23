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


def boundary_target(song, frame):
    times = np.arange(len(song["mfcc"])) * frame  # librosa frames are centered
    target = np.zeros(len(times))
    for time, _ in song["boundaries"]:
        target = np.maximum(target, np.exp(-0.5 * ((times - time) / BUMP_STD) ** 2))
    return target


def _match_pairs(predicted, true, tolerance):
    pairs = sorted((abs(p - t), i, j) for i, p in enumerate(predicted) for j, t in enumerate(true)
                   if abs(p - t) <= tolerance)
    used_predicted, matched = set(), np.zeros(len(true), dtype=bool)
    for _, i, j in pairs:
        if i not in used_predicted and not matched[j]:
            used_predicted.add(i)
            matched[j] = True
    return matched


def match(predicted, true, tolerance=TOLERANCE):
    """
    One-to-one matching of predicted and true times (closest pairs first): boolean "matched" mask of true.
    When the nearest predictions of the true times within tolerance are all different, this matching gives each of them
    its nearest prediction (computed with searchsorted); otherwise the pairs are sorted by distance (slower).
    """
    predicted, true = np.sort(np.asarray(predicted, dtype=float)), np.asarray(true, dtype=float)
    if len(predicted) == 0 or len(true) == 0:
        return np.zeros(len(true), dtype=bool)
    index = np.searchsorted(predicted, true)
    left, right = np.clip(index - 1, 0, len(predicted) - 1), np.clip(index, 0, len(predicted) - 1)
    left_distance, right_distance = np.abs(true - predicted[left]), np.abs(true - predicted[right])
    nearest = np.where(left_distance <= right_distance, left, right)
    matched = np.minimum(left_distance, right_distance) <= tolerance
    if len(np.unique(nearest[matched])) < matched.sum():
        return _match_pairs(predicted, true, tolerance)
    return matched


def detect(output, threshold, frame):
    peaks, _ = find_peaks(output, height=threshold, distance=max(1, round(MIN_PEAK_DISTANCE / frame)))
    return peaks * frame


def scores(outputs, songs, waves, threshold, frame):
    nb_predicted, matched, silence = 0, [], []
    for output, wave in zip(outputs, waves):
        predicted = detect(output, threshold, frame)
        true = songs[wave]["boundaries"]
        nb_predicted += len(predicted)
        matched.append(match(predicted, [t for t, _ in true]))
        silence.append(np.array([s for _, s in true], dtype=bool))
    matched, silence = np.concatenate(matched), np.concatenate(silence)
    precision = matched.sum() / max(nb_predicted, 1)
    recall = matched.mean() if len(matched) else 0.0
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return dict(f1=f1, precision=precision, recall=recall,
                recall_silence=matched[silence].mean() if silence.any() else np.nan,
                recall_no_silence=matched[~silence].mean() if (~silence).any() else np.nan)


def load_dataset(max_songs=None):
    """canary.common.load_dataset, and the boundary target of each song: songs[wave]["target"]."""
    songs, train_songs, test_songs, frame = load_songs_dataset(max_songs)
    for song in songs.values():
        song["target"] = boundary_target(song, frame)
    return songs, train_songs, test_songs, frame
