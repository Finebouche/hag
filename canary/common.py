"""
Shared parts of the syllable classification (syllables_classification): songs and their train/test split, MFCC
features and HAG pretraining phrases. Same songs, split and MFCC (cached in FEATURES_CACHE, also read by
next_phrase_prediction) as the next-phrase prediction.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.model_selection import GroupShuffleSplit

from hag.datasets.load_categorical_forecasting import (CANARY_SPLIT_SEED, build_phrase_sequences, canary_mfcc_config,
                                                       download_canary_if_needed)
from hag.datasets.spectral_decomposition import generate_multivariate_dataset

# native sampling rate of the recordings, as Trouvain & Hinaut (ICANN 2021): MFCC up to 8 kHz without resampling
SAMPLING_RATE = 44100
IGNORED_LABELS = ("SIL", "TRASH", "call")
SILENCE_GAP = 0.005  # s, phrases further apart than this are separated by a silence
RIDGES = 10.0 ** np.arange(-14, 3)
NB_PRETRAIN_PHRASES = 500
OUTPUTS = Path(__file__).resolve().parents[1] / "outputs" / "canary"
FEATURES_CACHE = Path(__file__).resolve().parents[1] / "hag" / "datasets" / "Canary" / f"boundary_features_{SAMPLING_RATE}.pkl"


def annotations_of_songs():
    annotations_dir, _ = download_canary_if_needed()
    return pd.concat([pd.read_csv(f, index_col=0) for f in sorted(annotations_dir.rglob("*.csv"))])


def song_phrases(annotations, wave, ignored_labels=IGNORED_LABELS):
    """[(start, end, label)] of the phrases of a song: segments of ignored_labels removed, repeated labels merged."""
    song = annotations[annotations["wave"] == wave].sort_values("start")
    song = song[~song["syll"].isin(ignored_labels)]
    phrases = []
    for start, end, label in song[["start", "end", "syll"]].itertuples(index=False, name=None):
        if phrases and phrases[-1][2] == label:
            phrases[-1] = (phrases[-1][0], end, label)
        else:
            phrases.append((start, end, label))
    return phrases


def load_songs(max_songs=None):
    """
    Songs of the next-phrase dataset with its train/test split by song (same songs as load_canary_dataset).
    Returns (songs, train_songs, test_songs): songs[wave] = dict(audio, phrases=[(start, end, label)],
    boundaries=[(time, silence)]) (the boundaries: same format of the cache as next_phrase_prediction).
    """
    _, audio_dir = download_canary_if_needed()
    annotations = annotations_of_songs()
    pairs = build_phrase_sequences(annotations)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=CANARY_SPLIT_SEED)
    train_idx, test_idx = next(gss.split(pairs, groups=pairs["wave"]))
    train_songs, test_songs = sorted(set(pairs["wave"].iloc[train_idx])), sorted(set(pairs["wave"].iloc[test_idx]))
    if max_songs is not None:
        train_songs, test_songs = train_songs[:max_songs], test_songs[:max(1, max_songs // 4)]

    wav_paths = {path.name: path for path in audio_dir.rglob("*.wav")}
    import librosa
    songs = {}
    for wave in train_songs + test_songs:
        phrases = song_phrases(annotations, wave)
        boundaries = [(prev[1], nxt[0] - prev[1] > SILENCE_GAP) for prev, nxt in zip(phrases[:-1], phrases[1:])]

        audio, sampling_rate = sf.read(wav_paths[wave], dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if sampling_rate != SAMPLING_RATE:
            audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=SAMPLING_RATE)
        songs[wave] = dict(audio=audio, phrases=phrases, boundaries=boundaries)
    return songs, train_songs, test_songs


def compute_features(songs):
    """MFCC (+ deltas) of each song: songs[wave]["mfcc"], frame duration."""
    config = canary_mfcc_config(SAMPLING_RATE)
    waves = list(songs)
    mfccs = generate_multivariate_dataset([songs[w]["audio"].reshape(-1, 1) for w in waves], True, "mfcc",
                                          mfcc_config=config, sampling_rate=SAMPLING_RATE, verbosity=0)
    for wave, features in zip(waves, mfccs):
        songs[wave]["mfcc"] = features
    return config["hop_length"] / SAMPLING_RATE


def scale(songs, train_songs):
    """MinMax scaling (0, 1) of the MFCC, fitted on the train songs."""
    train = np.concatenate([songs[w]["mfcc"] for w in train_songs])
    low, high = train.min(axis=0), train.max(axis=0)
    for song in songs.values():
        song["mfcc"] = (song["mfcc"] - low) / np.where(high > low, high - low, 1)


def pretrain_phrases(songs, train_songs, frame):
    """MFCC of NB_PRETRAIN_PHRASES phrases of the train songs (HAG's pretraining instances), drawn with numpy's RNG."""
    phrases = [songs[w]["mfcc"][int(start / frame):int(end / frame) + 1]
               for w in train_songs for start, end, _ in songs[w]["phrases"]]
    phrases = [p for p in phrases if len(p) > 1]
    indices = np.random.choice(len(phrases), min(NB_PRETRAIN_PHRASES, len(phrases)), replace=False)
    return np.array(phrases, dtype=object)[indices]


def load_dataset(max_songs=None):
    """
    Songs ready for the classification: songs[wave] = dict(mfcc (scaled on the train songs), phrases, boundaries),
    train and test songs, frame duration. The MFCC of all the songs are cached in FEATURES_CACHE (no audio kept).
    """
    import pickle
    if FEATURES_CACHE.exists():
        with open(FEATURES_CACHE, "rb") as f:
            songs, train_songs, test_songs, frame = pickle.load(f)
    else:
        songs, train_songs, test_songs = load_songs()
        frame = compute_features(songs)
        for song in songs.values():
            del song["audio"]
        with open(FEATURES_CACHE, "wb") as f:
            pickle.dump((songs, train_songs, test_songs, frame), f)
    if max_songs is not None:
        train_songs, test_songs = train_songs[:max_songs], test_songs[:max(1, max_songs // 4)]
        songs = {w: songs[w] for w in train_songs + test_songs}
    scale(songs, train_songs)
    return songs, train_songs, test_songs, frame
