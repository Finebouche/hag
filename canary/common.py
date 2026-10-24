"""
Songs of the canary tasks (canary.next_phrase_prediction, canary.syllables_classification): train/test split, MFCC
features (cached in FEATURES_CACHE), phrases and HAG pretraining phrases.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.preprocessing import MinMaxScaler

from hag.datasets.load_categorical_forecasting import (build_phrase_sequences, canary_mfcc_config,
                                                       download_canary_if_needed, split_canary_songs)
from hag.datasets.preprocessing import scale_data
from hag.datasets.spectral_decomposition import generate_multivariate_dataset

# native sampling rate of the recordings, as Trouvain & Hinaut (ICANN 2021): MFCC up to 8 kHz without resampling
SAMPLING_RATE = 44100
IGNORED_LABELS = ("SIL", "TRASH", "call")
SILENCE_GAP = 0.005  # s, phrases further apart than this are separated by a silence
RIDGES = 10.0 ** np.arange(-14, 3)
NB_PRETRAIN_PHRASES = 500
OUTPUTS = Path(__file__).resolve().parents[1] / "outputs" / "canary"
FEATURES_CACHE = Path(__file__).resolve().parents[1] / "hag" / "datasets" / "Canary" / f"boundary_features_{SAMPLING_RATE}.pkl"
# MFCC without the floor at 80 dB below the maximum of the whole song (load_dataset(top_db=None))
NO_FLOOR_FEATURES_CACHE = FEATURES_CACHE.with_name(f"boundary_features_{SAMPLING_RATE}_no_floor.pkl")


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


def load_songs():
    """
    Songs of the next-phrase dataset with its train/test split by song (same songs as load_canary_dataset).
    Returns (songs, train_songs, test_songs): songs[wave] = dict(audio, phrases=[(start, end, label)],
    boundaries=[(time, silence)]) (boundaries between the phrases, silent or not).
    """
    _, audio_dir = download_canary_if_needed()
    annotations = annotations_of_songs()
    train_songs, test_songs = split_canary_songs(build_phrase_sequences(annotations)["wave"])

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


def pretrain_phrases(songs, train_songs, frame):
    """MFCC of NB_PRETRAIN_PHRASES phrases of the train songs (HAG's pretraining instances), drawn with numpy's RNG."""
    phrases = [songs[w]["mfcc"][int(start / frame):int(end / frame) + 1]
               for w in train_songs for start, end, _ in songs[w]["phrases"]]
    phrases = [p for p in phrases if len(p) > 1]
    indices = np.random.choice(len(phrases), min(NB_PRETRAIN_PHRASES, len(phrases)), replace=False)
    return np.array(phrases, dtype=object)[indices]


def load_dataset(top_db=80.0):
    """
    Songs of the tasks: songs[wave] = dict(mfcc (MFCC + deltas, MinMax-scaled (0, 1) on the train songs), phrases,
    boundaries), train and test songs, frame duration. The MFCC of all the songs are cached in FEATURES_CACHE (no
    audio kept). top_db: floor of the dB of the mel spectrogram below the maximum of the whole song (librosa's
    default, 80); None: no floor (cached in NO_FLOOR_FEATURES_CACHE), for the next-phrase prediction, since this
    maximum depends on the future of each frame.
    """
    import pickle
    if top_db not in (80.0, None):
        raise ValueError(f"top_db={top_db}: only 80 (FEATURES_CACHE) and None (NO_FLOOR_FEATURES_CACHE) are cached")
    cache = FEATURES_CACHE if top_db is not None else NO_FLOOR_FEATURES_CACHE
    if cache.exists():
        with open(cache, "rb") as f:
            songs, train_songs, test_songs, frame = pickle.load(f)
    else:
        songs, train_songs, test_songs = load_songs()
        config = dict(canary_mfcc_config(SAMPLING_RATE), top_db=top_db)
        mfccs = generate_multivariate_dataset([song["audio"].reshape(-1, 1) for song in songs.values()], True, "mfcc",
                                              mfcc_config=config, sampling_rate=SAMPLING_RATE, verbosity=0)
        for song, features in zip(songs.values(), mfccs):
            song["mfcc"] = features
            del song["audio"]
        frame = config["hop_length"] / SAMPLING_RATE
        with open(cache, "wb") as f:
            pickle.dump((songs, train_songs, test_songs, frame), f)
    other_songs = [w for w in songs if w not in set(train_songs)]
    train, _, other = scale_data([songs[w]["mfcc"] for w in train_songs], None,
                                 [songs[w]["mfcc"] for w in other_songs], MinMaxScaler(feature_range=(0, 1)), True)
    for wave, mfcc in zip(train_songs + other_songs, train + other):
        songs[wave]["mfcc"] = mfcc
    return songs, train_songs, test_songs, frame
