"""
Categorical forecasting datasets: predict the category of the NEXT element of a sequence (next-symbol prediction)
from the signal of the current one. Instances are handled like classification instances.

Canary song dataset (M1-2016-spring, https://zenodo.org/records/6521932): canary songs are sequences of phrases
(A -> A -> B -> C ...). Each instance is the raw audio of the current phrase and its target is the label of the NEXT
phrase in the same song: the reservoir "forecasts" from the current phrase, and the readout classifies what comes next.
"""
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

DATASETS_DIR = Path(__file__).resolve().parent
CANARY_DIR = DATASETS_DIR / "Canary"

ZENODO_URL = "https://zenodo.org/api/records/6521932/files/{}/content"
ANNOTATIONS_ZIP = "M1-2016-spring_csv_annotations.zip"
AUDIO_ZIP = "M1-2016-sping_audio.zip"  # sic, typo in the Zenodo record
# Fixed so that the HPO (train split) and the final evaluation (test split) always see the same songs
CANARY_SPLIT_SEED = 0


def canary_mfcc_config(sampling_rate):
    """
    MFCC features used for canary songs by Trouvain & Hinaut (ICANN 2021, ReservoirPy tutorial 3), given at 44.1 kHz:
    23 ms window, 11.6 ms hop, 500-8000 Hz, 13 MFCC + delta + delta-delta. Window sizes are rescaled to sampling_rate.
    """
    scale = sampling_rate / 44100
    return dict(sr=sampling_rate, n_mfcc=13, win_length=round(1024 * scale), hop_length=round(512 * scale),
                n_fft=round(2048 * scale), fmin=500, fmax=min(8000, sampling_rate / 2), lifter=40, deltas=True)


def _download_and_extract(zip_name, target_dir):
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = CANARY_DIR / zip_name
    print(f"Downloading {zip_name} to {zip_path} ...")
    urllib.request.urlretrieve(ZENODO_URL.format(zip_name), str(zip_path))
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(target_dir)
    zip_path.unlink(missing_ok=True)


def download_canary_if_needed():
    annotations_dir = CANARY_DIR / "annotations"
    audio_dir = CANARY_DIR / "audio"
    if not any(annotations_dir.rglob("*.csv")):
        _download_and_extract(ANNOTATIONS_ZIP, annotations_dir)
    if not any(audio_dir.rglob("*.wav")):
        _download_and_extract(AUDIO_ZIP, audio_dir)  # ~855 MB
    return annotations_dir, audio_dir


def build_phrase_sequences(annotations, ignored_labels=("SIL", "TRASH"), merge_repeats=False, end_label=None):
    """
    Turn the phrase annotations into (current phrase, next label) pairs, song by song.

    Parameters:
    - annotations: DataFrame with columns "wave", "start", "end", "syll" (one row per annotated segment).
    - ignored_labels: labels removed from the sequence before pairing (they are neither inputs nor targets),
      e.g. A -> SIL -> B becomes the pair (A, B).
    - merge_repeats: if True, consecutive segments with the same label are merged into one phrase
      (A -> A -> B becomes A -> B). If False, A -> A is kept as a valid transition.
    - end_label: if given (e.g. "END"), the last phrase of each song gets this target; otherwise it is dropped.

    Returns a DataFrame with one row per instance: "wave", "start", "end", "current", "next".
    """
    pairs = []
    for wave, song in annotations.groupby("wave", sort=True):
        song = song.sort_values("start")
        song = song[~song["syll"].isin(ignored_labels)]
        phrases = list(song[["start", "end", "syll"]].itertuples(index=False, name=None))

        if merge_repeats:
            merged = []
            for start, end, label in phrases:
                if merged and merged[-1][2] == label:
                    merged[-1] = (merged[-1][0], end, label)
                else:
                    merged.append((start, end, label))
            phrases = merged

        for k, (start, end, label) in enumerate(phrases):
            if k + 1 < len(phrases):
                next_label = phrases[k + 1][2]
            elif end_label is not None:
                next_label = end_label
            else:
                continue
            pairs.append((wave, start, end, label, next_label))

    return pd.DataFrame(pairs, columns=["wave", "start", "end", "current", "next"])


def load_canary_dataset(test_split=0.2, seed=CANARY_SPLIT_SEED, target_sampling_rate=16000, ignored_labels=("SIL", "TRASH"),
                        merge_repeats=False, end_label=None):
    """
    Load the canary dataset as a next-phrase classification problem.

    The train/test split is done by song, so that phrases of the same song never end up on both sides.

    Returns:
    - sampling_rate: sampling rate of the returned audio.
    - X_train, X_test: lists of (T, 1) float arrays, raw audio of the current phrase.
    - Y_train, Y_test: one-hot encoded label of the next phrase.
    - groups: song id of each training instance (for StratifiedGroupKFold).
    - classes: label names, in the order of the one-hot columns.
    """
    annotations_dir, audio_dir = download_canary_if_needed()

    annotations = pd.concat([pd.read_csv(f, index_col=0) for f in sorted(annotations_dir.rglob("*.csv"))])
    pairs = build_phrase_sequences(annotations, ignored_labels, merge_repeats, end_label)
    wav_paths = {path.name: path for path in audio_dir.rglob("*.wav")}

    X = []
    for wave, song_pairs in pairs.groupby("wave", sort=False):
        audio, sampling_rate = sf.read(wav_paths[wave], dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)  # mono
        if target_sampling_rate is not None and target_sampling_rate != sampling_rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=target_sampling_rate)
            sampling_rate = target_sampling_rate
        for start, end in song_pairs[["start", "end"]].itertuples(index=False, name=None):
            X.append(audio[int(start * sampling_rate):int(end * sampling_rate)].reshape(-1, 1))
    # pairs are already contiguous per song, so X is in the same order as pairs

    X = np.array(X, dtype=object)
    groups = pairs["wave"].to_numpy()

    le = LabelEncoder()
    Y_encoded = le.fit_transform(pairs["next"])
    ohe = OneHotEncoder(sparse_output=False)
    Y_one_hot = ohe.fit_transform(Y_encoded.reshape(-1, 1))

    gss = GroupShuffleSplit(n_splits=1, test_size=test_split, random_state=seed)
    train_idx, test_idx = next(gss.split(X, Y_one_hot, groups))

    print("Number of songs =", len(np.unique(groups)))
    print("Number of instances (train / test) =", len(train_idx), "/", len(test_idx))
    print("Number of classes =", len(le.classes_))

    return (sampling_rate, list(X[train_idx]), list(X[test_idx]), Y_one_hot[train_idx], Y_one_hot[test_idx],
            groups[train_idx], le.classes_)


def load_dataset_categorical_forecasting(name, visualize=True, seed=CANARY_SPLIT_SEED):
    if name == "Canary":
        sampling_rate, X_train, X_test, Y_train, Y_test, groups, classes = load_canary_dataset(seed=seed)
        print("Classes (next phrase) =", list(classes))
        is_multivariate = False
        use_spectral_representation = False
        return use_spectral_representation, is_multivariate, sampling_rate, X_train, X_test, Y_train, Y_test, groups

    raise ValueError(f"The dataset with name '{name}' is not loadable")
