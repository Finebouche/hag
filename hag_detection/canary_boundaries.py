"""
Phrase boundary detection in canary songs (M1-2016-spring, "Marron1").

A canary phrase is a repetition of one syllable type, so a phrase change is a change of syllable type. A ridge readout
is trained on the states of a reservoir, frame by frame on whole songs, to output a bump (Gaussian, BUMP_STD) at each
phrase end followed by a phrase of another label; boundaries are the peaks of its output. Calls, SIL and TRASH are
ignored, and consecutive segments of the same label are one phrase (annotation splits, not transitions).

Conditions: MFCC alone (no reservoir), and reservoirs built with the best hyperparameters of the canary next-phrase
studies (legacy_studies/tpe_canary_db.sqlite3; unaffected by the MFCC fix): random_ee, var_hag, mean_hag, and
mean_hag_proj (HAG's matrices without the HAG learning). HAG learns on phrases of the train songs.

The ridge coefficient and the peak threshold are chosen on validation songs (taken from the train songs), then the
readout is trained again on all train songs. Score: F1 of the boundaries matched one to one within TOLERANCE, plus the
recall of the boundaries with and without silence between the two phrases.

Usage: python -m hag_detection.canary_boundaries [conditions] [nb_seeds] [max_songs] [network_size]
network_size: number of neurons of the reservoirs instead of the study's (500); results in canary_boundaries_<size>.csv
"""
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import soundfile as sf
from scipy.signal import find_peaks
from sklearn.model_selection import GroupShuffleSplit

from hag.datasets.load_categorical_forecasting import (CANARY_SPLIT_SEED, build_phrase_sequences, canary_mfcc_config,
                                                       download_canary_if_needed)
from hag.datasets.spectral_decomposition import generate_multivariate_dataset
from hag.hpo.utility import LEGACY_STUDIES
from hag.performances.esn_model_evaluation import run_reservoir
from hag.performances.evaluation_esn import build_reservoir_from_study

SAMPLING_RATE = 16000
IGNORED_LABELS = ("SIL", "TRASH", "call")
SILENCE_GAP = 0.005  # s, phrases further apart than this are separated by a silence
BUMP_STD = 0.025  # s
TOLERANCE = 0.05  # s
MIN_PEAK_DISTANCE = 0.1  # s
RIDGES = 10.0 ** np.arange(-14, 3)
THRESHOLDS = np.linspace(0.05, 0.95, 19)
NB_PRETRAIN_PHRASES = 500
STUDIES_DB = f"sqlite:///{LEGACY_STUDIES / 'tpe_canary_db.sqlite3'}"
CONDITIONS = ["mfcc", "random_ee", "var_hag", "mean_hag", "mean_hag_proj"]
OUTPUT = Path(__file__).resolve().parents[1] / "outputs" / "detection_results" / "canary_boundaries.csv"


def load_songs(max_songs=None):
    """
    Songs of the next-phrase dataset with its train/test split by song (same songs as load_canary_dataset).
    Returns (songs, train_songs, test_songs): songs[wave] = dict(audio, phrases=[(start, end, label)],
    boundaries=[(time, silence)]).
    """
    annotations_dir, audio_dir = download_canary_if_needed()
    annotations = pd.concat([pd.read_csv(f, index_col=0) for f in sorted(annotations_dir.rglob("*.csv"))])
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
        song = annotations[annotations["wave"] == wave].sort_values("start")
        song = song[~song["syll"].isin(IGNORED_LABELS)]
        phrases = []
        for start, end, label in song[["start", "end", "syll"]].itertuples(index=False, name=None):
            if phrases and phrases[-1][2] == label:
                phrases[-1] = (phrases[-1][0], end, label)
            else:
                phrases.append((start, end, label))
        boundaries = [(prev[1], nxt[0] - prev[1] > SILENCE_GAP) for prev, nxt in zip(phrases[:-1], phrases[1:])]

        audio, sampling_rate = sf.read(wav_paths[wave], dtype="float32", always_2d=True)
        audio = librosa.resample(audio.mean(axis=1), orig_sr=sampling_rate, target_sr=SAMPLING_RATE)
        songs[wave] = dict(audio=audio, phrases=phrases, boundaries=boundaries)
    return songs, train_songs, test_songs


def compute_features(songs):
    """MFCC (+ deltas) of each song, MinMax scaled (0, 1) on the train songs: songs[wave]["mfcc"], frame duration."""
    config = canary_mfcc_config(SAMPLING_RATE)
    waves = list(songs)
    mfccs = generate_multivariate_dataset([songs[w]["audio"].reshape(-1, 1) for w in waves], True, "mfcc",
                                          mfcc_config=config, sampling_rate=SAMPLING_RATE, verbosity=0)
    for wave, features in zip(waves, mfccs):
        songs[wave]["mfcc"] = features
    return config["hop_length"] / SAMPLING_RATE


def scale(songs, train_songs):
    train = np.concatenate([songs[w]["mfcc"] for w in train_songs])
    low, high = train.min(axis=0), train.max(axis=0)
    for song in songs.values():
        song["mfcc"] = (song["mfcc"] - low) / np.where(high > low, high - low, 1)


def boundary_target(song, frame):
    times = np.arange(len(song["mfcc"])) * frame  # librosa frames are centered
    target = np.zeros(len(times))
    for time, _ in song["boundaries"]:
        target = np.maximum(target, np.exp(-0.5 * ((times - time) / BUMP_STD) ** 2))
    return target


def pretrain_phrases(songs, train_songs, frame):
    """MFCC of NB_PRETRAIN_PHRASES phrases of the train songs (HAG's pretraining instances), drawn with numpy's RNG."""
    phrases = [songs[w]["mfcc"][int(start / frame):int(end / frame) + 1]
               for w in train_songs for start, end, _ in songs[w]["phrases"]]
    phrases = [p for p in phrases if len(p) > 1]
    indices = np.random.choice(len(phrases), min(NB_PRETRAIN_PHRASES, len(phrases)), replace=False)
    return np.array(phrases, dtype=object)[indices]


def match(predicted, true, tolerance=TOLERANCE):
    """One-to-one matching of predicted and true times (closest pairs first): boolean "matched" mask of true."""
    pairs = sorted((abs(p - t), i, j) for i, p in enumerate(predicted) for j, t in enumerate(true)
                   if abs(p - t) <= tolerance)
    used_predicted, matched = set(), np.zeros(len(true), dtype=bool)
    for _, i, j in pairs:
        if i not in used_predicted and not matched[j]:
            used_predicted.add(i)
            matched[j] = True
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


def ridge_fit(states, targets, ridge):
    """Ridge readout (with bias) accumulated song by song: Wout of shape (units + 1,)."""
    dim = states[0].shape[1] + 1
    XtX, XtY = np.zeros((dim, dim)), np.zeros(dim)
    for x, y in zip(states, targets):
        x = np.hstack([x, np.ones((len(x), 1))])
        XtX += x.T @ x
        XtY += x.T @ y
    return np.linalg.solve(XtX + ridge * np.eye(dim), XtY)


def ridge_run(states, Wout):
    return [np.hstack([x, np.ones((len(x), 1))]) @ Wout for x in states]


def evaluate(condition, songs, train_songs, test_songs, frame, seed, network_size=None):
    np.random.seed(seed)
    if condition == "mfcc":
        states = {w: songs[w]["mfcc"] for w in songs}
    else:
        function_name = condition.removesuffix("_proj")
        study = optuna.load_study(study_name=f"{function_name}_Canary_normal_multi", storage=STUDIES_DB)
        hyperparams = dict(study.best_trial.params)
        if network_size is not None:
            hyperparams["network_size"] = network_size
        reservoir, _ = build_reservoir_from_study(function_name, hyperparams, pretrain_phrases(songs, train_songs, frame),
                                                  True, plasticity=not condition.endswith("_proj"))
        waves = list(songs)
        states = dict(zip(waves, run_reservoir(reservoir, [songs[w]["mfcc"] for w in waves], reset=True)))
    targets = {w: boundary_target(songs[w], frame) for w in songs}

    # ridge and threshold on validation songs
    rng = np.random.default_rng(seed)
    validation = set(rng.choice(train_songs, max(1, len(train_songs) // 5), replace=False))
    fit_songs = [w for w in train_songs if w not in validation]
    validation = sorted(validation)
    best = (-1, None, None)
    for ridge in RIDGES:
        Wout = ridge_fit([states[w] for w in fit_songs], [targets[w] for w in fit_songs], ridge)
        outputs = ridge_run([states[w] for w in validation], Wout)
        for threshold in THRESHOLDS:
            f1 = scores(outputs, songs, validation, threshold, frame)["f1"]
            if f1 > best[0]:
                best = (f1, ridge, threshold)
    _, ridge, threshold = best

    Wout = ridge_fit([states[w] for w in train_songs], [targets[w] for w in train_songs], ridge)
    result = scores(ridge_run([states[w] for w in test_songs], Wout), songs, test_songs, threshold, frame)
    size = None if condition == "mfcc" else hyperparams["network_size"]
    return dict(condition=condition, seed=seed, network_size=size, ridge=ridge, threshold=threshold, validation_f1=best[0], **result)


def main():
    conditions = sys.argv[1].split(",") if len(sys.argv) > 1 else CONDITIONS
    nb_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    max_songs = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] != "all" else None
    network_size = int(sys.argv[4]) if len(sys.argv) > 4 else None
    output = OUTPUT if network_size is None else OUTPUT.with_stem(f"{OUTPUT.stem}_{network_size}")

    songs, train_songs, test_songs = load_songs(max_songs)
    frame = compute_features(songs)
    scale(songs, train_songs)
    print(f"{len(train_songs)} train / {len(test_songs)} test songs, frame = {frame * 1000:.1f} ms, "
          f"{sum(len(songs[w]['boundaries']) for w in test_songs)} test boundaries")

    results = []
    for condition in conditions:
        for seed in range(nb_seeds):
            result = evaluate(condition, songs, train_songs, test_songs, frame, seed, network_size)
            print({k: round(v, 3) if isinstance(v, float) else v for k, v in result.items()}, flush=True)
            results.append(result)

    results = pd.DataFrame(results)
    print(results.groupby("condition")[["f1", "precision", "recall", "recall_silence", "recall_no_silence"]]
          .agg(["mean", "std"]).round(3))
    if max_songs is None:
        output.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output, index=False)
        print("Saved to", output)


if __name__ == "__main__":
    main()
