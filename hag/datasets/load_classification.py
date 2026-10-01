from pathlib import Path
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import urllib.request
import zipfile
import glob

from aeon.datasets import load_classification

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from torchaudio.datasets import SPEECHCOMMANDS

DATASETS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DATASETS_DIR.parent


def _first_existing(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _require_path(path: Path, message: str | None = None) -> Path:
    if not path.exists():
        raise FileNotFoundError(message or f"Path does not exist: {path}")
    return path


def process_audio(file_path):
    filename = tf.strings.split(file_path, '/')[-1]
    label = tf.strings.split(filename, '_')[0]
    speaker = tf.strings.split(filename, '_')[1]
    audio = tf.io.read_file(file_path)
    audio, sampling_rate = tf.audio.decode_wav(audio, desired_channels=1)

    return {
        'label': label,
        'audio': audio,
        'audio/filename': filename,
        'speaker': speaker,
    }, sampling_rate




def download_speechcommands_if_needed() -> Path:
    speech_root = DATASETS_DIR

    # Download/extract if needed
    SPEECHCOMMANDS(root=str(speech_root), download=True, subset="training")

    extracted_dir = speech_root / "SpeechCommands" / "speech_commands_v0.02"
    if not extracted_dir.exists():
        raise FileNotFoundError(
            f"SPEECHCOMMANDS folder not found after download: {extracted_dir}"
        )

    return extracted_dir


def load_SPEECHCOMMANDS():
    base_dir = download_speechcommands_if_needed()

    testing_list = base_dir / "testing_list.txt"
    validation_list = base_dir / "validation_list.txt"

    with open(testing_list, "r") as f:
        testing_relpaths = {line.strip() for line in f if line.strip()}

    with open(validation_list, "r") as f:
        validation_relpaths = {line.strip() for line in f if line.strip()}

    X_train_raw, Y_train_raw = [], []
    X_test_raw, Y_test_raw = [], []
    sampling_rates = []

    for wav_path in base_dir.rglob("*.wav"):
        rel_path = wav_path.relative_to(base_dir).as_posix()

        if rel_path.startswith("_background_noise_/"):
            continue

        audio = tf.io.read_file(str(wav_path))
        audio, sr = tf.audio.decode_wav(audio, desired_channels=1)

        x = audio.numpy()
        sampling_rates.append(int(sr.numpy()))
        label = wav_path.parent.name

        if rel_path in testing_relpaths:
            X_test_raw.append(x)
            Y_test_raw.append(label)
        else:
            # train + validation together
            X_train_raw.append(x)
            Y_train_raw.append(label)

    sampling_rate = int(np.mean(sampling_rates))

    le = LabelEncoder()
    Y_train_raw = le.fit_transform(Y_train_raw).reshape(-1, 1)
    Y_test = le.transform(Y_test_raw).reshape(-1, 1)

    ohe = OneHotEncoder(sparse_output=False)
    Y_train_raw = ohe.fit_transform(Y_train_raw)
    Y_test = ohe.transform(Y_test)

    groups = None
    return X_train_raw, Y_train_raw, X_test_raw, Y_test, sampling_rate, groups

def visualize_groups_distribution(groups):
    unique_speakers = np.unique(groups)
    count = [np.sum(groups == speaker) for speaker in unique_speakers]

    fig, ax = plt.subplots(1, figsize=(6, 2))
    ax.bar(unique_speakers, count)
    ax.set_xlabel('Group')
    ax.set_ylabel('Count')
    plt.tight_layout()
    plt.show()


def download_fsdd() -> Path | None:
    # The GitHub archive extracts to free-spoken-digit-dataset-master/recordings
    fsdd_dir = DATASETS_DIR / "fsdd"
    fsdd_zip = DATASETS_DIR / "fsdd.zip"
    print(f"FSDD dataset not found locally. Downloading to {fsdd_dir} ...")
    fsdd_dir.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(
        "https://github.com/Jakobovski/free-spoken-digit-dataset/archive/refs/heads/master.zip",
        str(fsdd_zip),
    )
    with zipfile.ZipFile(fsdd_zip, "r") as zip_ref:
        zip_ref.extractall(fsdd_dir)
    fsdd_zip.unlink(missing_ok=True)
    return _first_existing(fsdd_dir / "free-spoken-digit-dataset-master" / "recordings")


def download_japanese_vowels_if_needed() -> Path:
    # reservoirpy downloads from cdn.uci-ics-mlr-prod.aws.uci.edu (expired certificate since 2026-09-23):
    # fetch the same UCI archive from archive.ics.uci.edu instead, reservoirpy then reads the local files
    from reservoirpy.datasets._japanese_vowels import REMOTE_FILES  # files reservoirpy expects in data_folder

    data_dir = DATASETS_DIR / "JapaneseVowels"
    if not all((data_dir / file_name).exists() for file_name in REMOTE_FILES.values()):
        print(f"JapaneseVowels dataset not found locally. Downloading to {data_dir} ...")
        data_dir.mkdir(parents=True, exist_ok=True)
        zip_path = DATASETS_DIR / "JapaneseVowels.zip"
        urllib.request.urlretrieve("https://archive.ics.uci.edu/static/public/128/japanese+vowels.zip", str(zip_path))
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)
        zip_path.unlink(missing_ok=True)

        # the description file is not used, but reservoirpy re-downloads everything if it is missing
        descr = data_dir / REMOTE_FILES["DESCR"]
        if not descr.exists():
            descr.write_text("Japanese Vowels, UCI dataset 128: https://archive.ics.uci.edu/dataset/128/japanese+vowels\n")
        missing = [f for f in REMOTE_FILES.values() if not (data_dir / f).exists()]
        if missing:
            raise FileNotFoundError(f"JapaneseVowels archive did not contain {missing} (in {data_dir})")
    return data_dir


# FSDD test speakers (2 of the 6 speakers, never seen in training nor in the hyperparameter optimization): the split
# of seed=1111 (GroupShuffleSplit, test_size=1/3), fixed on 2025-08-29 (8bf3ebc) and used for the results of August
# 2025. The seed was lost on 2026-02-26 (6d2a057): the split by speaker was then drawn at random at each loading,
# which changed the test accuracy by ~20 points.
FSDD_TEST_SPEAKERS = ("nicolas", "yweweler")


def load_FSDD_dataset(data_dir: Path, test_speakers=FSDD_TEST_SPEAKERS, visualize=False):
    data_dir = _require_path(
        Path(data_dir),
        f"FSDD recordings folder not found: {data_dir}"
    )

    # sorted: iterdir's order depends on the file system
    audio_files = sorted(str(data_dir / file) for file in data_dir.iterdir() if file.suffix == ".wav")
    print("Number of audio files:", len(audio_files))

    audio_files_dataset = tf.data.Dataset.from_tensor_slices(audio_files)
    audio_files_dataset = audio_files_dataset.map(process_audio, num_parallel_calls=tf.data.AUTOTUNE)

    feature_dict = []
    sampling_rates = []

    for data, sampling_rate in audio_files_dataset:
        feature_dict.append(data)
        sampling_rates.append(sampling_rate.numpy())

    sampling_rate = np.mean(np.array(sampling_rates))
    print("Mean sampling rate:", sampling_rate)

    features = []
    labels = []
    speakers = []

    for item in feature_dict:
        features.append(item['audio'].numpy())
        labels.append(item['label'].numpy())
        speakers.append(item['speaker'].numpy())

    X = np.array(features, dtype=object)
    Y = np.array(labels)
    groups = np.array([s.decode() if isinstance(s, bytes) else str(s) for s in speakers])

    le = LabelEncoder()
    Y_encoded = le.fit_transform(Y)

    ohe = OneHotEncoder(sparse_output=False)
    Y_one_hot = ohe.fit_transform(Y_encoded.reshape(-1, 1))

    unknown = set(test_speakers) - set(groups)
    if unknown:
        raise ValueError(f"Unknown FSDD test speakers {sorted(unknown)} (speakers: {sorted(set(groups))})")
    is_test = np.isin(groups, test_speakers)
    train_val_idx, test_idx = np.where(~is_test)[0], np.where(is_test)[0]
    print(f"FSDD test speakers: {sorted(test_speakers)}, train speakers: {sorted(map(str, set(groups[train_val_idx])))}")
    X_train, X_test = X[train_val_idx], X[test_idx]
    Y_train, Y_test = Y_one_hot[train_val_idx], Y_one_hot[test_idx]
    train_speakers, test_speakers = groups[train_val_idx], groups[test_idx]

    if visualize:
        visualize_groups_distribution(train_speakers)
        visualize_groups_distribution(test_speakers)

    return sampling_rate, X_train, X_test, Y_train, Y_test, groups[train_val_idx]


def load_haart_dataset(train_path: Path, test_path: Path):
    haart_dir = DATASETS_DIR / "HAART"
    haart_zip = DATASETS_DIR / "HAART.zip"

    if not haart_dir.exists():
        print(f"HAART dataset not found locally. Downloading to {haart_zip} ...")
        urllib.request.urlretrieve(
            'https://www.cs.ubc.ca/labs/spin/data/HAART%20DataSet.zip',
            str(haart_zip),
        )
        with zipfile.ZipFile(haart_zip, 'r') as zip_ref:
            zip_ref.extractall(haart_dir)
        haart_zip.unlink(missing_ok=True)

    train_path = _require_path(Path(train_path), f"HAART train file not found: {train_path}")
    test_path = _require_path(Path(test_path), f"HAART test file not found: {test_path}")

    df_train = pd.read_csv(train_path, header=0)
    df_test = pd.read_csv(test_path, header=0)

    grouped = df_train.groupby(['ParticipantNo', ' "Substrate"', ' "Cover"', ' "Gesture"'])
    X_train = [group.iloc[:, 4:68].values.astype(np.float64) for _, group in grouped]
    Y_train = [name[-1] for name, _ in grouped]

    grouped = df_test.groupby(['ParticipantID', 'Substrate', 'Cover', 'Gesture'])
    X_test = [group.iloc[:, 4:68].values.astype(np.float64) for _, group in grouped]
    Y_test = [name[-1] for name, _ in grouped]

    le = LabelEncoder()
    Y_train_encoded = le.fit_transform(Y_train)
    Y_test_encoded = le.transform(Y_test)

    ohe = OneHotEncoder(sparse_output=False)
    Y_train = ohe.fit_transform(Y_train_encoded.reshape(-1, 1))
    Y_test = ohe.transform(Y_test_encoded.reshape(-1, 1))

    sampling_rate = 54
    return sampling_rate, X_train, Y_train, X_test, Y_test


def load_aoen_dataset(dataset_name, seed=None):
    dataset_dir = DATASETS_DIR / dataset_name

    def _load_local_ts_files():
        from aeon.datasets import load_from_ts_file

        train_files = glob.glob(str(dataset_dir / "**" / f"{dataset_name}_TRAIN.ts"), recursive=True)
        test_files = glob.glob(str(dataset_dir / "**" / f"{dataset_name}_TEST.ts"), recursive=True)

        if not train_files or not test_files:
            raise FileNotFoundError(
                f"Could not find .ts files for {dataset_name} in {dataset_dir}"
            )

        X_train_unprocessed, Y_train_raw = load_from_ts_file(train_files[0])
        X_test_unprocessed, Y_test_raw = load_from_ts_file(test_files[0])

        return X_train_unprocessed, Y_train_raw, X_test_unprocessed, Y_test_raw, None

    try:
        X_train_unprocessed, Y_train_raw, meta_data = load_classification(
            dataset_name,
            return_metadata=True,
            load_equal_length=True,
            load_no_missing=True,
            split="train",
        )
        X_test_unprocessed, Y_test_raw, meta_data = load_classification(
            dataset_name,
            return_metadata=True,
            load_equal_length=True,
            load_no_missing=True,
            split="test",
        )

    except (ValueError, OSError) as exc:
        print(f"Aeon direct loading failed for '{dataset_name}': {exc}")

        if dataset_dir.exists():
            print(f"Dataset '{dataset_name}' found locally at {dataset_dir}. Loading local .ts files...")
            X_train_unprocessed, Y_train_raw, X_test_unprocessed, Y_test_raw, meta_data = _load_local_ts_files()

        else:
            print(f"Dataset '{dataset_name}' not found locally. Downloading manually...")

            dataset_dir.mkdir(parents=True, exist_ok=True)

            zip_path = DATASETS_DIR / f"{dataset_name}.zip"
            url = f"https://www.timeseriesclassification.com/aeon-toolkit/{dataset_name}.zip"

            urllib.request.urlretrieve(url, str(zip_path))

            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(dataset_dir)

            zip_path.unlink(missing_ok=True)

            X_train_unprocessed, Y_train_raw, X_test_unprocessed, Y_test_raw, meta_data = _load_local_ts_files()

    groups = None

    X_train_raw = [x.T for x in X_train_unprocessed]
    X_test_raw = [x.T for x in X_test_unprocessed]

    le = LabelEncoder()
    Y_train_raw = le.fit_transform(Y_train_raw).reshape(-1, 1)
    Y_test = le.transform(Y_test_raw).reshape(-1, 1)

    ohe = OneHotEncoder(sparse_output=False)
    Y_train_raw = ohe.fit_transform(Y_train_raw)
    Y_test = ohe.transform(Y_test)

    return X_train_raw, Y_train_raw, X_test_raw, Y_test, groups, meta_data


def load_dataset_classification(name, visualize=True, seed=None):
    if name in ["SpokenArabicDigits", "CatsDogs", "LSST", "ECG5000", "AbnormalHeartbeat"]:
        X_train, Y_train, X_test, Y_test, groups, meta_data = load_aoen_dataset(name, seed)
        sampling_rate = 10000

        is_multivariate = X_train[0].shape[1] != 1
        use_spectral_representation = is_multivariate

        print("Number of instances =", len(X_train))
        print("Shape of X =", X_train[0].shape)
        print("Shape of y =", Y_train.shape)
        print("Meta data =", meta_data)
        print("Multivariate =", is_multivariate)

        return use_spectral_representation, is_multivariate, sampling_rate, X_train, X_test, Y_train, Y_test, groups

    if name == "SPEECHCOMMANDS":
        is_multivariate = False
        use_spectral_representation = False
        X_train, Y_train, X_test, Y_test, sampling_rate, groups = load_SPEECHCOMMANDS()
        return use_spectral_representation, is_multivariate, sampling_rate, X_train, X_test, Y_train, Y_test, groups

    if name == "FSDD":
        fsdd_recordings = _first_existing(
            DATASETS_DIR / "fsdd" / "free-spoken-digit-dataset-master" / "recordings",
            DATASETS_DIR / "FSDD" / "free-spoken-digit-dataset-master" / "recordings",
        )
        if fsdd_recordings is None:
            fsdd_recordings = download_fsdd()
        if fsdd_recordings is None:
            raise FileNotFoundError(
                "FSDD recordings folder not found. Expected one of:\n"
                f"- {DATASETS_DIR / 'fsdd' / 'free-spoken-digit-dataset-master' / 'recordings'}\n"
                f"- {DATASETS_DIR / 'FSDD' / 'free-spoken-digit-dataset-master' / 'recordings'}"
            )

        sampling_rate, X_train, X_test, Y_train, Y_test, groups = load_FSDD_dataset(
            data_dir=fsdd_recordings,
            visualize=visualize,
        )

        is_multivariate = False
        use_spectral_representation = False
        return use_spectral_representation, is_multivariate, sampling_rate, X_train, X_test, Y_train, Y_test, groups

    if name == "HAART":
        sampling_rate, X_train_band, Y_train, X_test_band, Y_test = load_haart_dataset(
            train_path=DATASETS_DIR / "HAART" / "training.csv",
            test_path=DATASETS_DIR / "HAART" / "testWITHLABELS.csv",
        )
        is_multivariate = True
        groups = None
        use_spectral_representation = False
        return use_spectral_representation, is_multivariate, sampling_rate, X_train_band, X_test_band, Y_train, Y_test, groups

    if name == "JapaneseVowels":
        from reservoirpy.datasets import japanese_vowels

        X_train_band, X_test_band, Y_train, Y_test = japanese_vowels(data_folder=download_japanese_vowels_if_needed())
        is_multivariate = True
        groups = None
        sampling_rate = 10000
        Y_train = np.squeeze(np.array(Y_train), axis=1)
        Y_test = np.squeeze(np.array(Y_test), axis=1)
        use_spectral_representation = True
        return use_spectral_representation, is_multivariate, sampling_rate, X_train_band, X_test_band, Y_train, Y_test, groups

    raise ValueError(f"The dataset with name '{name}' is not loadable")
