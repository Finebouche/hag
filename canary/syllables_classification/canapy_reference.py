"""
canapy (https://github.com/birds-canopy/canapy) on our split of the canary songs: its default model for canaries
(config/default, the same as its canary preset: 1000 units, leak 0.1, spectral radius 0.4, input scaling per feature
group, ridge 1e-8) trained on our train songs, and its frame accuracy (canapy.metrics.frame_error_rate: raw frame
predictions, all the frames, silence included) on our test songs, for its annotators syn, nsyn and their ensemble.

Two steps, in two environments:
  python -m canary.syllables_classification.canapy_reference split <directory>
      (hag environment) our train and test songs (canary.common.load_songs) as two directories of links to
      their audio and annotation files: <directory>/{train,test}/{audio,annotations}
  python canary/syllables_classification/canapy_reference.py run <directory> [seeds]
      (canapy environment, no hag import) training and frame accuracy, results in <directory>/canapy_frame_accuracy.csv
canapy keeps at least one song of its training corpus for its own internal test (test_ratio = 0: 1 song).
"""
import sys
from pathlib import Path


def split(directory):
    from hag.datasets.load_categorical_forecasting import download_canary_if_needed
    import pandas as pd

    annotations_dir, audio_dir = download_canary_if_needed()
    _, train_songs, test_songs = _split_only()
    wavs = {p.name: p for p in audio_dir.rglob("*.wav")}
    csvs = {}
    for f in annotations_dir.rglob("*.csv"):
        for wave in pd.read_csv(f, index_col=0)["wave"].unique():
            csvs[wave] = f
    for name, waves in (("train", train_songs), ("test", test_songs)):
        for kind in ("audio", "annotations"):
            (directory / name / kind).mkdir(parents=True, exist_ok=True)
        for wave in waves:
            for source, kind in ((wavs[wave], "audio"), (csvs[wave], "annotations")):
                link = directory / name / kind / source.name
                if not link.exists():
                    link.symlink_to(source.resolve())
        print(f"{name}: {len(waves)} songs in {directory / name}")


def _split_only():
    """Train and test songs of canary.common.load_songs without loading the audio."""
    import pandas as pd
    from hag.datasets.load_categorical_forecasting import (build_phrase_sequences, download_canary_if_needed,
                                                           split_canary_songs)
    annotations_dir, _ = download_canary_if_needed()
    annotations = pd.concat([pd.read_csv(f, index_col=0) for f in sorted(annotations_dir.rglob("*.csv"))])
    return None, *split_canary_songs(build_phrase_sequences(annotations)["wave"])


def run(directory, seeds):
    import numpy as np
    import pandas as pd
    from canapy.annotator.ensemble import Ensemble
    from canapy.annotator.nsynannotator import NSynAnnotator
    from canapy.annotator.synannotator import SynAnnotator
    from canapy.corpus import Corpus
    from canapy.metrics import frame_error_rate
    from config import default_config

    rows = []
    for seed in seeds:
        # canapy's Config returns copies of its sections on attribute access (config.misc.seed = ... would change a
        # copy): the values are set by key, in default_config itself, which the corpora use too
        config = default_config
        config["misc"]["seed"] = seed
        config["transforms"]["training"]["test_ratio"] = 0.0  # train on our train songs (canapy keeps one for itself)
        assert config.misc.seed == seed and config.transforms.training.test_ratio == 0.0
        train = Corpus.from_directory(audio_directory=directory / "train" / "audio",
                                      annots_directory=directory / "train" / "annotations",
                                      spec_directory=directory / "spec_train")
        test = Corpus.from_directory(audio_directory=directory / "test" / "audio",
                                     annots_directory=directory / "test" / "annotations",
                                     spec_directory=directory / "spec_test")
        predictions = {}
        for name, annotator in (("syn", SynAnnotator(config)), ("nsyn", NSynAnnotator(config))):
            annotator.fit(train)
            predictions[name] = annotator.predict(test, return_raw=True)
            accuracy = 1 - frame_error_rate(test, predictions[name])
            rows.append(dict(annotator=name, seed=seed, frame_accuracy=accuracy))
            print(f"seed {seed} {name}: frame accuracy {100 * accuracy:.2f} %", flush=True)
        try:
            ensemble = Ensemble(config)
            ensemble.fit(train)
            prediction = ensemble.predict([predictions["syn"], predictions["nsyn"]])
            accuracy = 1 - frame_error_rate(test, prediction)
            rows.append(dict(annotator="ensemble", seed=seed, frame_accuracy=accuracy))
            print(f"seed {seed} ensemble: frame accuracy {100 * accuracy:.2f} %", flush=True)
        except Exception as error:  # the ensemble is a bonus: report and keep the two annotators
            print(f"seed {seed} ensemble failed: {error!r}", flush=True)
        pd.DataFrame(rows).to_csv(directory / "canapy_frame_accuracy.csv", index=False)
    results = pd.DataFrame(rows)
    summary = results.groupby("annotator").frame_accuracy.agg(["mean", "std"]).mul(100).round(2)
    summary["seeds"] = results.groupby("annotator").size()
    print(summary)


if __name__ == "__main__":
    command, directory = sys.argv[1], Path(sys.argv[2]).resolve()
    if command == "split":
        split(directory)
    elif command == "run":
        run(directory, [int(s) for s in sys.argv[3].split(",")] if len(sys.argv) > 3 else [42])
    else:
        raise ValueError("command: split or run")
