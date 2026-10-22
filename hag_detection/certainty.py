"""
Does the certainty of the syllable classification rise along a phrase? After training the readout of the syllable
classification (hag_detection.syllables), on every phrase of the test songs (merged annotated segments of one syllable
type, SIL, TRASH and calls excluded: songs[w]["phrases"]), frame by frame:
  - true_output: output of the readout unit of the true class;
  - margin: true_output minus the largest output of the other classes (> 0: frame correctly classified);
  - correct: the frame is correctly classified (canapy's frame accuracy).
These are averaged per phrase in bins of the relative position in the phrase (0 = onset, 1 = offset), and of the time
since the onset, then over the phrases. For each phrase, the mean true_output of its last third minus that of its first
third (Wilcoxon signed-rank test over the phrases): > 0 if the unit of the class is more active at the end.

Reservoirs: best parameters of the studies of hag_detection.evaluate.conditions (syllable studies with
HAG_DETECTION_TASK=syllables, boundary studies otherwise), ridge of the best trial of the study (or chosen by the
cross-validation of hag_detection.syllables). Results: outputs/detection/certainty_<size>.csv (bins),
certainty_<size>_phrases.csv (first / last third per condition and seed) and certainty_<size>.png.

Usage: python -m hag_detection.certainty [size] [conditions] [nb_seeds] [max_songs]
  (default 1000, mean_hag,var_hag,random_ee,random_ee_dense, 3 seeds)
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from hag_detection.common import OUTPUTS, load_dataset, pretrain_phrases
from hag_detection.evaluate import conditions
from hag_detection.models import build, normal_equations, ridge_readout, state_chunks
from hag_detection.syllables import add_frame_labels, choose_ridge

RELATIVE_BINS = 10
ONSET_BIN = 0.05  # s
ONSET_MAX = 1.0  # s
MIN_FRAMES = 6  # phrases shorter than this (in frames) are skipped
COLORS = {"mean_hag": "#2a78d6", "var_hag": "#eb6834", "random_ee": "#1baf7a", "random_ee_dense": "#eda100"}


def phrase_curves(output, labels_index, phrases, frame):
    """Per phrase: (relative position bins, onset time bins) of the true output, margin and correctness."""
    curves = []
    for start, end, label in phrases:
        f0, f1 = int(np.floor(start / frame)), min(int(np.floor(end / frame)), len(output))
        if f1 - f0 < MIN_FRAMES or label not in labels_index:
            continue
        c = labels_index[label]
        out = output[f0:f1]
        true = out[:, c]
        others = np.delete(out, c, axis=1).max(axis=1)
        margin = true - others
        relative = (np.arange(f1 - f0) + 0.5) / (f1 - f0)
        since_onset = np.arange(f1 - f0) * frame
        curves.append(dict(label=label, duration=(f1 - f0) * frame, true=true, margin=margin, correct=margin > 0,
                           relative=np.minimum((relative * RELATIVE_BINS).astype(int), RELATIVE_BINS - 1),
                           onset=(since_onset / ONSET_BIN).astype(int)))
    return curves


def summarize(curves, condition, seed):
    """Rows of the bins (mean over the phrases of the per-phrase bin means) and the first / last third per phrase."""
    rows = []
    for axis, n_bins in (("relative", RELATIVE_BINS), ("onset", int(ONSET_MAX / ONSET_BIN))):
        for b in range(n_bins):
            per_phrase = [(c["true"][c[axis] == b].mean(), c["margin"][c[axis] == b].mean(),
                           c["correct"][c[axis] == b].mean()) for c in curves if np.any(c[axis] == b)]
            if not per_phrase:
                continue
            true, margin, correct = np.array(per_phrase).T
            position = (b + 0.5) / n_bins if axis == "relative" else (b + 0.5) * ONSET_BIN
            rows.append(dict(condition=condition, seed=seed, axis=axis, position=position, true_output=true.mean(),
                             margin=margin.mean(), accuracy=correct.mean(), n_phrases=len(per_phrase)))
    first = np.array([c["true"][c["relative"] < RELATIVE_BINS / 3].mean() for c in curves])
    last = np.array([c["true"][c["relative"] >= 2 * RELATIVE_BINS / 3].mean() for c in curves])
    first_correct = np.array([c["correct"][c["relative"] < RELATIVE_BINS / 3].mean() for c in curves])
    last_correct = np.array([c["correct"][c["relative"] >= 2 * RELATIVE_BINS / 3].mean() for c in curves])
    phrases = dict(condition=condition, seed=seed, n_phrases=len(curves), first_third=first.mean(),
                   last_third=last.mean(), last_minus_first=(last - first).mean(),
                   share_rising=float(np.mean(last > first)), wilcoxon_p=wilcoxon(last - first).pvalue,
                   accuracy_first_third=first_correct.mean(), accuracy_last_third=last_correct.mean())
    return rows, phrases


def run(condition, model, params, size, ridge, songs, train_songs, test_songs, frame, classes, seed):
    np.random.seed(seed)
    reservoirs = build(model, params, size, pretrain_phrases(songs, train_songs, frame))
    Wout = ridge_readout(normal_equations(reservoirs, songs, train_songs, target="Y"), ridge)
    labels_index = {label: k for k, label in enumerate(classes)}
    curves = []
    for part, part_states in state_chunks(reservoirs, songs, test_songs):
        for x, w in zip(part_states, part):
            output = x.astype(np.float64) @ Wout[:-1] + Wout[-1]
            curves += phrase_curves(output, labels_index, songs[w]["phrases"], frame)
    return summarize(curves, condition, seed)


def plot(bins, size, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = bins[bins.axis == "relative"].groupby(["condition", "position"])[["true_output", "accuracy"]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), facecolor="#fcfcfb")
    for ax, column, title in ((axes[0], "true_output", "Output of the true-class unit"),
                              (axes[1], "accuracy", "Frame accuracy")):
        ax.set_facecolor("#fcfcfb")
        for condition in data.index.get_level_values(0).unique():
            curve = data.loc[condition, column]
            color = COLORS.get(condition, "#4a3aa7")
            ax.plot(curve.index, curve.values, color=color, linewidth=2, marker="o", markersize=4, label=condition)
            ax.annotate(condition, (curve.index[-1], curve.values[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8, color="#0b0b0b")
        ax.set_title(title, fontsize=10, color="#0b0b0b", loc="left")
        ax.set_xlabel("Relative position in the phrase (0 = onset, 1 = offset)", fontsize=8, color="#52514a")
        ax.set_xlim(0, 1.25)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c3c2b7")
        ax.tick_params(colors="#52514a", labelsize=8)
    axes[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle(f"Certainty along the phrases, test songs ({size} neurons, mean over seeds)", fontsize=11,
                 color="#0b0b0b", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    wanted = sys.argv[2].split(",") if len(sys.argv) > 2 else list(COLORS)
    nb_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    max_songs = int(sys.argv[4]) if len(sys.argv) > 4 else None

    songs, train_songs, test_songs, frame = load_dataset(max_songs)
    classes = add_frame_labels(songs, frame)
    available = {c[0]: c for c in conditions(size)}
    bins, phrases = [], []
    for condition in wanted:
        if condition not in available:
            print(f"{condition} ({size}): no study, skipped")
            continue
        _, model, params, ridge, _ = available[condition]
        if ridge is None:
            ridge, _ = choose_ridge(model, params, size, songs, train_songs, frame, classes)
        for seed in range(nb_seeds):
            rows, summary = run(condition, model, params, size, ridge, songs, train_songs, test_songs, frame, classes,
                                seed)
            bins += rows
            phrases.append(summary)
            print(f"{condition} ({size}) seed {seed}: true-class output first third {summary['first_third']:.3f} -> "
                  f"last third {summary['last_third']:.3f} (rising in {100 * summary['share_rising']:.0f} % of "
                  f"{summary['n_phrases']} phrases, Wilcoxon p {summary['wilcoxon_p']:.1e}); frame accuracy "
                  f"{100 * summary['accuracy_first_third']:.1f} -> {100 * summary['accuracy_last_third']:.1f} %",
                  flush=True)

    bins, phrases = pd.DataFrame(bins), pd.DataFrame(phrases)
    suffix = f"_{size}" if max_songs is None else f"_{size}_check"
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    bins.to_csv(OUTPUTS / f"certainty{suffix}.csv", index=False)
    phrases.to_csv(OUTPUTS / f"certainty{suffix}_phrases.csv", index=False)
    plot(bins, size, OUTPUTS / f"certainty{suffix}.png")
    print("Saved to", OUTPUTS / f"certainty{suffix}.csv", OUTPUTS / f"certainty{suffix}_phrases.csv",
          OUTPUTS / f"certainty{suffix}.png")

