"""
Certainty of a readout along the phrases of the test songs, shared by canary.next_phrase_prediction.certainty (next
phrase) and canary.syllables_classification.certainty (class of the syllable). On the frames of every phrase:
  - true_output: output of the readout unit of the true class;
  - margin: true_output minus the largest output of the other classes (> 0: frame correctly classified);
  - correct: the frame is correctly classified.
These are averaged per phrase in bins of the relative position in the phrase (0 = onset, 1 = end of its frames), and of
the time since the onset, then over the phrases. For each phrase, the mean true_output of its last third minus that of
its first third (Wilcoxon signed-rank test over the phrases): > 0 if the unit of the class is more active at the end.

Reservoirs: best parameters of the studies of the conditions of the task (canary.evaluate.conditions), ridge of the
best trial of the study (or chosen by the cross-validation of the task). Results: outputs/canary/<name>_<size>.csv (bins),
<name>_<size>_phrases.csv (first / last third per condition and seed) and <name>_<size>.png.
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from canary.common import OUTPUTS

RELATIVE_BINS = 10
ONSET_BIN = 0.05  # s
ONSET_MAX = 1.0  # s
MIN_FRAMES = 6  # phrases shorter than this (in frames) are skipped
COLORS = {"mean_hag": "#2a78d6", "var_hag": "#eb6834", "random_ee": "#1baf7a", "random_ee_dense": "#eda100",
          "random_ei": "#4a3aa7"}


def curve(out, c, label, frame):
    """Curves of the true output, margin and correctness of the frames out (frames, classes) of a phrase, true class c,
    with the relative position and onset time bins of the frames."""
    true = out[:, c]
    margin = true - np.delete(out, c, axis=1).max(axis=1)
    relative = (np.arange(len(out)) + 0.5) / len(out)
    return dict(label=label, true=true, margin=margin, correct=margin > 0,
                relative=np.minimum((relative * RELATIVE_BINS).astype(int), RELATIVE_BINS - 1),
                onset=(np.arange(len(out)) * frame / ONSET_BIN).astype(int))


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


def evaluate_conditions(conditions, size, wanted, nb_seeds, songs, train_songs, test_songs, frame, classes, select_ridge,
                        run_seed):
    """(bins, phrases) of the conditions wanted of size neurons (conditions(size) of the task), for nb_seeds seeds:
    run_seed(condition, model, params, size, ridge, songs, train_songs, test_songs, frame, classes, seed) ->
    summarize(...) of the task; the ridge of the conditions without study is chosen by select_ridge(model, params, size,
    songs, train_songs, frame, classes)."""
    available = {c[0]: c for c in conditions(size)}
    bins, phrases = [], []
    for condition in wanted:
        if condition not in available:
            print(f"{condition} ({size}): no study, skipped")
            continue
        _, model, params, ridge, _ = available[condition]
        if ridge is None:
            ridge, _ = select_ridge(model, params, size, songs, train_songs, frame, classes)
        for seed in range(nb_seeds):
            rows, summary = run_seed(condition, model, params, size, ridge, songs, train_songs, test_songs, frame,
                                     classes, seed)
            bins += rows
            phrases.append(summary)
            print(f"{condition} ({size}) seed {seed}: true-class output first third {summary['first_third']:.3f} -> "
                  f"last third {summary['last_third']:.3f} (rising in {100 * summary['share_rising']:.0f} % of "
                  f"{summary['n_phrases']} phrases, Wilcoxon p {summary['wilcoxon_p']:.1e}); accuracy "
                  f"{100 * summary['accuracy_first_third']:.1f} -> {100 * summary['accuracy_last_third']:.1f} %",
                  flush=True)
    return bins, phrases


def save(bins, phrases, name, size, title, labels):
    """outputs/canary/<name>_<size>.csv, ..._phrases.csv and .png (nothing if no condition was evaluated)."""
    if not phrases:
        sys.exit("No condition evaluated: nothing saved")
    bins, phrases = pd.DataFrame(bins), pd.DataFrame(phrases)
    stem = f"{name}_{size}"
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    bins.to_csv(OUTPUTS / f"{stem}.csv", index=False)
    phrases.to_csv(OUTPUTS / f"{stem}_phrases.csv", index=False)
    # mean true output (labels[0]) and accuracy (labels[1]) along the phrases (relative position), per condition
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = bins[bins.axis == "relative"].groupby(["condition", "position"])[["true_output", "accuracy"]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), facecolor="#fcfcfb")
    for ax, column, title_ax in ((axes[0], "true_output", labels[0]), (axes[1], "accuracy", labels[1])):
        ax.set_facecolor("#fcfcfb")
        for condition in data.index.get_level_values(0).unique():
            values = data.loc[condition, column]
            color = COLORS.get(condition, "#4a3aa7")
            ax.plot(values.index, values.values, color=color, linewidth=2, marker="o", markersize=4, label=condition)
            ax.annotate(condition, (values.index[-1], values.values[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8, color="#0b0b0b")
        ax.set_title(title_ax, fontsize=10, color="#0b0b0b", loc="left")
        ax.set_xlabel("Relative position in the phrase (0 = onset, 1 = end)", fontsize=8, color="#52514a")
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
    fig.suptitle(f"{title}, test songs ({size} neurons, mean over seeds)", fontsize=11,
                 color="#0b0b0b", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUTPUTS / f"{stem}.png", dpi=150)
    plt.close(fig)
    print("Saved to", OUTPUTS / f"{stem}.csv", OUTPUTS / f"{stem}_phrases.csv", OUTPUTS / f"{stem}.png")
