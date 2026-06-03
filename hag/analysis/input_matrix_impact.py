import os
import re
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from hag.analysis.commons import (
    dataset_label_map,
    evaluate_dataset_on_test,
    function_mapping,
    functions_order,
    load_data,
)
from hag.hpo.utility import retrieve_best_model


# ====================== CONFIG ======================
OUTFILE = Path("../../outputs/input_strategy.csv")
IMG_DIR = Path("../../outputs/figures")

DATASETS = ["CatsDogs", "FSDD", "JapaneseVowels"]
DATA_TYPE = "normal"   # or "noisy"
NOISE_STD = 0.001

FUNCTIONS = [
    "random_ee",
    "random_ei",
    "ip_correct",
    "anti-oja_fast",
    "ip-anti-oja_fast",
    "desp",
    "hadsp",
]

# (random_projection_experiment, mapping_label)
MAPPINGS = [
    (True, "random"),
    (False, "modular"),
]

COLUMNS = [
    "Dataset",
    "Function",
    "Mapping",
    "Representation",
    "Average Score",
    "Standard Deviation",
    "Date",
]


# ====================== IO HELPERS ======================
def ensure_output_paths() -> None:
    OUTFILE.parent.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)


def load_results_csv() -> pd.DataFrame:
    """Load existing results, repairing missing columns if needed."""
    if not OUTFILE.exists() or OUTFILE.stat().st_size == 0:
        df = pd.DataFrame(columns=COLUMNS)
        df.to_csv(OUTFILE, index=False)
        return df

    try:
        df = pd.read_csv(OUTFILE)
    except (EmptyDataError, FileNotFoundError):
        df = pd.DataFrame(columns=COLUMNS)

    for col in COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    return df[COLUMNS].copy()


def write_results_csv(existing_df: pd.DataFrame, new_rows: list[dict]) -> None:
    """Atomically append new rows to the results CSV."""
    if not new_rows:
        print("No new studies found; nothing to append.")
        return

    new_df = pd.DataFrame(new_rows).reindex(columns=COLUMNS)
    combined = pd.concat([existing_df, new_df], ignore_index=True)

    tmp_path = OUTFILE.with_suffix(OUTFILE.suffix + ".tmp")
    combined.to_csv(tmp_path, index=False)
    os.replace(tmp_path, OUTFILE)

    print("\n== New results ==")
    print(new_df)
    print(f"\nResults saved to {OUTFILE}.")


def build_existing_keys(df: pd.DataFrame) -> set[tuple[str, str, str, str]]:
    return set(
        df[["Dataset", "Function", "Mapping", "Representation"]]
        .dropna()
        .astype(str)
        .itertuples(index=False, name=None)
    )


# ====================== DATA LOADING ======================
dataset_policy_cache: dict[str, dict[str, bool]] = {}
data_cache: dict[tuple[str, str], tuple] = {}


def _load_dataset(dataset_name: str, representation: str):
    """Thin wrapper around commons.load_data."""
    return load_data(
        dataset_name,
        representation,
        DATA_TYPE,
        NOISE_STD,
        visualize=False,
    )


def get_dataset_policy(dataset_name: str) -> dict[str, bool]:
    """
    Infer dataset policy once and cache it.

    Rules:
    - If 'none' loads -> non-spectral-only
    - Else if 'mfcc' loads -> spectral-only
    - Else -> fail
    """
    if dataset_name in dataset_policy_cache:
        return dataset_policy_cache[dataset_name]

    try:
        out = _load_dataset(dataset_name, "none")
        data_cache[(dataset_name, "none")] = out
        policy = {
            "spectral_only": False,
            "is_instances_classification": bool(out[6]),
        }
        dataset_policy_cache[dataset_name] = policy
        return policy
    except Exception as err_none:
        print(
            f"[policy] {dataset_name}: 'none' failed -> "
            f"{type(err_none).__name__}: {err_none}"
        )

    try:
        out = _load_dataset(dataset_name, "mfcc")
        data_cache[(dataset_name, "mfcc")] = out
        policy = {
            "spectral_only": True,
            "is_instances_classification": bool(out[6]),
        }
        dataset_policy_cache[dataset_name] = policy
        print(f"[policy] {dataset_name}: spectral-only -> only 'mfcc'")
        return policy
    except Exception as err_mfcc:
        raise RuntimeError(
            f"Could not load dataset '{dataset_name}' with 'none' or 'mfcc'. "
            f"'none' failed with {type(err_none).__name__}: {err_none}. "
            f"'mfcc' failed with {type(err_mfcc).__name__}: {err_mfcc}."
        ) from err_mfcc


def get_representations_for_dataset(dataset_name: str) -> list[str]:
    policy = get_dataset_policy(dataset_name)

    if policy["spectral_only"]:
        return ["mfcc"]

    if policy["is_instances_classification"]:
        return ["mfcc", "custom", "none"]

    return ["stft", "mfcc", "custom", "none"]


def get_loaded_data(dataset_name: str, representation: str):
    """Load once per (dataset, representation) in this run."""
    key = (dataset_name, representation)
    if key not in data_cache:
        data_cache[key] = _load_dataset(dataset_name, representation)
    return data_cache[key]


# ====================== SKIP LOGIC ======================
def expected_keys_for_representation(
    dataset_name: str,
    representation: str,
) -> set[tuple[str, str, str, str]]:
    return {
        (str(dataset_name), str(function_name), str(mapping_label), str(representation))
        for _, mapping_label in MAPPINGS
        for function_name in FUNCTIONS
    }


def representation_already_done(
    dataset_name: str,
    representation: str,
    existing_keys: set[tuple[str, str, str, str]],
) -> bool:
    """
    If all (function, mapping) combinations already exist for this
    dataset/representation, do not call load_data at all.
    """
    return expected_keys_for_representation(
        dataset_name,
        representation,
    ).issubset(existing_keys)


# ====================== EVALUATION ======================
def format_scores(avg: float, std: float, is_instances_classification: bool) -> tuple[str, str]:
    if is_instances_classification:
        return f"{avg * 100:.5f} %", f"± {std * 100:.5f} %"
    return f"{avg:.5f}", f"± {std:.5f}"


def evaluate_one_configuration(
    dataset_name: str,
    representation: str,
    function_name: str,
    random_projection_experiment: bool,
    mapping_label: str,
    loaded_data: tuple,
) -> dict | None:
    (
        pretrain_data,
        train_data,
        test_data,
        y_train,
        y_test,
        is_multivariate,
        is_instances_classification,
    ) = loaded_data

    prefix =  f"rdn-proj_tpe_{representation}" if random_projection_experiment  else f"mod-proj_tpe_{representation}"

    study = retrieve_best_model(
        function_name=function_name,
        dataset_name=dataset_name,
        is_multivariate=is_multivariate,
        variate_type="multi",
        data_type=DATA_TYPE,
        prefix=prefix,
        db_dir="../../",  # or any folder you want
    )
    if study is None:
        print(f"No study found for {dataset_name} | {representation} | {function_name} | {mapping_label}")
        return None

    scores = evaluate_dataset_on_test(
        study=study,
        dataset_name=dataset_name,
        function_name=function_name,
        pretrain_data=pretrain_data,
        train_data=train_data,
        test_data=test_data,
        Y_train=y_train,
        Y_test=y_test,
        is_instances_classification=is_instances_classification,
        nb_trials=8,
        record_metrics=False,
        random_projection_experiment=random_projection_experiment,
    )

    avg = float(np.mean(scores))
    std = float(np.std(scores))
    avg_fmt, std_fmt = format_scores(avg, std, is_instances_classification)

    return {
        "Dataset": dataset_name,
        "Function": function_name,
        "Mapping": mapping_label,
        "Representation": representation,
        "Average Score": avg_fmt,
        "Standard Deviation": std_fmt,
        "Date": datetime.now().strftime("%Y-%m-%d"),
    }


# ====================== VISUALIZATION ======================
_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def to_num(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace("±", "").replace("%", "").strip()
    m = _NUM_RE.search(s)
    return float(m.group(0)) if m else np.nan


def luminance(rgb):
    r, g, b = rgb[:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def safe_slug(value: str) -> str:
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "rep"


def create_heatmaps() -> None:
    if not OUTFILE.exists():
        return

    df = pd.read_csv(OUTFILE)
    df.columns = [c.strip() for c in df.columns]

    if "Random Input Mapping" in df.columns and "Mapping" not in df.columns:
        df = df.rename(columns={"Random Input Mapping": "Mapping"})

    for col in ["Dataset", "Function", "Mapping", "Representation"]:
        if col not in df.columns:
            df[col] = np.nan

    df["Mapping"] = df["Mapping"].astype(str).str.strip().str.lower()
    df["Score"] = df["Average Score"].apply(to_num)

    reps = sorted(df["Representation"].dropna().unique())

    mpl.rcParams.update({"figure.dpi": 150, "savefig.dpi": 600})

    for rep in reps:
        dfr = df[df["Representation"] == rep].copy()
        if dfr.empty:
            continue

        wide = (
            dfr.pivot_table(
                index=["Dataset", "Function"],
                columns="Mapping",
                values="Score",
                aggfunc="mean",
            )
            .reset_index()
        )

        if {"modular", "random"}.issubset(wide.columns):
            wide["delta"] = wide["modular"] - wide["random"]
        else:
            wide["delta"] = np.nan

        dataset_order = [d for d in ["JapaneseVowels", "CatsDogs", "FSDD"] if d in wide["Dataset"].unique()]
        if not dataset_order:
            dataset_order = list(wide["Dataset"].unique())

        wide["Function"] = wide["Function"].map(function_mapping).fillna(wide["Function"])

        ordered_functions = [f for f in functions_order if f in wide["Function"].unique()]
        if not ordered_functions:
            ordered_functions = sorted(wide["Function"].unique())

        heat = (
            wide.pivot_table(index="Function", columns="Dataset", values="delta")
            .reindex(index=ordered_functions, columns=dataset_order)
        )

        col_labels = [dataset_label_map.get(d, d) for d in heat.columns]
        row_labels = list(heat.index)

        v = np.nanmax(np.abs(heat.values))
        if not np.isfinite(v) or v == 0:
            v = 1.0

        norm = mpl.colors.TwoSlopeNorm(vmin=-v, vcenter=0.0, vmax=v)
        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad("#eeeeee")

        data = np.ma.masked_invalid(heat.values)

        fig, ax = plt.subplots(
            figsize=(1.9 * len(col_labels) + 2.8, 0.55 * len(row_labels) + 2.5),
            layout="constrained",
        )
        im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")
        ax.set_title(f"Δ (Modular − Random) — Representation: {rep}", pad=10)

        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_xticklabels(col_labels)
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels)

        ax.set_xticks(np.arange(-0.5, data.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, data.shape[0], 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)

        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = heat.values[i, j]
                if np.isnan(val):
                    continue
                rgba = cmap(norm(val))
                txt_color = "white" if luminance(rgba) < 0.45 else "black"
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=10, color=txt_color)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ticks = np.linspace(-v, v, 5)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{t:.2f}" for t in ticks])
        cbar.set_label("Δ (Modular − Random)")

        pdf_out = IMG_DIR / f"heatmap_deltas_{safe_slug(rep)}.pdf"
        fig.savefig(pdf_out, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {pdf_out}")

        by_ds = wide.groupby("Dataset")["delta"].mean().reindex(dataset_order)
        overall = wide["delta"].mean()

        print(f"\n[{rep}] Average gain of Modular over Random:")
        for ds, val in by_ds.items():
            print(f"  {dataset_label_map.get(ds, ds)}: {val:+.2f}")
        print(f"  Overall: {overall:+.2f}")


# ====================== MAIN ======================
if __name__ == "__main__":
    ensure_output_paths()

    previous_results = load_results_csv()
    existing_keys = build_existing_keys(previous_results)
    new_rows: list[dict] = []

    for dataset_name in DATASETS:
        representations = get_representations_for_dataset(dataset_name)

        for representation in representations:
            print(f"\n===== DATASET: {dataset_name} | REP: {representation} =====")

            # IMPORTANT:
            # Skip before calling load_data(...), so datasets are not rebuilt if already done.
            if representation_already_done(dataset_name, representation, existing_keys):
                print("All combinations already in CSV -> skipping load_data and evaluation.")
                continue

            loaded_data = get_loaded_data(dataset_name, representation)

            for random_projection_experiment, mapping_label in MAPPINGS:
                print(f"--- Mapping: {mapping_label} ---")

                for function_name in FUNCTIONS:
                    key = (
                        str(dataset_name),
                        str(function_name),
                        str(mapping_label),
                        str(representation),
                    )

                    if key in existing_keys:
                        print(f"Skipping already computed: {key}")
                        continue

                    print(f"Evaluating: {function_name}")

                    row = evaluate_one_configuration(
                        dataset_name=dataset_name,
                        representation=representation,
                        function_name=function_name,
                        random_projection_experiment=random_projection_experiment,
                        mapping_label=mapping_label,
                        loaded_data=loaded_data,
                    )

                    if row is None:
                        continue

                    new_rows.append(row)
                    existing_keys.add(key)

    write_results_csv(previous_results, new_rows)
    create_heatmaps()

