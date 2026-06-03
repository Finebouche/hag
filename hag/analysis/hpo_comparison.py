import os
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from hag.analysis.commons import (
    evaluate_dataset_on_test,
    function_mapping,
    functions_order,
    function_colors,
    load_data,
)
from hag.hpo.utility import retrieve_best_model


columns = ["Dataset", "Function", "Sampler", "Average Score", "Standard Deviation", "Date"]
variate_type = "multi"
file_name = "../../outputs/hpo_strategy.csv"
figures_path = "../../outputs/figures/hpo_comparaison.pdf"

DATASETS = ["JapaneseVowels"]
SAMPLERS = ["cmaes", "tpe"]
FUNCTIONS = ["random_ee", "random_ei", "desp", "hadsp", "ip-anti-oja_fast", "anti-oja_fast", "ip_correct"]


def ensure_results_file(path: str) -> pd.DataFrame:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path) and os.path.getsize(path) > 0:
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=columns)
        df.to_csv(path, index=False)

    for col in columns:
        if col not in df.columns:
            df[col] = np.nan

    return df[columns].copy()


def build_existing_keys(df: pd.DataFrame) -> set[tuple[str, str, str]]:
    return set(
        df[["Dataset", "Function", "Sampler"]]
        .dropna()
        .astype(str)
        .itertuples(index=False, name=None)
    )


def format_scores(avg: float, std: float, is_instances_classification: bool) -> tuple[str, str]:
    if is_instances_classification:
        return f"{avg * 100:.5f} %", f"± {std * 100:.5f} %"
    return f"{avg:.5f}", f"± {std:.5f}"


previous_results = ensure_results_file(file_name)
existing_keys = build_existing_keys(previous_results)
all_new_rows = []

for dataset_name in DATASETS:
    dataset_keys = {
        (str(dataset_name), str(function_name), str(sampler_name))
        for sampler_name in SAMPLERS
        for function_name in FUNCTIONS
    }

    # Skip the whole dataset if everything is already computed
    if dataset_keys.issubset(existing_keys):
        print(f"===== DATASET: {dataset_name} =====")
        print("All sampler/function combinations already present -> skipping.")
        continue

    print(f"===== DATASET: {dataset_name} =====")
    spectral_representation = (
        "mfcc"
        if dataset_name in ["CatsDogs", "FSDD", "JapaneseVowels", "SPEECHCOMMANDS", "SpokenArabicDigits"]
        else "stft"
    )

    # Load data only if at least one result is missing
    (
        pretrain_data,
        train_data,
        test_data,
        Y_train,
        Y_test,
        is_multivariate,
        is_instances_classification,
    ) = load_data(dataset_name, spectral_representation, visualize=True)

    for sampler_name in SAMPLERS:
        print(f"--- Sampler: {sampler_name} ---")

        for function_name in FUNCTIONS:
            key = (str(dataset_name), str(function_name), str(sampler_name))

            if key in existing_keys:
                print(f"Skipping already computed: {key}")
                continue

            print("Function:", function_name)

            study = retrieve_best_model(
                function_name=function_name,
                dataset_name=dataset_name,
                is_multivariate=is_multivariate,
                variate_type=variate_type,
                data_type="normal",
                prefix=sampler_name,
                db_dir="../../",
            )

            scores = evaluate_dataset_on_test(
                study,
                dataset_name,
                function_name,
                pretrain_data,
                train_data,
                test_data,
                Y_train,
                Y_test,
                is_instances_classification,
                record_metrics=False,
            )

            average_score = float(np.mean(scores))
            std_deviation = float(np.std(scores))
            formatted_average, formatted_std = format_scores(
                average_score,
                std_deviation,
                is_instances_classification,
            )

            all_new_rows.append(
                {
                    "Dataset": dataset_name,
                    "Function": function_name,
                    "Sampler": sampler_name,
                    "Average Score": formatted_average,
                    "Standard Deviation": formatted_std,
                    "Date": datetime.now().strftime("%Y-%m-%d"),
                }
            )

            existing_keys.add(key)

if all_new_rows:
    new_results = pd.DataFrame(all_new_rows).reindex(columns=columns)
    print("\n== New results ==")
    print(new_results)

    total_results = pd.concat([previous_results, new_results], ignore_index=True)

    tmp_file = file_name + ".tmp"
    total_results.to_csv(tmp_file, index=False)
    os.replace(tmp_file, file_name)

    print(f"Results saved to {file_name}.\n")
else:
    print("No missing test scores to compute.")

# ===================== PLOT =====================
df = pd.read_csv(file_name)

df["Average Score"] = pd.to_numeric(
    df["Average Score"].astype(str).str.replace("%", "", regex=False).str.strip(),
    errors="coerce",
)
df["Standard Deviation"] = pd.to_numeric(
    df["Standard Deviation"]
    .astype(str)
    .str.replace("±", "", regex=False)
    .str.replace("%", "", regex=False)
    .str.strip(),
    errors="coerce",
)

df["Function"] = df["Function"].replace(function_mapping)
df["Dataset"] = df["Dataset"].replace({
    "JapaneseVowels": "Japanese\nVowels",
})

sampler_hatching = {
    "tpe": "",
    "cmaes": "///",
}

functions = [f for f in functions_order if f in df["Function"].unique()]
samplers = ["tpe", "cmaes"]

datasets = df["Dataset"].unique()
datasets.sort()

x = np.arange(len(datasets))
width = 0.08

fig, ax = plt.subplots(figsize=(12, 6))
fontsize = 12

for i, func in enumerate(functions):
    for j, sampler in enumerate(samplers):
        offset = i * (2 * width) + j * width
        sub_df = df[(df["Function"] == func) & (df["Sampler"] == sampler)]
        merged = pd.DataFrame({"Dataset": datasets}).merge(sub_df, on="Dataset", how="left")

        ax.bar(
            x + offset,
            merged["Average Score"],
            width,
            yerr=merged["Standard Deviation"],
            capsize=4,
            color=function_colors.get(func, "gray"),
            hatch=sampler_hatching.get(sampler, ""),
            edgecolor="black",
        )

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="both", labelsize=fontsize)
ax.set_xlabel("Dataset", fontsize=fontsize)
ax.set_ylabel("Average Score", fontsize=fontsize)

total_functions = len(functions)
total_samplers = len(samplers)
group_width = total_functions * total_samplers * width
ax.set_xticks(x + group_width / 2 - (width / 2))
ax.set_xticklabels(datasets, rotation=0)

function_legend = [
    Patch(facecolor=function_colors[f], edgecolor="black", label=f) for f in functions
]
sampler_legend = [
    Patch(facecolor="white", edgecolor="black", hatch=sampler_hatching[s], label=s.upper())
    for s in samplers
]

first_legend = ax.legend(handles=function_legend, title="Function", loc="upper left", fontsize=fontsize)
ax.add_artist(first_legend)
ax.legend(handles=sampler_legend, title="Sampler", loc="lower left", fontsize=fontsize)

plt.tight_layout()
os.makedirs(os.path.dirname(figures_path), exist_ok=True)
fig.savefig(figures_path, bbox_inches="tight")
plt.show()