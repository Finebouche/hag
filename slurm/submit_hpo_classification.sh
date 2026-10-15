#!/bin/bash
# Hyperparameter optimizations of a classification dataset on PlaFRIM, one job per function (job array), then the merge
# of their databases into the one of the dataset (hag/hpo/corrected_studies). Run from the root of a checkout of main:
#   bash slurm/submit_hpo_classification.sh <dataset> [function ...]
# Default functions: those of hag/hpo/hpo_esn.py.
set -euo pipefail
DATASET=$1
shift
FUNCTIONS=("$@")
[ ${#FUNCTIONS[@]} -eq 0 ] && FUNCTIONS=(random_ee random_ei var_hag mean_hag ip_correct anti-oja ip-anti-oja)
NODES=(--constraint="zen4|zen3|zonda" --exclude="sirocco[21-25],enbata[01-02]")
cd "$(dirname "$0")/.."
mkdir -p slurm/logs slurm/tasks

TASKS=slurm/tasks/hpo_${DATASET}_$(date +%Y%m%d_%H%M%S).txt
printf "%s\n" "${FUNCTIONS[@]}" > "$TASKS"
n=${#FUNCTIONS[@]}
hpo=$(sbatch --parsable "${NODES[@]}" --array=0-$((n - 1)) --job-name="hpo-$DATASET" \
    --export=ALL,HAG_HPO_DATASET="$DATASET",HAG_HPO_FUNCTIONS="$TASKS" slurm/hpo_classification.sbatch)
echo "$DATASET: job array $hpo ($n functions, $TASKS)"

merge=$(sbatch --parsable "${NODES[@]}" --cpus-per-task=1 --mem=4G --time=00:30:00 --dependency=afterany:$hpo \
    --job-name="merge-$DATASET" --output=slurm/logs/%x-%j.out \
    --wrap="set +u; module load micromamba; export MAMBA_ROOT_PREFIX=\$HOME/micromamba; eval \"\$(micromamba shell hook -s bash)\"; micromamba activate hag_env; set -u; PYTHONPATH=\$PWD python -m hag.hpo.merge_studies $DATASET")
echo "$DATASET: merge of the databases, job $merge (after $hpo)"
