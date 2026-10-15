#!/bin/bash
# Test evaluation of a dataset on PlaFRIM, one job per function (job array, slurm/test_evaluation.sbatch), then the
# concatenation of their rows into outputs/hag/test_results/test_results_classification_<dataset>.csv. Run from the root
# of a checkout of main:
#   bash slurm/submit_test_evaluation.sh <dataset> [function ...]
set -euo pipefail
DATASET=$1
shift
FUNCTIONS=("$@")
[ ${#FUNCTIONS[@]} -eq 0 ] && FUNCTIONS=(random_ee random_ei ip_correct anti-oja ip-anti-oja mean_hag var_hag)
NODES=(--constraint="zen4|zen3|zonda" --exclude="sirocco[21-25],enbata[01-02]")
cd "$(dirname "$0")/.."
mkdir -p slurm/logs slurm/tasks outputs/hag/test_results/parts

TASKS=slurm/tasks/test_${DATASET}_$(date +%Y%m%d_%H%M%S).txt
printf "%s\n" "${FUNCTIONS[@]}" > "$TASKS"
n=${#FUNCTIONS[@]}
array=$(sbatch --parsable "${NODES[@]}" --array=0-$((n - 1)) --job-name="test-$DATASET" \
    --export=ALL,HAG_TEST_DATASET="$DATASET",HAG_TEST_FUNCTIONS="$TASKS" slurm/test_evaluation.sbatch)
echo "$DATASET: test evaluation, job array $array ($n functions, $TASKS)"

OUT=outputs/hag/test_results/test_results_classification_$DATASET.csv
merge=$(sbatch --parsable "${NODES[@]}" --cpus-per-task=1 --mem=2G --time=00:10:00 --dependency=afterany:$array \
    --job-name="test-merge-$DATASET" --output=slurm/logs/%x-%j.out \
    --wrap="set +u; module load micromamba; export MAMBA_ROOT_PREFIX=\$HOME/micromamba; eval \"\$(micromamba shell hook -s bash)\"; micromamba activate hag_env; python -c \"import glob, pandas as pd; files = sorted(glob.glob('outputs/hag/test_results/parts/${DATASET}_*.csv')); pd.concat([pd.read_csv(f) for f in files]).to_csv('$OUT', index=False); print(len(files), 'functions ->', '$OUT')\"")
echo "$DATASET: concatenation into $OUT, job $merge (after $array)"
