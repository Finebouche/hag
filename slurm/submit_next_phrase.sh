#!/bin/bash
# Prediction of the next phrase of canary songs (next_phrase_prediction.prediction) on PlaFRIM, one job per condition
# (reservoirs of the next-phrase studies, outputs/canary/studies), then the merge of their results and the paired
# comparisons. Run from the root of a checkout of the detection branch:
#   HAG_CANARY_TASK=next_phrase [HAG_CANARY_LEAK=1] bash slurm/submit_next_phrase.sh [sizes] [nb_seeds]   (default 500,1000 and 10 seeds)
set -euo pipefail
SIZES=${1:-500,1000}
NB_SEEDS=${2:-10}
MAX_RUNNING=${MAX_RUNNING:-12}
NODES=(--constraint="zen4|zen3|zonda" --exclude="sirocco[21-25],enbata[01-02]")
cd "$(dirname "$0")/.."
mkdir -p slurm/logs slurm/tasks

set +u
module load micromamba
export MAMBA_ROOT_PREFIX=$HOME/micromamba
eval "$(micromamba shell hook -s bash)"
micromamba activate hag_env
set -u
TASKS=slurm/tasks/next_phrase_$(date +%Y%m%d_%H%M%S).txt
HAG_CANARY_STUDIES=$PWD/outputs/canary/studies PYTHONPATH=$PWD \
    python -m next_phrase_prediction.prediction list "$SIZES" 2>/dev/null | grep -E "^[^ ]+ [0-9]+$" > "$TASKS"
n=$(wc -l < "$TASKS")

array=$(sbatch --parsable "${NODES[@]}" --array=0-$((n - 1))%$MAX_RUNNING --time=1-00:00:00 \
    --job-name=canary-next-phrase --export=ALL,HAG_CANARY_MODE=next_phrase,HAG_CANARY_WORKERS=1,HAG_CANARY_TASKS="$TASKS" \
    slurm/canary.sbatch "$NB_SEEDS")
echo "next phrase: job array $array ($n conditions, $TASKS)"
merge=$(sbatch --parsable "${NODES[@]}" --cpus-per-task=1 --mem=8G --time=01:00:00 --dependency=afterany:$array \
    --job-name=next-phrase-merge --export=ALL,HAG_CANARY_MODE=next_phrase,HAG_CANARY_WORKERS=1 \
    --output=slurm/logs/%x-%j.out slurm/canary.sbatch merge)
echo "merge and comparisons: job $merge (after $array)"
