#!/bin/bash
# Full canary boundary detection experiment (hag_detection), run from the root of a checkout of the detection branch:
#   1. optimization of the pure models (hag_detection.models.PURE_MODELS) at each size of SIZES, job array;
#   2. joint optimization of the hybrids <hag>+<esn> at each size of HYBRID_SIZES, starting from the parts optimized
#      alone at half the size (after 1);
#   3. final evaluation with NB_SEEDS seeds and the paired comparisons (after 2).
# N_TRIALS trials per study (Optuna, TPE). Usage: bash slurm/submit_detection.sh
set -euo pipefail
N_TRIALS=${N_TRIALS:-400}
NB_SEEDS=${NB_SEEDS:-10}
MAX_RUNNING=${MAX_RUNNING:-12}
SIZES=(250 500 1000)
HYBRID_SIZES=(500 1000)
PURE=(mean_hag var_hag random_ee random_ee_dense)
HYBRIDS=(mean_hag+random_ee mean_hag+random_ee_dense var_hag+random_ee var_hag+random_ee_dense)
NODES=(--constraint="zen4|zen3|zonda" --exclude="sirocco[21-25],enbata[01-02]")
cd "$(dirname "$0")/.."
mkdir -p slurm/logs slurm/tasks outputs/detection/studies

# the MFCC cache, once, before the jobs (hag/datasets/Canary/boundary_features.pkl)
set +u
module load micromamba
export MAMBA_ROOT_PREFIX=$HOME/micromamba
eval "$(micromamba shell hook -s bash)"
micromamba activate hag_env
set -u
PYTHONPATH=$PWD python -c "from hag_detection.common import load_dataset; load_dataset()" > /dev/null

stamp=$(date +%Y%m%d_%H%M%S)
PURE_TASKS=slurm/tasks/detection_pure_$stamp.txt
HYBRID_TASKS=slurm/tasks/detection_hybrid_$stamp.txt
for size in "${SIZES[@]}"; do for model in "${PURE[@]}"; do echo "$model $size"; done; done > "$PURE_TASKS"
for size in "${HYBRID_SIZES[@]}"; do for model in "${HYBRIDS[@]}"; do echo "$model $size"; done; done > "$HYBRID_TASKS"

n=$(wc -l < "$PURE_TASKS")
pure=$(sbatch --parsable "${NODES[@]}" --array=0-$((n - 1))%$MAX_RUNNING --job-name=detection-pure \
    --export=ALL,HAG_DETECTION_TASKS="$PURE_TASKS" slurm/detection.sbatch "$N_TRIALS")
echo "pure models: job array $pure ($n studies, $PURE_TASKS)"

n=$(wc -l < "$HYBRID_TASKS")
hybrid=$(sbatch --parsable "${NODES[@]}" --array=0-$((n - 1))%$MAX_RUNNING --job-name=detection-hybrid \
    --dependency=afterok:$pure --export=ALL,HAG_DETECTION_TASKS="$HYBRID_TASKS" slurm/detection.sbatch "$N_TRIALS")
echo "joint hybrids: job array $hybrid ($n studies, $HYBRID_TASKS), after $pure"

evaluation=$(sbatch --parsable "${NODES[@]}" --job-name=detection-evaluate --dependency=afterok:$hybrid \
    --export=ALL,HAG_DETECTION_MODE=evaluate,HAG_DETECTION_WORKERS=1 --output=slurm/logs/%x-%j.out \
    slurm/detection.sbatch "$(IFS=,; echo "${SIZES[*]}")" "$NB_SEEDS")
echo "final evaluation: job $evaluation, after $hybrid"
