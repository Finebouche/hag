#!/bin/bash
# Full canary boundary detection experiment (hag_detection), run from the root of a checkout of the detection branch:
#   1. optimization of the pure models (hag_detection.models.PURE_MODELS) at each size of SIZES, job array;
#   2. joint optimization of the hybrids <hag>+<esn> at each size of HYBRID_SIZES, starting from the parts optimized
#      alone at half the size (after 1);
#   3. final evaluation with NB_SEEDS seeds and the paired comparisons (after 2).
# hag_detection.hpo.N_TRIALS trials per study (Optuna, TPE). With HAG_DETECTION_TASK=syllables, the studies optimize
# the frame accuracy of the syllable classification ("syllables_<model>_<size>") and the final evaluation is
# slurm/submit_syllables.sh.
# With HAG_DETECTION_TASK=next_phrase, they optimize the accuracy of the prediction of the next phrase
# ("next_phrase_<model>_<size>") and the final evaluation is slurm/submit_next_phrase.sh.
# With HAG_DETECTION_INPUT_GAINS=1 (syllables, next_phrase), the gain of each group of input features is searched too,
# and with HAG_DETECTION_LEAK=1 (syllables, next_phrase), the leak rate of each reservoir.
# Usage: [HAG_DETECTION_TASK=syllables|next_phrase] [HAG_DETECTION_INPUT_GAINS=1] [HAG_DETECTION_LEAK=1] [PURE="..."]
#        [HYBRIDS="..."] [SIZES="..."] [HYBRID_SIZES="..."] bash slurm/submit_detection.sh
set -euo pipefail
NB_SEEDS=${NB_SEEDS:-10}
MAX_RUNNING=${MAX_RUNNING:-12}
read -r -a SIZES <<< "${SIZES:-250 500 1000}"
read -r -a HYBRID_SIZES <<< "${HYBRID_SIZES:-500 1000}"
# models (space-separated lists, PURE and HYBRIDS environment variables)
read -r -a PURE <<< "${PURE:-mean_hag var_hag random_ee random_ee_dense}"
read -r -a HYBRIDS <<< "${HYBRIDS:-mean_hag+random_ee mean_hag+random_ee_dense var_hag+random_ee var_hag+random_ee_dense}"
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
    --export=ALL,HAG_DETECTION_TASKS="$PURE_TASKS" slurm/detection.sbatch)
echo "pure models: job array $pure ($n studies, $PURE_TASKS)"

n=$(wc -l < "$HYBRID_TASKS")
hybrid=$(sbatch --parsable "${NODES[@]}" --array=0-$((n - 1))%$MAX_RUNNING --job-name=detection-hybrid \
    --dependency=afterok:$pure --export=ALL,HAG_DETECTION_TASKS="$HYBRID_TASKS" slurm/detection.sbatch)
echo "joint hybrids: job array $hybrid ($n studies, $HYBRID_TASKS), after $pure"

if [ "${HAG_DETECTION_TASK:-boundaries}" = syllables ]; then
    # syllable classification: one job per condition, submitted once the studies exist (slurm/submit_syllables.sh)
    evaluation=$(sbatch --parsable "${NODES[@]}" --cpus-per-task=1 --mem=4G --time=00:30:00 --dependency=afterok:$hybrid \
        --job-name=submit-syllables --output=slurm/logs/%x-%j.out --wrap="bash slurm/submit_syllables.sh $(IFS=,; echo "${SIZES[*]}") $NB_SEEDS")
    echo "final evaluation (syllable classification, one job per condition): submitted by job $evaluation, after $hybrid"
elif [ "${HAG_DETECTION_TASK:-boundaries}" = next_phrase ]; then
    evaluation=$(sbatch --parsable "${NODES[@]}" --cpus-per-task=1 --mem=4G --time=00:30:00 --dependency=afterok:$hybrid \
        --job-name=submit-next-phrase --output=slurm/logs/%x-%j.out --wrap="bash slurm/submit_next_phrase.sh $(IFS=,; echo "${SIZES[*]}") $NB_SEEDS")
    echo "final evaluation (next phrase, one job per condition): submitted by job $evaluation, after $hybrid"
else
    evaluation=$(sbatch --parsable "${NODES[@]}" --job-name=detection-evaluate --dependency=afterok:$hybrid \
        --export=ALL,HAG_DETECTION_MODE=evaluate,HAG_DETECTION_WORKERS=1 --output=slurm/logs/%x-%j.out \
        slurm/detection.sbatch "$(IFS=,; echo "${SIZES[*]}")" "$NB_SEEDS")
    echo "final evaluation: job $evaluation, after $hybrid"
fi
