#!/bin/bash
# Hyperparameter optimizations of a classification dataset on PlaFRIM, one job per function (job array), then the merge
# of their databases into the one of the dataset (hag/hpo/corrected_studies). Run from the root of a checkout of main:
#   [HAG_HPO_MODULE=hag.hpo.hpo_rnn] bash slurm/submit_hpo_classification.sh <dataset> [function ...]
# Default functions: those of hag/hpo/hpo_esn.py, or lstm_last and gru with HAG_HPO_MODULE=hag.hpo.hpo_rnn (on the
# GPU nodes).
# Variants of the studies of hag.hpo.hpo_esn (environment variables): HAG_HPO_SAMPLER=cmaes, HAG_HPO_REPRESENTATION=
# mfcc|custom|none and HAG_HPO_INPUT=block|dense, e.g.
#   HAG_HPO_REPRESENTATION=custom HAG_HPO_INPUT=dense bash slurm/submit_hpo_classification.sh FSDD
set -euo pipefail
DATASET=$1
shift
FUNCTIONS=("$@")
MODULE=${HAG_HPO_MODULE:-hag.hpo.hpo_esn}
if [ ${#FUNCTIONS[@]} -eq 0 ]; then
    if [ "$MODULE" = hag.hpo.hpo_rnn ]; then
        FUNCTIONS=(lstm_last gru)
    else
        FUNCTIONS=(random_ee random_ei var_hag mean_hag ip_correct anti-oja ip-anti-oja)
    fi
fi
NODES=(--constraint="zen4|zen3|zonda" --exclude="sirocco[21-25],enbata[01-02]")
GPU_NODES=(--constraint="nvidia&ampere" --cpus-per-task=16 --mem=64G)  # A100 nodes (sirocco21-25)
cd "$(dirname "$0")/.."
mkdir -p slurm/logs slurm/tasks

# variant (prefix of its database, as hag.hpo.utility.study_variant), empty for the default studies
SAMPLER=${HAG_HPO_SAMPLER:-tpe}
VARIANT=""
if [ -n "${HAG_HPO_REPRESENTATION:-}" ] || [ "${HAG_HPO_INPUT:-block}" = dense ]; then
    PROJECTION=mod
    [ "${HAG_HPO_INPUT:-block}" = dense ] && PROJECTION=rdn
    VARIANT=${PROJECTION}-proj_${SAMPLER}_${HAG_HPO_REPRESENTATION:-mfcc}
elif [ "$SAMPLER" != tpe ]; then
    VARIANT=$SAMPLER
fi
TASKS=slurm/tasks/hpo_${DATASET}${VARIANT:+_$VARIANT}_$(date +%Y%m%d_%H%M%S).txt
printf "%s\n" "${FUNCTIONS[@]}" > "$TASKS"
n=${#FUNCTIONS[@]}
HPO_NODES=("${NODES[@]}")
[ "$MODULE" = hag.hpo.hpo_rnn ] && HPO_NODES=("${GPU_NODES[@]}")
hpo=$(sbatch --parsable "${HPO_NODES[@]}" --array=0-$((n - 1)) --job-name="hpo-$DATASET${VARIANT:+-$VARIANT}" \
    --export=ALL,HAG_HPO_DATASET="$DATASET",HAG_HPO_FUNCTIONS="$TASKS",HAG_HPO_MODULE="$MODULE",HAG_HPO_VARIANT="$VARIANT" \
    slurm/hpo_classification.sbatch)
echo "$DATASET: job array $hpo ($n functions, $TASKS)"

merge=$(sbatch --parsable "${NODES[@]}" --cpus-per-task=1 --mem=4G --time=00:30:00 --dependency=afterany:$hpo \
    --job-name="merge-$DATASET${VARIANT:+-$VARIANT}" --output=slurm/logs/%x-%j.out \
    --wrap="set +u; module load micromamba; export MAMBA_ROOT_PREFIX=\$HOME/micromamba; eval \"\$(micromamba shell hook -s bash)\"; micromamba activate hag_env; set -u; PYTHONPATH=\$PWD python -m hag.hpo.merge_studies $DATASET $VARIANT")
echo "$DATASET: merge of the databases, job $merge (after $hpo)"
