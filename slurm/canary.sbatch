#!/bin/bash
# Canary boundary detection on PlaFRIM (next_phrase_prediction), from a checkout of the detection branch:
#   - HAG_CANARY_MODE=hpo (default): hyperparameter optimization python -m next_phrase_prediction.hpo <model> <size>,
#     HAG_CANARY_WORKERS processes sharing the cores; in a job array, "<model> <size>" is the line
#     SLURM_ARRAY_TASK_ID (from 0) of the file HAG_CANARY_TASKS;
#   - HAG_CANARY_MODE=evaluate: final evaluation python -m next_phrase_prediction.evaluate [sizes] [nb_seeds];
#   - HAG_CANARY_MODE=syllables: frame-by-frame syllable classification python -m syllables_classification.classification [sizes]
#     [nb_seeds] (reservoirs of the detection studies);
#   - HAG_CANARY_MODE=next_phrase: prediction of the next phrase python -m next_phrase_prediction.prediction [sizes]
#     [nb_seeds] (reservoirs of the next-phrase studies, HAG_CANARY_TASK=next_phrase);
#   - HAG_CANARY_MODE=certainty: certainty along the phrases python -m syllables_classification.certainty [size]
#     [conditions] [nb_seeds] (next_phrase_prediction.certainty with HAG_CANARY_TASK=next_phrase).
# The Optuna databases are in /dev/shm during the job (SQLite does not support the network file systems well, and the
# suroit nodes have no local disk), copied from and back to outputs/canary/studies (every 10 minutes and at the
# end, without replacing more recent files). Submit everything with slurm/submit_canary.sh.
#SBATCH --job-name=canary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=slurm/logs/%x-%A_%a.out
set -euo pipefail
if [ -n "${HAG_CANARY_TASKS:-}" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    read -r MODEL SIZE < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$HAG_CANARY_TASKS")
    set -- "$MODEL" "$SIZE" "$@"
fi
set +u
module load micromamba
export MAMBA_ROOT_PREFIX=$HOME/micromamba
eval "$(micromamba shell hook -s bash)"
micromamba activate hag_env
set -u
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=$PWD  # this checkout, not the editable install of ~/hag

# threads of each worker (NumPy and JAX)
WORKERS=${HAG_CANARY_WORKERS:-4}
THREADS=$((SLURM_CPUS_PER_TASK / WORKERS))
export HAG_CANARY_WORKERS=$WORKERS OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=$THREADS"
# numba cache of librosa per job: the jobs writing it at the same time in the shared environment fail (stale file handle)
export NUMBA_CACHE_DIR=/dev/shm/numba_cache_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}

STUDIES=$PWD/outputs/canary/studies
export HAG_CANARY_STUDIES=/dev/shm/hag_canary_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}
mkdir -p "$STUDIES" "$HAG_CANARY_STUDIES" slurm/logs
rsync -a "$STUDIES/" "$HAG_CANARY_STUDIES/"
sync_back() { rsync -a --update "$HAG_CANARY_STUDIES/" "$STUDIES/"; }
trap 'sync_back; rm -rf "$HAG_CANARY_STUDIES" "$NUMBA_CACHE_DIR"' EXIT
trap 'exit 143' TERM
(while true; do sleep 600; sync_back; done) &

echo "[slurm] job ${SLURM_JOB_ID} ${SLURM_ARRAY_TASK_ID:-}: ${HAG_CANARY_MODE:-hpo} $*, on $(hostname)," \
    "$SLURM_CPUS_PER_TASK cores, $WORKERS workers"
if [ "${HAG_CANARY_MODE:-hpo}" = evaluate ]; then
    python -m next_phrase_prediction.evaluate "$@"
elif [ "${HAG_CANARY_MODE:-hpo}" = certainty ]; then
    if [ "${HAG_CANARY_TASK:-}" = next_phrase ]; then
        python -m next_phrase_prediction.certainty "$@"
    else
        python -m syllables_classification.certainty "$@"
    fi
elif [ "${HAG_CANARY_MODE:-hpo}" = syllables ]; then
    # in a job array: one condition ("<condition> <size>" of HAG_CANARY_TASKS)
    if [ -n "${HAG_CANARY_TASKS:-}" ]; then set -- run "$@"; fi
    python -m syllables_classification.classification "$@"
elif [ "${HAG_CANARY_MODE:-hpo}" = next_phrase ]; then
    # in a job array: one condition ("<condition> <size>" of HAG_CANARY_TASKS)
    if [ -n "${HAG_CANARY_TASKS:-}" ]; then set -- run "$@"; fi
    python -m next_phrase_prediction.prediction "$@"
else
    python -m next_phrase_prediction.hpo "$@"
fi
