#!/bin/bash
# Canary boundary detection on PlaFRIM (hag_detection), from a checkout of the detection branch:
#   - HAG_DETECTION_MODE=hpo (default): hyperparameter optimization python -m hag_detection.hpo <model> <size>
#     [n_trials], HAG_DETECTION_WORKERS processes sharing the cores; in a job array, "<model> <size>" is the line
#     SLURM_ARRAY_TASK_ID (from 0) of the file HAG_DETECTION_TASKS;
#   - HAG_DETECTION_MODE=evaluate: final evaluation python -m hag_detection.evaluate [sizes] [nb_seeds];
#   - HAG_DETECTION_MODE=syllables: frame-by-frame syllable classification python -m hag_detection.syllables [sizes]
#     [nb_seeds] (reservoirs of the detection studies).
# The Optuna databases are in /dev/shm during the job (SQLite does not support the network file systems well, and the
# suroit nodes have no local disk), copied from and back to outputs/detection/studies (every 10 minutes and at the
# end, without replacing more recent files). Submit everything with slurm/submit_detection.sh.
#SBATCH --job-name=detection
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=slurm/logs/%x-%A_%a.out
set -euo pipefail
if [ -n "${HAG_DETECTION_TASKS:-}" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    read -r MODEL SIZE < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$HAG_DETECTION_TASKS")
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
WORKERS=${HAG_DETECTION_WORKERS:-4}
THREADS=$((SLURM_CPUS_PER_TASK / WORKERS))
export HAG_DETECTION_WORKERS=$WORKERS OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=$THREADS"
# numba cache of librosa per job: the jobs writing it at the same time in the shared environment fail (stale file handle)
export NUMBA_CACHE_DIR=/dev/shm/numba_cache_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}

STUDIES=$PWD/outputs/detection/studies
export HAG_DETECTION_STUDIES=/dev/shm/hag_detection_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}
mkdir -p "$STUDIES" "$HAG_DETECTION_STUDIES" slurm/logs
rsync -a "$STUDIES/" "$HAG_DETECTION_STUDIES/"
sync_back() { rsync -a --update "$HAG_DETECTION_STUDIES/" "$STUDIES/"; }
trap 'sync_back; rm -rf "$HAG_DETECTION_STUDIES" "$NUMBA_CACHE_DIR"' EXIT
trap 'exit 143' TERM
(while true; do sleep 600; sync_back; done) &

echo "[slurm] job ${SLURM_JOB_ID} ${SLURM_ARRAY_TASK_ID:-}: ${HAG_DETECTION_MODE:-hpo} $*, on $(hostname)," \
    "$SLURM_CPUS_PER_TASK cores, $WORKERS workers"
if [ "${HAG_DETECTION_MODE:-hpo}" = evaluate ]; then
    python -m hag_detection.evaluate "$@"
elif [ "${HAG_DETECTION_MODE:-hpo}" = syllables ]; then
    # in a job array: one condition ("<condition> <size>" of HAG_DETECTION_TASKS)
    if [ -n "${HAG_DETECTION_TASKS:-}" ]; then set -- run "$@"; fi
    python -m hag_detection.syllables "$@"
else
    python -m hag_detection.hpo "$@"
fi
