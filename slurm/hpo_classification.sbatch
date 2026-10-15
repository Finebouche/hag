#!/bin/bash
# Hyperparameter optimization of HAG's classification studies on PlaFRIM (python -m hag.hpo.hpo_esn <dataset>
# <function>), one function per task of a job array: the function of the task is the line SLURM_ARRAY_TASK_ID (from 0)
# of the file HAG_HPO_FUNCTIONS, the dataset is HAG_HPO_DATASET. Run from the root of a checkout of main.
# Each function has its own database during the run (several jobs writing the same SQLite file would corrupt it):
# hag/hpo/corrected_studies/parts/<dataset>/<function>/, merged into the database of the dataset afterwards
# (python -m hag.hpo.merge_studies <dataset>). The database is in /dev/shm during the job (SQLite does not support the
# network file systems well), copied back every 10 minutes and at the end. A job stopped by its time limit can be
# resubmitted: the study resumes with its completed trials. Submit with slurm/submit_hpo_classification.sh.
# HAG_HPO_MODULE: the module of the optimization, hag.hpo.hpo_esn (default) or hag.hpo.hpo_rnn (LSTM, GRU).
#SBATCH --job-name=hpo-classification
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=3-00:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=slurm/logs/%x-%A_%a.out
set -euo pipefail
: "${HAG_HPO_DATASET:?}" "${HAG_HPO_FUNCTIONS:?}"
FUNCTION=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$HAG_HPO_FUNCTIONS")
set +u
module load micromamba
export MAMBA_ROOT_PREFIX=$HOME/micromamba
eval "$(micromamba shell hook -s bash)"
micromamba activate hag_env
set -u
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=$PWD  # this checkout, not the editable install of ~/hag

# parallel studies (processes) x threads per trial = cores of the job
export HAG_HPO_PARALLEL_STUDIES=${HAG_HPO_PARALLEL_STUDIES:-4}
THREADS=$((SLURM_CPUS_PER_TASK / HAG_HPO_PARALLEL_STUDIES))
export HAG_HPO_JOBS_PER_TRIAL=$THREADS OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=$THREADS"
# numba cache of librosa per job: jobs writing it at the same time in the shared environment fail (stale file handle)
export NUMBA_CACHE_DIR=/dev/shm/numba_cache_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}

PART=$PWD/hag/hpo/corrected_studies/parts/$HAG_HPO_DATASET/$FUNCTION
export HAG_HPO_STUDIES=/dev/shm/hag_hpo_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$PART" "$HAG_HPO_STUDIES" slurm/logs
rsync -a "$PART/" "$HAG_HPO_STUDIES/"
sync_back() { rsync -a "$HAG_HPO_STUDIES/" "$PART/"; }
trap 'sync_back; rm -rf "$HAG_HPO_STUDIES" "$NUMBA_CACHE_DIR"' EXIT
trap 'exit 143' TERM
(while true; do sleep 600; sync_back; done) &

echo "[slurm] job $SLURM_JOB_ID $SLURM_ARRAY_TASK_ID: $HAG_HPO_DATASET $FUNCTION, on $(hostname)," \
    "$SLURM_CPUS_PER_TASK cores, $HAG_HPO_PARALLEL_STUDIES parallel studies x $THREADS threads"
python -m "${HAG_HPO_MODULE:-hag.hpo.hpo_esn}" "$HAG_HPO_DATASET" "$FUNCTION"
