#!/bin/bash
# Test evaluation of the corrected HPO studies on PlaFRIM (python -m hag.performances.run_test_evaluation <dataset>
# <function> [nb_trials]), one function per task of a job array: the function is the line SLURM_ARRAY_TASK_ID (from 0)
# of the file HAG_TEST_FUNCTIONS, the dataset is HAG_TEST_DATASET. Each task writes its rows to
# outputs/hag/test_results/parts/<dataset>_<function>.csv. Submit with slurm/submit_test_evaluation.sh.
#SBATCH --job-name=test-evaluation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm/logs/%x-%A_%a.out
set -euo pipefail
: "${HAG_TEST_DATASET:?}" "${HAG_TEST_FUNCTIONS:?}"
FUNCTION=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$HAG_TEST_FUNCTIONS")
set +u
module load micromamba
export MAMBA_ROOT_PREFIX=$HOME/micromamba
eval "$(micromamba shell hook -s bash)"
micromamba activate hag_env
set -u
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=$PWD  # this checkout, not the editable install of ~/hag
export NUMBA_CACHE_DIR=/dev/shm/numba_cache_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
trap 'rm -rf "$NUMBA_CACHE_DIR"' EXIT
mkdir -p outputs/hag/test_results/parts
export HAG_TEST_RESULTS=$PWD/outputs/hag/test_results/parts/${HAG_TEST_DATASET}_${FUNCTION}.csv
echo "[slurm] job $SLURM_JOB_ID $SLURM_ARRAY_TASK_ID: test evaluation $HAG_TEST_DATASET $FUNCTION, on $(hostname)"
python -m hag.performances.run_test_evaluation "$HAG_TEST_DATASET" "$FUNCTION" "$@"
