#!/bin/bash
# An analysis of hag.analysis on PlaFRIM (python -m <HAG_ANALYSIS> [arguments]), results in outputs/hag/analysis_results.
# Run from the root of a checkout of main:
#   sbatch --export=ALL,HAG_ANALYSIS=hag.analysis.compare_jax --job-name=compare-jax slurm/analysis.sbatch CatsDogs FSDD
#SBATCH --job-name=analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH --constraint=zen4|zen3|zonda
#SBATCH --output=slurm/logs/%x-%j.out
set -euo pipefail
: "${HAG_ANALYSIS:?}"
set +u
module load micromamba
export MAMBA_ROOT_PREFIX=$HOME/micromamba
eval "$(micromamba shell hook -s bash)"
micromamba activate hag_env
set -u
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=$PWD  # this checkout, not the editable install of ~/hag
# numba cache of librosa per job: jobs writing it at the same time in the shared environment fail (stale file handle)
export NUMBA_CACHE_DIR=/dev/shm/numba_cache_${SLURM_JOB_ID}
trap 'rm -rf "$NUMBA_CACHE_DIR"' EXIT
mkdir -p slurm/logs
echo "[slurm] job $SLURM_JOB_ID: $HAG_ANALYSIS $*, on $(hostname)"
python -m "$HAG_ANALYSIS" "$@"
