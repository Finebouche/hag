#!/bin/bash
# Canary phrase boundary detection (python -m hag_detection.canary_boundaries [conditions] [nb_seeds]), results in
# outputs/canary/boundaries.csv. Run from the repository root (a checkout of the detection branch), data in hag/datasets/Canary:
#   sbatch slurm/canary_boundaries.sbatch [conditions] [nb_seeds]
#SBATCH --job-name=canary-boundaries
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --constraint=zen4|zen3|zonda
#SBATCH --exclude=sirocco[21-25],enbata[01-02]
#SBATCH --output=slurm/logs/%x-%j.out
set -eo pipefail
module load micromamba
export MAMBA_ROOT_PREFIX=$HOME/micromamba
eval "$(micromamba shell hook -s bash)"
micromamba activate hag_env
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH=$PWD  # this checkout, not the editable install of ~/hag
python -c "import hag; print(hag.__file__)"
python -m hag_detection.canary_boundaries "$@"
