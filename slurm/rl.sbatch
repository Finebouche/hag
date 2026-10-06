#!/bin/bash
# Reinforcement learning experiments of hag (hag/rl) on PlaFRIM, for one benchmark (environment variable HAG_RL_ENV,
# see hag.rl.envs.BENCHMARKS), on one node: hyperparameter optimization (hag.rl.hpo, all the studies or the ones given
# as arguments) then final training (hag.rl.train), N_WORKERS being the cores of the job. The results are written on the
# local disk of the node (SQLite does not support the network file systems well) and copied to
# outputs/rl_results/<benchmark> every 10 minutes and at the end of the job.
#
# Environment: slurm/setup_env.sh (micromamba environment hag_env).
# Submit from the repository root (all the benchmarks: slurm/submit_rl.sh):
#   sbatch --export=ALL,HAG_RL_ENV=<benchmark> --job-name=rl-<benchmark> slurm/rl.sbatch [study ...]
#SBATCH --job-name=hag-rl
#SBATCH --partition=routage
#SBATCH --constraint=zonda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=1-00:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=slurm/logs/%x-%j.out

set -euo pipefail
: "${HAG_RL_ENV:?set the benchmark: sbatch --export=ALL,HAG_RL_ENV=<benchmark> slurm/rl.sbatch}"
module load micromamba
export MAMBA_ROOT_PREFIX=$HOME/micromamba
eval "$(micromamba shell hook -s bash)"
micromamba activate hag_env

RESULTS=$SLURM_SUBMIT_DIR/outputs/rl_results/$HAG_RL_ENV
export HAG_RL_RESULTS=/tmp/hag_rl_$SLURM_JOB_ID
mkdir -p "$RESULTS" "$HAG_RL_RESULTS"
rsync -a "$RESULTS/" "$HAG_RL_RESULTS/"
sync_back() { rsync -a "$HAG_RL_RESULTS/" "$RESULTS/"; }
# copy back at the end, also when the job is cancelled or reaches its time limit (TERM 2 minutes before)
trap 'sync_back; rm -rf "$HAG_RL_RESULTS"' EXIT
trap 'exit 143' TERM
(while true; do sleep 600; sync_back; done) &

echo "[slurm] job $SLURM_JOB_ID: $HAG_RL_ENV on $(hostname), $SLURM_CPUS_PER_TASK workers, results in $RESULTS"
# (in the background with wait: bash only handles the signals between two commands)
python -m hag.rl.hpo "$@" & wait $!
python -m hag.rl.train & wait $!
