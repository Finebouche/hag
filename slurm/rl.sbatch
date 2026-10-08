#!/bin/bash
# Reinforcement learning experiments of hag (hag/rl) on PlaFRIM, for one benchmark (environment variable HAG_RL_ENV,
# see hag.rl.envs.BENCHMARKS) and one method (environment variable HAG_RL_METHOD: "ppo", default, "dqn", "lspi" or
# "bc", the latter without hyperparameter optimization), on one
# node: hyperparameter optimization (hag.rl.<method>.hpo, all the studies or the ones given as arguments) then final
# training (hag.rl.<method>.train); or, with HAG_RL_METHOD=sweep, the sweep of the readout and of the reservoir size
# (hag.rl.sweep, its methods given as arguments, all by default); N_WORKERS being the cores of the job (a whole suroit node: 48 AMD
# Zen4 cores, 256 GB). The results are written in /dev/shm (local memory of the node: SQLite does not support the
# network file systems well, and the suroit nodes have no local disk) and copied to outputs/rl_results/<benchmark>
# every 10 minutes and at the end of the job.
# Usage rules of PlaFRIM: large experiments at night or during holidays, no job longer than 72 h outside week-ends, no
# more than 5 hours on all the cores of a sub-cluster per working day (9:00-19:00); slurm/submit_rl.sh limits the
# number of jobs running at the same time.
#
# Environment: slurm/setup_env.sh (micromamba environment hag_env).
# Submit from the repository root (all the benchmarks: slurm/submit_rl.sh):
#   sbatch --export=ALL,HAG_RL_ENV=<benchmark>,HAG_RL_METHOD=<method> --job-name=rl-<benchmark> slurm/rl.sbatch \
#       [study ...]
#SBATCH --job-name=hag-rl
#SBATCH --partition=routage
#SBATCH --constraint=suroit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
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
export HAG_RL_RESULTS=/dev/shm/hag_rl_$SLURM_JOB_ID
mkdir -p "$RESULTS" "$HAG_RL_RESULTS"
rsync -a "$RESULTS/" "$HAG_RL_RESULTS/"
sync_back() { rsync -a "$HAG_RL_RESULTS/" "$RESULTS/"; }
# copy back at the end, also when the job is cancelled or reaches its time limit (TERM 2 minutes before)
trap 'sync_back; rm -rf "$HAG_RL_RESULTS"' EXIT
trap 'exit 143' TERM
(while true; do sleep 600; sync_back; done) &

METHOD=${HAG_RL_METHOD:-ppo}
echo "[slurm] job $SLURM_JOB_ID: $HAG_RL_ENV, $METHOD, on $(hostname), $SLURM_CPUS_PER_TASK workers," \
    "results in $RESULTS"
# (in the background with wait: bash only handles the signals between two commands)
if [ "$METHOD" = sweep ]; then
    python -m hag.rl.sweep "$@" & wait $!
elif [ "$METHOD" = bc ]; then
    # (no hyperparameter optimization: features of PPO's)
    python -m hag.rl.bc.train & wait $!
else
    python -m "hag.rl.$METHOD.hpo" "$@" & wait $!
    python -m "hag.rl.$METHOD.train" & wait $!
fi
