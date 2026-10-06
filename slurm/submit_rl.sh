#!/bin/bash
# Submit one job of slurm/rl.sbatch per benchmark (all the benchmarks of hag.rl.envs.BENCHMARKS, or the ones given as
# arguments). Run from the repository root:  bash slurm/submit_rl.sh [benchmark ...]
set -euo pipefail
cd "$(dirname "$0")/.."
BENCHMARKS=("$@")
if [ ${#BENCHMARKS[@]} -eq 0 ]; then
    BENCHMARKS=(Acrobot-v1 LunarLander-v3 HalfCheetah-P Hopper-P Walker2d-P Ant-P PositionOnlyCartPole
                NoisyPositionOnlyCartPole PositionOnlyPendulum NoisyPositionOnlyPendulum RepeatPrevious CountRecall
                Autoencode)
fi
mkdir -p slurm/logs
for benchmark in "${BENCHMARKS[@]}"; do
    sbatch --export=ALL,HAG_RL_ENV="$benchmark" --job-name="rl-$benchmark" slurm/rl.sbatch
done
