#!/bin/bash
# Submit one job of slurm/rl.sbatch per benchmark (all the benchmarks of hag.rl.envs.BENCHMARKS, or the ones given as
# arguments), at most MAX_CONCURRENT running at the same time (the next ones wait for the end of a previous one, Slurm
# dependencies: no flooding of the platform). Run from the repository root:  bash slurm/submit_rl.sh [benchmark ...]
set -euo pipefail
MAX_CONCURRENT=8
cd "$(dirname "$0")/.."
BENCHMARKS=("$@")
if [ ${#BENCHMARKS[@]} -eq 0 ]; then
    BENCHMARKS=(Acrobot-v1 LunarLander-v3 HalfCheetah-P Hopper-P Walker2d-P Ant-P PositionOnlyCartPole
                NoisyPositionOnlyCartPole PositionOnlyPendulum NoisyPositionOnlyPendulum RepeatPrevious CountRecall
                Autoencode)
fi
mkdir -p slurm/logs
JOBS=()
for k in "${!BENCHMARKS[@]}"; do
    benchmark=${BENCHMARKS[$k]}
    dependency=()
    if [ "$k" -ge "$MAX_CONCURRENT" ]; then
        dependency=(--dependency="afterany:${JOBS[$((k - MAX_CONCURRENT))]}")
    fi
    job=$(sbatch --parsable "${dependency[@]}" --export=ALL,HAG_RL_ENV="$benchmark" --job-name="rl-$benchmark" \
        slurm/rl.sbatch)
    JOBS+=("$job")
    echo "$benchmark: job $job ${dependency[*]}"
done
