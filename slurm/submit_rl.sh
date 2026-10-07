#!/bin/bash
# Submit one job of slurm/rl.sbatch per benchmark (all the benchmarks of hag.rl.envs.BENCHMARKS, or the ones given as
# arguments), at most MAX_CONCURRENT running at the same time (the next ones wait for the end of a previous one, Slurm
# dependencies: no flooding of the platform). Method: environment variable HAG_RL_METHOD ("ppo", default, or "lspi";
# LSPI: only the benchmarks with discrete actions are kept). Run from the repository root:
#   [HAG_RL_METHOD=lspi] bash slurm/submit_rl.sh [benchmark ...]
set -euo pipefail
MAX_CONCURRENT=8
cd "$(dirname "$0")/.."
BENCHMARKS=("$@")
if [ ${#BENCHMARKS[@]} -eq 0 ]; then
    BENCHMARKS=(Acrobot-v1 LunarLander-v3 HalfCheetah-P Hopper-P Walker2d-P Ant-P PositionOnlyCartPole
                NoisyPositionOnlyCartPole PositionOnlyPendulum NoisyPositionOnlyPendulum RepeatPrevious CountRecall
                Autoencode)
fi
method=${HAG_RL_METHOD:-ppo}
if [ "$method" = lspi ]; then
    # LSPI: benchmarks with discrete actions only (hag.rl.envs), in the environment of the jobs
    set +u
    module load micromamba
    export MAMBA_ROOT_PREFIX=$HOME/micromamba
    eval "$(micromamba shell hook -s bash)"
    micromamba activate hag_env
    set -u
    BENCHMARKS=($(python -m hag.rl.envs --discrete "${BENCHMARKS[@]}" 2>/dev/null))
    echo "LSPI, benchmarks with discrete actions: ${BENCHMARKS[*]}"
fi
mkdir -p slurm/logs
JOBS=()
for k in "${!BENCHMARKS[@]}"; do
    benchmark=${BENCHMARKS[$k]}
    dependency=()
    if [ "$k" -ge "$MAX_CONCURRENT" ]; then
        dependency=(--dependency="afterany:${JOBS[$((k - MAX_CONCURRENT))]}")
    fi
    job=$(sbatch --parsable ${dependency[@]+"${dependency[@]}"} --export=ALL,HAG_RL_ENV="$benchmark",HAG_RL_METHOD="$method" \
        --job-name="$method-$benchmark" slurm/rl.sbatch)
    JOBS+=("$job")
    echo "$benchmark: job $job ${dependency[*]+${dependency[*]}}"
done
