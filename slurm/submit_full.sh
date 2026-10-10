#!/bin/bash
# Full experiment from scratch, in its own results root (HAG_RL_RESULTS_ROOT, default outputs/rl_results_v2): for each
# benchmark of the sweep (python -m hag.rl.sweep --benchmarks, or the ones given as arguments), the hyperparameter
# optimizations of the methods METHODS, one task of a job array per benchmark, method and study (hag.rl.search.STUDIES,
# one database per study), HPO_CPUS cores each, at most MAX_HPO tasks running at the same time; then the sweep of the
# benchmark (hag.rl.sweep, SWEEP_CPUS cores) as soon as all its optimizations have succeeded. Nodes: the CPU sub-clusters suroit (Zen4), diablo06-09 (Zen3) and zonda (Zen2), the first free ones,
# without the nodes with GPUs (sirocco, enbata): the jobs are spread over several sub-clusters, as asked by the usage
# rules of PlaFRIM (no more than the equivalent of 5 hours on all the cores of a sub-cluster per working day).
# Run from the repository root:
#   bash slurm/submit_full.sh [benchmark ...]
set -euo pipefail
MAX_HPO=36
HPO_CPUS=16
SWEEP_CPUS=48
METHODS=(ppo lspi fqi openai_es nac)  # (BC has no hyperparameter optimization)
NODES=(--constraint="zen4|zen3|zonda" --exclude="sirocco[21-25],enbata[01-02]")
cd "$(dirname "$0")/.."
export HAG_RL_RESULTS_ROOT=${HAG_RL_RESULTS_ROOT:-$PWD/outputs/rl_results_v2}

set +u
module load micromamba
export MAMBA_ROOT_PREFIX=$HOME/micromamba
eval "$(micromamba shell hook -s bash)"
micromamba activate hag_env
set -u
BENCHMARKS=("$@")
if [ ${#BENCHMARKS[@]} -eq 0 ]; then
    BENCHMARKS=($(python -m hag.rl.sweep --benchmarks 2>/dev/null))
fi
STUDIES=($(python -c "from hag.rl import search; print(*search.STUDIES)" 2>/dev/null))

mkdir -p slurm/logs slurm/tasks "$HAG_RL_RESULTS_ROOT"
TASKS=slurm/tasks/hpo_$(date +%Y%m%d_%H%M%S).txt
for benchmark in "${BENCHMARKS[@]}"; do
    for method in "${METHODS[@]}"; do
        for study in "${STUDIES[@]}"; do
            echo "$benchmark $method $study" >> "$TASKS"
        done
    done
done
n_tasks=$(wc -l < "$TASKS")
hpo=$(sbatch --parsable "${NODES[@]}" --cpus-per-task=$HPO_CPUS --array=0-$((n_tasks - 1))%$MAX_HPO \
    --export=ALL,HAG_RL_TASKS="$TASKS",HAG_RL_TRAIN=0 --job-name=hpo --output=slurm/logs/%x-%A_%a.out slurm/rl.sbatch)
echo "hyperparameter optimizations: job array $hpo ($n_tasks tasks, $TASKS), results in $HAG_RL_RESULTS_ROOT"

per_benchmark=$((${#METHODS[@]} * ${#STUDIES[@]}))
for k in "${!BENCHMARKS[@]}"; do
    benchmark=${BENCHMARKS[$k]}
    tasks=$(seq -s : $((k * per_benchmark)) $(((k + 1) * per_benchmark - 1)) \
        | sed -e "s/:$//" -e "s/[0-9][0-9]*/${hpo}_&/g")
    dependency="afterok:$tasks"
    job=$(sbatch --parsable "${NODES[@]}" --cpus-per-task=$SWEEP_CPUS --time=2-00:00:00 --dependency="$dependency" \
        --export=ALL,HAG_RL_ENV="$benchmark",HAG_RL_METHOD=sweep --job-name="sweep-$benchmark" slurm/rl.sbatch)
    echo "$benchmark: sweep $job (after its $per_benchmark optimizations)"
done
