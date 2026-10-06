"""Benchmark, folder of the results and number of parallel processes of the reinforcement learning experiments
(hag/rl), set with environment variables (e.g. by the Slurm jobs of slurm/rl.sbatch)."""
import os
from pathlib import Path

from hag.hpo.utility import PROJECT_ROOT

# results folder, can be redirected with the environment variable HAG_RL_RESULTS (e.g. to the local disk of a cluster
# node: SQLite databases do not support network file systems well)
RL_RESULTS = Path(os.environ.get("HAG_RL_RESULTS", PROJECT_ROOT / "outputs" / "rl_results"))
# benchmark (see hag.rl.envs.BENCHMARKS)
BENCHMARK_NAME = os.environ.get("HAG_RL_ENV", "CartPole-v1")
# parallel processes (one thread each): the cores allocated by Slurm, else 8
N_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
