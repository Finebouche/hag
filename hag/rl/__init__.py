"""Reinforcement learning with HAG as preprocessing: partially observable gymnasium environments, causal filter bank,
reservoir (random ESN or HAG) features and PPO agents (stable-baselines3)."""
import os

# one thread per process: the experiments are parallelized over processes (N_WORKERS of hag.rl.train and hag.rl.hpo)
# and their networks and matrices are small, multithreading would only add contention (and slows PPO down). The flags
# of XLA (jax) must be set before jax is imported.
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

# scikit-learn loads the OpenMP runtime of the environment (conda's llvm-openmp) before torch (imported by
# stable-baselines3), which brings its own copy: loading torch first aborts with "OMP: Error #15".
import sklearn  # noqa: E402, F401
import torch  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

torch.set_num_threads(1)
threadpool_limits(1)  # BLAS of numpy
