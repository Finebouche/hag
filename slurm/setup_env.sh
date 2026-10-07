#!/bin/bash
# Micromamba environment hag_env for the reinforcement learning experiments of hag on PlaFRIM (CPU only), with the
# development version of reservoirpy (fork, branch v0.4.3-dev) next to the repository: ~/reservoirpy and ~/hag.
#   module load git
#   git clone https://github.com/Finebouche/hag.git ~/hag
#   git clone -b v0.4.3-dev https://github.com/Finebouche/reservoirpy.git ~/reservoirpy
#   bash ~/hag/slurm/setup_env.sh
# Run again to update the environment (it is created only if it does not exist).
set -euo pipefail
module load micromamba
export MAMBA_ROOT_PREFIX=$HOME/micromamba
eval "$(micromamba shell hook -s bash)"
micromamba env list | grep -q "^ *hag_env " || micromamba create -y -n hag_env -c conda-forge python=3.12 pip
micromamba activate hag_env
pip install --upgrade pip
# torch and torchaudio without CUDA (the RL experiments run on CPU)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
# dependencies of hag (pyproject.toml, with the extra "rl"), without its pinned reservoirpy: the fork is installed
pip install aeon cmaes dcor imageio jax joblib librosa matplotlib networkx numpy optuna pandas scikit-learn scipy \
    seaborn svgpath2mpl tensorflow-cpu tqdm threadpoolctl "gymnasium[classic-control,box2d,mujoco]" stable-baselines3 \
    popgym
pip install -e "$HOME/reservoirpy" --no-deps
pip install -e "$HOME/hag" --no-deps
python -c "import hag.rl.ppo.hpo, hag.rl.lspi.hpo, reservoirpy, mujoco, Box2D, popgym; print('hag_env ok:', \
reservoirpy.__file__)"
