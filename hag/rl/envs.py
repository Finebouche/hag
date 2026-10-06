"""
Partially observable benchmarks (BENCHMARKS), with their budgets:
  - classic control and Box2D (gymnasium) without their velocities: the agent must infer them from the history of the
    observations (CartPole, Acrobot, LunarLander),
  - "-P" variants of the MuJoCo locomotion tasks (Ni et al., 2022, "Recurrent model-free RL can be a strong baseline for
    many POMDPs"): positions only, the velocities removed (HalfCheetah, Hopper, Walker2d, Ant),
  - POPGym (Morad et al., 2023), easy versions: positions only (and noisy) CartPole and Pendulum, and memory tasks on
    cards (RepeatPrevious, CountRecall, Autoencode), whose discrete observations are one-hot encoded.
"""
from dataclasses import dataclass, field
from typing import Optional

import gymnasium as gym
import numpy as np


@dataclass(frozen=True)
class Benchmark:
    gym_id: str                                    # gymnasium id
    keep: Optional[tuple] = None                   # observation indices kept (None: all)
    kwargs: dict = field(default_factory=dict)     # arguments of gym.make
    hpo_timesteps: int = 100_000                   # PPO budget per seed of a trial of the hyperparameter optimization
    train_timesteps: int = 300_000                 # PPO budget per seed of the final training (hag.rl.train)
    n_trials: int = 200                            # trials per study of the hyperparameter optimization
    normalize_reward: bool = False                 # normalization of the rewards for PPO (VecNormalize)


BENCHMARKS = {
    # classic control and Box2D, velocities removed
    "CartPole-v1": Benchmark("CartPole-v1", keep=(0, 2)),  # cart position, pole angle
    "Acrobot-v1": Benchmark("Acrobot-v1", keep=(0, 1, 2, 3), hpo_timesteps=200_000, train_timesteps=500_000),
    "LunarLander-v3": Benchmark("LunarLander-v3", keep=(0, 1, 4, 6, 7),  # x, y, angle, legs contacts
                                hpo_timesteps=500_000, train_timesteps=2_000_000, normalize_reward=True),
    # MuJoCo, positions only (the observations of the v5 environments start with the positions)
    **{f"{name}-P": Benchmark(f"{name}-v5", keep=tuple(range(n_positions)), kwargs=kwargs, hpo_timesteps=300_000,
                              train_timesteps=1_000_000, n_trials=150, normalize_reward=True)
       for name, n_positions, kwargs in [("HalfCheetah", 8, {}), ("Hopper", 5, {}), ("Walker2d", 8, {}),
                                         ("Ant", 13, dict(include_cfrc_ext_in_observation=False))]},
    # POPGym, easy versions
    **{name: Benchmark(f"popgym-{name}Easy-v0", hpo_timesteps=200_000, train_timesteps=1_000_000)
       for name in ["PositionOnlyCartPole", "NoisyPositionOnlyCartPole", "PositionOnlyPendulum",
                    "NoisyPositionOnlyPendulum"]},
    **{name: Benchmark(f"popgym-{name}Easy-v0", hpo_timesteps=300_000, train_timesteps=1_000_000)
       for name in ["RepeatPrevious", "CountRecall", "Autoencode"]},
}


class MaskObservation(gym.ObservationWrapper):
    """Keep only the observation components of indices."""

    def __init__(self, env: gym.Env, indices):
        super().__init__(env)
        self.indices = np.asarray(indices)
        low, high = env.observation_space.low[self.indices], env.observation_space.high[self.indices]
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

    def observation(self, observation):
        return observation[self.indices].astype(np.float32)


def make_env(name: str) -> gym.Env:
    """Environment of the benchmark name (see BENCHMARKS), with continuous observations: velocities removed, discrete
    observations one-hot encoded."""
    benchmark = BENCHMARKS[name]
    if benchmark.gym_id.startswith("popgym"):
        import popgym  # noqa: F401  (registers the POPGym environments)
    env = gym.make(benchmark.gym_id, **benchmark.kwargs)
    if benchmark.keep is not None:
        env = MaskObservation(env, benchmark.keep)
    if not isinstance(env.observation_space, gym.spaces.Box):
        env = gym.wrappers.FlattenObservation(env)  # (one-hot encoding of the discrete observations)
    return env
