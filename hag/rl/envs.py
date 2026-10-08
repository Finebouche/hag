"""
Partially observable benchmarks (BENCHMARKS), with their budgets:
  - classic control and Box2D (gymnasium) without their velocities: the agent must infer them from the history of the
    observations (CartPole, Acrobot, LunarLander),
  - "-P" variants of the MuJoCo locomotion tasks (Ni et al., 2022, "Recurrent model-free RL can be a strong baseline for
    many POMDPs"): positions only, the velocities removed (HalfCheetah, Hopper, Walker2d, Ant),
  - POPGym (Morad et al., 2023), easy versions: positions only (and noisy) CartPole and Pendulum, and memory tasks on
    cards (RepeatPrevious, CountRecall, Autoencode), whose discrete observations are one-hot encoded; medium and hard
    versions of RepeatPrevious (card of k = 32 and 64 steps before, instead of 4) and CountRecall (more decks, and more
    card values for the hard one), the easy ones being solved by most conditions.
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
    train_timesteps: int = 300_000                 # PPO budget per seed of the final training (hag.rl.ppo.train)
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
    # POPGym, medium and hard versions of the memory tasks
    **{f"{name}{level}": Benchmark(f"popgym-{name}{level}-v0", hpo_timesteps=300_000, train_timesteps=1_000_000)
       for name in ["RepeatPrevious", "CountRecall"] for level in ["Medium", "Hard"]},
}


class MaskObservation(gym.ObservationWrapper):
    """Keep only the observation components of indices (the full observation is kept in the attribute full)."""

    def __init__(self, env: gym.Env, indices):
        super().__init__(env)
        self.indices = np.asarray(indices)
        low, high = env.observation_space.low[self.indices], env.observation_space.high[self.indices]
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)
        self.full = None

    def observation(self, observation):
        self.full = np.asarray(observation)
        return observation[self.indices].astype(np.float32)


def hidden_state(env: gym.Env) -> np.ndarray:
    """Hidden state at the last observation of an environment of make_env, what the agent has to infer: the components
    removed from the observations (velocities), or the state of the POPGym environments (flattened: one-hot encoding
    of its discrete parts)."""
    unwrapped = env.unwrapped
    if hasattr(unwrapped, "get_state"):
        return gym.spaces.flatten(unwrapped.state_space, unwrapped.get_state()).astype(float)
    while not isinstance(env, MaskObservation):
        env = env.env
    return np.delete(env.full, env.indices).astype(float)


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


class ExpertObservation(gym.ObservationWrapper):
    """Observations of an environment of make_env followed by its hidden state (see hidden_state): what an expert
    needs to act without memory. The observation of the agent is the first n_agent components."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.n_agent = env.observation_space.shape[0]
        env.reset(seed=0)
        n = self.n_agent + len(hidden_state(env))
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(n,), dtype=np.float32)

    def observation(self, observation):
        return np.concatenate([observation, hidden_state(self.env)]).astype(np.float32)


def make_expert_env(name: str) -> ExpertObservation:
    """Environment of the benchmark name with the expert observations (see ExpertObservation)."""
    return ExpertObservation(make_env(name))


def has_discrete_actions(name: str) -> bool:
    """Whether the actions of the benchmark name are discrete (as LSPI needs)."""
    env = make_env(name)
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    env.close()
    return discrete


if __name__ == "__main__":
    # names of the benchmarks given as arguments (all by default), only those with discrete actions with --discrete,
    # one per line (used by slurm/submit_rl.sh)
    import sys

    names = [arg for arg in sys.argv[1:] if arg != "--discrete"] or list(BENCHMARKS)
    for name in names:
        if "--discrete" not in sys.argv[1:] or has_discrete_actions(name):
            print(name)
