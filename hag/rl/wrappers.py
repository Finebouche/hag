"""
stable-baselines3 vectorized environment whose observations are the online features of hag.rl.features.FeaturePipeline.
"""
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper

from hag.rl.features import FeaturePipeline


class FeatureVecEnv(VecEnvWrapper):
    """Replace the observations of venv by their features. The vectorized environments of stable-baselines3 reset an
    environment as soon as its episode ends and return the first observation of the next one: the state of the
    pipeline of this environment is then reset, and the features of the terminal observation (used by PPO to bootstrap
    the value of truncated episodes) are computed from the state of the finished episode."""

    def __init__(self, venv: VecEnv, pipeline: FeaturePipeline):
        pipeline.set_n_envs(venv.num_envs)
        self.pipeline = pipeline
        observation_space = spaces.Box(-np.inf, np.inf, shape=(pipeline.n_features,), dtype=np.float32)
        super().__init__(venv, observation_space=observation_space)

    def reset(self):
        observations = self.venv.reset()
        self.pipeline.reset()
        return self.pipeline.step(observations)

    def step_wait(self):
        observations, rewards, dones, infos = self.venv.step_wait()
        for i in np.flatnonzero(dones):
            terminal = infos[i].get("terminal_observation")
            if terminal is not None:
                infos[i]["terminal_observation"] = self.pipeline.step(terminal[None, :], indices=[i], commit=False)[0]
        self.pipeline.reset(np.flatnonzero(dones))
        return self.pipeline.step(observations), rewards, dones, infos
