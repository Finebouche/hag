"""
Linear readouts of the features evaluated as policies, one episode per readout, for the methods searching the readout
directly (evolution strategies: hag.rl.cmaes, hag.rl.openai_es).

A readout is a matrix theta of shape (n_outputs, F + 1) applied to the standardized features [x, 1] (statistics of the
pretraining episodes, see hag.rl.lspi.train.Features):
  - discrete actions: one score per action, the action of highest score;
  - continuous actions (Box): action = low + (tanh(theta [x, 1]) + 1) / 2 * (high - low).
A population of readouts runs its episodes at the same time (one environment per readout, features computed in batch),
all with the same reset seed (common random numbers: the readouts are compared on the same episodes).
"""
import numpy as np
from gymnasium import spaces

from hag.rl.envs import make_env


class Features:
    """Standardized features [x, 1] of the observations of n environments (states of the pipeline: one per
    environment)."""

    def __init__(self, pipeline, episodes: list):
        from hag.rl.probe import episode_features  # (one environment)
        pipeline.set_n_envs(1)
        X = np.concatenate([episode_features(pipeline, episode) for episode in episodes])
        self.mean, self.std = X.mean(axis=0), X.std(axis=0) + 1e-8
        self.pipeline = pipeline
        self.n_features = X.shape[1] + 1

    def set_n_envs(self, n: int):
        self.pipeline.set_n_envs(n)

    def reset(self, indices=None):
        self.pipeline.reset(indices)

    def __call__(self, observations: np.ndarray, indices=None) -> np.ndarray:
        x = (self.pipeline.step(np.asarray(observations, dtype=float), indices=indices) - self.mean) / self.std
        return np.concatenate([x, np.ones((len(x), 1))], axis=1)


def n_outputs(action_space) -> int:
    """Number of rows of a readout for an action space."""
    return action_space.n if isinstance(action_space, spaces.Discrete) else int(np.prod(action_space.shape))


def actions(thetas: np.ndarray, x: np.ndarray, action_space) -> np.ndarray:
    """Actions of readouts thetas (n, n_outputs, F + 1) for features x (n, F + 1)."""
    scores = np.einsum("nof,nf->no", thetas, x)
    if isinstance(action_space, spaces.Discrete):
        return np.argmax(scores, axis=1)
    low, high = action_space.low.ravel(), action_space.high.ravel()
    return (low + (np.tanh(scores) + 1) / 2 * (high - low)).reshape((len(x),) + action_space.shape)


class Population:
    """Environments of a population of readouts, evaluated together."""

    def __init__(self, env_id: str, features: Features, size: int):
        self.envs = [make_env(env_id) for _ in range(size)]
        self.features = features
        self.action_space = self.envs[0].action_space

    def evaluate(self, thetas: np.ndarray, seed: int) -> tuple:
        """Return of one episode of each readout of thetas (n, n_outputs, F + 1), n <= size, all with the reset seed
        seed. Returns (returns (n,), number of environment steps)."""
        n = len(thetas)
        self.features.set_n_envs(n)
        observations = np.stack([env.reset(seed=seed)[0] for env in self.envs[:n]])
        running, returns, steps = np.ones(n, dtype=bool), np.zeros(n), 0
        while running.any():
            indices = np.flatnonzero(running)
            x = self.features(observations[indices], indices=indices)
            for i, action in zip(indices, actions(thetas[indices], x, self.action_space)):
                observation, reward, terminated, truncated, _ = self.envs[i].step(action)
                observations[i] = observation
                returns[i] += reward
                running[i] = not (terminated or truncated)
            steps += len(indices)
        return returns, steps

    def evaluate_one(self, theta: np.ndarray, n_episodes: int, seed: int) -> list:
        """Returns of n_episodes episodes of the readout theta (reset seeds seed, seed + 1, ...), run in batch."""
        returns = []
        for start in range(0, n_episodes, len(self.envs)):
            n = min(len(self.envs), n_episodes - start)
            self.features.set_n_envs(n)
            observations = np.stack([env.reset(seed=seed + start + k)[0] for k, env in enumerate(self.envs[:n])])
            running, total = np.ones(n, dtype=bool), np.zeros(n)
            while running.any():
                indices = np.flatnonzero(running)
                x = self.features(observations[indices], indices=indices)
                thetas = np.broadcast_to(theta, (len(indices),) + theta.shape)
                for i, action in zip(indices, actions(thetas, x, self.action_space)):
                    observation, reward, terminated, truncated, _ = self.envs[i].step(action)
                    observations[i] = observation
                    total[i] += reward
                    running[i] = not (terminated or truncated)
            returns += list(total)
        return returns


def run_strategy(strategy, env_id: str, features: Features, seed: int, total_timesteps: int, n_selections: int,
                 n_selection_episodes: int, n_eval_episodes: int) -> dict:
    """Evolution strategy on the readouts: generations of strategy.ask() (readouts (n, d), d = n_outputs * (F + 1)),
    evaluated on one episode each (same reset seed within a generation), then strategy.tell(readouts, returns), until
    total_timesteps environment steps. The mean readout of the strategy (strategy.mean) is evaluated on
    n_selection_episodes episodes n_selections times during the training ("selection_curve"); the best one is then
    evaluated on n_eval_episodes episodes (other seeds). The selection and evaluation episodes are not counted in the
    budget."""
    population = Population(env_id, features, strategy.popsize)
    shape = (n_outputs(population.action_space), features.n_features)
    steps, generation, train_returns, curve, best = 0, 0, [], [], (-np.inf, None, -1)
    next_selection = 0
    while steps < total_timesteps:
        thetas = strategy.ask()
        returns, n = population.evaluate(thetas.reshape((-1,) + shape), seed=1_000_000 * seed + generation)
        strategy.tell(thetas, returns)
        steps, generation = steps + n, generation + 1
        train_returns += list(returns)
        if steps >= next_selection or steps >= total_timesteps:
            mean = strategy.mean.reshape(shape).copy()
            selection = population.evaluate_one(mean, n_selection_episodes, seed=500_000 + 100 * len(curve))
            curve.append(float(np.mean(selection)))
            if curve[-1] > best[0]:
                best = (curve[-1], mean, steps)
            next_selection += total_timesteps / n_selections
    evaluation = population.evaluate_one(best[1], n_eval_episodes, seed=10_000 + seed)
    train_returns = np.asarray(train_returns)
    return {"eval_return_mean": float(np.mean(evaluation)), "eval_return_std": float(np.std(evaluation)),
            "train_return_mean": float(train_returns.mean()),
            "train_return_last10%": float(train_returns[-max(1, len(train_returns) // 10):].mean()),
            "greedy_return_mean": float(np.mean(curve)), "best_timestep": best[2], "selection_curve": curve,
            "generations": generation}
