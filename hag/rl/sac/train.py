"""
Soft actor-critic (SAC), with linear actor and critics (NET_ARCH: linear readouts of the features, as the readout of a
reservoir), on the benchmark and conditions of hag.rl.experiment (features fitted on its pretraining episodes), instead
of PPO (hag.rl.ppo): off-policy, a replay buffer of the features and entropy-regularized policy and Q-functions.
  - Continuous actions: stable-baselines3's SAC (Haarnoja et al., 2018), its default hyperparameters except the
    learning rate (searched) and the size of the replay buffer (BUFFER_SIZE).
  - Discrete actions: discrete SAC (DiscreteSAC, Christodoulou, 2019, "Soft actor-critic for discrete action
    settings"), not in stable-baselines3, with the same defaults: softmax policy, twin Q-functions giving the values of
    all the actions (the expectations over the actions are exact, no reparameterization), target networks updated by
    Polyak averaging (TAU), entropy coefficient tuned towards the target entropy TARGET_ENTROPY_RATIO * log(n_actions)
    (searched; the 0.98 of the paper, close to the maximal entropy, makes the coefficient diverge: on CartPole, solved
    in 50,000 steps with 0.7, not with 0.98).
For each condition and seed: returns of the training episodes (sample efficiency; returns of the environment, also when
the rewards are normalized) and final evaluation of the deterministic policy. With USE_HPO_PARAMS, each condition uses
the best parameters of the hyperparameter optimization of SAC (hag.rl.sac.hpo: features, learning rate and, for
discrete actions, target entropy) if available, else the features of the best ones of PPO (hag.rl.ppo.hpo) with
LEARNING_RATE and TARGET_ENTROPY_RATIO, else the default ones (hag.rl.experiment.DEFAULT_PARAMS).
Memory: the replay buffer holds 2 * BUFFER_SIZE * n_features float32 (observations and next observations), 0.8 GB for
1000 features.

Results: <RL_RESULTS>/sac_<name>.csv (columns of hag.rl.ppo.train), learning curves in <RL_RESULTS>/rl_curves/ (name:
RESULTS_NAME, RL_RESULTS: see hag.rl.utils).
Run from the repository root:  HAG_RL_ENV=<benchmark> python -m hag.rl.sac.train [condition ...]
"""
import copy
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

import numpy as np
import pandas as pd
import torch
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize

from hag.rl.envs import make_env
from hag.rl.experiment import (BENCHMARK, CONDITIONS, DEFAULT_PARAMS, ENV_ID, RESULTS_NAME, SEEDS, UNITS,
                               condition_kind, pretraining_episodes)
from hag.rl.pretrain import make_pipeline
from hag.rl.ppo.train import EpisodeReturns
from hag.rl.utils import N_WORKERS, RL_RESULTS
from hag.rl.wrappers import FeatureVecEnv

# =============================== PARAMETERS ===============================
TOTAL_TIMESTEPS = BENCHMARK.train_timesteps
N_EVAL_EPISODES = 50
N_EVAL_ENVS = 8
USE_HPO_PARAMS = True
NET_ARCH = []               # hidden layers of the actor and of the critics (none: linear readouts)
# stable-baselines3's defaults, except the size of the replay buffer (1e6: 8 GB for 1000 features)
LEARNING_RATE = 3e-4        # (default, searched)
BUFFER_SIZE = 100_000
LEARNING_STARTS = 100       # random actions before the first update
BATCH_SIZE = 256
TAU = 0.005
GAMMA = 0.99
TRAIN_FREQ = 1              # environment steps between two updates
GRADIENT_STEPS = 1          # gradient steps per update
TARGET_ENTROPY_RATIO = 0.7  # discrete SAC: target entropy, fraction of the maximal one log(n_actions) (default, searched)
SAC_PARAMS = ("learning_rate", "target_entropy_ratio")  # parameters of SAC in the parameters of a condition
# ===========================================================================


def mlp(n_inputs: int, n_outputs: int, hidden: list) -> torch.nn.Sequential:
    sizes = [n_inputs] + list(hidden)
    layers = [layer for a, b in zip(sizes, sizes[1:]) for layer in (torch.nn.Linear(a, b), torch.nn.ReLU())]
    return torch.nn.Sequential(*layers, torch.nn.Linear(sizes[-1], n_outputs))


class DiscreteSAC:
    """Discrete SAC (Christodoulou, 2019) on a vectorized environment with discrete actions (see the module
    docstring). predict has the interface of stable-baselines3's models (for evaluate_policy)."""

    def __init__(self, env, seed: int, net_arch=NET_ARCH, learning_rate=LEARNING_RATE, buffer_size=BUFFER_SIZE,
                 learning_starts=LEARNING_STARTS, batch_size=BATCH_SIZE, tau=TAU, gamma=GAMMA, train_freq=TRAIN_FREQ,
                 gradient_steps=GRADIENT_STEPS, target_entropy_ratio=TARGET_ENTROPY_RATIO):
        torch.manual_seed(seed)
        self.env, self.rng = env, np.random.default_rng(seed)
        self.n_actions, n_features = env.action_space.n, env.observation_space.shape[0]
        self.actor = mlp(n_features, self.n_actions, net_arch)
        self.critics = torch.nn.ModuleList([mlp(n_features, self.n_actions, net_arch) for _ in range(2)])
        self.targets = copy.deepcopy(self.critics).requires_grad_(False)
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.optimizers = [torch.optim.Adam(params, lr=learning_rate)
                           for params in (self.actor.parameters(), self.critics.parameters(), [self.log_alpha])]
        self.target_entropy = target_entropy_ratio * np.log(self.n_actions)
        self.learning_starts, self.batch_size, self.tau, self.gamma = learning_starts, batch_size, tau, gamma
        self.train_freq, self.gradient_steps = train_freq, gradient_steps
        self.buffer = dict(obs=np.zeros((buffer_size, n_features), np.float32),
                           next_obs=np.zeros((buffer_size, n_features), np.float32),
                           action=np.zeros(buffer_size, np.int64), reward=np.zeros(buffer_size, np.float32),
                           terminated=np.zeros(buffer_size, np.float32))
        self.size, self.position = 0, 0

    def _store(self, obs, actions, rewards, next_obs, terminated):
        for k in range(len(actions)):
            for name, value in zip(self.buffer, (obs[k], next_obs[k], actions[k], rewards[k], terminated[k])):
                self.buffer[name][self.position] = value
            self.position = (self.position + 1) % len(self.buffer["action"])
            self.size = min(self.size + 1, len(self.buffer["action"]))

    def _update(self):
        indices = self.rng.integers(self.size, size=self.batch_size)
        batch = {name: torch.as_tensor(values[indices]) for name, values in self.buffer.items()}
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():  # soft value of the next states, exact expectation over the actions
            log_probs = torch.log_softmax(self.actor(batch["next_obs"]), dim=1)
            q = torch.min(*(target(batch["next_obs"]) for target in self.targets))
            value = (log_probs.exp() * (q - alpha * log_probs)).sum(dim=1)
            target_q = batch["reward"] + self.gamma * (1 - batch["terminated"]) * value
        actions = batch["action"][:, None]
        critic_loss = sum(torch.nn.functional.mse_loss(critic(batch["obs"]).gather(1, actions)[:, 0], target_q)
                          for critic in self.critics)
        log_probs = torch.log_softmax(self.actor(batch["obs"]), dim=1)
        probs = log_probs.exp()
        with torch.no_grad():
            q = torch.min(*(critic(batch["obs"]) for critic in self.critics))
        actor_loss = (probs * (alpha * log_probs - q)).sum(dim=1).mean()
        entropy = -(probs * log_probs).sum(dim=1).detach()
        alpha_loss = (self.log_alpha * (entropy - self.target_entropy)).mean()
        for optimizer, loss in zip(self.optimizers, (actor_loss, critic_loss, alpha_loss)):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            for target, critic in zip(self.targets.parameters(), self.critics.parameters()):
                target.mul_(1 - self.tau).add_(self.tau * critic)

    def learn(self, total_timesteps: int, returns: EpisodeReturns = None) -> "DiscreteSAC":
        """Train for total_timesteps environment steps; returns: records the returns of the training episodes."""
        obs, n_envs = self.env.reset(), self.env.num_envs
        for step in range(0, total_timesteps, n_envs):
            if step < self.learning_starts:
                actions = self.rng.integers(self.n_actions, size=n_envs)
            else:
                actions = self.predict(obs)[0]
            next_obs, rewards, dones, infos = self.env.step(actions)
            # (the vectorized environments reset at the end of an episode: its last observation is in the infos)
            real_next = next_obs.copy()
            terminated = np.zeros(n_envs)
            for i in np.flatnonzero(dones):
                real_next[i] = infos[i]["terminal_observation"]
                terminated[i] = not infos[i].get("TimeLimit.truncated", False)
                if returns is not None and "episode" in infos[i]:
                    returns.returns.append(infos[i]["episode"]["r"])
                    returns.timesteps.append(step + n_envs)
            self._store(obs, actions, rewards, real_next, terminated)
            obs = next_obs
            if step >= self.learning_starts and (step // n_envs) % self.train_freq == 0:
                for _ in range(self.gradient_steps):
                    self._update()
        return self

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        with torch.no_grad():
            logits = self.actor(torch.as_tensor(np.asarray(observation, np.float32)))
        if deterministic:
            return logits.argmax(dim=1).numpy(), state
        probs = torch.softmax(logits, dim=1).numpy().astype(float)
        return np.array([self.rng.choice(self.n_actions, p=p / p.sum()) for p in probs]), state


def condition_params(condition):
    """Parameters of a condition (features and SAC_PARAMS): best of the hyperparameter optimization of SAC if available
    (and USE_HPO_PARAMS), else the features of the best ones of PPO, else DEFAULT_PARAMS. The control "proj" takes the
    parameters of HAG."""
    kind = condition_kind(condition)
    if USE_HPO_PARAMS:
        from hag.rl.ppo import hpo as ppo_hpo  # (avoids a circular import)
        from hag.rl.sac import hpo
        best = hpo.best_params(kind)
        if best is not None:
            return best
        best = ppo_hpo.best_params(kind)
        if best is not None:
            return {key: value for key, value in best.items() if key != "learning_rate"}
    return DEFAULT_PARAMS[kind]


def run(condition, seed, episodes, params=None, total_timesteps=TOTAL_TIMESTEPS, evaluate=True, save_curves=True,
        check=None, units=UNITS, net_arch=NET_ARCH):
    """Train SAC with the features of a condition, fitted on the pretraining episodes, and return the results of the
    run. params: parameters of the features, "learning_rate" and "target_entropy_ratio" (discrete actions) of SAC
    (default: condition_params; LEARNING_RATE and TARGET_ENTROPY_RATIO if missing). check: function called with the feature pipeline before the training (e.g. to stop a run whose reservoir
    is not suitable by raising an exception). units: reservoir size, net_arch: hidden layers of the actor and of the
    critics (default: linear readouts)."""
    start = time.time()
    params = condition_params(condition) if params is None else params
    sac = dict(dict(learning_rate=LEARNING_RATE, target_entropy_ratio=TARGET_ENTROPY_RATIO),
               **{key: params[key] for key in SAC_PARAMS if key in params})
    features = {key: value for key, value in params.items() if key not in SAC_PARAMS}
    pipeline = make_pipeline(condition, episodes, units, seed, features)
    if check is not None:
        check(pipeline)
    train_env = FeatureVecEnv(make_vec_env(lambda: make_env(ENV_ID), n_envs=1, seed=seed), copy.deepcopy(pipeline))
    if BENCHMARK.normalize_reward:
        train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True)
    eval_env = FeatureVecEnv(make_vec_env(lambda: make_env(ENV_ID), n_envs=N_EVAL_ENVS, seed=10_000 + seed),
                             copy.deepcopy(pipeline))

    returns = EpisodeReturns()
    if isinstance(train_env.action_space, spaces.Discrete):
        model = DiscreteSAC(train_env, seed, net_arch=net_arch, **sac).learn(total_timesteps, returns)
    else:
        model = SAC("MlpPolicy", train_env, learning_rate=sac["learning_rate"], buffer_size=BUFFER_SIZE,
                    learning_starts=LEARNING_STARTS, batch_size=BATCH_SIZE, tau=TAU, gamma=GAMMA,
                    train_freq=TRAIN_FREQ, gradient_steps=GRADIENT_STEPS, seed=seed, device="cpu", verbose=0,
                    policy_kwargs=dict(net_arch=dict(pi=list(net_arch), qf=list(net_arch))))
        model.learn(total_timesteps, callback=returns)
    eval_mean, eval_std = (evaluate_policy(model, eval_env, n_eval_episodes=N_EVAL_EPISODES, deterministic=True)
                           if evaluate else (np.nan, np.nan))

    if save_curves:
        curves = RL_RESULTS / "rl_curves"
        curves.mkdir(parents=True, exist_ok=True)
        np.savez(curves / f"sac_{RESULTS_NAME}_{condition}_seed{seed}.npz", returns=returns.returns,
                 timesteps=returns.timesteps)
    train = np.asarray(returns.returns)
    W = None if pipeline.reservoir is None else pipeline.reservoir["W"]
    return {"env": ENV_ID, "condition": condition, "seed": seed, "n_features": pipeline.n_features,
            "connections_per_neuron": np.nan if W is None else np.count_nonzero(W) / len(W),
            "eval_return_mean": eval_mean, "eval_return_std": eval_std,
            "train_return_mean": float(train.mean()) if len(train) else np.nan,  # whole training: sample efficiency
            "train_return_last10%": float(train[-max(1, len(train) // 10):].mean()) if len(train) else np.nan,
            "time_s": round(time.time() - start), "params": dict(features, **sac)}


def main(conditions):
    output = RL_RESULTS / f"sac_{RESULTS_NAME}.csv"
    rows = []
    with ProcessPoolExecutor(N_WORKERS, mp_context=get_context("spawn")) as pool:
        episodes = dict(zip(SEEDS, pool.map(pretraining_episodes, SEEDS)))
        futures = [pool.submit(run, condition, seed, episodes[seed]) for seed in SEEDS for condition in conditions]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(f"[sac] {ENV_ID} {row['condition']:16s} seed {row['seed']}: eval {row['eval_return_mean']:7.3f} ± "
                  f"{row['eval_return_std']:5.3f} | train mean {row['train_return_mean']:7.3f}, last 10% "
                  f"{row['train_return_last10%']:7.3f} | {row['n_features']} features | {row['time_s']}s", flush=True)
            # saved after each run, so that an interrupted study keeps its results
            output.parent.mkdir(parents=True, exist_ok=True)
            rows.sort(key=lambda row: (row["seed"], conditions.index(row["condition"])))
            pd.DataFrame(rows).to_csv(output, index=False)

    results = pd.DataFrame(rows)
    metrics = ["eval_return_mean", "train_return_mean", "train_return_last10%"]
    print(results.groupby("condition")[metrics].agg(["mean", "std"]).round(3).to_string())
    print(f"Results saved to {output}")


if __name__ == "__main__":
    main(sys.argv[1:] or CONDITIONS)
