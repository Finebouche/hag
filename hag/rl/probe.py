"""
Supervised probe of the features of hag.rl.experiment: how well a linear readout (ridge) of the features decodes the
hidden state of the benchmark, what the agent has to infer (see hag.rl.envs.hidden_state: the velocities removed from
the observations, or the state of the POPGym environments), without reinforcement learning. Score: R² of the ridge on
held-out pretraining episodes, averaged over the components of the hidden state (each one standardized).

Validation of the probe as a proxy of the reinforcement learning objective: for completed and pruned trials of the
hyperparameter optimization of PPO (hag.rl.ppo.hpo, N_PROBED_TRIALS per study, drawn at random), the features of the
trial are rebuilt on the pretraining episodes of the first seed of the search (same reservoir as the first run of the
trial) and probed. The probe scores are compared (Spearman correlation) with the returns of the trial: its first run
(same reservoir) and its objective (mean over the seeds, completed trials). Results: <RL_RESULTS>/probe_validation.csv.
The pretraining episodes are collected again: their policy may differ slightly from the one of the search when the
search ran on another machine (PPO is not bitwise reproducible across processors).

Run from the repository root (results of the search in <RL_RESULTS>, see hag.rl.utils):
  HAG_RL_ENV=<benchmark> HAG_RL_RESULTS=outputs/rl_results/<benchmark> python -m hag.rl.probe
"""
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
import optuna
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

from hag.rl import experiment, search
from hag.rl.ppo import hpo
from hag.rl.pretrain import make_pipeline
from hag.rl.utils import N_WORKERS, RL_RESULTS

# =============================== PARAMETERS ===============================
N_PROBED_TRIALS = 60        # trials per study of the search
TEST_FRACTION = 0.3         # held-out episodes of the probe
ALPHAS = np.logspace(-3, 4, 15)  # regularizations of the ridge (chosen by generalized cross-validation)
SEED = 0                    # draw of the trials and of the held-out episodes
# ===========================================================================


def episode_features(pipeline, episode: np.ndarray) -> np.ndarray:
    """Features (T, n_features) of the observations (T, D) of one episode, as seen by PPO (FeaturePipeline.step)."""
    inputs = pipeline.transform_episode(episode)
    reservoir = pipeline.reservoir
    if reservoir is None:
        return inputs
    W, lr = reservoir["W"], reservoir["lr"]
    drive = inputs @ reservoir["Win"].T + reservoir["bias"]
    x, states = np.zeros(len(W)), np.empty((len(inputs), len(W)))
    for t in range(len(inputs)):
        x = (1 - lr) * x + lr * np.tanh(W @ x + drive[t])
        states[t] = x
    return states


def probe_score(features: list, targets: list, seed: int = SEED) -> float:
    """R² on held-out episodes (TEST_FRACTION) of a ridge decoding the targets (list of (T, H)) from the features (list
    of (T, F)), averaged over the targets (standardized; constant targets ignored)."""
    test = np.random.default_rng(seed).permutation(len(features))[:max(1, round(TEST_FRACTION * len(features)))]
    is_test = np.isin(np.arange(len(features)), test)
    X_train, X_test = (np.concatenate([f for f, t in zip(features, is_test) if t == flag]) for flag in (False, True))
    Y_train, Y_test = (np.concatenate([y for y, t in zip(targets, is_test) if t == flag]) for flag in (False, True))
    mean, std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8
    varying = Y_train.std(axis=0) > 1e-8
    y_mean, y_std = Y_train[:, varying].mean(axis=0), Y_train[:, varying].std(axis=0)
    ridge = RidgeCV(alphas=ALPHAS).fit((X_train - mean) / std, (Y_train[:, varying] - y_mean) / y_std)
    prediction = ridge.predict((X_test - mean) / std)
    return float(r2_score((Y_test[:, varying] - y_mean) / y_std, prediction))


def probe_trial(study_name: str, number: int, state: str, values: dict, rl_first: float, rl_value: float,
                episodes: list, hiddens: list) -> dict:
    """Probe of the features of a trial of the search (in a worker process)."""
    start = time.time()
    params = hpo.trial_params(study_name, values)
    features = {key: value for key, value in params.items() if key != "learning_rate"}
    kind = search.study_kind(study_name)
    pipeline = make_pipeline(search.study_condition(study_name), episodes, experiment.UNITS, search.SEED, features)
    W = None if pipeline.reservoir is None else pipeline.reservoir["W"]
    score = probe_score([episode_features(pipeline, episode) for episode in episodes], hiddens)
    return {"benchmark": experiment.ENV_ID, "study": study_name, "kind": kind, "trial": number, "state": state,
            "rl_first_run": rl_first, "rl_value": rl_value, "probe_r2": score, "n_features": pipeline.n_features,
            "connections_per_neuron": np.nan if W is None else np.count_nonzero(W) / len(W),
            "time_s": round(time.time() - start, 1)}


def summary(results: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlations between the probe scores and the returns, per study and over all the trials."""
    rows = []
    for study, group in list(results.groupby("study")) + [("all", results)]:
        completed = group[group.state == "COMPLETE"]
        rows.append({"study": study, "trials": len(group),
                     "rho_first_run": spearmanr(group.probe_r2, group.rl_first_run).statistic,
                     "rho_objective": spearmanr(completed.probe_r2, completed.rl_value).statistic
                     if len(completed) > 2 else np.nan,
                     "probe_r2_median": group.probe_r2.median(), "probe_r2_max": group.probe_r2.max()})
    return pd.DataFrame(rows)


def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = search.storage(hpo.DB_NAME)
    rng = np.random.default_rng(SEED)
    tasks = []
    for study_name in optuna.get_all_study_names(storage):
        trials = [t for t in optuna.load_study(study_name=study_name, storage=storage).trials
                  if t.state.name in ("COMPLETE", "PRUNED") and 0 in t.intermediate_values]
        for k in rng.permutation(len(trials))[:N_PROBED_TRIALS]:
            t = trials[k]
            tasks.append((study_name, t.number, t.state.name, t.params, t.intermediate_values[0],
                          t.value if t.state.name == "COMPLETE" else np.nan))
    print(f"[probe] {experiment.ENV_ID}: {len(tasks)} trials to probe", flush=True)
    output = RL_RESULTS / "probe_validation.csv"
    rows = []
    with ProcessPoolExecutor(N_WORKERS, mp_context=get_context("spawn")) as pool:
        episodes, hiddens = pool.submit(experiment.pretraining_episodes, search.SEED, True).result()
        futures = [pool.submit(probe_trial, *task, episodes, hiddens) for task in tasks]
        for k, future in enumerate(futures):
            rows.append(future.result())
            if (k + 1) % 50 == 0 or k + 1 == len(futures):
                pd.DataFrame(rows).to_csv(output, index=False)
                print(f"[probe] {experiment.ENV_ID}: {k + 1}/{len(futures)} trials probed", flush=True)
    results = pd.DataFrame(rows)
    print(summary(results).round(3).to_string(index=False))
    print(f"Results saved to {output}")


if __name__ == "__main__":
    main()
