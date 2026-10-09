"""
Sweep of the readout and of the reservoir size on the benchmark of hag.rl.experiment (HAG_RL_ENV): are the reservoirs
(and HAG) limited by the linear readout of PPO, by the 100 units, or by PPO itself (vs DQN, LSPI, evolution strategies,
natural actor-critic, and behavioral cloning, which needs no RL)?
Grid (one run per cell and seed of SEEDS):
  - PPO (hag.rl.ppo.train): networks of the policy and of the value function NET_ARCHS (linear readouts as in
    hag.rl.ppo.train, or MLPs), learning rates LEARNING_RATES (the one of the hyperparameter optimization was searched
    for linear readouts only: the best learning rate of each cell is selected afterwards, see summary),
  - DQN (hag.rl.dqn.train, benchmarks with discrete actions): Q-networks DQN_NET_ARCHS (linear readouts), learning
    rates DQN_LEARNING_RATES (best one per cell selected afterwards),
  - LSPI (hag.rl.lspi.train, benchmarks with discrete actions): linear readout, "ridge" and "gamma" of its
    hyperparameter optimization,
  - BC (hag.rl.bc.train, benchmarks with discrete actions): ridge readout cloning an expert that sees the hidden state
    (DAgger), the upper bound of the linear readouts; one expert per seed, shared by the cells,
  - CMA-ES (hag.rl.cmaes.train) and OpenAI-ES (hag.rl.openai_es.train): linear readout searched on the returns of
    episodes, initial step sizes CMAES_SIGMAS and learning rates OPENAI_ES_LEARNING_RATES (best one per cell selected
    afterwards),
  - NAC (hag.rl.nac.train, benchmarks with discrete actions): softmax policy updated along the natural gradient given
    by an LSTD-Q(lambda) critic, step sizes NAC_STEP_SIZES (best one per cell selected afterwards),
  - conditions of hag.rl.experiment.CONDITIONS and HYBRID_CONDITIONS (hybrids of an ESN and HAG, see
    hag.rl.pretrain.build_reservoir), reservoir sizes UNITS_GRID for the reservoirs ("esn", "hag", their control
    "proj" and the hybrids; rounded up as in hag.rl.pretrain.build_reservoir, the actual size is the column
    "n_features").
Features: best parameters of the hyperparameter optimization of the method (searched with UNITS units and linear
readouts), on the benchmark PARAMS_FROM[ENV_ID] for the harder variants without their own optimization (read from
PARAMS_ROOT/<benchmark>/); without optimization of the method (DQN, LSPI, and BC which has none), the features of the
best parameters of PPO (LSPI: with its default RIDGE and GAMMA). The hybrids take the parameters of the ESN (filter
bank, inputs, spectral radius and connectivity) and, in "hag", those of HAG (plasticity, and inputs of the HAG half of
"esn_hag").
LSPI and NAC runs whose system has more than LSPI_MAX_SIZE unknowns (n_actions * (n_features + 1) for LSPI,
(n_actions + 1) * (n_features + 1) for NAC) are skipped (column "skipped"): their dense solve is too slow and too large.
The runs already in the results are skipped (a job stopped by its time limit can be resubmitted).

Results: <RL_RESULTS>/sweep_<name>.csv (columns of hag.rl.ppo.train and hag.rl.lspi.train, and of the grid: "method",
"units", "arch", "learning_rate": the step size of the method, STEP_PARAMS, NaN for LSPI and BC).
Run from the repository root:  HAG_RL_ENV=<benchmark> python -m hag.rl.sweep [method ...]  (METHODS by default)
Benchmarks of the sweep (used by slurm/submit_rl.sh):  python -m hag.rl.sweep --benchmarks
"""
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

from hag.hpo.utility import PROJECT_ROOT
from hag.rl.bc import train as bc_train
from hag.rl.envs import has_discrete_actions, make_env
from hag.rl.experiment import CONDITIONS, ENV_ID, RESULTS_NAME, SEEDS, UNITS, condition_kind, pretraining_episodes
from hag.rl.pretrain import HYBRIDS
from hag.rl.utils import N_WORKERS, RL_RESULTS

# =============================== PARAMETERS ===============================
METHODS = ("ppo", "dqn", "lspi", "bc", "cmaes", "openai_es", "nac")
CONTINUOUS_METHODS = ("ppo", "cmaes", "openai_es")  # methods of the benchmarks with continuous actions
# benchmarks with discrete actions (all the methods), where the conditions differ or memory matters
SWEEP_BENCHMARKS = ("PositionOnlyCartPole", "LunarLander-v3", "RepeatPrevious", "RepeatPreviousMedium",
                    "RepeatPreviousHard", "CountRecall", "CountRecallMedium", "Autoencode")
UNITS_GRID = (100, 300, 1000)
HYBRID_CONDITIONS = tuple(f"filterbank+{hybrid}" for hybrid in HYBRIDS)
NET_ARCHS = {"linear": [], "64": [64], "64x64": [64, 64]}  # hidden layers of the policy and of the value function
LEARNING_RATES = (3e-4, 1e-3, 3e-3, 1e-2)                  # 3e-4: stable-baselines3's default (for its 64x64 MLP)
DQN_NET_ARCHS = ("linear",)
DQN_LEARNING_RATES = (1e-4, 3e-4, 1e-3, 3e-3)              # 1e-4: stable-baselines3's default
CMAES_SIGMAS = (0.03, 0.1, 0.3)
OPENAI_ES_LEARNING_RATES = (0.003, 0.01, 0.03)
NAC_STEP_SIZES = (0.03, 0.1, 0.3)
# parameter of each method stored in the column "learning_rate" (its swept step size)
STEP_PARAMS = {"ppo": "learning_rate", "dqn": "learning_rate", "openai_es": "learning_rate", "cmaes": "sigma0",
               "nac": "step_size"}
LSPI_MAX_SIZE = 10_000
# features of the harder variants: those of the hyperparameter optimization of their easy version
PARAMS_FROM = {"RepeatPreviousMedium": "RepeatPrevious", "RepeatPreviousHard": "RepeatPrevious",
               "CountRecallMedium": "CountRecall", "CountRecallHard": "CountRecall"}
PARAMS_ROOT = PROJECT_ROOT / "outputs" / "rl_results"     # results of the benchmarks, <benchmark>/<database>.sqlite3
# ===========================================================================

KEY = ("method", "condition", "units", "arch", "learning_rate", "seed")


class TooLarge(Exception):
    pass


def method_params(method: str) -> dict:
    """Parameters of the features (and of the method) per kind of features: best ones of the hyperparameter
    optimization of the method, else of PPO, on the benchmark PARAMS_FROM.get(ENV_ID, ENV_ID) (learning rate removed:
    it is swept). The databases are copied to a temporary folder first (SQLite does not support the network file
    systems well)."""
    from hag.rl.cmaes import hpo as cmaes_hpo
    from hag.rl.dqn import hpo as dqn_hpo
    from hag.rl.lspi import hpo as lspi_hpo
    from hag.rl.nac import hpo as nac_hpo
    from hag.rl.openai_es import hpo as openai_es_hpo
    from hag.rl.ppo import hpo as ppo_hpo

    own = {"dqn": dqn_hpo, "lspi": lspi_hpo, "cmaes": cmaes_hpo, "openai_es": openai_es_hpo, "nac": nac_hpo}
    searches = [own[method], ppo_hpo] if method in own else [ppo_hpo]
    source = PARAMS_FROM.get(ENV_ID, ENV_ID)
    params = {}
    with tempfile.TemporaryDirectory() as root:
        for hpo in searches:
            path = PARAMS_ROOT / source / f"{hpo.DB_PREFIX}_{source}.sqlite3"
            if path.exists():
                shutil.copy(path, root)
        for kind in ("obs", "filterbank", "esn", "hag"):
            for hpo in searches:
                best = hpo.best_params(kind, f"{hpo.DB_PREFIX}_{source}", Path(root))
                if best is not None:
                    break
            if best is None:
                raise FileNotFoundError(f"No hyperparameter optimization of {method} or PPO for {kind} in "
                                        f"{PARAMS_ROOT / source}")
            # (features of PPO for LSPI: hag.rl.lspi.train.run uses its default ridge and gamma)
            params[kind] = {key: value for key, value in best.items()
                            if key not in ("learning_rate", STEP_PARAMS.get(method))}
    for hybrid in HYBRIDS:  # parameters of the ESN, and of HAG in "hag"
        params[hybrid] = dict(params["esn"], hag=params["hag"])
    return params


def grid(methods) -> list:
    """Cells and seeds of the sweep, the longest runs first (largest reservoirs and networks)."""
    specs = []
    for method in methods:
        for condition in CONDITIONS + list(HYBRID_CONDITIONS):
            reservoir = condition_kind(condition) in ("esn", "hag") + HYBRIDS
            for units in (UNITS_GRID if reservoir else (0,)):  # 0: no reservoir
                cells = {"ppo": [(arch, lr) for arch in NET_ARCHS for lr in LEARNING_RATES],
                         "dqn": [(arch, lr) for arch in DQN_NET_ARCHS for lr in DQN_LEARNING_RATES],
                         "cmaes": [("linear", sigma0) for sigma0 in CMAES_SIGMAS],
                         "openai_es": [("linear", lr) for lr in OPENAI_ES_LEARNING_RATES],
                         "nac": [("linear", step_size) for step_size in NAC_STEP_SIZES]}.get(
                    method, [("linear", np.nan)])
                specs += [dict(method=method, condition=condition, units=units, arch=arch, learning_rate=lr, seed=seed)
                          for arch, lr in cells for seed in SEEDS]
    return sorted(specs, key=lambda spec: (-spec["units"], -len(NET_ARCHS[spec["arch"]])))


def key(row) -> tuple:
    return (row["method"], row["condition"], int(row["units"]), row["arch"], str(float(row["learning_rate"])),
            int(row["seed"]))


def run(spec: dict, episodes: list, params: dict, expert_info: dict = None) -> dict:
    """Run of one cell and seed (in a worker process). expert_info: expert of the seed (BC, see
    hag.rl.bc.train.expert)."""
    from hag.rl.dqn import train as dqn_train
    from hag.rl.lspi import train as lspi_train
    from hag.rl.ppo import train as ppo_train

    condition, seed, units = spec["condition"], spec["seed"], spec["units"] or UNITS
    arch = NET_ARCHS[spec["arch"]]
    if spec["method"] == "ppo":
        row = ppo_train.run(condition, seed, episodes, params=dict(params, learning_rate=spec["learning_rate"]),
                            save_curves=False, units=units, net_arch=dict(pi=list(arch), vf=list(arch)))
    elif spec["method"] == "dqn":
        row = dqn_train.run(condition, seed, episodes, params=dict(params, learning_rate=spec["learning_rate"]),
                            save_curves=False, units=units, net_arch=list(arch))
    elif spec["method"] == "bc":
        row = bc_train.run(condition, seed, episodes, params=params, save_curves=False, units=units,
                           expert_info=expert_info)
    elif spec["method"] in ("cmaes", "openai_es"):
        from importlib import import_module
        train = import_module(f"hag.rl.{spec['method']}.train")
        step = {STEP_PARAMS[spec["method"]]: spec["learning_rate"]}
        row = train.run(condition, seed, episodes, params=dict(params, **step), save_curves=False, units=units)
    else:  # (LSPI and NAC: dense linear systems)
        from hag.rl.nac import train as nac_train
        n_actions = make_env(ENV_ID).action_space.n
        n_rows = n_actions if spec["method"] == "lspi" else n_actions + 1

        def check(pipeline):
            if n_rows * (pipeline.n_features + 1) > LSPI_MAX_SIZE:
                raise TooLarge(pipeline.n_features)

        try:
            if spec["method"] == "lspi":
                row = lspi_train.run(condition, seed, episodes, params=params, save_curves=False, check=check,
                                     units=units)
            else:
                row = nac_train.run(condition, seed, episodes, params=dict(params, step_size=spec["learning_rate"]),
                                    save_curves=False, check=check, units=units)
        except TooLarge as error:
            row = {"env": ENV_ID, "n_features": error.args[0],
                   "skipped": f"{spec['method'].upper()} system larger than {LSPI_MAX_SIZE} unknowns"}
    return dict(row, **spec)


def summary(results: pd.DataFrame) -> pd.DataFrame:
    """Per cell (method, condition, units, arch): learning rate of PPO with the best mean return of the training
    episodes (sample efficiency, over the seeds), and the returns of its runs (mean and std over the seeds)."""
    results = results[results.get("skipped", pd.Series(np.nan, index=results.index)).isna()]
    cells = ["method", "condition", "units", "arch"]
    metrics = ["eval_return_mean", "train_return_mean", "n_features"]
    means = results.fillna({"learning_rate": 0}).groupby(cells + ["learning_rate"])[metrics].agg(["mean", "std"])
    best = means[("train_return_mean", "mean")].groupby(level=cells).idxmax()
    best = means.loc[best.values].round(3).reset_index(level="learning_rate")
    best["learning_rate"] = [f"{lr:.0e}" if lr else "-" for lr in best["learning_rate"]]  # ("-": LSPI)
    return best


def main(methods):
    output = RL_RESULTS / f"sweep_{RESULTS_NAME}.csv"
    rows = pd.read_csv(output).to_dict("records") if output.exists() else []
    done = {key(row) for row in rows}
    specs = [spec for spec in grid(methods) if key(spec) not in done]
    print(f"[sweep] {ENV_ID}: {len(specs)} runs ({len(done)} already done), {N_WORKERS} workers", flush=True)
    params = {method: method_params(method) for method in methods}
    with ProcessPoolExecutor(N_WORKERS, mp_context=get_context("spawn")) as pool:
        # experts of BC (trained if not saved yet), at the same time as the pretraining episodes
        experts = {seed: pool.submit(bc_train.expert, seed) for seed in SEEDS} if "bc" in methods else {}
        episodes = dict(zip(SEEDS, pool.map(pretraining_episodes, SEEDS)))
        experts = {seed: future.result() for seed, future in experts.items()}
        futures = [pool.submit(run, spec, episodes[spec["seed"]],
                               params[spec["method"]][condition_kind(spec["condition"])], experts.get(spec["seed"]))
                   for spec in specs]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(f"[sweep] {ENV_ID} {row['method']} {row['condition']:16s} units {row['units']:4d} {row['arch']:6s} "
                  f"lr {row['learning_rate']:.0e} seed {row['seed']}: "
                  + (row["skipped"] if isinstance(row.get("skipped"), str) else
                     f"eval {row['eval_return_mean']:7.3f} | train mean {row['train_return_mean']:7.3f} | "
                     f"{row['n_features']} features | {row['time_s']}s"), flush=True)
            # saved after each run, so that an interrupted sweep keeps its results
            output.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(output, index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(summary(pd.DataFrame(rows)).to_string())
    print(f"Results saved to {output}")


if __name__ == "__main__":
    if "--benchmarks" in sys.argv[1:]:
        print("\n".join(SWEEP_BENCHMARKS))
    else:
        main(sys.argv[1:] or list(METHODS if has_discrete_actions(ENV_ID) else CONTINUOUS_METHODS))
