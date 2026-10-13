"""
Reservoirs of the canary boundary detection, in JAX: HAG learns with the JAX HAGReservoir (hag.models.jax_hag_reservoir,
same algorithm as run_algorithm with the most correlated partners, random choices from Jax's generator), and all the
reservoirs run as reservoirpy.jax nodes (hag.performances.esn_model_evaluation.run_reservoir). The other matrices are
those of hag.performances.evaluation_esn.build_reservoir_from_study (HAG's evaluation).

Models (name -> reservoirs):
  - "mfcc": no reservoir (the readout sees the MFCC).
  - pure models of SIZE neurons: "mean_hag", "var_hag" (HAG learning on phrases of the train songs, empty initial W),
    "random_ee" (ESN), "random_ee_dense" (ESN with a dense random input matrix instead of the block one: by default all
    the models map each MFCC coefficient to its own block of neurons, see hag.models.reservoir.init_matrices).
  - "<hag>_proj": the matrices of <hag> without the HAG learning (control).
  - hybrids "<a>+<b>": two reservoirs of SIZE / 2 neurons each, run side by side without connections between them;
    the readout sees the concatenation of their states. Their parameters are those of the parts, prefixed by
    "<a>:" and "<b>:".
Real sizes are multiples of the number of MFCC features (39): build_reservoir_from_study rounds up.
"""
import numpy as np

from hag.hpo.utility import hag_reservoir_from_hyperparameters
from hag.models.activation_functions import tanh
from hag.performances.esn_model_evaluation import init_reservoir, run_reservoir, to_jax_node
from hag.performances.evaluation_esn import build_reservoir_from_study

HAG_MODELS = ("mean_hag", "var_hag")
ESN_MODELS = ("random_ee", "random_ee_dense")
PURE_MODELS = HAG_MODELS + ESN_MODELS


def parts(model):
    """[(part model, parameter prefix)] of a pure or hybrid model."""
    if "+" in model:
        return [(part, f"{part}:") for part in model.split("+")]
    return [(model, "")]


def suggest(trial, model, max_phrase_length):
    """Parameters of the model (pure or hybrid) suggested by an Optuna trial, same ranges as hag.hpo.hpo_esn."""
    params = {}
    for part, prefix in parts(model):
        def name(key):
            return prefix + key
        params[name("input_scaling")] = trial.suggest_float(name("input_scaling"), 0.01, 0.2, step=0.005)
        params[name("bias_scaling")] = trial.suggest_float(name("bias_scaling"), 0, 0.2, step=0.005)
        if part in HAG_MODELS:
            params[name("weight_increment")] = trial.suggest_float(name("weight_increment"), 0.001, 0.1, step=0.001)
            params[name("use_full_instance")] = trial.suggest_categorical(name("use_full_instance"), [True, False])
            params[name("min_increment")] = trial.suggest_int(name("min_increment"), 3, max_phrase_length)
            params[name("max_increment")] = trial.suggest_int(name("max_increment"), params[name("min_increment")],
                                                               max_phrase_length * 5)
            if part == "mean_hag":
                params[name("target_rate")] = trial.suggest_float(name("target_rate"), 0.5, 1, step=0.01)
                params[name("rate_spread")] = trial.suggest_float(name("rate_spread"), 0.01, 0.4, step=0.005)
            else:
                params[name("variance_target")] = trial.suggest_float(name("variance_target"), 0.001, 0.02, step=0.001)
                params[name("variance_spread")] = trial.suggest_float(name("variance_spread"), 0.001, 0.05, step=0.002)
                params[name("intrinsic_saturation")] = trial.suggest_float(name("intrinsic_saturation"), 0.8, 0.98,
                                                                           step=0.02)
                params[name("intrinsic_coef")] = trial.suggest_float(name("intrinsic_coef"), 0.8, 0.98, step=0.02)
        elif part in ESN_MODELS:
            params[name("connectivity")] = trial.suggest_float(name("connectivity"), 0, 1)
            params[name("spectral_radius")] = trial.suggest_float(name("spectral_radius"), 0.4, 1.6, step=0.01)
        else:
            raise ValueError(f"Unknown model {part}")
    return params


def build(model, params, size, pretrain):
    """Reservoirs of the model with SIZE neurons in all (None for "mfcc"), matrices drawn with numpy's global RNG."""
    if model == "mfcc":
        return None
    plasticity = not model.endswith("_proj")
    model = model.removesuffix("_proj")
    model_parts = parts(model)
    reservoirs = []
    for part, prefix in model_parts:
        hyperparams = {key[len(prefix):]: value for key, value in params.items() if key.startswith(prefix)
                       and ":" not in key[len(prefix):]}
        hyperparams.update(network_size=size // len(model_parts), leaky_rate=1, input_connectivity=1)
        if part in HAG_MODELS:
            hyperparams["connectivity"] = 0
        if part in HAG_MODELS and plasticity:
            # JAX HAG learning, initial matrices of HAG's evaluation (init_matrices) drawn with this seed
            node = hag_reservoir_from_hyperparameters(hyperparams, part, pretrain[0].shape[1],
                                                      seed=int(np.random.randint(0, 2 ** 31 - 2 ** 16)), jax=True)
            node.fit(list(pretrain))
            reservoir = init_reservoir(np.asarray(node.W), np.asarray(node.Win), np.ravel(np.asarray(node.bias)), 1,
                                       tanh)
        else:
            reservoir, _ = build_reservoir_from_study(part.removesuffix("_dense"), hyperparams, pretrain, True,
                                                      random_projection_experiment=part.endswith("_dense"),
                                                      plasticity=plasticity)
        if to_jax_node(reservoir) is None:
            raise RuntimeError(f"{part}: no reservoirpy.jax equivalent, it would run in NumPy")
        reservoirs.append(reservoir)
    return reservoirs


def states(reservoirs, inputs):
    """States (float32) of the reservoirs on each input series, concatenated (the inputs themselves without reservoir)."""
    if reservoirs is None:
        return [np.asarray(x, dtype=np.float32) for x in inputs]
    runs = [run_reservoir(reservoir, inputs, reset=True) for reservoir in reservoirs]
    return [np.hstack([np.asarray(run[k], dtype=np.float32) for run in runs]) for k in range(len(inputs))]


CHUNK = 32  # songs per batched run: the states of all the songs at once do not fit in memory with 1000 neurons


def state_chunks(reservoirs, songs, waves, chunk=CHUNK):
    """(waves, states) by chunks of songs of similar lengths (less padding in the batched JAX runs)."""
    order = sorted(waves, key=lambda w: len(songs[w]["mfcc"]))
    for i in range(0, len(order), chunk):
        part = order[i:i + chunk]
        yield part, states(reservoirs, [songs[w]["mfcc"] for w in part])


def normal_equations(reservoirs, songs, waves, target="target"):
    """
    Centered statistics of the ridge readout on the targets songs[w][target] (T,) or (T, k), accumulated song by song:
    (scatter matrix of the states, cross-scatter with the targets, mean state, mean target). Each song gives its centered
    statistics, combined with the formula of the parallel variance (Chan et al.): no subtraction of n * mean mean^T from
    the uncentered X^T X, which loses the precision of nearly dependent states with a large mean (the readout then
    depends on rounding errors with a small ridge).
    """
    Sxx = Sxy = x_mean = y_mean = None
    total = 0
    for part, part_states in state_chunks(reservoirs, songs, waves):
        for x, w in zip(part_states, part):
            x, y = np.asarray(x, dtype=np.float64), np.asarray(songs[w][target], dtype=np.float64)
            n, xm, ym = len(x), x.mean(axis=0), y.mean(axis=0)
            xc, yc = x - xm, y - ym
            if Sxx is None:
                Sxx, Sxy, x_mean, y_mean = np.zeros((x.shape[1], x.shape[1])), np.zeros((x.shape[1],) + y.shape[1:]), xm, ym
                Sxx += xc.T @ xc
                Sxy += xc.T @ yc
                total = n
                continue
            new_total = total + n
            dx, dy = xm - x_mean, ym - y_mean
            Sxx += xc.T @ xc + (total * n / new_total) * np.outer(dx, dx)
            Sxy += xc.T @ yc + (total * n / new_total) * (np.outer(dx, dy) if dy.ndim else dx * dy)
            x_mean = x_mean + dx * n / new_total
            y_mean = y_mean + dy * n / new_total
            total = new_total
    return Sxx, Sxy, x_mean, y_mean


def ridge_readout(statistics, ridge):
    """Ridge readout (W stacked on its bias, as a last row) from the centered statistics of normal_equations."""
    Sxx, Sxy, x_mean, y_mean = statistics
    W = np.linalg.solve(Sxx + ridge * np.eye(len(Sxx)), Sxy)
    return np.vstack([W, (y_mean - x_mean @ W)[None]]) if W.ndim == 2 else np.append(W, y_mean - x_mean @ W)


def readout_outputs(reservoirs, songs, waves, Wouts):
    """{wave: (T, len(Wouts)) outputs of the ridge readouts Wouts (columns, with bias)} on the songs, by chunks."""
    Wouts = np.column_stack(Wouts)
    outputs = {}
    for part, part_states in state_chunks(reservoirs, songs, waves):
        for x, w in zip(part_states, part):
            outputs[w] = x.astype(np.float64) @ Wouts[:-1] + Wouts[-1]
    return outputs
