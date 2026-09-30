import numpy as np

from sklearn.metrics import accuracy_score
from hag.performances.losses import nrmse_multivariate
from reservoirpy.nodes import Reservoir, IPReservoir, Ridge, RLS, LMS, NVAR, LocalPlasticityReservoir
from hag.models.intrinsicSynapticPlasticityReservoir import IPLocalPlasticityReservoir
from scipy import sparse
from reservoirpy import activationsfunc
from reservoirpy.jax import nodes as jax_nodes
from hag.models import activation_functions
from hag.models.jax_local_plasticity_reservoir import (LocalPlasticityReservoir as JaxLocalPlasticityReservoir,
                                                    IPLocalPlasticityReservoir as JaxIPLocalPlasticityReservoir)
from hag.performances.batched_reservoir import run_batched

# NumPy activation functions and their reservoirpy.jax name
JAX_ACTIVATION_NAMES = {
    activation_functions.tanh: "tanh", activationsfunc.tanh: "tanh", np.tanh: "tanh",
    activation_functions.sigmoid: "sigmoid", activationsfunc.sigmoid: "sigmoid",
}

# NumPy reservoirpy reservoir -> (reservoirpy.jax node with the same forward step, fitted attributes to copy)
# (IPLocalPlasticityReservoir's forward step does not use its IP parameters a, b: same step as its parent)
JAX_NODES = {
    Reservoir: (jax_nodes.Reservoir, ()),
    IPReservoir: (jax_nodes.IPReservoir, ("a", "b")),
    LocalPlasticityReservoir: (JaxLocalPlasticityReservoir, ()),
    IPLocalPlasticityReservoir: (JaxLocalPlasticityReservoir, ()),
}


def _dense(m):
    return m.toarray() if sparse.issparse(m) else np.asarray(m, dtype=float)


def to_jax_node(reservoir):
    """
    reservoirpy.jax node with the same parameters as a NumPy reservoirpy reservoir (plastic reservoirs: their current,
    already fitted parameters), or None if not supported (reservoirpy then runs it): unknown reservoir or activation
    (e.g. a lambda), matrices or fitted attributes not initialized yet.
    """
    jax_type, fitted = JAX_NODES.get(type(reservoir), (None, ()))
    activation = JAX_ACTIVATION_NAMES.get(reservoir.activation)
    if (jax_type is None or activation is None
            or any(callable(getattr(reservoir, name)) for name in ("W", "Win", "bias"))
            or any(getattr(reservoir, name, None) is None for name in fitted)):
        return None

    W, Win = _dense(reservoir.W), _dense(reservoir.Win)
    node = jax_type(units=W.shape[0], W=W, Win=Win, bias=np.ravel(_dense(reservoir.bias)), lr=reservoir.lr,
                    activation=activation)
    node.initialize(np.zeros((1, Win.shape[1])))
    for name in fitted:
        setattr(node, name, np.asarray(getattr(reservoir, name)))
    return node


# NumPy reservoir learning a local plasticity rule -> (JAX node learning it, attributes to copy in both directions)
JAX_LEARNERS = {
    LocalPlasticityReservoir: (JaxLocalPlasticityReservoir, ()),
    IPLocalPlasticityReservoir: (JaxIPLocalPlasticityReservoir, ("mu", "sigma", "ip_learning_rate", "activation_type",
                                                                 "a", "b")),
}


def fit_reservoir(reservoir, x, warmup=0):
    """
    reservoir.fit(x, warmup=warmup). Local plasticity reservoirs learn in JAX (same weights, much faster) and the
    learned parameters are written back into the NumPy reservoir; the others are fitted by reservoirpy.
    """
    jax_type, ip_attributes = JAX_LEARNERS.get(type(reservoir), (None, ()))
    activation = JAX_ACTIVATION_NAMES.get(reservoir.activation)
    if jax_type is None or activation is None or any(callable(getattr(reservoir, name)) for name in ("W", "Win", "bias")):
        return reservoir.fit(x, warmup=warmup)
    if not reservoir.initialized:
        reservoir.initialize(x)

    node = jax_type(W=_dense(reservoir.W), Win=_dense(reservoir.Win), bias=np.ravel(_dense(reservoir.bias)),
                    lr=reservoir.lr, activation=activation, local_rule=reservoir.increment.__name__.replace("_", "-"),
                    eta=reservoir.eta, bcm_theta=reservoir.bcm_theta,
                    synapse_normalization=reservoir.synapse_normalization, epochs=reservoir.epochs,
                    **{name: getattr(reservoir, name) for name in ip_attributes})
    node.initialize()
    node.state = {key: np.asarray(reservoir.state[key], dtype=float) for key in ("internal", "out")}
    node.fit(x, warmup=warmup)

    reservoir.W = np.asarray(node.W)
    reservoir.state = {key: np.asarray(value) for key, value in node.state.items()}
    for name in ip_attributes:
        setattr(reservoir, name, np.asarray(getattr(node, name)) if name in ("a", "b") else getattr(node, name))
    return reservoir


def run_reservoir(reservoir, x, reset=True):
    """
    States of the reservoir on x, like reservoir.run(x): x is one (T, D) series or a list of series, each series
    starting from a zero state (reset=True) or from the reservoir's current state (reset=False).
    Run as a reservoirpy.jax node, batched and compiled (see batched_reservoir), for the supported reservoirs.
    """
    node = to_jax_node(reservoir)
    single = isinstance(x, np.ndarray) and x.dtype != object and x.ndim == 2
    if node is None:
        if reset and hasattr(reservoir, "state"):
            reservoir.reset()
        return reservoir.run(x)

    initial_state = None
    if not reset and getattr(reservoir, "state", None) is not None:
        initial_state = {key: reservoir.state[key] for key in node.state}
    states = run_batched(node, [x] if single else list(x), return_all=True, initial_state=initial_state)
    return states[0] if single else states


class BatchedESN:
    """Stands for `reservoir >> readout`, with the reservoir states computed by run_reservoir."""

    def __init__(self, reservoir, readout):
        self.reservoir, self.readout = reservoir, readout

    def run(self, x, **kwargs):
        # reservoirpy's reset / stateful arguments are accepted and ignored: each call starts from a zero state, as
        # the reservoir is reset after training
        return self.readout.run(run_reservoir(self.reservoir, x))


def init_readout(ridge_coef=None, rls=False, lms=False):
    """Select the proper readout according to flags."""
    if rls:
        return RLS()
    if lms:
        return LMS()
    return Ridge(ridge=ridge_coef)


def init_nvar_model(delay, order, strides=1):
    nvar_reservoir = NVAR(delay=delay, order=order, strides=strides)
    return nvar_reservoir


def init_ip_reservoir(W, Win, bias, mu, sigma, learning_rate, leaking_rate):
    bias = np.asarray(bias).ravel()   # (units,)
    ip_reservoir = IPReservoir(
        units=bias.size,
        mu=mu,
        sigma=sigma,
        learning_rate=learning_rate,
        W=np.asarray(W, dtype=np.float64),
        Win=Win,
        lr=leaking_rate,
        bias=bias,                    # <- dense 1D
        activation="tanh",
    )
    return ip_reservoir


def init_reservoir(W, Win, bias, leaking_rate, activation_function):
    bias = np.asarray(bias).ravel()   # (units,)
    reservoir = Reservoir(
        units=bias.size,
        W=np.asarray(W, dtype=np.float64),
        Win=Win,
        lr=leaking_rate,
        bias=bias,                    # <- dense 1D
        activation=activation_function,
    )
    return reservoir


def init_local_rule_reservoir(W, Win, bias, local_rule, eta, synapse_normalization, bcm_theta, leaking_rate, activation_function):
    bias = np.asarray(bias).ravel()  # (units,)
    local_rule_reservoir = LocalPlasticityReservoir(
        units=bias.size,
        local_rule=local_rule,
        eta=eta,
        synapse_normalization=synapse_normalization,
        bcm_theta=bcm_theta,
        W=np.asarray(W, dtype=np.float64),
        Win=Win,
        lr=leaking_rate,
        bias=bias,
        activation=activation_function,
    )
    return local_rule_reservoir

def init_ip_local_rule_reservoir(W, Win, bias, mu, sigma, learning_rate, local_rule, eta, synapse_normalization, bcm_theta, leaking_rate):
    bias = np.asarray(bias).ravel()  # (units,)
    ip_local_rule_reservoir = IPLocalPlasticityReservoir(
        units=bias.size,
        local_rule=local_rule,
        eta=eta,
        synapse_normalization=synapse_normalization,
        bcm_theta=bcm_theta,
        mu=mu,
        sigma=sigma,
        ip_learning_rate = learning_rate,  # IP learning rate
        W=np.asarray(W, dtype=np.float64),
        Win=Win,
        lr=leaking_rate,
        bias=bias,
        activation="tanh",
    )
    return ip_local_rule_reservoir

def train_model_for_prediction(reservoir, readout, X_train, Y_train, n_jobs, warmup=2, rls=False, lms=False, verbosity=0):
    # IMPORTANT: name trainable nodes to satisfy reservoirpy's check_unnamed_trainable
    reservoir.name = "reservoir"
    readout.name = "readout"

    if verbosity > 0:
        print("X_train.shape:", X_train.shape)
        print("Y_train.shape:", Y_train.shape)

    # Run reservoir from a zero state to obtain states (already-fitted plastic reservoirs keep their parameters)
    states = run_reservoir(reservoir, X_train)

    # Train only the readout, not the reservoir.
    states_train = states[warmup:]
    y_train = Y_train[warmup:]

    if rls or lms:
        for x_t, y_t in zip(states_train, y_train):
            readout.train(x_t, y_t)
    else:
        readout.fit(states_train, y_train)

    # Reset again so validation does not start from the final training state.
    if hasattr(reservoir, "state"):
        reservoir.reset()

    return BatchedESN(reservoir, readout)

def train_model_for_classification(reservoir, readout, X_train, Y_train, mode, warmup=2):
    if mode == "sequence-to-vector":
        states_to_train_on = _last_states_per_sequence(reservoir, X_train)
        readout.fit(states_to_train_on, Y_train)
        return readout
    elif mode == "sequence-to-sequence":
        Y_train_seq = [np.array([Y_train[i]] * len(x)) for i, x in enumerate(X_train)]

        all_states = run_reservoir(reservoir, X_train)
        readout.fit(all_states, Y_train_seq, warmup=warmup)

        return BatchedESN(reservoir, readout)
    else:
        raise ValueError(f"Invalid mode: {mode}")


def predict_model_for_classification(reservoir, readout, X_test, esn=None, mode="sequence-to-vector"):
    if mode == "sequence-to-vector":
        states_to_predict = _last_states_per_sequence(reservoir, X_test)
        Y_pred = readout.run(states_to_predict)
        Y_pred = [y for y in Y_pred]  # convert to list if needed
    elif mode == "sequence-to-sequence":
        Y_pred = esn.run(X_test, stateful=False)
    else:
        raise ValueError(f"Invalid mode: {mode}")

    return Y_pred


def _last_states_per_sequence(reservoir, sequences):
    # Supported reservoirs: batched reservoirpy.jax run from a zero state, same states much faster
    node = to_jax_node(reservoir)
    if node is not None:
        return run_batched(node, list(sequences))

    last_states = []
    for sequence in sequences:
        if hasattr(reservoir, "state"):
            reservoir.reset()
        states = reservoir.run(sequence)
        last_states.append(states[-1])
    if hasattr(reservoir, "state"):
        reservoir.reset()
    return np.vstack(last_states)


def compute_score(Y_pred, Y_test, is_instances_classification, model_name="", verbosity=0):
    if is_instances_classification:
        Y_pred_class = [np.argmax(y_p) for y_p in Y_pred]
        Y_test_class = [np.argmax(y_t) for y_t in Y_test]

        score = accuracy_score(Y_test_class, Y_pred_class)
    else:
        if len(Y_test.shape) == 1:
            Y_test = Y_test.reshape(-1, 1)
        if len(Y_pred.shape) == 1:
            Y_pred = Y_pred.reshape(-1, 1)
        score = float(nrmse_multivariate(Y_test, Y_pred))

    if verbosity > 0:
        print(f"Accuracy for {model_name}: {score * 100:.3f} %")
    return score
