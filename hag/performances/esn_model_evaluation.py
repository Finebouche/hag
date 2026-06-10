import numpy as np

from sklearn.metrics import accuracy_score
from hag.performances.losses import nrmse_multivariate
from reservoirpy.nodes import Reservoir, IPReservoir, Ridge, RLS, LMS, NVAR, LocalPlasticityReservoir
from hag.models.intrinsicSynapticPlasticityReservoir import IPLocalPlasticityReservoir

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

    # Reset only if the reservoir has already been initialized.
    # Plain Reservoir objects do not have `.state` before first run.
    if hasattr(reservoir, "state"):
        reservoir.reset()

    # Run reservoir to obtain states.
    # This initializes plain Reservoirs, and uses already-fitted plastic Reservoirs.
    states = reservoir.run(X_train)

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

    return reservoir >> readout

def train_model_for_classification(reservoir, readout, X_train, Y_train, mode, warmup=2):
    if mode == "sequence-to-vector":
        states_to_train_on = _last_states_per_sequence(reservoir, X_train)
        readout.fit(states_to_train_on, Y_train)
        return readout
    elif mode == "sequence-to-sequence":
        Y_train_seq = [np.array([Y_train[i]] * len(x)) for i, x in enumerate(X_train)]

        all_states = reservoir.run(X_train)
        readout.fit(all_states, Y_train_seq, warmup=warmup)

        return reservoir >> readout
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
