import numpy as np

from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score
from performances.losses import nrmse_multivariate
from reservoirpy.nodes import Reservoir, IPReservoir, Ridge, RLS, LMS, NVAR, LocalPlasticityReservoir
from models.intrinsicSynapticPlasticityReservoir import IPLocalPlasticityReservoir
from reservoirpy import ESN

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


def init_ip_reservoir(W, Win, bias, mu, sigma, learning_rate, leaking_rate, activation_function):
    bias = np.asarray(bias).ravel()   # (units,)
    ip_reservoir = IPReservoir(
        units=bias.size,
        mu=mu,
        sigma=sigma,
        learning_rate=learning_rate,
        W=csr_matrix(W),
        Win=Win,
        lr=leaking_rate,
        bias=bias,                    # <- dense 1D
        activation=activation_function,
    )
    return ip_reservoir


def init_reservoir(W, Win, bias, leaking_rate, activation_function):
    bias = np.asarray(bias).ravel()   # (units,)
    reservoir = Reservoir(
        units=bias.size,
        W=csr_matrix(W),
        Win=Win,
        lr=leaking_rate,
        bias=bias,                    # <- dense 1D
        activation=activation_function,
    )
    return reservoir


def init_local_rule_reservoir(W, Win, bias, local_rule, eta, synapse_normalization, bcm_theta, leaking_rate, activation_function):
    local_rule_reservoir = LocalPlasticityReservoir(
        units=bias.size,
        local_rule=local_rule,
        eta=eta,
        synapse_normalization=synapse_normalization,
        bcm_theta=bcm_theta,
        W=csr_matrix(W),
        Win=Win,
        lr=leaking_rate,
        bias=csr_matrix(bias).T,
        activation=activation_function,
    )
    return local_rule_reservoir

def init_ip_local_rule_reservoir(W, Win, bias, mu, sigma, learning_rate, local_rule, eta, synapse_normalization, bcm_theta, leaking_rate, activation_function):
    ip_local_rule_reservoir = IPLocalPlasticityReservoir(
        units=bias.size,
        local_rule=local_rule,
        eta=eta,
        synapse_normalization=synapse_normalization,
        bcm_theta=bcm_theta,
        mu=mu,
        sigma=sigma,
        ip_learning_rate = learning_rate,  # IP learning rate
        W=csr_matrix(W),
        Win=Win,
        lr=leaking_rate,
        bias=csr_matrix(bias).T,
        activation="tanh",
    )
    return ip_local_rule_reservoir

def train_model_for_prediction(reservoir, readout, X_train, Y_train, n_jobs, warmup=2, rls=False, lms=False, verbosity=0):
    # IMPORTANT: name trainable nodes to satisfy reservoirpy's check_unnamed_trainable
    reservoir.name = "reservoir"
    readout.name = "readout"

    esn = ESN(reservoir=reservoir, readout=readout, workers=n_jobs)

    if rls or lms:
        # warmup once (see issue #2 below)
        if warmup > 0:
            esn.run(X_train[:warmup])

        esn.train(X_train[warmup:], Y_train[warmup:])
    else:
        if verbosity > 0:
            print(X_train.shape)
        esn.fit(X_train, Y_train, warmup=warmup)

    return esn


def train_model_for_classification(reservoir, readout, X_train, Y_train, mode, warmup=2):
    if mode == "sequence-to-vector":
        # Run ALL sequences in one call (batched)
        all_states = reservoir.run(X_train)
        # Extract last state of each sequence
        states_to_train_on = np.array([s[-1] for s in all_states])
        readout.fit(states_to_train_on, Y_train)
        return readout
    elif mode == "sequence-to-sequence":
        # Repeat targets to match sequence lengths
        Y_train_seq = [np.array([Y_train[i]] * len(x)) for i, x in enumerate(X_train)]
        esn = reservoir >> readout
        esn.fit(X_train, Y_train_seq, stateful=False, warmup=warmup)
        return esn
    else:
        raise ValueError(f"Invalid mode: {mode}")


def predict_model_for_classification(reservoir, readout, X_test, esn=None, mode="sequence-to-vector"):
    if mode == "sequence-to-vector":
        all_states = reservoir.run(X_test)
        states_to_predict = np.vstack([s[-1] for s in all_states])
        Y_pred = readout.run(states_to_predict)
        Y_pred = [y for y in Y_pred]  # convert to list if needed
    elif mode == "sequence-to-sequence":
        Y_pred = esn.run(X_test, stateful=False)
    else:
        raise ValueError(f"Invalid mode: {mode}")

    return Y_pred


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
