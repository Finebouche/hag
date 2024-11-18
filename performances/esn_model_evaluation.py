import numpy as np

from scipy.sparse import csr_matrix
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from performances.losses import nrmse_multivariate
from reservoirpy.nodes import Reservoir, IPReservoir, Ridge, RLS, LMS, NVAR
import reservoirpy

reservoirpy.verbosity(level=0)

def init_nvar_model(delay, order, strides=1, ridge_coef=None, rls=False, lms=False):
    nvar_reservoir = NVAR(delay=delay, order=order, strides=strides)
    if rls:
        readout = RLS()
    elif lms:
        readout = LMS()
    else:
        readout = Ridge(ridge=ridge_coef)
    return nvar_reservoir, readout


def init_ip_reservoir_model(W, Win, bias, mu, sigma, leaking_rate, activation_function, ridge_coef=None, rls=False, lms=False):
    ip_reservoir = IPReservoir(
        units=bias.size,
        mu=mu,
        sigma=sigma,
        W=csr_matrix(W),
        Win=Win,
        lr=leaking_rate,
        bias=csr_matrix(bias).T,
        activation="tanh",
    )
    if rls:
        readout = RLS()
    elif lms:
        readout = LMS()
    else:
        readout = Ridge(ridge=ridge_coef)
    return ip_reservoir, readout


def init_reservoir_model(W, Win, bias, leaking_rate, activation_function, ridge_coef=None, rls=False, lms=False):
    reservoir = Reservoir(units=bias.size,
                          W=csr_matrix(W),
                          Win=Win,
                          lr=leaking_rate,
                          bias=csr_matrix(bias).T,
                          activation=activation_function,
                          equation='external')

    if rls:
        readout = RLS()
    elif lms:
        readout = LMS()
    else:
        readout = Ridge(ridge=ridge_coef)
    return reservoir, readout

def train_model_for_prediction(reservoir, readout, X_train, Y_train, warmup=2, rls=False, lms=False):
    esn = reservoir >> readout
    if rls or lms:
        for i in range(warmup):
            esn.run(X_train[:warmup])
        esn.train(X_train[warmup:], Y_train[warmup:])
    else:
        esn.fit(X_train, Y_train, warmup=warmup)

    return esn


def train_model_for_classification(reservoir, readout, X_train, Y_train, n_jobs, mode, warmup=2, hide_progress=False):
    if mode == "sequence-to-vector":
        def compute_state(x):
            import reservoirpy
            reservoirpy.verbosity(level=0)

            return reservoir.run(x, reset=True)[-1, np.newaxis].flatten()

        states_to_train_on = Parallel(n_jobs=n_jobs)(
            delayed(compute_state)(x) for x in tqdm(X_train, desc="Processing", dynamic_ncols=True, disable=hide_progress)
        )

        states_to_train_on = np.array(states_to_train_on)
        readout.fit(states_to_train_on, Y_train)

    elif mode == "sequence-to-sequence":
        # make Y_train repeat_targets
        # for each sequence in X_train, the corresponding target is Ytrain repeated as many times as the sequence length
        Y_train = [np.array([Y_train[i]] * len(x)) for i, x in enumerate(X_train)]
        esn = reservoir >> readout
        esn.fit(X_train, Y_train, stateful=False, warmup=warmup)
        return esn
    else:
        raise ValueError(f"Invalid mode: {mode}")


def predict_model_for_classification(reservoir, readout, X_test, esn=None, n_jobs=1, mode="sequence-to-vector", hide_progress=False):
    if mode == "sequence-to-vector":
        def predict(x):
            import reservoirpy
            reservoirpy.verbosity(level=0)

            states = reservoir.run(x, reset=True)[-1, np.newaxis].flatten() # read from the last state of the reservoir
            y = readout.run(states)
            return y

        Y_pred = Parallel(n_jobs=n_jobs)(delayed(predict)(x) for x in tqdm(X_test, desc="Evaluating", disable=hide_progress))
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
