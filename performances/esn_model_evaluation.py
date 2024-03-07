import numpy as np

from scipy.sparse import csr_matrix
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from reservoirpy.nodes import Reservoir, Ridge, ESN
import reservoirpy
reservoirpy.verbosity(level=0)

def init_model(W, Win, bias, activation_function, ridge_coef):
    reservoir = Reservoir(units=bias.size,
                          W=csr_matrix(W),
                          Win=Win,
                          bias=csr_matrix(bias).T,
                          activation=activation_function,
                          equation='external')
    readout = Ridge(ridge=ridge_coef)
    return reservoir, readout

def train_model_for_classification(reservoir, readout, X_train, Y_train, n_jobs):
    def compute_state(x):
        import reservoirpy
        reservoirpy.verbosity(level=0)

        return reservoir.run(x, reset=True)[-1, np.newaxis].flatten()

    states_to_train_on = Parallel(n_jobs=n_jobs)(
        delayed(compute_state)(x) for x in tqdm(X_train, desc="Processing", dynamic_ncols=True)
    )
    readout.fit(np.array(states_to_train_on), Y_train)

def predict_model_for_classification(reservoir, readout, X_test, n_jobs):
    def predict(x):
        import reservoirpy
        reservoirpy.verbosity(level=0)

        states = reservoir.run(x, reset=True)
        y = readout.run(states[-1, np.newaxis]) # read from the last state of the reservoir
        return y

    Y_pred = Parallel(n_jobs=n_jobs)(delayed(predict)(x) for x in tqdm(X_test, desc="Evaluating"))
    return Y_pred

def init_and_train_model_for_classification(W, Win, bias, activation_function, ridge_coef, X_train, Y_train, n_jobs):

    reservoir, readout = init_model(W, Win, bias, activation_function, ridge_coef)

    train_model_for_classification(reservoir, readout, X_train, Y_train, n_jobs)

    return reservoir, readout


def init_and_train_model_for_prediction(W, Win, bias, activation_function, ridge_coef, X_train, Y_train):
    reservoir, readout = init_model(W, Win, bias, activation_function, ridge_coef)

    states_to_train_on = reservoir.run(X_train)
    readout.fit(np.array(states_to_train_on), Y_train)

    return reservoir, readout

def compute_score(Y_pred, Y_test, model_name, verbosity=1):
    Y_pred_class = [np.argmax(y_p) for y_p in Y_pred]
    Y_test_class = [np.argmax(y_t) for y_t in Y_test]

    score = accuracy_score(Y_test_class, Y_pred_class)

    if verbosity > 0:
        print(f"Accuracy for {model_name}: {score * 100:.3f} %")
    return score