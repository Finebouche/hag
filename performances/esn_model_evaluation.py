import numpy as np

from scipy.sparse import csr_matrix
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from reservoirpy.nodes import Reservoir, Ridge, ESN


def train_and_predict_model(W, Win, bias, activation_function, ridge_coef, X_train, X_test, Y_train, n_jobs):
    # To remember :
    #  For reservoirpy   pre_s = W @ r + Win @ (u + noise_gen(dist=dist, shape=u.shape, gain=g_in)) + bias

    reservoir = Reservoir(units=bias.size,
                          W =csr_matrix(W),
                          Win=csr_matrix(np.diag(Win.toarray().flatten())),
                          bias=csr_matrix(bias).T,
                          activation=activation_function,
                          equation='external'
                         )
    readout = Ridge(ridge=ridge_coef)

    def compute_state(x):
        import reservoirpy
        reservoirpy.verbosity(level=0)

        return reservoir.run(x, reset=True)[-1, np.newaxis].flatten()

    states_train = Parallel(n_jobs=n_jobs)(
        delayed(compute_state)(x) for x in tqdm(X_train, desc="Processing", dynamic_ncols=True)
    )
    readout.fit(np.array(states_train), Y_train)

    def predict(x):
        import reservoirpy
        reservoirpy.verbosity(level=0)

        states = reservoir.run(x, reset=True)
        y = readout.run(states[-1, np.newaxis])
        return y

    Y_pred = Parallel(n_jobs=n_jobs)(delayed(predict)(x) for x in X_test)

    return Y_pred

def compute_score(Y_pred, Y_test, model_name, verbosity=1):
    Y_pred_class = [np.argmax(y_p) for y_p in Y_pred]
    Y_test_class = [np.argmax(y_t) for y_t in Y_test]

    score = accuracy_score(Y_test_class, Y_pred_class)

    if verbosity > 0:
        print(f"Accuracy for {model_name}: {score * 100:.3f} %")
    return score