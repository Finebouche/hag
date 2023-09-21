import numpy as np
from reservoirpy.nodes import Reservoir, Ridge, Input, ESN
from scipy.sparse import csr_matrix
from joblib import Parallel, delayed
from sklearn.metrics import mutual_info_score
from sklearn.metrics import accuracy_score

def train_and_predict_model(W, Win, bias, activation_function, ridge_coef, X_train, X_test, Y_train, Y_test, n_jobs):
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
    model = ESN(reservoir=reservoir, readout=readout)
    
    states_train = []

    def compute_state(x):
        return reservoir.run(x, reset=True)[-1, np.newaxis].flatten()
    
    states_train = Parallel(n_jobs=n_jobs)(delayed(compute_state)(x) for x in X_train)

    readout.fit(np.array(states_train), Y_train)

    Y_pred = []
    def predict(x):
        states = reservoir.run(x, reset=True)
        y = readout.run(states[-1, np.newaxis])
        return y

    Y_pred = Parallel(n_jobs=n_jobs)(delayed(predict)(x) for x in X_test)

    return Y_pred

def compute_score(Y_pred, Y_test, model_name):
    Y_pred_class = [np.argmax(y_p) for y_p in Y_pred]
    Y_test_class = [np.argmax(y_t) for y_t in Y_test]

    score = accuracy_score(Y_test_class, Y_pred_class)

    print(f"Accuracy for {model_name}: {score * 100:.3f} %")
    return score