from scipy import linalg, sparse, stats
import numpy as np

def update_ei_reservoir(W_ee, W_ie, W_ei, Win, u, r_e, r_i, leaky_rate, bias_e, bias_i, activation_function):
    pre_s_e = (1 - leaky_rate) * r_e + leaky_rate * (W_ee @ r_e - W_ie @ r_i) + (Win.A.flatten() * u) + bias_e
    pre_s_i = (1 - leaky_rate) * r_i + leaky_rate * W_ei @ r_e + bias_i
    return activation_function(pre_s_e), activation_function(pre_s_i)



def ridge_regression(R, Ytrain, ridge_coef):
    # R          : the states' matrix
    # Ytrain     : the data to reproduce
    # ridge_coef : the ridge coefficient

    # Part to add b_out
    R = np.concatenate((np.ones((1, R.shape[1])), R))

    # W_out = (Ytrain@R.T) @ (linalg.inv(R@R.T + ridge_coef*np.eye(R.shape[0])))
    if ridge_coef == 0:
        W_out = linalg.solve(R @ R.T, R @ Ytrain.T).T
    else:
        I = np.eye(R.shape[0])
        I[0:0] = 0
        W_out = linalg.solve(R @ R.T + ridge_coef * I, R @ Ytrain.T).T

    b_out = W_out[:, 0]
    W_out = W_out[:, 1:]

    return W_out, b_out



def train_ei(W_ee, W_ie, W_ei, Win_e, bias_e, bias_i, Utrain, Ytrain, activation_function, ridge_coef=1e-8,
             init_len=0, leaky_rate=1, state_e=None, state_i=None):
    # state_e         : the initial state before the train (We start from a null state)

    n_e = Win_e.shape[0]
    n_i = W_ie.shape[1]
    # We initialize the states to random if there is no state provided
    if state_e is None:
        state_e = np.random.uniform(-1, 1, n_e)
    if state_i is None:
        state_i = np.random.uniform(-1, 1, n_i)

    # run the reservoir with the data and collect R = (u, state)
    seq_len = len(Utrain)
    if Utrain.ndim == 2 and Utrain.shape[1] > 1:
        seq_len = len(Utrain[1,:])

    R = np.zeros((n_e, seq_len - init_len))
    for t in range(seq_len):
        if Utrain.ndim == 1 or Utrain.shape[1] == 1 or Utrain.shape[0] == 1:
            u = Utrain[t]
        elif Utrain.ndim == 2:
            u = Utrain[:, t]

        state_e, state_i = update_ei_reservoir(W_ee, W_ie, W_ei, Win_e, u, state_e, state_i, leaky_rate, bias_e, bias_i,
                                               activation_function)
        # we collect after the initialisation of the reservoir (default = 0)
        if t > init_len:
            R[:, t - init_len] = state_e
    #             R[:,t-init_len] = np.concatenate((u, state))

    # Ensure that for Ytrain and R : columns -> iteration and rows -> neurons
    # Ytrain and R  have the "standard" shape for Ridge Regression
    Ytrain = Ytrain[init_len:, :]
    Y = Ytrain.T
    # Compute Wout using Ridge Regression
    Wout, b_out = ridge_regression(R, Y, ridge_coef)

    return Wout, b_out, state_e, state_i


def run_ei(W_ee, W_ie, W_ei, Win_e, bias_e, bias_i, Wout, U, activation_function, b_out=None, leaky_rate=1, last_state_e=None, last_state_i=None):
    # We start from previous state or else from uniform random state
    n_e = Win_e.shape[0]
    n_i = W_ie.shape[1]
    # We initialize the states to random if there is no state provided
    if last_state_e is None:
        state_e = np.random.uniform(-1, 1, n_e)
    else:
        state_e = last_state_e

    if last_state_i is None:
        state_i = np.random.uniform(-1, 1, n_i)
    else:
        state_i = last_state_i

    seq_len = len(U)
    if U.ndim == 2 and U.shape[1] > 1:
        seq_len = len(U[1,:])

    R = np.zeros((n_e, seq_len))
    for t in range(seq_len):
        if U.ndim == 1 or U.shape[1] == 1 or U.shape[0] == 1:
            u = U[t]
        elif U.ndim == 2 and U.shape[1] > 1:
            u = U[:,t]
        state_e, state_i = update_ei_reservoir(W_ee, W_ie, W_ei, Win_e, u, state_e, state_i, leaky_rate, bias_e, bias_i,
                                               activation_function)
        #         R[:, t ] =  np.concatenate((u,state))
        R[:, t] = state_e
    y = Wout @ R + b_out
    return y.T
