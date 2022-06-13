from scipy import linalg, sparse, stats
import numpy as np
from numpy.random import Generator, PCG64


def update_reservoir(W, Win, u, r, leaky_rate, bias, activation_function):
    pre_s = (1 - leaky_rate) * r + leaky_rate * W @ r + Win @ u + bias
    return activation_function(pre_s)


def ridge_regression(R, Ytrain, ridge_coef):
    # R          : the states' matrix
    # Ytrain     : the data to reproduce
    # ridge_coef : the ridge coefficient

    ## Partie pour ajouter b_out
    R = np.concatenate((np.ones((1, R.shape[1])), R))
    I = np.eye(R.shape[0])
    I[0:0] = 0

    # W_out = (Ytrain@R.T) @ (linalg.inv(R@R.T + ridge_coef*np.eye(R.shape[0])))
    W_out = linalg.solve(R @ R.T + ridge_coef * I, R @ Ytrain.T).T

    b_out = W_out[:, 0]
    W_out = W_out[:, 1:]

    return W_out, b_out


def train(W, Win, bias, Utrain, Ytrain, activation_function, ridge_coef=1e-8, init_len=0, leaky_rate=1, state=None):
    # state         : the initial state before ethe train (We start from a null state)

    n = Win.shape[0]
    # We initialize the state to random if there is no state provided
    if state is None:
        state = np.random.uniform(-1, 1, n)

    # run the reservoir with the data and collect R = (u, state)
    seq_len = len(Utrain)
    R = np.zeros((n, seq_len - init_len))
    for t in range(seq_len):
        u = Utrain[t]
        state = update_reservoir(W, Win, u, state, leaky_rate, bias, activation_function)
        # we collect after the initialisation of the reservoir (default = 0)
        if t > init_len:
            R[:, t - init_len] = state
    #             R[:,t-init_len] = np.concatenate((u, state))

    # Ensure that for Ytrain and R : columns -> iteration and rows -> neurons
    # Ytrain and R  have the "standard" shape for Ridge Regression
    Ytrain = Ytrain[init_len:, :]
    Y = Ytrain.T
    # Compute Wout using Ridge Regression
    Wout, b_out = ridge_regression(R, Y, ridge_coef)

    return Wout, b_out, state


def run(W, Win, bias, Wout, U, activation_function, b_out=None, last_state=None, leaky_rate=1):
    # We start from previous state or else from uniform random state
    n = Win.shape[0]
    state = (np.random.uniform(0, 1, n) if last_state is None else last_state)

    seq_len = len(U)
    R = np.zeros((n, seq_len))
    for t in range(seq_len):
        u = U[t]
        state = update_reservoir(W, Win, u, state, leaky_rate, bias, activation_function)
        #         R[:, t ] =  np.concatenate((u,state))
        R[:, t] = state
    y = Wout @ R + b_out
    return y.T


def synaptic_scaling(W):
    W = sparse.lil_matrix(W)
    for i in range(W.shape[1]):
        W[i] = W[i] / np.sum(W[i])
    W = sparse.coo_matrix(W)
    return W


def activation_target_definition(W):
    synaptic_activation_target = []
    W = sparse.lil_matrix(W)
    for i in range(W.shape[1]):
        synaptic_activation_target.append(np.sum(W[i]))
    return np.array(synaptic_activation_target)


def constant_synaptic_scaling(W, synaptic_activation_target):
    #  Synaptic scaling to the initial value of sum of incoming value
    # synaptic_activation_target is computed with activation_target_definition
    W = sparse.lil_matrix(W)
    for i in range(W.shape[1]):
        W[i] = W[i] / np.sum(W[i]) * synaptic_activation_target[i]
    W = sparse.coo_matrix(W)
    return W


# https://stackoverflow.com/questions/16016959/scipy-stats-seed

def init_matrices(n, input_connectivity, connectivity, spectral_radius, input_scaling, seed):
    numpy_randomGen = Generator(PCG64(seed))
    #
    # The distribution generation functions
    #
    ## stats.norm(1, 0.5).rvs
    scipy_norm = stats.norm(1, 0.5)
    scipy_norm.random_state = numpy_randomGen
    ## stats.uniform(-1, 1).rvs
    scipy_uniform = stats.uniform(0, 1)
    scipy_uniform.random_state = numpy_randomGen
    ## stats.norm(1, 0.5).rvs
    scipy_binom = stats.binom(n=1, p=0.5)
    scipy_binom.random_state = numpy_randomGen

    #
    # The generation of the matrices
    #
    # Input matrice
    Win = sparse.random(n, 1, density=input_connectivity, random_state=seed, data_rvs=scipy_norm.rvs)
    Win = Win * input_scaling
    # Reservoir matrice
    W = sparse.random(n, n, density=connectivity, random_state=seed, data_rvs=scipy_uniform.rvs)
    W.setdiag(0)
    W.eliminate_zeros()
    # Set the spectral radius
    # source : p104 David Verstraeten : largest eigenvalue = spectral radius
    eigen = sparse.linalg.eigs(W, k=1, which="LM", maxiter=W.shape[0] * 20, tol=0.1, return_eigenvectors=False)
    sr = max(abs(eigen))
    W *= spectral_radius / sr
    # Bias matrice
    bias = scipy_uniform.rvs(size=n)

    return Win, W, bias
