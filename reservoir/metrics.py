from scipy import sparse
import numpy as np
from reservoir import update_reservoir

def uncoupled_dynamics(W, Win, bias, state, U_test1, leaky_rate=1, temp=5000, A=0.9):
    ST = []
    if state is None:
        state = np.random.uniform(-1, 1, n)

    for i in range(temp):
        ST.append(state)
        u = U_test1[i]
        state = update_reservoir(W, Win, u, state, leaky_rate, bias, activation_function)

    ST = np.array(ST)

    _, S, _ = np.linalg.svd(ST)
    R = []
    for s in S:
        R.append(s / np.sum(S))

    UD = 0
    frac = 0
    while frac < A:
        frac = frac + R[UD]
        UD += 1
    return UD, frac


def spectral_radius(W):
    eigen = sparse.linalg.eigs(W, k=1, which="LM", maxiter=W.shape[0] * 20, tol=0.1, return_eigenvectors=False)
    return max(abs(eigen))


def pearson(states1, states2):
    # Takes states and produces a pearson similarity
    mean1 = np.mean(states1)
    mean2 = np.mean(states2)

    numerator = np.sum((states1 - mean1) * (states2 - mean2))
    denominator = np.sqrt(np.sum((states1 - mean1) ** 2) * np.sum((states2 - mean2) ** 2))

    if denominator == 0:
        return 0

    return numerator / denominator
