import numpy as np
from connexion_generation.utility import switch_connexion


def add_random_connexion(W, low, high):
    n = W.shape[0]
    value = np.random.uniform(low, high)
    i = np.random.randint(low=0, high=n)
    j = np.random.randint(low=0, high=n)
    if W.getrow(i).getcol(j).toarray() == 0:
        W = switch_connexion(W, i, j, value)
    return W
