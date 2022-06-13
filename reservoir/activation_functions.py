from functools import wraps
import numpy as np


def _elementwise(func):
    """Vectorize a function to apply it on arrays. """
    vect = np.vectorize(func)

    @wraps(func)
    def vect_wrapper(*args, **kwargs):
        u = np.asanyarray(args)
        v = vect(u)
        return v[0]

    return vect_wrapper


def softmax(x: np.ndarray, beta=1) -> np.ndarray:
    return np.exp(beta * x) / np.exp(beta * x).sum()


@_elementwise
def softplus(x: np.ndarray) -> np.ndarray:
    return np.log(1 + np.exp(x))


@_elementwise
def sigmoid(x: np.ndarray) -> np.ndarray:
    if x < 0:
        u = np.exp(x)
        return u / (u + 1)
    return 1 / (1 + np.exp(-x))


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


@_elementwise
def identity(x: np.ndarray) -> np.ndarray:
    return x


@_elementwise
def relu(x: np.ndarray) -> np.ndarray:
    if x < 0:
        return 0
    return x


@_elementwise
def heaviside(x: np.ndarray) -> np.ndarray:
    if x < 0:
        return 0
    return 1
