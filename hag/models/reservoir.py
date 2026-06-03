from scipy import sparse, stats
import numpy as np
from numpy.random import Generator, PCG64
from scipy.sparse import block_diag


def update_reservoir(W, Win, u, r, leaky_rate, bias, activation_function):
    u = u.astype(np.float64)
    pre_s = (1 - leaky_rate) * r.flatten() + leaky_rate * (W @ r.flatten()) + (Win @ u.flatten()) + bias
    return activation_function(pre_s)

def init_matrices(n_neurons, input_connectivity, connectivity, K, spectral_radius=1, w_distribution=stats.uniform(loc=-1, scale=2),
                  win_distribution=stats.uniform(0, 1), use_block=False, seed=111, random_projection_experiment=False):
    # K is the number of times a single input is repeated to the models
    # The distribution generation functions #
    # stats.norm(1, 0.5)
    # stats.uniform(-1, 1)
    # stats.binom(n=1, p=0.5)

    # To ensure reproducibility
    numpy_randomGen = Generator(PCG64(seed))
    w_distribution.random_state = numpy_randomGen
    win_distribution.random_state = numpy_randomGen
    bias_distribution = stats.norm(0.1, 0.1)
    bias_distribution.random_state = numpy_randomGen

    if isinstance(n_neurons, int):
        reservoir_size = n_neurons
        W_shape = (n_neurons, n_neurons)
    else:
        reservoir_size = n_neurons[0]
        W_shape = n_neurons

    if reservoir_size <= 0:
        raise ValueError(f"n_neurons must be positive, got {reservoir_size}")

    if K <= 0:
        raise ValueError(f"K must be positive, got {K}")

    common_size = num_block = reservoir_size // K

    # Reservoir matrix
    if use_block:
        # Create block-diagonal models matrix W
        blocks = []
        for i in range(num_block):
            # Each block is a K x K random sparse matrix
            block = sparse.random(K, K, density=connectivity, random_state=seed + i, data_rvs=w_distribution.rvs)
            blocks.append(block)
        W = block_diag(blocks)
    else:
        W = sparse.random(reservoir_size, W_shape[1], density=connectivity, random_state=seed, data_rvs=w_distribution.rvs)

    # We set the diagonal to zero only for a square matrix
    if reservoir_size == W_shape[1] and reservoir_size > 0:
        W.setdiag(0)
        W.eliminate_zeros()
        # Set the spectral radius
        # source : p104 David Verstraeten : largest eigenvalue = spectral radius
        if connectivity > 0:
            eigen = sparse.linalg.eigs(W, k=1, which="LM", maxiter=W.shape[0] * 20, tol=0.1, return_eigenvectors=False)
            sr = max(abs(eigen))
            W *= spectral_radius / sr

    # Input matrix
    # We want the Win matrix to explicitly map each input directly to a specific segment of neurons,
    # with each segment receiving the same input value duplicated K times.
    Win = np.zeros((reservoir_size, common_size))
    if not random_projection_experiment:
        for i in range(common_size):
            start_index = i * K
            end_index = start_index + K
            Win[start_index:end_index, i] = win_distribution.rvs(K)
    else:
        # RANDOM PROJECTION: sparse random connections from all inputs to all neurons
        Win_sparse = sparse.random(
            reservoir_size, common_size,
            density=input_connectivity,         # choose ~1/common_size for ~1 nonzero per row
            random_state=seed,
            data_rvs=win_distribution.rvs,
            format="csr",
        )

        # (Optional but recommended) Guarantee at least one input per neuron
        zero_rows = np.where(Win_sparse.getnnz(axis=1) == 0)[0]
        if zero_rows.size > 0:
            for r in zero_rows:
                j = int(Generator(PCG64(seed + r)).integers(0, common_size))
                Win_sparse[r, j] = win_distribution.rvs()

        Win = Win_sparse.toarray()

    # Bias matrix
    bias = np.abs(bias_distribution.rvs(size=reservoir_size))

    return Win, W.toarray(), bias.flatten()