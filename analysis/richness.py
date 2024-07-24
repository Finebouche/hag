from scipy import sparse
import numpy as np
from matplotlib import pyplot as plt

def spectral_radius(W):
    eigen = sparse.linalg.eigs(W, k=1, which="LM", maxiter=W.shape[0] * 20, tol=0.1, return_eigenvectors=False)
    return max(abs(eigen))


def pearson(state_history, number_of_steps=None):
    state_history = np.array(state_history)

    if number_of_steps is None:
        number_of_steps = state_history.shape[0]
    n_time_points, n_neurons = state_history[:number_of_steps, :].shape
    mean_correlations = []
    std_correlations = []  # List to store the standard deviations

    # Define the window size for the correlation calculation
    size_window = 1000
    # Define step size for moving the window; this is optional, set to 1 for a classic rolling window
    step_size = int(size_window / 10)

    # Calculate the number of possible windows based on the step size
    num_windows = (n_time_points - size_window) // step_size + 1

    # Sliding window approach
    for i in range(num_windows):
        # Calculate the start index for each window
        start_index = i * step_size
        # Extract the window of data
        window = state_history[start_index:start_index + size_window, :]

        # Compute the Pearson correlation matrix for the current window
        correlation_matrix = np.corrcoef(window, rowvar=False)

        # Compute the mean and standard deviation of correlations, excluding the diagonal
        off_diagonal_correlations = correlation_matrix[np.triu_indices(n_neurons, k=1)]
        mean_corr = np.mean(off_diagonal_correlations)
        std_corr = np.std(off_diagonal_correlations)

        mean_correlations.append(mean_corr)
        std_correlations.append(std_corr)

    # Plotting the mean correlations with the standard deviation area
    plt.figure(figsize=(10, 5))
    time_windows = range(num_windows)
    plt.plot(time_windows, mean_correlations, marker='.', linestyle='-', color='b')
    plt.fill_between(time_windows, np.array(mean_correlations) - np.array(std_correlations),
                     np.array(mean_correlations) + np.array(std_correlations), color='blue', alpha=0.2)
    plt.title('Rolling Average Pearson Correlation Between Neurons')
    plt.xlabel('Rolling Window Steps')
    plt.ylabel('Average Correlation')
    plt.grid(True)


# Paper uses temp = 5000
def uncoupled_dynamics(STATE_H, temp=5000, A=0.9):
    # A : in (0, 1] and expresses the desired amount of explained variability
    # temps : int, the number of steps we want to evaluate the dynamics on
    STATE_H = np.array(STATE_H)[:temp, ]

    # compute the (sorted) singular values and the poucentage wise singular values
    _, singular_values, _ = np.linalg.svd(STATE_H)
    rel_sv = []
    for s in singular_values:
        rel_sv.append(s / np.sum(singular_values))

    # compute the number of singular values that explains A variability
    UD = 0
    frac = 0
    while frac < A:
        frac = frac + rel_sv[UD]
        UD += 1
    return UD, frac


# calculate estimator of Renyi’s quadratic entropy by the Parzen Window method using aGaussian kernel
def renyi_entropy(states, K=0.3):
    # K = 0.3 in the paper
    n = states.shape[1]
    states = states - np.mean(states, axis=0)
    states = states / np.std(states, axis=0)
    entropy = 0

    # the gaussian kernel use for approximation with the kernel size K
    def gaussian(x):
        return np.exp(-x ** 2 / (2 * K ** 2))

    for i in range(n):
        for j in range(n):
            entropy += gaussian(states[:, i] - states[:, j])
    return np.log(entropy / (n ** 2))
