from scipy import sparse
import numpy as np
from sklearn.decomposition import PCA
import dcor
from joblib import Parallel, delayed
from tqdm import tqdm

def spectral_radius(W):
    eigen = sparse.linalg.eigs(W, k=1, which="LM", maxiter=W.shape[0] * 20, tol=0.1, return_eigenvectors=False)
    return max(abs(eigen))


def pearson(state_history, size_window=500, step_size=None, num_windows=None, show_progress=True):
    state_history = np.array(state_history)
    number_of_steps = state_history.shape[0]
    n_time_points, n_neurons = state_history[:number_of_steps, :].shape

    # Define step size for moving the window; this is optional, set to 1 for a classic rolling window
    if step_size is None:
        step_size = int(size_window / 10)

    # Calculate the number of possible windows based on the step size
    if num_windows is None:
        num_windows = (n_time_points - size_window) // step_size + 1

    mean_correlations = []
    std_correlations = []
    # Sliding window approach
    for i in (tqdm(range(num_windows), desc="Pearson calculation") if show_progress else range(num_windows)):
        # Calculate the start index of the window and extract the window
        start_index = i * step_size

        # Compute the Pearson correlation matrix for the current window
        correlation_matrix = np.corrcoef(state_history[start_index:start_index + size_window, :], rowvar=False)

        # Compute the mean and standard deviation of correlations, excluding the diagonal
        off_diagonal_correlations = correlation_matrix[np.triu_indices(n_neurons, k=1)]
        mean_corr = np.mean(off_diagonal_correlations)
        std_corr = np.std(off_diagonal_correlations)

        mean_correlations.append(mean_corr)
        std_correlations.append(std_corr)

    return mean_correlations, std_correlations


def squared_uncoupled_dynamics(state_history, size_window=500, step_size=None, num_windows=None, A=0.9, show_progress=True):
    # A : in (0, 1] and expresses the desired amount of explained variability
    # state_history : {array-like, sparse matrix} of shape (n_samples, n_features)
    state_history = np.array(state_history)
    n_time_points, n_neurons = state_history.shape

    # Define step size for moving the window; this is optional, set to 1 for a classic rolling window
    if step_size is None:
        step_size = int(size_window / 10)

    # Calculate the number of possible windows based on the step size
    if num_windows is None:
        num_windows = (n_time_points - size_window) // step_size + 1


    # List to store the number of principal components for each window
    evolution_num_components = []

    # Sliding window PCA
    for i in (tqdm(range(num_windows), desc="Squared Uncoupled Dynamics calculation") if show_progress else range(num_windows)):
        # Calculate the start index of the window and extract the window
        start_index = i * step_size

        # Extract the current window of data
        window_data = state_history[start_index:start_index + size_window, :]

        # Standardize the data (important for PCA)
        window_data_std = (window_data - np.mean(window_data, axis=0)) / np.std(window_data, axis=0)

        # Perform PCA
        pca = PCA()
        pca.fit(window_data_std)

        # Get the cumulative explained variance
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

        # Find the number of components that explain at least A variance
        num_components = np.argmax(cumulative_variance >= A) + 1

        # Append the result for this window
        evolution_num_components.append(num_components)

    return evolution_num_components

# Should yield the same result as uncoupled_dynamics
def squared_uncoupled_dynamics_alternative(state_history, size_window=1000, step_size=None, num_windows=None, A=0.9, show_progress=True):
    state_history = np.array(state_history)
    n_time_points, n_neurons = state_history.shape

    # Define step size for moving the window; this is optional, set to 1 for a classic rolling window
    if step_size is None:
        step_size = int(size_window / 10)

    # Calculate the number of possible windows based on the step size
    if num_windows is None:
        num_windows = (n_time_points - size_window) // step_size + 1

    # List to store the number of components for each window
    evolution_num_components = []

    # Sliding window approach
    for i in (tqdm(range(num_windows), desc="Squared Uncoupled Dynamics calculation") if show_progress else range(num_windows)):
        # Calculate the start index of the window and extract the window
        start_index = i * step_size
        # Extract the current window of data
        window_data = state_history[start_index:start_index + size_window, :]
        window_data_std = (window_data - np.mean(window_data, axis=0)) / np.std(window_data, axis=0)

        # Compute the Singular Value Decomposition
        S = np.linalg.svd(window_data_std, compute_uv=False)

        # Calculate the normalized relevance R_j
        R = S**2 / np.sum(S**2)

        # Cumulative sum of R_j
        cumulative_R = np.cumsum(R)

        # Find the minimum d such that the cumulative sum >= theta
        num_components = np.searchsorted(cumulative_R, A) + 1

        evolution_num_components.append(num_components)

    return evolution_num_components


def linear_uncoupled_dynamics(state_history, size_window=1000, step_size=None, num_windows=None, theta=0.9, show_progress=True):
    state_history = np.array(state_history)
    n_time_points, n_neurons = state_history.shape

    # Define step size for moving the window; this is optional, set to 1 for a classic rolling window
    if step_size is None:
        step_size = int(size_window / 10)

    # Calculate the number of possible windows based on the step size
    if num_windows is None:
        num_windows = (n_time_points - size_window) // step_size + 1

    # List to store the number of components for each window
    evolution_num_components = []

    # Sliding window approach
    for i in (tqdm(range(num_windows), desc="Linear Uncoupled Dynamics calculation")if show_progress else range(num_windows)):
        # Calculate the start index of the window and extract the window
        start_index = i * step_size
        # Extract the current window of data
        window_data = state_history[start_index:start_index + size_window, :]

        # Compute the Singular Value Decomposition
        S = np.linalg.svd(window_data, compute_uv=False)

        # Calculate the normalized relevance R_j
        R = S / np.sum(S)

        # Cumulative sum of R_j
        cumulative_R = np.cumsum(R)

        # Find the minimum d such that the cumulative sum >= theta
        num_components = np.searchsorted(cumulative_R, theta) + 1

        evolution_num_components.append(num_components)

    return evolution_num_components



def condition_number(state_history, size_window=1000, step_size=None, num_windows=None, show_progress=True):
    state_history = np.array(state_history)
    n_time_points, n_neurons = state_history.shape

    # Define step size for moving the window; this is optional, set to 1 for a classic rolling window
    if step_size is None:
        step_size = int(size_window / 10)

    # Calculate the number of possible windows based on the step size
    if num_windows is None:
        num_windows = (n_time_points - size_window) // step_size + 1

    # List to store the number of components for each window
    evolution_condition_number = []

    # Sliding window approach
    for i in (tqdm(range(num_windows), desc="Condition Number calculation") if show_progress else range(num_windows)):
        # Calculate the start index of the window and extract the window
        start_index = i * step_size
        # Extract the current window of data
        window_data = state_history[start_index:start_index + size_window, :]

        # Compute the singular value decomposition and take the smallest and largest singular values
        S = np.linalg.svd(window_data, compute_uv=False)
        condition_number = np.max(S) / np.min(S)
        evolution_condition_number.append(condition_number)

    return evolution_condition_number


def distance_correlation(state_history, size_window=500, step_size=None, num_windows=None,
                         show_progress=True, method="auto", nb_jobs=1):
    """
    Computes the mean and standard deviation of pairwise distance correlations in sliding windows of the state_history.
    The computation over neuron pairs is always parallelized using the nb_jobs parameter.
    """
    state_history = np.array(state_history)
    n_time_points, n_neurons = state_history.shape

    # Define step size if not provided
    if step_size is None:
        step_size = int(size_window / 10)

    # Calculate the number of windows if not provided
    if num_windows is None:
        num_windows = (n_time_points - size_window) // step_size + 1

    mean_distance_correlations = []
    std_distance_correlations = []

    # Pre-generate all unique neuron pairs
    neuron_pairs = [(j, k) for j in range(n_neurons) for k in range(j + 1, n_neurons)]

    for i in (tqdm(range(num_windows), desc="Distance Correlation") if show_progress else range(num_windows)):
        start_index = i * step_size
        window_data = state_history[start_index:start_index + size_window, :]

        # Parallelize the computation over neuron pairs
        dcorr_values = Parallel(n_jobs=nb_jobs)(
            delayed(dcor.distance_correlation)(window_data[:, j], window_data[:, k], method=method)
            for j, k in neuron_pairs
        )

        # Compute mean and standard deviation for this window
        mean_distance_correlations.append(np.mean(dcorr_values))
        std_distance_correlations.append(np.std(dcorr_values))

    return mean_distance_correlations, std_distance_correlations



if __name__ == "__main__":
    import time
    # Generate mock data: e.g. 5000 time points and 50 neurons
    n_time_points = 5000
    n_neurons = 500
    state_history = np.random.randn(n_time_points, n_neurons)

    # Parameters for sliding windows
    size_window = 5000
    step_size = None  # defaults to size_window//10
    num_windows = None  # auto-computed

    # Test the 1D vector (pairwise) method
    print("Testing distance_correlation (1D vectors)...")
    start_time = time.time()
    mean_dcorr, std_dcorr = distance_correlation(state_history, size_window=size_window,
                                                 step_size=step_size, num_windows=num_windows,
                                                 show_progress=False, nb_jobs=10)
    elapsed_time = time.time() - start_time
    print("Execution time: {:.4f} seconds".format(elapsed_time))
    print("Mean over windows: {:.4f}  Std over windows: {:.4f}".format(np.mean(mean_dcorr), np.mean(std_dcorr)))

    # Test the multidimensional method
    print("\nTesting distance_correlation_multidim (multidimensional)...")
    start_time = time.time()
    multidim_dcorr_m, _ = distance_correlation(state_history, size_window=size_window,
                                                   step_size=step_size, num_windows=num_windows,
                                                   show_progress=False, method="mergesort", nb_jobs=10)
    elapsed_time = time.time() - start_time
    print("Execution time: {:.4f} seconds".format(elapsed_time))
    print("Mean correlation over windows: {:.4f}".format(np.mean(multidim_dcorr_m)))


    # For each window, the only pair in the 1D approach should match the multidimensional value.
    for i, (v1, v2) in enumerate(zip(mean_dcorr, multidim_dcorr_m)):
        equal = np.isclose(v1, v2, atol=1e-8)
        print(f"Window {i}: pairwise = {v1:.8f}, multidim = {v2:.8f} --> Equal: {equal}")