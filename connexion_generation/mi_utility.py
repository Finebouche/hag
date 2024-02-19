# Code for Mutual Information
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import mutual_info_score
import time  # Import the time module
from multiprocessing import cpu_count


# Function to convert an input to a list
def convert_to_list(input):
    if isinstance(input, int):
        # If the input is an integer, return a list containing just the integer
        return [input]
    elif isinstance(input, np.ndarray):
        # If the input is a NumPy array, convert to a list
        return input.tolist()
    elif isinstance(input, list):
        # If the input is already a list, return it as is
        return input
    else:
        # If the input is none of the above, raise an error
        raise ValueError("Input must be an integer, a NumPy array, or a list.")


def compute_mi_for_pair(activity1, activity2, bin_edges):
    # Compute the joint histogram as density
    joint_histogram, x_edges, y_edges = np.histogram2d(activity1, activity2, bins=bin_edges, density=True)

    # Compute the bin areas for normalization
    dx = np.diff(x_edges)
    dy = np.diff(y_edges)
    bin_areas = dx[:, None] * dy[None, :]

    # Convert density to probability by multiplying with bin area
    joint_probabilities = joint_histogram * bin_areas

    # Add a small value to avoid log(0) issues
    eps = 1e-10
    joint_probabilities += eps

    # Compute the marginal probabilities
    marginal_prob1 = np.sum(joint_probabilities, axis=1)
    marginal_prob2 = np.sum(joint_probabilities, axis=0)

    # Compute the mutual information
    mi = np.nansum(joint_probabilities * np.log2(
        joint_probabilities / (marginal_prob1[:, None] * marginal_prob2[None, :])))
    return mi


def compute_mutual_information(states, neuron_groups, bins=10, n_jobs=1):
    # Check if neuron_groups is one-dimensional
    if hasattr(neuron_groups, 'ndim') and neuron_groups.ndim == 1:
        # If so, it implies group1 and group2 are the same
        group1 = group2 = convert_to_list(neuron_groups)
    elif len(neuron_groups) != 2:
        raise ValueError(
            "neuron_groups must be a list containing two lists of neuron indices or a list of neuron indices.")
    else:
        group1, group2 = map(convert_to_list, neuron_groups)

    # Initialize the matrix to store mutual information values
    mutual_information = np.zeros((len(group1), len(group2)))

    # Create mappings from neuron ID to index in the mutual information matrix for each group
    id_to_index_group1 = {neuron_id: index for index, neuron_id in enumerate(group1)}
    id_to_index_group2 = {neuron_id: index for index, neuron_id in enumerate(group2)}

    # Compute bin edges
    all_neurons = set(group1) | set(group2)
    bin_edges = np.histogram_bin_edges(np.concatenate([states[neuron] for neuron in all_neurons]), bins=bins)

    # Generate pairs of neurons to compute MI for
    pairs = []
    to_symmetrise = []
    # Loop through each neuron in group1 and group2 to generate pairs
    for neuron1 in group1:
        for neuron2 in group2:
            if (neuron1 != neuron2):
                if (neuron2, neuron1) not in pairs:
                    pairs.append((neuron1, neuron2))
                else:
                    to_symmetrise.append((neuron1, neuron2))

    # Use joblib to parallelize the MI computation for each pair
    results = Parallel(n_jobs=n_jobs)(delayed(compute_mi_for_pair)(states[neuron1], states[neuron2], bin_edges) for neuron1, neuron2 in pairs)

    # Assign computed MI values to the mutual_information matrix
    for (neuron1, neuron2), mi_value in zip(pairs, results):
        i, j = id_to_index_group1[neuron1], id_to_index_group2[neuron2]
        mutual_information[i, j] = mi_value
        if (neuron2, neuron1) in to_symmetrise:
            symmetric_i, symmetric_j = id_to_index_group1[neuron2], id_to_index_group2[neuron1]
            mutual_information[symmetric_i, symmetric_j] = mi_value

    return mutual_information


def compute_mutual_information_sklearn(states, neuron_groups, n_jobs=1):
    if hasattr(neuron_groups, 'ndim') and neuron_groups.ndim == 1:
        group1 = group2 = convert_to_list(neuron_groups)
    elif len(neuron_groups) == 1:
        group1 = group2 = convert_to_list(neuron_groups)
    elif len(neuron_groups) != 2:
        raise ValueError(
            "neuron_groups must be a list containing two lists of neuron indices or a list of neuron indices.")
    else:  # two groups that are different from each other
        group1, group2 = map(convert_to_list, neuron_groups)

    mutual_information_sklearn = np.zeros((len(group1), len(group2)))
    id_to_index_group1 = {neuron_id: index for index, neuron_id in enumerate(group1)}
    id_to_index_group2 = {neuron_id: index for index, neuron_id in enumerate(group2)}

    # Compute bin edges
    all_neurons = set(group1) | set(group2)

    # Generate pairs of neurons to compute MI for
    pairs = []
    to_symmetrise = []
    # Loop through each neuron in group1 and group2 to generate pairs
    for neuron1 in group1:
        for neuron2 in group2:
            if (neuron1 != neuron2):
                if (neuron2, neuron1) not in pairs:
                    pairs.append((neuron1, neuron2))
                else:
                    to_symmetrise.append((neuron1, neuron2))

    # Use joblib to parallelize the MI computation for each pair
    results = Parallel(n_jobs=n_jobs)(
        delayed(mutual_info_score)(states[neuron1], states[neuron2]) for neuron1, neuron2 in pairs
    )

    # Assign computed MI values to the mutual_information matrix
    for (neuron1, neuron2), mi_value in zip(pairs, results):
        i, j = id_to_index_group1[neuron1], id_to_index_group2[neuron2]
        mutual_information_sklearn[i, j] = mi_value / np.log(2)
        if (neuron2, neuron1) in to_symmetrise:
            symmetric_i, symmetric_j = id_to_index_group1[neuron2], id_to_index_group2[neuron1]
            mutual_information_sklearn[symmetric_i, symmetric_j] = mi_value / np.log(2)


    return mutual_information_sklearn


# test code
if __name__ == "__main__":

    np.random.seed(0)  # Ensure reproducibility

    n_samples = 100000
    p = 0.5  # Base probability for the Bernoulli distribution
    base_state = np.random.binomial(1, p, n_samples)

    states = [base_state]  # Initialize states list with the base state

    # Generate 49 additional states with varying degrees of correlation
    number_of_states = 100
    for i in range(1, number_of_states):
        # Example strategy to vary correlation:
        # Linearly decrease flip probability with each state to simulate varying degrees of correlation
        flip_prob = 1 - (i / number_of_states)  # Decreases from ~0.98 to ~0.02
        noise = np.random.binomial(1, flip_prob, n_samples)
        new_state = np.logical_xor(base_state, noise).astype(int)
        states.append(new_state)

    # Update the neurons array to match the new states
    neurons = [np.arange(len(states)), [1]]

    # bins setting remains the same
    bins = 100

    # Store the timings
    custom_timings = []
    sklearn_timings = []

    max_jobs = cpu_count() # Determine the maximum number of jobs to test
    tolerance = 1e-4  # adjust based on your accuracy needs (True up to 1e-4)
    for n_jobs in range(1, max_jobs + 1):
        print(f"Testing with n_jobs = {n_jobs}")

        # Time the custom MI calculation
        start_time = time.time()
        mi_custom = compute_mutual_information(states, neurons, bins, n_jobs)
        custom_time = time.time() - start_time
        custom_timings.append((1, custom_time))
        print(f"Custom calculation time: {custom_time:.4f} seconds")

        # Time the Scikit-learn MI calculation
        start_time = time.time()
        mi_sklearn = compute_mutual_information_sklearn(states, neurons, n_jobs)  # Ensure this function uses n_jobs
        sklearn_time = time.time() - start_time
        sklearn_timings.append((1, sklearn_time))
        print(f"Scikit-learn calculation time: {sklearn_time:.4f} seconds")

        # Calculate the absolute difference between the two result matrices
        difference_matrix = np.abs(mi_custom - mi_sklearn)

        # Check if all differences are within the tolerance level
        if np.all(difference_matrix < tolerance):
            print(f"True")
        else:
            print(f"False")