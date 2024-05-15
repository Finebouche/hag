# Code for Mutual Information
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import mutual_info_score
import time  # Import the time module
from multiprocessing import cpu_count

# Function to convert an input to a list
def convert_to_lists(input):
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

def compute_pearson_corr(x, Y):
    x_mean = np.mean(x)
    # Compute the mean of each row of the 2D array
    Y_mean = np.mean(Y, axis=1)
    # Compute the numerator of the Pearson correlation
    numerator = np.sum((x - x_mean) * (Y - Y_mean[:, None]), axis=1)
    # Compute the denominator of the Pearson correlation
    denominator = np.sqrt(np.sum((x - x_mean) ** 2) * np.sum((Y - Y_mean[:, None]) ** 2, axis=1))
    # Compute the Pearson correlation
    corr = numerator / denominator
    return corr


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
    mi = np.nansum(joint_probabilities * np.log2(joint_probabilities / (marginal_prob1[:, None] * marginal_prob2[None, :])))
    return mi


def compute_mutual_information(states, neuron_groups, bins=10, n_jobs=1, sklearn=False):
    if isinstance(neuron_groups, set):
        optimized_pairs = neuron_groups
        all_neurons = {neuron for pair in optimized_pairs for neuron in pair}

    elif isinstance(neuron_groups, list):
        # Check if neuron_groups is one-dimensional
        if len(neuron_groups) == 1:
            # If so, it implies group1 and group2 are the same
            group1 = group2 = convert_to_lists(neuron_groups)
        elif len(neuron_groups) == 2:
            group1, group2 = map(convert_to_lists, neuron_groups)
        else:
            raise ValueError("neuron_groups must be one-dimensional or two-dimensional")

        # Generate pairs of neurons to compute MI for
        optimized_pairs = set((min(n1, n2), max(n1, n2)) for n1 in group1 for n2 in group2)
        all_neurons = set(group1) | set(group2)

    else:
        raise ValueError("neuron_groups must be a list or a set.")

    # Compute bin edges
    bin_edges = np.histogram_bin_edges(np.concatenate([states[neuron] for neuron in all_neurons]), bins=bins)

    if sklearn:
        results = Parallel(n_jobs=n_jobs)(
             delayed(mutual_info_score)(states[neuron1, :], states[neuron2, :]) for neuron1, neuron2 in optimized_pairs
        )
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_mi_for_pair)(states[neuron1, :], states[neuron2, :], bin_edges) for neuron1, neuron2 in optimized_pairs
        )

    nb_of_neurons = states.shape[0]
    # Initialize the matrix to store mutual information values
    mutual_information = np.zeros((nb_of_neurons, nb_of_neurons))
    # Assign computed MI values to the mutual_information matrix using advanced indexing
    neuron1_indices, neuron2_indices = zip(*optimized_pairs)
    mutual_information[neuron1_indices, neuron2_indices] = results
    mutual_information[neuron2_indices, neuron1_indices] = results

    return mutual_information


# test code
if __name__ == "__main__":
    np.random.seed(0)  # Ensure reproducibility

    n_time_step = 2000
    p = 0.5  # Base probability for the Bernoulli distribution
    base_state = np.random.binomial(1, p, n_time_step)

    states = [base_state]  # Initialize states list with the base state

    # Generate number_of_states additional states with varying degrees of correlation
    number_of_neurons = 500
    for i in range(1, number_of_neurons):
        # Example strategy to vary correlation:
        # Linearly decrease flip probability with each state to simulate varying degrees of correlation
        flip_prob = 1 - (i / number_of_neurons)  # Decreases from ~0.98 to ~0.02
        noise = np.random.binomial(1, flip_prob, n_time_step)
        new_state = np.logical_xor(base_state, noise).astype(int)
        states.append(new_state)

    states = np.array(states)
    print(states.shape)

    # Update the neurons array to match the new states
    neurons = [np.arange(number_of_neurons), np.arange(number_of_neurons)]

    # Store the timings
    custom_timings = []
    sklearn_timings = []

    max_jobs = cpu_count()  # Determine the maximum number of jobs to test

    for n_jobs in range(1, max_jobs + 1):
        print(f"Testing with n_jobs = {n_jobs}")

        # Time the custom MI calculation
        start_time = time.time()
        corr = compute_pearson_corr(states[0], states[1:])
        #mi_custom = compute_mutual_information(states, neurons, bins=10, n_jobs=n_jobs)
        print(corr.shape)
        custom_time = time.time() - start_time
        custom_timings.append((1, custom_time))
        print(f"Custom calculation time: {custom_time:.4f} seconds")

        # Time the Scikit-learn MI calculation
        start_time = time.time()
        corr_numpy = np.corrcoef(states[0], states[1:])[0, 1:].flatten()
        #mi_sklearn = compute_mutual_information(states, neurons, sklearn=True, n_jobs=n_jobs)
        sklearn_time = time.time() - start_time
        sklearn_timings.append((1, sklearn_time))
        print(f"Scikit-learn calculation time: {sklearn_time:.4f} seconds")

        # print(corr)
        # print(corr_numpy * corr[1] / corr_numpy[1])
        # print(mi_custom)
        # print(mi_sklearn * mi_custom[0, 1]/mi_sklearn[0, 1])
        # Calculate the absolute difference between the two result matrices
        # difference_matrix = np.abs(mi_custom - mi_sklearn * mi_custom[0, 1]/mi_sklearn[0, 1])
        difference_matrix = np.abs(corr - corr_numpy * corr[1]/corr_numpy[1])

        # Check if all differences are within the tolerance level
        tolerance = 1e-6  # adjust based on your accuracy needs (True up to 1e-4)
        if np.all(difference_matrix < tolerance):
            print(f"True")
        else:
            print(f"False")
