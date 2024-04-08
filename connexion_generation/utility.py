from scipy import sparse
import numpy as np
from connexion_generation.mi_utility import compute_mutual_information

class TwoDimArrayWrapper:
    def __init__(self, input_data):
        if input_data.ndim != 2:
            raise ValueError("Expected a 2D array.")
        self.A = input_data
        self.data = input_data
        self.shape = input_data.shape
        self.size = input_data.shape[1]
        self.flat_data = input_data.flatten()

    def __getitem__(self, key):
        # Handle slice access
        if isinstance(key, slice):
            sliced_data = self.data[:, key.start:key.stop:key.step]
            return TwoDimArrayWrapper(sliced_data)
        # Handle single element access
        return self.data[:, key]


def determine_connection_pairs(neurons_needing_new_connection, connectivity_matrix, states=None,
                               is_inter_matrix=False, mi_based=False, max_partners=12, random_seed=None, n_jobs=1):
    """
    Determine pairs of neurons for establishing new connections based on specified criteria.

    Returns:
    - A list of tuples, where each tuple represents a new connection (source_neuron, target_neuron).
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    if states is None and mi_based:
        raise ValueError("States must be provided if mutual information based pruning is used.")

    new_connections = []
    available_neurons = list(range(connectivity_matrix.shape[1])) if is_inter_matrix \
        else list(neurons_needing_new_connection)
    if len(available_neurons) <= 1:
        return []

    if mi_based:
        mi_for_available_neurons = compute_mutual_information(states, [available_neurons, neurons_needing_new_connection], n_jobs=n_jobs)
        for neuron in neurons_needing_new_connection:
            # If neuron already has more than MAX_NUMBER_OF_PARTNER partners:
            # the available neurons are the one that already have a connexion with it
            noun_zero = connectivity_matrix.getrow(neuron).nonzero()[1]
            if len(noun_zero) >= max_partners:
                available_for_this_neuron = noun_zero
            else:
                available_for_this_neuron = available_neurons.copy()
                if not is_inter_matrix:
                    # cannot add a connexion with itself
                    available_for_this_neuron.remove(neuron)

            if len(available_for_this_neuron) == 0:
                raise ValueError("No available neurons for connexion, this should not happen.")
            else:
                # select the connexion with the highest mutual information
                mi_interresting = mi_for_available_neurons[neuron, available_for_this_neuron]

                # Step 2: Get the indices in `mi_interresting` that correspond to this maximum value
                mi_interresting_max_indices = np.where(mi_interresting == np.max(mi_interresting))[0]

                # Step 3: Map these indices back to the original `available_for_this_neuron` array
                max_value_indices = [available_for_this_neuron[idx] for idx in mi_interresting_max_indices]
                incoming_connexion = np.random.choice(max_value_indices)

                new_connections.append((neuron, incoming_connexion))
    else:
        for neuron in neurons_needing_new_connection:
            # If neuron already has more than MAX_NUMBER_OF_PARTNER partners:
            # the available neurons are the one that already have a connexion with it
            if np.count_nonzero(connectivity_matrix.getrow(neuron).A) >= max_partners:
                available_for_this_neuron = connectivity_matrix.getrow(neuron).nonzero()[1]
            else:
                available_for_this_neuron = available_neurons.copy()
                if not is_inter_matrix:
                    # cannot add a connexion with itself
                    available_for_this_neuron.remove(neuron)
            if len(available_for_this_neuron) == 0:
                raise ValueError("No available neurons for connexion, this should not happen.")
            else:
                incoming_connexion = np.random.choice(available_for_this_neuron)
                new_connections.append((neuron, incoming_connexion))

    return new_connections


def determine_pruning_pairs(neurons_for_pruning, connectivity_matrix, states=None, mi_based=False, random_seed=None, n_jobs=1):
    """
    Identifies pairs of neurons for pruning from a connectivity matrix.

    Returns:
    - A list of tuples, where each tuple represents a pair (neuron, connection) to be pruned.
    """

    if random_seed is not None:
        np.random.seed(random_seed)
    if states is None and mi_based:
        raise ValueError("States must be provided if mutual information based pruning is used.")

    new_pruning_pairs = []
    if mi_based:
        for neuron in neurons_for_pruning:
            connections = connectivity_matrix.getrow(neuron).nonzero()[1]
            if len(connections) == 0:
                continue
            mi = compute_mutual_information(states, [connections, [neuron]], n_jobs=n_jobs)
            # We prune the connexion with the lowest mutual information
            # select the connexion with the highest mutual information

            # Step 1: Get the indices in `mi_interresting` that correspond to this minimum value
            mi_interresting = mi[neuron, connections]
            mi_interresting_min_indices = np.where(mi_interresting == np.min(mi_interresting))[0]

            # Step 2: Map these indices back to the original `available_for_this_neuron` array
            min_value_indices = [connections[idx] for idx in mi_interresting_min_indices]
            incoming_connexion = np.random.choice(min_value_indices)

            new_pruning_pairs.append((neuron, incoming_connexion))
    else:
        for neuron in neurons_for_pruning:
            connections = connectivity_matrix.getrow(neuron).nonzero()[1]
            if connections.size > 0:
                chosen_connection = np.random.choice(connections)
                new_pruning_pairs.append((neuron, chosen_connection))

    return new_pruning_pairs


def change_connexion(W, i, j, value):
    # i for rows, j for columns
    W = sparse.lil_matrix(W)
    W[i, j] = W[i, j] + value
    if W[i, j] < 0:
        W[i, j] = 0
    W = sparse.coo_matrix(W)
    W.eliminate_zeros()
    return W
