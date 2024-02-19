from scipy import sparse
import numpy as np
from connexion_generation.mi_utility import compute_mutual_information_sklearn


class TwoDimArrayWrapper:
    def __init__(self, input_data):
        if input_data.ndim != 2:
            raise ValueError("Expected a 2D array.")
        self.input_data = input_data
        self.shape = input_data.shape
        self.size = input_data.shape[1]
        self.flat_data = input_data.flatten()

    def __getitem__(self, key):
        # Handle slice access
        if isinstance(key, slice):
            sliced_data = self.input_data[:, key.start:key.stop:key.step]
            return TwoDimArrayWrapper(sliced_data)
        # Handle single element access
        return self.input_data[:, key]


def determine_connection_pairs(neurons_needing_new_connection, connectivity_matrix, is_inter_matrix=False,
                               max_partners=12, random_seed=None):
    """
    Determine pairs of neurons for establishing new connections based on specified criteria.

    Returns:
    - A list of tuples, where each tuple represents a new connection (source_neuron, target_neuron).
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    new_connections = []
    available_neurons = list(range(connectivity_matrix.shape[1])) if is_inter_matrix else list(
        neurons_needing_new_connection)

    for neuron in neurons_needing_new_connection:
        available_for_this_neuron = available_neurons.copy()
        if not is_inter_matrix:
            # cannot add a connexion with itself
            available_for_this_neuron.remove(neuron)
        # If neuron already has more than MAX_NUMBER_OF_PARTNER partners:
        # the available neurons are the one that already have a connexion with it
        if np.count_nonzero(connectivity_matrix.getrow(neuron).A) >= max_partners:
            available_for_this_neuron = connectivity_matrix.getrow(neuron).nonzero()[1]

        # select randomly
        if len(available_for_this_neuron) > 0:
            incoming_connexion = np.random.choice(available_for_this_neuron)
            new_connections.append((neuron, incoming_connexion))

    return new_connections


def determine_pruning_pairs(neurons_for_pruning, connectivity_matrix, states, mi_based=False, random_seed=None):
    """
    Identifies pairs of neurons for pruning from a connectivity matrix.

    Returns:
    - A list of tuples, where each tuple represents a pair (neuron, connection) to be pruned.
    """

    if random_seed is not None:
        np.random.seed(random_seed)

    new_pruning_pairs = []
    if mi_based:
        for neuron in neurons_for_pruning:
            connections = connectivity_matrix.getrow(neuron).nonzero()[1]
            mi = compute_mutual_information_sklearn(states, [connections, [neuron]])
            # We prune the connexion with the highest mutual information
            new_pruning_pairs.append((neuron, connections[np.argmax(mi)]))
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
