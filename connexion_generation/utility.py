from scipy import sparse
import numpy as np
from connexion_generation.correlation_utility import compute_mutual_information, compute_pearson_corr
from joblib import Parallel, delayed


def available_neurons(neuron, connectivity_matrix, neurons_pool, max_partners=12, is_inter_matrix=False):
    # If neuron already has more than MAX_NUMBER_OF_PARTNER partners:
    # the available neurons are the one that already have a connexion with it
    non_zeros = connectivity_matrix.getrow(neuron).nonzero()[1]

    if len(non_zeros) >= max_partners:
        available_for_this_neuron = non_zeros
    else:
        available_for_this_neuron = neurons_pool.copy()
        if not is_inter_matrix:
            # cannot add a connexion with itself
            available_for_this_neuron.remove(neuron)
    if len(available_for_this_neuron) == 0:
        raise ValueError("No available neurons for connection, this should not happen.")

    return available_for_this_neuron


def determine_connection_pairs(neurons_needing_new_connection, connectivity_matrix, states=None, method="random",
                               is_inter_matrix=False, max_partners=12, random_seed=None, n_jobs=1):
    """
    Determine pairs of neurons for establishing new connections based on specified criteria.

    Returns:
    - A list of tuples, where each tuple represents a new connection (source_neuron, target_neuron).
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    if states is None and (method == "mi" or method == "pearson"):
        raise ValueError("States must be provided if mutual information or pearson based pruning is used.")

    new_connections = []
    neurons_pool = list(range(connectivity_matrix.shape[1])) if is_inter_matrix \
        else list(neurons_needing_new_connection)
    if len(neurons_pool) <= 1:
        return []

    if method == "mi":
        def compute_mi_for_neuron(neuron):
            available_for_this_neuron = available_neurons(neuron, connectivity_matrix, neurons_pool, max_partners)
            mi_for_available_neurons = compute_mutual_information(states, [available_for_this_neuron, [neuron]])

            # select the connection with the highest mutual information
            mi_for_this_neuron = mi_for_available_neurons[neuron, available_for_this_neuron]
            max_value_indices = np.array(available_for_this_neuron)[
                np.isclose(mi_for_this_neuron, np.nanmax(mi_for_this_neuron))]

            incoming_connexion = np.random.choice(max_value_indices)
            return (neuron, incoming_connexion)

        new_connections = Parallel(n_jobs=n_jobs)(
            delayed(compute_mi_for_neuron)(neuron)
            for neuron in neurons_needing_new_connection
        )

    elif method == "pearson":
        for neuron in neurons_needing_new_connection:
            available_for_this_neuron = available_neurons(neuron, connectivity_matrix, neurons_pool, max_partners)
            # correlations = np.corrcoef(states[neuron, :], states[available_for_this_neuron, :])[0, 1:].flatten()
            correlations = np.corrcoef(states[neuron, 1:], states[available_for_this_neuron, :-1])[0, 1:]
            # Find the neuron with the maximum Pearson correlation (use isclose to handle floating point errors)

            corr_max = np.nanmax(correlations)
            max_value_indices = np.array(available_for_this_neuron)[np.isclose(correlations, corr_max)]

            if np.isnan(max_value_indices).all():
                max_value_indices = available_for_this_neuron
            try:
                incoming_connexion = np.random.choice(max_value_indices)
            except ValueError:
                # print every variable
                print("neuron", neuron)
                print("available_for_this_neuron", available_for_this_neuron)
                print("correlations", correlations)
                print("corr_max", corr_max)
                print("max_value_indices", max_value_indices)
                print("states", states)
                print("isclose", np.isclose(correlations, corr_max))

            new_connections.append((neuron, incoming_connexion))

        # def compute_pearson_for_neuron(neuron):
        #     available_for_this_neuron = available_neurons(neuron, connectivity_matrix, neurons_pool, max_partners)
        #     # correlations = np.corrcoef(states[neuron, :], states[available_for_this_neuron, :])[0, 1:].flatten()
        #     correlations = np.corrcoef(states[neuron, 1:], states[available_for_this_neuron, :-1])[0, 1:]
        #     # Find the neuron with the maximum Pearson correlation (use isclose to handle floating point errors)
        #
        #     corr_max = np.nanmax(correlations)
        #     max_value_indices = np.array(available_for_this_neuron)[np.isclose(correlations, corr_max)]
        #
        #     incoming_connexion = np.random.choice(max_value_indices)
        #     return (neuron, incoming_connexion)
        #
        #
        # new_connections = Parallel(n_jobs=n_jobs)(
        #     delayed(compute_pearson_for_neuron)(neuron)
        #     for neuron in neurons_needing_new_connection
        # )

    elif method == "random":
        for neuron in neurons_needing_new_connection:
            available_for_this_neuron = available_neurons(neuron, connectivity_matrix, neurons_pool, max_partners)

            incoming_connexion = np.random.choice(available_for_this_neuron)
            new_connections.append((neuron, incoming_connexion))

    else:
        raise ValueError("Invalid method. Must be one of 'mi', 'pearson', 'random'.")

    return new_connections


def determine_pruning_pairs(neurons_for_pruning, connectivity_matrix, states=None, method="random", random_seed=None,
                            n_jobs=1):
    """
    Identifies pairs of neurons for pruning from a connectivity matrix.

    Returns:
    - A list of tuples, where each tuple represents a pair (neuron, connection) to be pruned.
    """

    if random_seed is not None:
        np.random.seed(random_seed)
    if states is None and (method == "mi" or method == "pearson"):
        raise ValueError("States must be provided if mutual information or pearson based pruning is used.")

    new_pruning_pairs = []
    if method == "mi":
        for neuron in neurons_for_pruning:
            connections = connectivity_matrix.getrow(neuron).nonzero()[1]
            if len(connections) == 0:
                continue
            mi = compute_mutual_information(states, [connections, [neuron]])[neuron, connections]

            min_value_indices = np.array(connections)[np.isclose(mi, np.nanmin(mi))]
            chosen_connection = np.random.choice(min_value_indices)
            if chosen_connection is None:
                raise ValueError("No incoming connection found for neuron, this should not happen.")
            else:
                new_pruning_pairs.append((neuron, chosen_connection))

    elif method == "pearson":
        for neuron in neurons_for_pruning:
            connections = connectivity_matrix.getrow(neuron).nonzero()[1]
            if len(connections) == 0:
                continue

            # Compute Pearson correlations for all potential connections
            correlations = np.corrcoef(states[neuron, 1:], states[connections, :-1])[0, 1:]

            # Find the neuron with the maximum Pearson correlation
            corr_min = np.nanmin(correlations)
            if corr_min == np.nan:
                print("Correlation is nan")
                print("correlations", correlations)
                print("neuron", neuron)
                print("connections", connections)
                print("states", states)
            min_value_indices = np.array(connections)[np.isclose(correlations, corr_min)]
            chosen_connection = np.random.choice(min_value_indices)
            if chosen_connection is None:
                raise ValueError("No incoming connection found for neuron, this should not happen.")
            else:
                new_pruning_pairs.append((neuron, chosen_connection))

    elif method == "random":
        for neuron in neurons_for_pruning:
            connections = connectivity_matrix.getrow(neuron).nonzero()[1]
            if len(connections) == 0:
                continue
            chosen_connection = np.random.choice(connections)
            if chosen_connection is None:
                raise ValueError("No incoming connection found for neuron, this should not happen.")
            else:
                new_pruning_pairs.append((neuron, chosen_connection))
    else:
        raise ValueError("Invalid method. Must be one of 'mi', 'pearson', 'random'.")

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
