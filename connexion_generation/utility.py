from scipy import sparse
import numpy as np
from connexion_generation.correlation_utility import compute_mutual_information, compute_pearson_corr


def available_neurons(neuron, connectivity_matrix, neurons_pool, max_partners=12, is_inter_matrix=False):
    # If neuron already has more than MAX_NUMBER_OF_PARTNER partners:
    # the available neurons are the one that already have a connexion with it
    noun_zeros = connectivity_matrix.getrow(neuron).nonzero()[1]

    if len(noun_zeros) >= max_partners:
        available_for_this_neuron = noun_zeros
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
        set_of_possible_pairs = set()
        for neuron in neurons_needing_new_connection:
            available_for_this_neuron = available_neurons(neuron, connectivity_matrix, neurons_pool, max_partners)
            # add the possible pairs to the set
            set_of_possible_pairs.update(
                (min(neuron, available_neuron), max(neuron, available_neuron))
                for available_neuron in available_for_this_neuron
            )

        mi_for_available_neurons = compute_mutual_information(states, set_of_possible_pairs, n_jobs=n_jobs)

        for neuron in neurons_needing_new_connection:
            available_for_this_neuron = available_neurons(neuron, connectivity_matrix, neurons_pool, max_partners)

            # select the connexion with the highest mutual information
            mi_for_this_neuron = mi_for_available_neurons[neuron, available_for_this_neuron]
            # Step 2: Get the indices in `mi_interresting` that correspond to this maximum value
            max_value_indices = np.array(available_for_this_neuron)[
                np.isclose(mi_for_this_neuron, np.max(mi_for_this_neuron))]

            incoming_connexion = np.random.choice(max_value_indices)
            new_connections.append((neuron, incoming_connexion))

    elif method == "pearson":
        # print(neurons_needing_new_connection)

        for neuron in neurons_needing_new_connection:
            available_for_this_neuron = available_neurons(neuron, connectivity_matrix, neurons_pool, max_partners)

            # Compute Pearson correlations for all potential connections
            # correlations = np.corrcoef(states[neuron, :], states[available_for_this_neuron, :])[0, 1:].flatten()
            correlations = compute_pearson_corr(states[neuron, 1:], states[available_for_this_neuron, :-1])

            # Check that lenght of available_for_this_neuron and correlations are the same
            if len(available_for_this_neuron) != len(correlations):
                raise ValueError("Length of available_for_this_neuron and correlations are not the same.")
            # Step 2: Get the indices in available_for_this_neuron that correspond to this maximum value

            # Find the neuron with the maximum Pearson correlation (use isclose to handle floating point errors)
            max_value_indices = np.array(available_for_this_neuron)[np.isclose(correlations, np.max(correlations))]

            incoming_connexion = np.random.choice(max_value_indices)
            new_connections.append((neuron, incoming_connexion))

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
            mi = compute_mutual_information(states, [connections, [neuron]], n_jobs=n_jobs)

            mi_interresting = mi[neuron, connections]
            min_value_indices = np.array(connections)[np.isclose(mi_interresting, np.min(mi_interresting))]
            chosen_connection = np.random.choice(min_value_indices)

            new_pruning_pairs.append((neuron, chosen_connection))
    elif method == "pearson":
        for neuron in neurons_for_pruning:
            connections = connectivity_matrix.getrow(neuron).nonzero()[1]
            if len(connections) == 0:
                continue

            # Compute Pearson correlations for all potential connections
            correlations = np.corrcoef(states[neuron, :], states[connections, :])[0, 1:]

            # Find the neuron with the maximum Pearson correlation
            min_value_indices = np.array(connections)[np.isclose(correlations, np.min(correlations))]

            incoming_connexion = np.random.choice(min_value_indices)

            if incoming_connexion is None:
                raise ValueError("No incoming connection found for neuron, this should not happen.")
            else:
                new_pruning_pairs.append((neuron, incoming_connexion))

    elif method == "random":
        for neuron in neurons_for_pruning:
            connections = connectivity_matrix.getrow(neuron).nonzero()[1]
            if len(connections) == 0:
                continue
            chosen_connection = np.random.choice(connections)
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
