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
            mi = compute_mutual_information(states, [connections, [neuron]], n_jobs=n_jobs)[neuron, connections]

            min_value_indices = np.array(connections)[np.isclose(mi, np.min(mi))]
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
            correlations = np.corrcoef(states[neuron, :], states[connections, :])[0, 1:]

            # Find the neuron with the maximum Pearson correlation
            min_value_indices = np.array(connections)[np.isclose(correlations, np.min(correlations))]
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


def hadsp(W_e, states, delta_z, weight_increment, W_inhibitory=np.array([]), max_partners=12, method="random", n_jobs=1):
    states = np.array(states).T
    nb_neurons = W_e.shape[0]
    neurons = np.arange(nb_neurons)
    total_prun = 0
    total_add = 0
    assert states.shape[0] == nb_neurons, "Wrong state shape. "

    need_pruning = neurons[delta_z >= 1]
    # We prune excitatory connexion to drive delta_z down
    new_prune_pairs = determine_pruning_pairs(need_pruning, W_e, states, method, n_jobs=n_jobs)
    for connexion in new_prune_pairs:
        W_e = change_connexion(W_e, connexion[0], connexion[1], -weight_increment)
        total_prun += 1
    # We add inhibitory connexion to drive delta_z down
    if min(W_inhibitory.shape) > 0:
        new_connexion_pairs = determine_connection_pairs(need_pruning, W_inhibitory, states, method, is_inter_matrix=True)
        for connexion in new_connexion_pairs:
            W_inhibitory = change_connexion(W_inhibitory, connexion[0], connexion[1], weight_increment)
            total_add += 1

    need_increase = neurons[delta_z <= -1]
    # We add an excitatory connexion to drive delta_z up
    new_connexion_pairs = determine_connection_pairs(need_increase, W_e, states, method, max_partners=max_partners,
                                                     n_jobs=n_jobs)
    for connexion in new_connexion_pairs:
        W_e = change_connexion(W_e, connexion[0], connexion[1], weight_increment)
        total_add += 1
    # If needed we prune inhibitory connexion to increase the rate
    if min(W_inhibitory.shape) > 0:
        new_prune_pairs = determine_pruning_pairs(need_pruning, W_inhibitory, states, method)
        for connexion in new_prune_pairs:
            W_inhibitory = change_connexion(W_inhibitory, connexion[0], connexion[1], -weight_increment)
            total_prun += 1

    return W_e, W_inhibitory, total_add, total_prun


def change_connexion(W, i, j, value):
    # i for rows, j for columns
    W = sparse.lil_matrix(W)
    W[i, j] = W[i, j] + value
    if W[i, j] < 0:
        W[i, j] = 0
    W = sparse.coo_matrix(W)
    W.eliminate_zeros()
    return W
