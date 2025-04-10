import numpy as np
from connexion_generation.correlation_utility import compute_mutual_information, compute_pearson_corr
from joblib import Parallel, delayed


def available_neurons(neuron, connectivity_matrix, neurons_pool, max_partners=np.inf, is_inter_matrix=False):
    # If neuron already has more than MAX_NUMBER_OF_PARTNER partners:
    # the available neurons are the one that already have a connexion with it
    non_zeros = connectivity_matrix[neuron].nonzero()[0]

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
                               is_inter_matrix=False, max_partners=np.inf, random_seed=None, n_jobs=1):
    """
    Determine pairs of neurons for establishing new connections based on specified criteria.

    Returns:
    - A list of tuples, where each tuple represents a new connection (source_neuron, target_neuron).
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    if states is None and (method == "mi" or method == "pearson"):
        raise ValueError("States must be provided if mutual information or pearson based pruning is used.")

    neurons_pool = list(range(connectivity_matrix.shape[1])) if is_inter_matrix \
        else list(neurons_needing_new_connection)
    if len(neurons_pool) <= 1:
        return []

    def compute_new_connexion(neuron):
        available_for_neuron = available_neurons(neuron, connectivity_matrix, neurons_pool, max_partners)

        if method == "mi":
            mi_for_available_neurons = compute_mutual_information(states, [available_for_neuron, [neuron]])
            mi_for_neuron = mi_for_available_neurons[neuron, available_for_neuron]
            neuron_to_choose_from = np.array(available_for_neuron)[np.isclose(mi_for_neuron, np.nanmax(mi_for_neuron))]
        elif method == "pearson":
            # Alternative : np.corrcoef(states[neuron, 1:], states[available_for_neuron, :-1])[0, 1:]
            correlations = compute_pearson_corr(states[neuron, 1:], states[available_for_neuron, 1:])
            neuron_to_choose_from = np.array(available_for_neuron)[np.isclose(correlations, np.nanmax(correlations))]
        elif method == "random":
            neuron_to_choose_from = np.array(available_for_neuron)
        else:
            raise ValueError("Invalid method. Must be one of 'mi', 'pearson', 'random'.")

        if neuron_to_choose_from.size == 0:
            raise ValueError("No neuron_to_choose_from found for neuron in adding, this should not happen as"
                             f'list(neurons_needing_new_connection) is : {neurons_needing_new_connection}.')

        incoming_neuron = np.random.choice(neuron_to_choose_from)
        return neuron, incoming_neuron

    new_connections = Parallel(n_jobs=n_jobs)(
        delayed(compute_new_connexion)(neuron)
        for neuron in neurons_needing_new_connection
    )

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
    for neuron in neurons_for_pruning:
        connections = connectivity_matrix[neuron].nonzero()[0]
        if len(connections) == 0:
            continue
        # if method == "mi":
        #     mi = compute_mutual_information(states, [connections, [neuron]])[neuron, connections]
        #     neuron_to_choose_from = np.array(connections)[np.isclose(mi, np.nanmin(mi))]
        # elif method == "pearson":
        #     correlations = np.corrcoef(states[neuron, 1:], states[connections, :-1])[0, 1:]
        #     neuron_to_choose_from = np.array(connections)[np.isclose(correlations, np.nanmin(correlations))]
        # elif method == "random":
        #     neuron_to_choose_from = connections
        # else:
        #     raise ValueError("Invalid method. Must be one of 'mi', 'pearson', 'random'.")
        #
        # if neuron_to_choose_from.size == 0:
        #     raise ValueError("No neuron_to_choose_from found for neuron in pruning, this should not happen.")

        chosen_connection = np.random.choice(connections)
        new_pruning_pairs.append((neuron, chosen_connection))

    return new_pruning_pairs
