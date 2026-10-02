import numpy as np
from joblib import Parallel, delayed


def mutual_information_matrix(states, bins=10, chunk=64):
    """
    Mutual information (bits) between all the rows of states (neurons, T), from their joint histograms on `bins`
    bins shared by all neurons (edges of the histogram of all the states), 1e-10 being added to each joint
    probability. Vectorized: the joint histograms of all the pairs are the products of the one-hot encodings of the
    bins of each neuron, computed by blocks of `chunk` neurons to bound the memory.
    """
    n, T = states.shape
    edges = np.histogram_bin_edges(states, bins=bins)
    # bin of each state ([e_k, e_k+1), the last bin being closed, as np.histogram2d)
    idx = np.clip(np.searchsorted(edges, states, side="right") - 1, 0, bins - 1)
    onehot = np.zeros((n, bins, T))
    onehot[np.arange(n)[:, None], idx, np.arange(T)[None, :]] = 1
    flat = onehot.reshape(n * bins, T)
    mi = np.empty((n, n))
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        # joint probabilities of the pairs (i in the block, j): (block, n, bins, bins)
        joint = (flat[start * bins:stop * bins] @ flat.T).reshape(stop - start, bins, n, bins).transpose(0, 2, 1, 3)
        joint = joint / T + 1e-10
        marginal_i, marginal_j = joint.sum(axis=3), joint.sum(axis=2)
        mi[start:stop] = np.sum(joint * np.log2(joint / (marginal_i[..., :, None] * marginal_j[..., None, :])),
                                axis=(2, 3))
    return mi


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
                               is_inter_matrix=False, max_partners=np.inf, random_seed=None, n_jobs=1,
                               rng=None):
    """
    Determine pairs of neurons for establishing new connections based on specified criteria.
    rng: np.random.RandomState used for the random choices (default: numpy's global RNG).

    Returns:
    - A list of tuples, where each tuple represents a new connection (source_neuron, target_neuron).
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    rng = np.random if rng is None else rng
    if states is None and method in ("mi", "pearson"):
        raise ValueError("States must be provided if mutual information or pearson based pruning is used.")

    neurons_pool = list(range(connectivity_matrix.shape[1])) if is_inter_matrix else list(neurons_needing_new_connection)
    if len(neurons_pool) <= 1:
        return []

    if method == "pearson":
        # Pearson correlation between all neurons, computed once for all the neurons needing a connexion
        # (same as the correlation of states[neuron, 1:] with states[available, 1:] for each neuron, much faster)
        with np.errstate(divide="ignore", invalid="ignore"):
            pearson_corr = np.corrcoef(states[:, 1:])
    elif method == "mi":
        # Mutual information between the neurons of the pool (the available neurons of each neuron, with
        # max_partners = inf), computed once for all the neurons needing a connexion (bins shared by the pool)
        mi_neurons = np.array(neurons_pool) if np.isinf(max_partners) else np.arange(states.shape[0])
        mi = np.full((states.shape[0], states.shape[0]), np.nan)
        mi[np.ix_(mi_neurons, mi_neurons)] = mutual_information_matrix(states[mi_neurons])

    def compute_new_connexion(neuron):
        available_for_neuron = available_neurons(neuron, connectivity_matrix, neurons_pool, max_partners)
        if method == "mi":
            mi_for_neuron = mi[neuron, available_for_neuron]
            neuron_to_choose_from = np.array(available_for_neuron)[np.isclose(mi_for_neuron, np.nanmax(mi_for_neuron))]
        elif method == "hebbian":
            # Local Hebbian rule: co-activity x_i * x_j at a single time step (first of the window)
            correlations = states[neuron, 0] * states[available_for_neuron, 0]
            neuron_to_choose_from = np.array(available_for_neuron)[correlations > 0]
        elif method == "pearson":
            correlations = pearson_corr[neuron, available_for_neuron]
            # Alternative : np.corrcoef(states[neuron, 1:], states[available_for_neuron, :-1])[0, 1:]
            neuron_to_choose_from = np.array(available_for_neuron)[np.isclose(correlations, np.nanmax(correlations))]
        elif method == "random":
            neuron_to_choose_from = np.array(available_for_neuron)
        else:
            raise ValueError("Invalid method. Must be one of 'mi', 'pearson', 'random'.")

        if neuron_to_choose_from.size == 0:
            raise ValueError(
                "No candidate neuron survived connection selection. "
                f"neuron={neuron}, method={method}, "
                f"available_count={len(available_for_neuron)}, "
                f"neurons_needing_count={len(neurons_needing_new_connection)}"
            )
        incoming_neuron = rng.choice(neuron_to_choose_from)
        return neuron, incoming_neuron

    new_connections = Parallel(n_jobs=n_jobs)(
        delayed(compute_new_connexion)(neuron)
        for neuron in neurons_needing_new_connection
    )

    return new_connections


def determine_pruning_pairs(neurons_for_pruning, connectivity_matrix, states=None, method="random", random_seed=None,
                            n_jobs=1, rng=None):
    """
    Identifies pairs of neurons for pruning from a connectivity matrix.

    Returns:
    - A list of tuples, where each tuple represents a pair (neuron, connection) to be pruned.
    """

    if random_seed is not None:
        np.random.seed(random_seed)
    rng = np.random if rng is None else rng
    if states is None and (method == "mi" or method == "pearson"):
        raise ValueError("States must be provided if mutual information or pearson based pruning is used.")

    new_pruning_pairs = []
    for neuron in neurons_for_pruning:
        connections = connectivity_matrix[neuron].nonzero()[0]
        if len(connections) == 0:
            continue

        chosen_connection = rng.choice(connections)
        new_pruning_pairs.append((neuron, chosen_connection))

    return new_pruning_pairs
