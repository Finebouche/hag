from scipy import sparse
import numpy as np

class TwoDimArrayWrapper:
    def __init__(self, input_data):
        if input_data.ndim != 2:
            raise ValueError("Expected a 2D array.")
        self.input_data = input_data
        self.shape = input_data.shape
        self.size = input_data.shape[1]
        self.flat_data = input_data.flatten()

    def __getitem__(self, key):
        # Handle single element access
        return self.input_data[:, key]


def select_pairs_connexion(need_new, W, is_inter_matrix=False, max_connections=12):
    need_new = list(need_new)
    new_connexions = []

    if is_inter_matrix:
        # all neuron are available for to make a new connection (including the one it has already connection with)
        available_neurons = list(range(W.shape[1]))
    else:
        # only neurons that need a new connection are available
        available_neurons = list(need_new)

    for selected_neuron in need_new:
        available_for_this_neuron = available_neurons.copy()
        if not is_inter_matrix:
            # cannot add a connexion with itself
            available_for_this_neuron.remove(selected_neuron)
        # If neuron already has more than MAX_NUMBER_OF_PARTNER partners:
        # the available neurons are the one that already have a connexion with it
        if np.count_nonzero(W.getrow(selected_neuron).A) >= max_connections:
            available_for_this_neuron = W.getrow(selected_neuron).nonzero()[1]

        # select randomly
        if len(available_for_this_neuron) > 0:
            incoming_connexion = np.random.choice(available_for_this_neuron)
            new_connexions.append((selected_neuron, incoming_connexion))

    return new_connexions


def select_pairs_pruning(need_pruning, W):
    # select pruning pairs if they are pair matching
    # probably non-optimal
    new_pruning_pairs = []
    need_pruning = list(need_pruning)
    for selected_neuron in need_pruning:
        # for a selected neuron we look at the incoming connexion (the row)
        row = W.getrow(selected_neuron).nonzero()[1]
        if len(row) > 0:
            incoming_connexion = row[np.random.randint(len(row))]
            new_pruning_pairs.append((selected_neuron, incoming_connexion))
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
