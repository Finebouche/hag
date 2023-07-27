from scipy import sparse
import numpy as np

def compute_synaptic_change(states, target_activation_levels, growth_parameter, change_type="linear", average="QUEUE", queue_size = 5,
                            minimum_calcium_concentration=0.1):
    # Calculate the synaptic change based on https://doi.org/10.3389/fnana.2016.00057
    states = np.array(states)
    if change_type == "linear":
        delta_z = (target_activation_levels - states) / growth_parameter

    elif change_type == "gaussian":
        a = (target_activation_levels + minimum_calcium_concentration) / 2
        b = (target_activation_levels - minimum_calcium_concentration) / 1.65
        delta_z = growth_parameter * (2 * np.expm1(-(states - a) / b) - 1)
    else:
        raise ValueError('change_type must be "linear" or "gaussian"')

    # For the case where we average over a time window (weighted average for 0 size array)
    if average=="WHOLE":
        delta_z = np.ma.average(delta_z, axis=0)
    # else we take the last value from the time window
    elif average == "LAST":
        delta_z = delta_z[-1]
    # else we take the last queue_size values from the time window
    elif average=="QUEUE":
        delta_z = np.ma.average(delta_z[-queue_size:], axis=0)

    return np.trunc(delta_z)  # -1,5->-1 and 1.5->1


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

def set_connexion(W, i, j, value):
    W = sparse.lil_matrix(W)
    W[i, j] = value
    W = sparse.coo_matrix(W)
    if value == 0:
        W.eliminate_zeros()
    return W
