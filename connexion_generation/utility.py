from scipy import sparse
import numpy as np

def compute_synaptic_change(states, target_activation_levels, growth_parameter, change_type="linear", time_window=1,
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
    if time_window != 1 and len(delta_z) > 0:
        delta_z = np.ma.average(delta_z, axis=0)

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

def switch_connexion(W, i, j, value):
    W = sparse.lil_matrix(W)
    W[i, j] = value
    W = sparse.coo_matrix(W)
    if value == 0:
        W.eliminate_zeros()
    return W

# ARCHIVED CODE

# def intersection(arr1, arr2):
#     # Function to find intersection of two arrays
#     result = list(filter(lambda x: x in arr1, arr2))
#     return result


# def compare(W_old, W_new, Win, bias, Wout, activation_function, b_out, U_test, y_test, last_state, leaky_rate=1):
#     y_pred_old = run(W_old, Win, bias, Wout, U_test, activation_function, b_out, last_state, leaky_rate)
#     nrmse_old = nrmse(y_test, y_pred_old)
#
#     y_pred_new = run(W_new, Win, bias, Wout, U_test, activation_function, b_out, last_state)
#     nrmse_new = nrmse(y_test, y_pred_new)
#     if float(nrmse_new) < float(nrmse_old):
#         print(float(nrmse_new))
#         return W_new, 0
#     else:
#         return W_old, 1
