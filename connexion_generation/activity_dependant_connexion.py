import numpy as np
from connexion_generation.utility import switch_connexion, change_connexion


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


def select_pairs_connexion(need_new, W, is_inter_matrix=False):
    # Old way to do it
    #     # Select on neuron out of the one that need connexion but randomly
    #     row = np.where(W.getrow(selected_neuron).A == 0)[1]

    # New way
    need_new = list(need_new)
    new_connexion = []
    for selected_neuron in need_new:
        # all neuron are available for to make a new connection (including the one it has already connection with)
        available_neuron = list(range(W.shape[1]))
        if not is_inter_matrix:
            # cannot add a connexion with itself
            available_neuron.remove(selected_neuron)
        # select randomly
        incoming_connexion = available_neuron[np.random.randint(len(available_neuron))]
        new_connexion.append((selected_neuron, incoming_connexion))

    return new_connexion


def intersection(arr1, arr2):
    # Function to find intersection of two arrays
    result = list(filter(lambda x: x in arr1, arr2))
    return result


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


def add_activity_connexion(W, state, delta_z, value):
    neurons = np.arange(len(state))

    need_pruning = neurons[delta_z <= -1]
    # If the connexion increase is negative we do pruning
    new_prune_pairs = select_pairs_pruning(need_pruning, W)
    total_prun = 0
    for connexion in new_prune_pairs:
        W = change_connexion(W, connexion[0], connexion[1], -value)
        total_prun += 1

    need_new_connexion = neurons[delta_z >= 1]
    # If the connexion increase is positive we add a connexion
    new_connexion_pairs = select_pairs_connexion(need_new_connexion, W)
    total_add = 0
    for connexion in new_connexion_pairs:
        W = change_connexion(W, connexion[0], connexion[1], value)
        total_add += 1

    return W, total_add, total_prun


def add_activity_connexion_for_excitatory(W_e, W_ie, state, delta_z, value):
    neurons = np.arange(len(state))
    total_prun = 0
    total_add = 0

    # DECREASE THE RATE
    need_decrease = neurons[delta_z <= -1]
    # We prune excitatory connexion to decrease the rate
    new_prune_pairs = select_pairs_pruning(need_decrease, W_e)
    for connexion in new_prune_pairs:
        W_e = change_connexion(W_e, connexion[0], connexion[1], -value)
        total_prun += 1
    # We add inhibitory connexion to decrease the rate
    if min(W_ie.shape) > 0 :
        new_connexion_pairs = select_pairs_connexion(need_decrease, W_ie, True)
        for connexion in new_connexion_pairs:
            W_ie = change_connexion(W_ie, connexion[0], connexion[1], value)
            total_add += 1

    # INCREASE THE RATE
    need_increase = neurons[delta_z >= 1]
    # We add an excitatory connexion to increase the rate
    new_connexion_pairs = select_pairs_connexion(need_increase, W_e)
    for connexion in new_connexion_pairs:
        W_e = change_connexion(W_e, connexion[0], connexion[1], value)
        total_add += 1
    # We prune inhibitory connexion to increase the rate
    if min(W_ie.shape) > 0 :
        new_prune_pairs = select_pairs_pruning(need_decrease, W_ie)
        for connexion in new_prune_pairs:
            W_ie = change_connexion(W_ie, connexion[0], connexion[1], -value)
            total_prun += 1

    return W_e, W_ie, total_add, total_prun
