import numpy as np
from connexion_generation.utility import change_connexion, select_pairs_pruning



def select_pairs_connexion(need_new, W, is_inter_matrix=False):
    # Old way to do it
    #     # Select on neuron out of the one that need connexion but randomly
    #     row = np.where(W.getrow(selected_neuron).A == 0)[1]

    # New way
    need_new = list(need_new)
    new_connexion = []

    # all neuron are available for to make a new connection (including the one it has already connection with)
    available_neuron = list(range(W.shape[1]))

    for selected_neuron in need_new:
        available_for_this_neuron = available_neuron.copy()
        if not is_inter_matrix:
            # cannot add a connexion with itself
            available_for_this_neuron.remove(selected_neuron)

        # but we limit the number of connexion with one other neuron to 5
        is_superior = W.A[selected_neuron] > 1
        available_for_this_neuron = np.array(available_for_this_neuron)[~is_superior[available_for_this_neuron]]

        # select randomly
        incoming_connexion = available_for_this_neuron[np.random.randint(len(available_for_this_neuron))]
        new_connexion.append((selected_neuron, incoming_connexion))

    return new_connexion


def bounded_adsp(W_e, state, delta_z, value, W_outside_neurons=np.array([])):
    neurons = np.arange(len(state))
    total_prun = 0
    total_add = 0

    # DECREASE THE RATE
    need_pruning = neurons[delta_z <= -1]
    # We prune excitatory connexion to decrease the rate
    new_prune_pairs = select_pairs_pruning(need_pruning, W_e)
    for connexion in new_prune_pairs:
        W_e = change_connexion(W_e, connexion[0], connexion[1], -value)
        total_prun += 1
    # We add inhibitory connexion to decrease the rate
    if min(W_outside_neurons.shape) > 0:
        new_connexion_pairs = select_pairs_connexion(need_pruning, W_outside_neurons, True)
        for connexion in new_connexion_pairs:
            W_outside_neurons = change_connexion(W_outside_neurons, connexion[0], connexion[1], value)
            total_add += 1

    # INCREASE THE RATE
    need_increase = neurons[delta_z >= 1]
    # We add an excitatory connexion to increase the rate
    new_connexion_pairs = select_pairs_connexion(need_increase, W_e)
    # from the list of neuron that need a new connexion we select the ones that already have more than 20 connexion
    sum_connexion = np.sum(W_e.A, axis=0)
    is_superior = sum_connexion > 20
    need_pruning_first = need_increase[is_superior[need_increase]]
    for connexion in need_pruning_first:
        # We prune the connexion that have the lowest strictly positive weight
        connexion_to_prune = np.where(W_e.A[:, connexion] > 0)[0].min()
        W_e = change_connexion(W_e, connexion_to_prune, connexion, -value)
        total_prun += 1

    for connexion in new_connexion_pairs:
        W_e = change_connexion(W_e, connexion[0], connexion[1], value)
        total_add += 1
    # We prune inhibitory connexion to increase the rate
    if min(W_outside_neurons.shape) > 0:
        new_prune_pairs = select_pairs_pruning(need_pruning, W_outside_neurons)
        for connexion in new_prune_pairs:
            W_outside_neurons = change_connexion(W_outside_neurons, connexion[0], connexion[1], -value)
            total_prun += 1

    return W_e, W_outside_neurons, total_add, total_prun

