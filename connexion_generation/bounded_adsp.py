import numpy as np
from connexion_generation.utility import change_connexion, set_connexion, select_pairs_pruning
from connexion_generation.utility import compute_synaptic_change
from reservoir.reservoir import update_reservoir

def select_pairs_connexion(need_new, W, is_inter_matrix=False, max_connections = 12):
    # Old way to do it
    #     # Select on neuron out of the one that need connexion but randomly
    #     row = np.where(W.getrow(selected_neuron).A == 0)[1]

    # New way
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


        # # but we limit the number of connexion with one other neuron to 8
        # neurons_values = W.getrow(selected_neuron).A[:, available_for_this_neuron]
        # # select from available_for_this_neuron the one that have neuron_values < 0.8 and that are not already connected
        # available_for_this_neuron = np.array([available_for_this_neuron])[neurons_values < 0.8]


        # select randomly
        if len(available_for_this_neuron) > 0:
            incoming_connexion = np.random.choice(available_for_this_neuron)
            new_connexions.append((selected_neuron, incoming_connexion))

    return new_connexions


def bounded_adsp(W_e, state, delta_z, value, W_inhibitory_connexions=np.array([]), max_connections = 12):
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
    if min(W_inhibitory_connexions.shape) > 0:
        new_connexion_pairs = select_pairs_connexion(need_pruning, W_inhibitory_connexions, True)
        for connexion in new_connexion_pairs:
            W_inhibitory_connexions = change_connexion(W_inhibitory_connexions, connexion[0], connexion[1], value)
            total_add += 1


    # INCREASE THE RATE
    need_increase = neurons[delta_z >= 1]

    # We add an excitatory connexion to increase the rate
    new_connexion_pairs = select_pairs_connexion(need_increase, W_e, max_connections=max_connections)

    for connexion in new_connexion_pairs:
        W_e = change_connexion(W_e, connexion[0], connexion[1], value)
        total_add += 1

    # If neuron already has more than MAX_NUMBER_OF_PARTNER partners:
    # the available neurons are the one that already have a connexion with it
    # MAX_NUMBER_OF_PARTNER = 8
    # is_superior = np.count_nonzero(W_e.A, axis=1) > MAX_NUMBER_OF_PARTNER
    # need_decrease = need_increase[is_superior[need_increase]]
    # for neuron in need_decrease:
    #     # select randomly one of the connexion to prune
    #     connexion_to_prune = np.random.choice(W_e.getrow(neuron).nonzero()[1])
    #     W_e = set_connexion(W_e, neuron, connexion_to_prune, 0)
    #     total_prun += 1

    # If needed we prune inhibitory connexion to increase the rate
    if min(W_inhibitory_connexions.shape) > 0:
        new_prune_pairs = select_pairs_pruning(need_pruning, W_inhibitory_connexions)
        for connexion in new_prune_pairs:
            W_inhibitory_connexions = change_connexion(W_inhibitory_connexions, connexion[0], connexion[1], -value)
            total_prun += 1

    return W_e, W_inhibitory_connexions, total_add, total_prun
    

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
        
def run_HADSP_algorithm(W, Win, bias, leaky_rate, activation_function, input_data, increment, value, target_rate, growth_parameter, visualize=False):    # last_state
    state = np.random.uniform(0, 1, bias.size)
    state_history = []
    
    total_add = 0
    total_prun = 0
    add = []
    prun = []
    step=0

    for i in range(increment*5):
        state = update_reservoir(W, Win, input_data[i], state, leaky_rate, bias, activation_function)
        state_history.append(state)

    # size of simulation 
    number_steps = int((input_data.size-increment*5)/increment)
    for k in range(number_steps): 
        delta_z = compute_synaptic_change(state_history[-increment:], target_rate, growth_parameter, average="WHOLE")
        W, _, nb_new_add, nb_new_prun = bounded_adsp(W, state, delta_z, value)
    
        for i in range(increment):
            state = update_reservoir(W, Win, input_data[increment*(k+5)+i], state, leaky_rate, bias, activation_function)
            state_history.append(state)
            
        total_add += nb_new_add
        total_prun += nb_new_prun
        add.append(total_add)
        prun.append(total_prun)
        step += 1
        
    add = np.array(add)
    prun = np.array(prun)

    if visualize:
        plt.figure()
        plt.plot(np.arange(step)*INCREMENT, add, label="total number of added connexion")
        plt.plot(np.arange(step)*INCREMENT, prun, label="total number of prunned connexion")
        plt.plot(np.arange(step)*INCREMENT, add-prun, label="difference")
        plt.plot(np.arange(step)*INCREMENT, [0]*step, linestyle=(0, (1, 10)))
        plt.legend()
        plt.grid()
    return W