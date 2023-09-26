import numpy as np
from connexion_generation.utility import change_connexion, select_pairs_pruning, select_pairs_connexion
from reservoir.reservoir import update_reservoir
import matplotlib.pyplot as plt
from tqdm import tqdm

def compute_synaptic_change(states, target_activation_levels, growth_parameter, change_type="linear", average="QUEUE",
                            queue_size=5, minimum_calcium_concentration=0.1):
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
    if average == "WHOLE":
        delta_z = np.ma.average(delta_z, axis=0)
    # else we take the last value from the time window
    elif average == "LAST":
        delta_z = delta_z[-1]
    # else we take the last queue_size values from the time window
    elif average == "QUEUE":
        delta_z = np.ma.average(delta_z[-queue_size:], axis=0)

    return np.trunc(delta_z)  # -1,5->-1 and 1.5->1

def bounded_adsp(W_e, state, delta_z, value, W_inhibitory_connexions=np.array([]), max_connections=12):
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

    # If needed we prune inhibitory connexion to increase the rate
    if min(W_inhibitory_connexions.shape) > 0:
        new_prune_pairs = select_pairs_pruning(need_pruning, W_inhibitory_connexions)
        for connexion in new_prune_pairs:
            W_inhibitory_connexions = change_connexion(W_inhibitory_connexions, connexion[0], connexion[1], -value)
            total_prun += 1

    return W_e, W_inhibitory_connexions, total_add, total_prun

def run_HADSP_algorithm(W, Win, bias, leaky_rate, activation_function, input_data, increment, value, target_rate,
                        growth_parameter, visualize=False):  # last_state
    state = np.random.uniform(0, 1, bias.size)
    state_history = []

    total_add = 0
    total_prun = 0
    add = []
    prun = []
    step = 0

    for i in range(increment * 5):
        state = update_reservoir(W, Win, input_data[i], state, leaky_rate, bias, activation_function)
        state_history.append(state)

    # size of simulation
    number_steps = int((input_data.size - increment * 5) / increment)
    for k in tqdm(range(number_steps)):
        delta_z = compute_synaptic_change(state_history[-increment:], target_rate, growth_parameter, average="WHOLE")
        W, _, nb_new_add, nb_new_prun = bounded_adsp(W, state, delta_z, value)

        for i in range(increment):
            state = update_reservoir(W, Win, input_data[increment * (k + 5) + i], state, leaky_rate, bias,
                                     activation_function)
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
        plt.plot(np.arange(step) * increment, add, label="total number of added connexion")
        plt.plot(np.arange(step) * increment, prun, label="total number of prunned connexion")
        plt.plot(np.arange(step) * increment, add - prun, label="difference")
        plt.plot(np.arange(step) * increment, [0] * step, linestyle=(0, (1, 10)))
        plt.legend()
        plt.grid()
    return W
