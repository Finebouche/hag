# Code for Homeostatic Activity Dependant Structural Plasticity

import numpy as np
from connexion_generation.utility import change_connexion, determine_pruning_pairs, determine_connection_pairs
from reservoir.reservoir import update_reservoir
import matplotlib.pyplot as plt
from tqdm import tqdm


def compute_synaptic_change(states, target_activation_levels, growth_parameter, change_type="linear", average="WHOLE",
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

    # We average over a time window (weighted average for 0 size array)
    if average == "WHOLE":
        delta_z = np.ma.average(delta_z, axis=0)
    # else we take the last value from the time window
    elif average == "LAST":
        delta_z = delta_z[-1]
    # else we take the last queue_size values from the time window
    elif average == "QUEUE":
        delta_z = np.ma.average(delta_z[-queue_size:], axis=0)

    return np.trunc(delta_z)  # -1,5->-1 and 1.5->1


def bounded_hadsp(W_e, states, delta_z, weight_increment, W_inhibitory_connexions=np.array([]), max_partners=12, mi_based=False):
    neurons = np.arange(len(states[0]))
    total_prun = 0
    total_add = 0

    # DECREASE THE RATE
    need_pruning = neurons[delta_z <= -1]
    # We prune excitatory connexion to decrease the rate
    new_prune_pairs = determine_pruning_pairs(need_pruning, W_e, states, mi_based)
    for connexion in new_prune_pairs:
        W_e = change_connexion(W_e, connexion[0], connexion[1], -weight_increment)
        total_prun += 1
    # We add inhibitory connexion to decrease the rate
    if min(W_inhibitory_connexions.shape) > 0:
        new_connexion_pairs = determine_connection_pairs(need_pruning, W_inhibitory_connexions, True)
        for connexion in new_connexion_pairs:
            W_inhibitory_connexions = change_connexion(W_inhibitory_connexions, connexion[0], connexion[1], weight_increment)
            total_add += 1

    # INCREASE THE RATE
    need_increase = neurons[delta_z >= 1]
    # We add an excitatory connexion to increase the rate
    new_connexion_pairs = determine_connection_pairs(need_increase, W_e, max_partners=max_partners)
    for connexion in new_connexion_pairs:
        W_e = change_connexion(W_e, connexion[0], connexion[1], weight_increment)
        total_add += 1
    # If needed we prune inhibitory connexion to increase the rate
    if min(W_inhibitory_connexions.shape) > 0:
        new_prune_pairs = determine_pruning_pairs(need_pruning, W_inhibitory_connexions, states, mi_based)
        for connexion in new_prune_pairs:
            W_inhibitory_connexions = change_connexion(W_inhibitory_connexions, connexion[0], connexion[1], -weight_increment)
            total_prun += 1

    return W_e, W_inhibitory_connexions, total_add, total_prun


def run_hadsp_algorithm(W, Win, bias, leaky_rate, activation_function, input_data, time_increment, weight_increment, target_rate,
                        growth_parameter, max_increment=None, mi_based=False, average="WHOLE",
                        instances=False, common_index=0, visualize=False):
    state = np.random.uniform(0, 1, bias.size)
    state_history = []

    total_add = 0
    total_prun = 0
    add = [0]
    prun = [0]
    step = 0
    steps = []

    if max_increment is None:
        int_logspace = [time_increment]
        max_increment = time_increment
    else:
        logspace = np.logspace(np.log10(time_increment), np.log10(max_increment), num=10)
        int_logspace = np.round(logspace).astype(int)

    if instances:   # if is true, take the next instance of the instance array input_data
        # check that input data comon dimension is the same
        assert len(set([instance.shape[1] for instance in input_data])) == 1,  "common dimension must be 1"
        init_array = np.concatenate(input_data[:3], axis=0)
        input_data = input_data[3:]
    else:
        # randomly select the increment size
        init_length = time_increment * 5
        init_array = input_data[:init_length]
        input_data = input_data[init_length:]

    # initialization
    for input_value in init_array:
        state = update_reservoir(W, Win, input_value, state, leaky_rate, bias, activation_function)
        state_history.append(state)

    pbar = tqdm(total=len(input_data))
    while (len(input_data) > max_increment and not instances) or (len(input_data) > 0 and instances):
        if instances:   # if is true, take the next instance of the instance array input_data
            input_array = input_data[0]
            input_data = input_data[1:]
            inc = 1
        else:
            # randomly select the increment size
            inc = np.random.choice(int_logspace)
            input_array = input_data[:inc]
            input_data = input_data[inc:]


        for input_value in input_array:
            state = update_reservoir(W, Win, input_value, state, leaky_rate, bias, activation_function)
            state_history.append(state)

        delta_z = compute_synaptic_change(state_history[-inc:], target_rate, growth_parameter, average=average)
        W, _, nb_new_add, nb_new_prun = bounded_hadsp(W, state_history[-inc:], delta_z, weight_increment, mi_based=mi_based)

        total_add += nb_new_add
        total_prun += nb_new_prun
        add.append(total_add)
        prun.append(total_prun)
        step += inc
        steps.append(step)
        pbar.update(inc)

    add = np.array(add)
    prun = np.array(prun)
    pbar.close()

    if visualize:
        plt.figure()
        plt.plot(steps, add, label="total number of added connexion")
        plt.plot(steps, prun, label="total number of prunned connexion")
        plt.plot(steps, add - prun, label="difference")
        plt.plot(steps, steps, linestyle=(0, (1, 10)))
        plt.legend()
        plt.grid()
    return W, state_history
