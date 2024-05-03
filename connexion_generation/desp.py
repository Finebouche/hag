# Code for Dynamic Equilibrium Structural Plasticity
import numpy as np
from connexion_generation.utility import change_connexion, determine_pruning_pairs, determine_connection_pairs
from reservoir.reservoir import update_reservoir
import matplotlib.pyplot as plt
from tqdm import tqdm


def compute_variance(states, average="WHOLE", queue_size=10):
    # Calculate the synaptic change based on variance as per user requirement
    states = np.array(states)

    # we calculate SEM over a time window
    if average == "WHOLE":
        sem = np.std(states, axis=0)
    # else we take the variance of the last value from the time window, which doesn't make sense for a single value
    # so for LAST, it's either not applicable or we keep as is, because variance of a single value is always 0.
    # else we take the variance of the last queue_size values from the time window
    elif average == "QUEUE":
        sem = np.std(states[-queue_size:], axis=0)
    else:
        raise ValueError("Average must be one of 'WHOLE', 'QUEUE'")

    return sem  # Apply truncation similar to the original implementation


def bounded_desp(W_e, states, variance, min_variance, max_variance, weight_increment,
                 W_inhibitory_connexions=np.array([]), max_partners=12, mi_based=False, n_jobs=1):
    neurons = np.arange(W_e.shape[0])
    total_prun = 0
    total_add = 0
    assert states.shape[0] == W_e.shape[
        0], "The number of neurons in the states and the number of rows in W_e must be the same."

    # DECREASE THE VARIANCE
    need_pruning = neurons[variance >= max_variance]

    # select the connexion with the highest mutual information
    new_prune_pairs = determine_pruning_pairs(need_pruning, W_e, states, mi_based=mi_based, n_jobs=n_jobs)

    for connexion in new_prune_pairs:
        W_e = change_connexion(W_e, connexion[0], connexion[1], -weight_increment)
        total_prun += 1
    # We add inhibitory connexion to decrease the rate
    if min(W_inhibitory_connexions.shape) > 0:
        new_connexion_pairs = determine_connection_pairs(need_pruning, W_inhibitory_connexions, True)
        for connexion in new_connexion_pairs:
            W_inhibitory_connexions = change_connexion(W_inhibitory_connexions, connexion[0], connexion[1],
                                                       weight_increment)
            total_add += 1

    # INCREASE THE VARIANCE
    need_increase = neurons[variance <= min_variance]
    # select a neuron with the highest mutual information
    new_connexion_pairs = determine_connection_pairs(need_increase, W_e, states, mi_based=mi_based,
                                                     max_partners=max_partners, n_jobs=n_jobs)
    for connexion in new_connexion_pairs:
        W_e = change_connexion(W_e, connexion[0], connexion[1], weight_increment)
        total_add += 1
    # If needed we prune inhibitory connexion to increase the rate
    if min(W_inhibitory_connexions.shape) > 0:
        new_prune_pairs = determine_pruning_pairs(need_pruning, W_inhibitory_connexions)
        for connexion in new_prune_pairs:
            W_inhibitory_connexions = change_connexion(W_inhibitory_connexions, connexion[0], connexion[1],
                                                       -weight_increment)
            total_prun += 1

    return W_e, W_inhibitory_connexions, total_add, total_prun


def run_desp_algorithm(W, Win, bias, leaky_rate, activation_function, input_data, time_increment, weight_increment,
                       min_variance, max_variance, max_increment=None, max_partners=12, mi_based=False, average="WHOLE",
                       instances=False, n_jobs=1, visualize=False):
    state = np.random.uniform(0, 1, bias.size)
    state_history = []
    variance_history = []

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
        assert len(set([instance.shape[0] for instance in input_data])) == 1,  "common dimension must be 0"
        init_array = np.concatenate(input_data[:3], axis=1).T
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
            input_array = input_data[0].T
            input_data = input_data[1:]
            inc = 1
            state_inc = inc*input_array.shape[0]
        else:
            # randomly select the increment size
            inc = np.random.choice(int_logspace)
            input_array = input_data[:inc]
            input_data = input_data[inc:]
            state_inc = inc

        for input_value in input_array:
            state = update_reservoir(W, Win, input_value, state, leaky_rate, bias, activation_function)
            state_history.append(state)

        variance = compute_variance(state_history[-state_inc:], average=average)
        # happened variance to variance_history for a number of inc
        variance_history.extend([variance] * state_inc)

        W, _, nb_new_add, nb_new_prun = bounded_desp(W, np.array(state_history[-inc:]).T, variance, min_variance,
                                                     max_variance, weight_increment, max_partners=max_partners,
                                                     mi_based=mi_based, n_jobs=n_jobs)

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

    return W, state_history, variance_history