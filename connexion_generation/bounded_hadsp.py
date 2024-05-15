# Code for Homeostatic Activity Dependant Structural Plasticity

import numpy as np
from connexion_generation.utility import hadsp
from reservoir.reservoir import update_reservoir
import matplotlib.pyplot as plt
from tqdm import tqdm


def compute_synaptic_change(states, target_activation_levels, rate_spread, change_type="linear", average="WHOLE",
                            queue_size=5, minimum_calcium_concentration=0.1):
    # Calculate the synaptic change based on https://doi.org/10.3389/fnana.2016.00057
    states = np.array(states)
    if change_type == "linear":
        delta_z = (states - target_activation_levels) / rate_spread

    elif change_type == "gaussian":
        a = (target_activation_levels + minimum_calcium_concentration) / 2
        b = (target_activation_levels - minimum_calcium_concentration) / 1.65
        delta_z = rate_spread * (2 * np.expm1(-(states - a) / b) - 1)
    else:
        raise ValueError('change_type must be "linear" or "gaussian"')

    # We average over a time window (weighted average for 0 size array)
    if average == "WHOLE":
        delta_z = np.ma.average(delta_z, axis=0)
    elif average == "QUEUE":
        delta_z = np.ma.average(delta_z[-queue_size:], axis=0)
    else:
        raise ValueError("Average must be one of 'WHOLE', 'QUEUE'")

    return delta_z # -1,5->-1 and 1.5->1


def run_hadsp_algorithm(W, Win, bias, leaky_rate, activation_function, input_data, time_increment, weight_increment,
                        target_rate, rate_spread, instances, max_increment=None, max_partners=12, method="random",
                        average="WHOLE", n_jobs=1, visualize=False, record_history=False):
    state = np.random.uniform(0, 1, bias.size)
    state_history = []
    delta_z_history = []

    if visualize:
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

    pbar = tqdm(total=len(input_data), desc="HADSP")
    while (len(input_data) > max_increment and not instances) or (len(input_data) > 0 and instances):
        if instances:   # if is true, take the next instance of the instance array input_data
            input_array = input_data[0]
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

        delta_z = compute_synaptic_change(state_history[-state_inc:], target_rate, rate_spread, average=average)
        W, _, nb_new_add, nb_new_prun = hadsp(W, state_history[-state_inc:], delta_z, weight_increment, max_partners=max_partners, method=method, n_jobs=n_jobs)

        if not record_history:
            state_history = []
        else:  # happened variance to variance_history for a number of inc
            delta_z_history.extend([delta_z] * 10)

        if visualize:
            total_add += nb_new_add
            total_prun += nb_new_prun
            add.append(total_add)
            prun.append(total_prun)
        step += inc
        steps.append(step)
        pbar.update(inc)

    pbar.close()

    if visualize:
        add = np.array(add)
        prun = np.array(prun)
        plt.figure()
        plt.plot(steps, add, label="total number of added connexion")
        plt.plot(steps, prun, label="total number of prunned connexion")
        plt.plot(steps, add - prun, label="difference")
        plt.plot(steps, steps, linestyle=(0, (1, 10)))
        plt.legend()
        plt.grid()
    return W, state_history, delta_z_history
