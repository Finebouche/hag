import numpy as np
from hag.neuron_selection import determine_connection_pairs, determine_pruning_pairs
from models.reservoir import update_reservoir
import matplotlib.pyplot as plt
from tqdm import tqdm


def compute_variance(states, target_variance, variance_spread, average="WHOLE", queue_size=10):
    # Calculate the synaptic change based on variance as per user requirement
    states = np.array(states)

    # we calculate SEM over a time window
    if average == "WHOLE":
        sem = np.std(states, axis=0)
    elif average == "QUEUE":
        sem = np.std(states[-queue_size:], axis=0)
    else:
        raise ValueError("Average must be one of 'WHOLE', 'QUEUE'")

    return (sem - target_variance) / variance_spread


def compute_synaptic_change(states, target_rate, rate_spread, change_type="linear", average="WHOLE",
                            queue_size=5, minimum_calcium_concentration=0.1):
    # Calculate the synaptic change based on https://doi.org/10.3389/fnana.2016.00057
    states = np.array(states)
    if change_type == "linear":
        delta_z = (states - target_rate) / rate_spread

    elif change_type == "gaussian":
        a = (target_rate + minimum_calcium_concentration) / 2
        b = (target_rate - minimum_calcium_concentration) / 1.65
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

    return delta_z  # -1,5->-1 and 1.5->1


def run_algorithm(W, Win, bias, leaky_rate, activation_function, input_data, weight_increment,
                  target, spread, algorithm_type, multiple_instances, min_increment, max_increment=None, use_full_instance=False,
                  max_partners=np.inf, method="random", intrinsic_saturation=0.9, intrinsic_coef=0.9, average="WHOLE",
                  n_jobs=1, visualize=False, record_history=False):
    neurons_state = np.random.uniform(0, 1, bias.size)
    states_history = []
    delta_z_history = []
    W_history = []

    if visualize:
        total_add = 0
        total_prun = 0
        add = [0]
        prun = [0]
    step = 0
    steps = []

    if max_increment is None:
        time_increment = min_increment
        int_logspace = [time_increment]
        max_increment = time_increment
    else:
        logspace = np.logspace(np.log10(min_increment), np.log10(max_increment), num=10)
        int_logspace = np.unique(np.round(logspace).astype(int))

    if not multiple_instances:
        use_full_instance = False

    if use_full_instance:  # if is true, take the next instance of the instance array input_data
        # check that input data comon dimension is the same
        assert len(set([instance.shape[1] for instance in input_data])) == 1, "common dimension must be 1"
        init_array = np.concatenate(input_data[:3], axis=0)
        input_data = input_data[3:]
    else:
        if multiple_instances:
            input_data = np.concatenate(input_data, axis=0)
        init_length = min_increment * 5
        init_array = input_data[:init_length]
        input_data = input_data[init_length:]

    # initialization
    for input_value in init_array:
        neurons_state = update_reservoir(W, Win, input_value, neurons_state, leaky_rate, bias, activation_function)
        states_history.append(neurons_state)

    pbar = tqdm(total=len(input_data), desc="HAG algorithm")
    while (len(input_data) > max_increment and not use_full_instance) or (len(input_data) > 0 and use_full_instance):
        if use_full_instance:  # if is true, take the next instance of the instance array input_data
            input_array = input_data[0]
            input_data = input_data[1:]
            inc = 1
            state_inc = inc * input_array.shape[0]
        else:
            # randomly select the increment size
            inc = np.random.choice(int_logspace)
            input_array = input_data[:inc]
            input_data = input_data[inc:]
            state_inc = inc

        for input_value in input_array:
            neurons_state = update_reservoir(W, Win, input_value, neurons_state, leaky_rate, bias, activation_function)
            states_history.append(neurons_state)

        if algorithm_type in ("hadsp", "mean_hag_marked", "rnn-mean_hag"):
            delta_z = compute_synaptic_change(states_history[-state_inc:], target, spread, average=average)
        elif algorithm_type in ("desp", "var_hag_marked"):
            delta_z = compute_variance(states_history[-state_inc:], target, spread, average=average)
        else:
            raise ValueError("type must be one of 'hadsp', 'desp'")

        W, _, nb_new_add, nb_new_prun = hag_step(W, states_history[-state_inc:], delta_z, weight_increment,
                                                 max_partners=max_partners, method=method, n_jobs=n_jobs)

        if algorithm_type in ("desp", "var_hag_marked"):
            # implement intrinsic homeostatic plasticity based on saturation of states
            neurons_states = np.array(states_history[-state_inc:]).T
            for neuron_states, i in zip(neurons_states, range(bias.size)):
                if np.all(neuron_states >= intrinsic_saturation):
                    W[i, :] = W[i, :] * intrinsic_coef  # Now you can modify it directly

        if not record_history:
            states_history = []
        else:
            W_history.append((np.copy(W)))
            if use_full_instance:  # happened variance to variance_history for a number of inc
                delta_z_history.extend([delta_z] * 10)
            else:
                delta_z_history.extend([delta_z] * inc)

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
    return W, [states_history, delta_z_history, W_history]


def hag_step(W_e, states, delta_z, weight_increment, W_inhibitory=np.array([]), max_partners=np.inf, method="random",
             n_jobs=1):
    states = np.array(states).T
    nb_neurons = W_e.shape[0]
    neurons = np.arange(nb_neurons)
    total_prun = 0
    total_add = 0
    assert states.shape[0] == nb_neurons, "Wrong state shape. "

    need_pruning = neurons[delta_z >= 1]
    # We prune excitatory connexion to drive delta_z down
    new_prune_pairs = determine_pruning_pairs(need_pruning, W_e, states, method, n_jobs=n_jobs)
    for connexion in new_prune_pairs:
        W_e = change_connexion(W_e, connexion[0], connexion[1], -weight_increment)
        total_prun += 1
    # We add inhibitory connexion to drive delta_z down
    if min(W_inhibitory.shape) > 0:
        new_connexion_pairs = determine_connection_pairs(need_pruning, W_inhibitory, states, method,
                                                         is_inter_matrix=True)
        for connexion in new_connexion_pairs:
            W_inhibitory = change_connexion(W_inhibitory, connexion[0], connexion[1], weight_increment)
            total_add += 1

    need_increase = neurons[delta_z <= -1]
    # We add an excitatory connexion to drive delta_z up
    new_connexion_pairs = determine_connection_pairs(need_increase, W_e, states, method, max_partners=max_partners,
                                                     n_jobs=n_jobs)
    for connexion in new_connexion_pairs:
        W_e = change_connexion(W_e, connexion[0], connexion[1], weight_increment)
        total_add += 1
    # If needed we prune inhibitory connexion to increase the rate
    if min(W_inhibitory.shape) > 0:
        new_prune_pairs = determine_pruning_pairs(need_pruning, W_inhibitory, states, method)
        for connexion in new_prune_pairs:
            W_inhibitory = change_connexion(W_inhibitory, connexion[0], connexion[1], -weight_increment)
            total_prun += 1

    return W_e, W_inhibitory, total_add, total_prun


def change_connexion(W, i, j, value):
    # i for rows, j for columns
    W[i, j] = W[i, j] + value
    if W[i, j] < 0:
        W[i, j] = 0
    return W
