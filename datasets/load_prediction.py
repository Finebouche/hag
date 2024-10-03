import matplotlib.pyplot as plt
import pandas as pd
from reservoirpy.datasets import to_forecasting


def load_mackey_glass_dataset(step_ahead=5, visualize=True):
    from reservoirpy.datasets import mackey_glass
    train_steps = 15000
    test_steps = 5000
    timesteps = train_steps + test_steps
    mg_inputs = mackey_glass(timesteps + step_ahead, tau=17, a=0.2, b=0.1, n=10, x0=1.2, h=1, seed=None)

    # Define the time step of your Mackey-Glass system
    dt = 0.00001
    # Compute the equivalent sampling rate
    sampling_rate = 1 / dt

    X_train = mg_inputs[:train_steps]
    X_test = mg_inputs[train_steps:timesteps]
    Y_train = mg_inputs[step_ahead:train_steps + step_ahead]
    Y_test = mg_inputs[train_steps + step_ahead:timesteps + step_ahead]

    if visualize:
        fig, ax = plt.subplots(figsize=(16, 5))
        ax.plot(range(500), X_test[:500])
        plt.plot(range(500), Y_test[:500], c="orange")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', labelsize=20)
        plt.show()

    return sampling_rate, X_train, X_test, Y_train, Y_test


def load_lorenz_dataset(step_ahead=5, visualize=True):
    from reservoirpy.datasets import lorenz
    train_steps = 15000
    test_steps = 5000
    timesteps = train_steps + test_steps
    dt = 0.03
    lorenz_inputs = lorenz(timesteps + step_ahead, rho=28.0, sigma=10.0, beta=2.6666666666666665, x0=[1.0, 1.0, 1.0],
                           h=dt, seed=None)
    # Compute the equivalent sampling rate
    sampling_rate = 1 / dt
    X_train = lorenz_inputs[:15000]
    X_test = lorenz_inputs[15000:20000]
    Y_train = lorenz_inputs[step_ahead:15000 + step_ahead]
    Y_test = lorenz_inputs[15000 + step_ahead:timesteps + step_ahead]

    if visualize:
        fig, ax = plt.subplots(figsize=(16, 8))
        colors_x = ['lightblue', 'blue', 'darkblue']  # Different nuances of blue
        colors_y = ['peachpuff', 'orange', 'pink']  # Different nuances of orange
        for i in range(3):
            ax.plot(range(500), X_test[:500, i], color=colors_x[i])
        for i in range(3):
            ax.plot(range(500), Y_test[:500, i], color=colors_y[i])
        # Customize the plot
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', labelsize=20)
        plt.legend()
        plt.show()

    return sampling_rate, X_train, X_test, Y_train, Y_test


def load_sunspot_dataset(step_ahead=5, visualize=True):
    sunspots = pd.read_csv('datasets/Sunspot/SN_ms_tot_V2.0.csv', sep=';', header=None)
    # Define the time step of your Mackey-Glass system

    sunspots = sunspots.values[:, 3].reshape(-1, 1)

    dt = 10000
    # Compute the equivalent sampling rate
    sampling_rate = 1 / dt

    test_size = sunspots.shape[0] // 10

    X_train, X_test, Y_train, Y_test = to_forecasting(sunspots, forecast=1, axis=0, test_size=test_size)

    if visualize:
        fig, ax = plt.subplots(figsize=(16, 5))
        ax.plot(range(test_size), X_test[:test_size])
        plt.plot(range(test_size), Y_test[:test_size], c="orange")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', labelsize=20)
        plt.show()

    return sampling_rate, X_train, X_test, Y_train, Y_test


def load_dataset_prediction(name, step_ahead=5, visualize=True):
    if name == "MackeyGlass":
        sampling_rate, X_train, X_test, Y_train, Y_test = load_mackey_glass_dataset(step_ahead, visualize)
        is_multivariate = False
        return is_multivariate, sampling_rate, X_train, X_test, Y_train, Y_test
    elif name == "Lorenz":
        sampling_rate, X_train, X_test, Y_train, Y_test = load_lorenz_dataset(step_ahead, visualize)
        is_multivariate = True
        return is_multivariate, sampling_rate, X_train, X_test, Y_train, Y_test
    elif name == "Sunspot":
        sampling_rate, X_train, X_test, Y_train, Y_test = load_sunspot_dataset(step_ahead, visualize)
        is_multivariate = False
        return is_multivariate, sampling_rate, X_train, X_test, Y_train, Y_test
    else:
        ValueError("The dataset with name {} is not loadable".format(name))
