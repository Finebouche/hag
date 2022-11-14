import numpy as np
import matplotlib.pyplot as plt


def plot_mackey_glass(X, sample, tau):
    fig = plt.figure(figsize=(13, 5))
    N = sample

    ax = plt.subplot((121))
    t = np.linspace(0, N, N)
    for i in range(N - 1):
        ax.plot(t[i:i + 2], X[i:i + 2], color=plt.cm.magma(255 * i // N), lw=1.0)

    plt.title(f"Timeseries - {N} timesteps")
    plt.xlabel("$t$")
    plt.ylabel("$P(t)$")

    ax2 = plt.subplot((122))
    ax2.margins(0.05)
    for i in range(N - 1):
        ax2.plot(X[i:i + 2], X[i + tau:i + tau + 2], color=plt.cm.magma(255 * i // N), lw=1.0)

    plt.title(f"Phase diagram: $P(t) = f(P(t-\\tau))$")
    plt.xlabel("$P(t-\\tau)$")
    plt.ylabel("$P(t)$")

    plt.tight_layout()
    plt.show()


def plot_train_test(X_train, y_train, X_test, y_test):
    sample = 500
    test_len = X_test.shape[0]
    fig = plt.figure(figsize=(15, 5))
    plt.plot(np.arange(0, 500), X_train[-sample:], label="Training data")
    plt.plot(np.arange(0, 500), y_train[-sample:], label="Training objective")
    plt.plot(np.arange(500, 500 + test_len), X_test, label="Testing data")
    plt.plot(np.arange(500, 500 + test_len), y_test, label="Testing objective")
    plt.legend()
    plt.show()


def plot_readout(readout, that="Wout"):
    if hasattr(readout, 'Wout'):
        Wout = readout.Wout
    else:
        Wout = readout
    if hasattr(readout, 'bias'):
        bias = readout.bias
        Wout = np.r_[bias, Wout]

    fig = plt.figure(figsize=(15, 5))

    ax = fig.add_subplot(111)
    ax.grid(axis="y")
    ax.set_ylabel('Coefs. de {string}'.format(string=that))
    ax.set_xlabel("Neurones du reservoir")
    ax.bar(np.arange(Wout.size), Wout.ravel()[::-1])

    plt.show()


def plot_results(y_pred, y_test, sample=500):
    fig = plt.figure(figsize=(15, 7))
    plt.subplot(211)
    plt.plot(np.arange(sample), y_pred[:sample], lw=3, label="ESN prediction")
    plt.plot(np.arange(sample), y_test[:sample], linestyle="--", lw=2, label="True value")
    plt.plot(np.abs(y_test[:sample] - y_pred[:sample]), label="Absolute deviation")

    plt.legend()
    plt.show()
