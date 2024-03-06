import numpy as np
import matplotlib.pyplot as plt


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


def plot_results(y_pred, y_test, start=0, end=500):
    sample = slice(start, end)
    x_corrd = np.arange(start, end)
    print(sample)
    fig = plt.figure(figsize=(15, 7))
    plt.subplot(211)
    plt.plot(x_corrd, y_pred[sample], lw=3, label="ESN prediction")
    plt.plot(x_corrd, y_test[sample], linestyle="--", lw=2, label="True value")
    plt.plot(x_corrd, np.abs(y_test[sample] - y_pred[sample]), label="Absolute deviation")

    #plot legend outside the graph
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
