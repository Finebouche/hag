import numpy as np
import matplotlib.pyplot as plt

def plot_prediction_vs_actual(y_pred, y_test, start=0, end=500):
    sample = slice(start, end)
    x_corrd = np.arange(start, end)
    fig = plt.figure(figsize=(15, 7))
    plt.subplot(211)
    plt.plot(x_corrd, y_pred[sample], lw=3, label="ESN prediction")
    plt.plot(x_corrd, y_test[sample], linestyle="--", lw=2, label="True value")
    plt.plot(x_corrd, np.abs(y_test[sample] - y_pred[sample]), label="Absolute deviation")

    #plot legend outside the graph
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
