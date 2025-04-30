from seaborn import heatmap, color_palette
from matplotlib import pyplot as plt
import numpy as np

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


def show_matrice(W, ax = None, palette = "vlag"):
    if ax is None:
        fig, ax = plt.figure(figsize=(9,7))
    heatmap(W.todense(), cmap=color_palette(palette, as_cmap=True), ax = ax)
    plt.show()

def show_ei_matrix(Wee, Wei, Wi, palette = "vlag"):
    plt.figure(figsize=(7,7))
    n_e = Wei.shape[1]
    n_i = Wei.shape[0]
    
    ax0 = plt.subplot2grid((n_e+n_i, n_e+n_i), (0, 0), colspan=n_e, rowspan=n_e)
    ax0.imshow(Wee, cmap=color_palette(palette, as_cmap=True))
    ax0.axis('off')

    if n_i > 0:
        ax1 = plt.subplot2grid((n_e+n_i, n_e+n_i), (0, n_e), colspan=n_i, rowspan=n_e)
        ax1.imshow(Wie, cmap=color_palette(palette, as_cmap=True))
        ax1.axis('off')

        ax2 = plt.subplot2grid((n_e+n_i, n_e+n_i), (n_e, 0), colspan=n_e, rowspan=n_i)
        ax2.imshow(Wei.toarray(), cmap=color_palette(palette, as_cmap=True))
        ax2.axis('off')

        ax3 = plt.subplot2grid((n_e+n_i, n_e+n_i), (n_e, n_e), colspan=n_i, rowspan=n_i)
        ax3.imshow(np.zeros((n_i, n_i)), cmap=color_palette(palette, as_cmap=True))
        ax3.axis('off')

    plt.show()