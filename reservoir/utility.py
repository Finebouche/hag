from seaborn import heatmap, color_palette
from matplotlib import pyplot as plt
import numpy as np

def show_matrice(W, ax = None):
    if ax is None:
        fig, ax = plt.figure(figsize=(9,7))
    heatmap(W.todense(), cmap=color_palette("vlag", as_cmap=True), ax = ax)
    plt.show()

def show_ei_matrix(Wee, Wei, Wie):
    plt.figure(figsize=(7,7))
    n_e = Wei.shape[1]
    n_i = Wei.shape[0]
    
    ax0 = plt.subplot2grid((n_e+n_i, n_e+n_i), (0, 0), colspan=n_e, rowspan=n_e)
    ax0.imshow(Wee.A, cmap=color_palette("vlag", as_cmap=True))
    ax0.axis('off')

    if n_i > 0:
        ax1 = plt.subplot2grid((n_e+n_i, n_e+n_i), (0, n_e), colspan=n_i, rowspan=n_e)
        ax1.imshow(Wie.A, cmap=color_palette("vlag", as_cmap=True))
        ax1.axis('off')

        ax2 = plt.subplot2grid((n_e+n_i, n_e+n_i), (n_e, 0), colspan=n_e, rowspan=n_i)
        ax2.imshow(Wei.toarray(), cmap=color_palette("vlag", as_cmap=True))
        ax2.axis('off')

        ax3 = plt.subplot2grid((n_e+n_i, n_e+n_i), (n_e, n_e), colspan=n_i, rowspan=n_i)
        ax3.imshow(np.zeros((n_i, n_i)), cmap=color_palette("vlag", as_cmap=True))
        ax3.axis('off')

    plt.show()