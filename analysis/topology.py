import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from io import BytesIO

import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def motif_distribution(W):
    # N is a tuple (row, col), since W might not be a square matrix
    N = W.shape[0]

    # Define binary connectivity matrix
    M = np.where(W > 0, 1, 0)
    M = M - np.diag(np.diag(M))

    M = M.T
    S = np.where(M == M.T, M, 0)
    A = M - S
    Mtilde = np.ones((N, N)) - M
    Stilde = np.where(Mtilde == Mtilde.T, Mtilde, 0)
    Stilde = Stilde - np.diag(np.diag(Stilde))

    # Motifs calculation
    Mot1 = np.sum(np.sum((Stilde @ Stilde) * Stilde)) / 6  # subgraph
    Mot2 = np.sum(np.sum((Stilde @ Stilde) * A))  # subgraph
    Mot3 = np.sum(np.sum((Stilde @ Stilde) * S)) / 2  # subgraph
    Mot4 = (np.sum(np.sum((A.T @ A) * Mtilde * Mtilde.T)) - np.trace(A.T @ A)) / 2  # subgraph 6
    Mot5 = (np.sum(np.sum((A @ A.T) * Mtilde * Mtilde.T)) - np.trace(A @ A.T)) / 2  # subgraph 36
    Mot6 = np.sum(np.sum((A @ A) * Mtilde * Mtilde.T))  # subgraph 12
    Mot7 = np.sum(np.sum((S @ A.T) * Mtilde * Mtilde.T))  # subgraph 74
    Mot8 = np.sum(np.sum((S @ A) * Mtilde * Mtilde.T))  # subgraph 14
    Mot9 = (np.sum(np.sum((S @ S) * Mtilde * Mtilde.T)) - np.trace(S @ S)) / 2  # subgraph 78
    Mot10 = np.sum(np.sum((A @ A) * A))  # subgraph 38
    Mot11 = np.sum(np.sum((A.T @ A.T) * A)) / 3  # subgraph 98
    Mot12 = np.sum(np.sum((A.T @ A) * S)) / 2  # subgraph 108
    Mot13 = np.sum(np.sum((A @ A) * S))  # subgraph 102
    Mot14 = np.sum(np.sum((A @ A.T) * S)) / 2  # subgraph 46
    Mot15 = np.sum(np.sum((S @ S) * A))  # subgraph 110
    Mot16 = np.sum(np.sum((S @ S) * S)) / 6  # subgraph 238

    # Sanity check should be (n choose 3) = N * (N-1) * (N-2) / 6
    assert Mot1 + Mot2 + Mot3 + Mot4 + Mot5 + Mot6 + Mot7 + Mot8 + Mot9 + Mot10 + Mot11 + Mot12 + Mot13 + Mot14 + Mot15 + Mot16 == N * (N-1) * (N-2) / 6

    return [Mot1, Mot2, Mot3, Mot4, Mot5, Mot6, Mot7, Mot8, Mot9, Mot10, Mot11, Mot12, Mot13, Mot14, Mot15, Mot16]


# List of motifs (as edges)
MOTIFS_EDGES = [
    [],
    [(0, 1)],
    [(0, 1), (1, 0)],
    [(0, 1), (0, 2)],  # subgraph 6
    [(1, 0), (2, 0)],  # subgraph 36
    [(1, 0), (0, 2)],  # subgraph 12
    [(1, 0), (0, 1), (2, 0)],  # subgraph 74
    [(1, 0), (0, 1), (0, 2)],  # subgraph 14
    [(1, 0), (0, 1), (0, 2), (2, 0)],  # subgraph 78
    [(0, 1), (0, 2), (1, 2)],  # subgraph 38
    [(1, 0), (1, 2), (2, 0)],  # subgraph 98
    [(0, 1), (0, 2), (1, 2), (2, 1)],  # subgraph 108
    [(0, 1), (2, 0), (1, 2), (2, 1)],  # subgraph 102
    [(1, 0), (2, 0), (1, 2), (2, 1)],  # subgraph 46
    [(0, 1), (2, 0), (0, 2), (1, 2), (2, 1)],  # subgraph 110
    [(0, 1), (1, 0), (2, 0), (0, 2), (1, 2), (2, 1)],  # subgraph 238
]


def draw_motifs_distribution(motifs_count):
    # Define a function to draw a motif and return the image
    def get_motif_image(edges):
        fig, ax = plt.subplots(figsize=(1, 1))  # Smaller figsize
        G = nx.DiGraph()
        G.add_edges_from(edges)
        pos = nx.shell_layout(G)
        nx.draw(G, pos, node_size=50, arrowsize=20, node_color='black', ax=ax)
        plt.axis('off')
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        return mpimg.imread(buf)

    def offset_image(coord, img, ax):
        imagebox = OffsetImage(img, zoom=0.4)
        imagebox.image.axes = ax
        ab = AnnotationBbox(imagebox, (0, 0), xybox=(39 + coord * 32, -20), frameon=False, xycoords='axes points',
                            boxcoords="axes points")
        ax.add_artist(ab)

    # Plot the histogram
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.rainbow(np.linspace(0, 1, len(motifs_count)))
    bars = ax.bar(range(len(motifs_count)), motifs_count, color=colors, alpha=0.7, log=True)
    ax.get_xaxis().set_ticklabels([])

    # Plot motifs above each bar as labels
    for i, (rect, edges) in enumerate(zip(bars, MOTIFS_EDGES)):
        img = get_motif_image(edges)
        offset_image(i, img, ax)

    plt.show()
