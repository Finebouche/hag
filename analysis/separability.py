import numpy as np
from scipy.spatial.distance import pdist
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)


def inter_intra_class_distance(X, y):
    # 1) sanity check
    assert X.shape[0] == y.shape[0], \
        f"X has {X.shape[0]} samples but y has length {y.shape[0]}"

    classes, counts = np.unique(y, return_counts=True)

    # 2) compute centroids
    centroids = {
        cls: X[y == cls].mean(axis=0)
        for cls in classes
    }

    # 3) inter-class: pairwise distances between centroids
    inter_vals = []
    for i, cls1 in enumerate(classes):
        for cls2 in classes[i + 1:]:
            c1, c2 = centroids[cls1], centroids[cls2]
            inter_vals.append(np.linalg.norm(c1 - c2))
    avg_inter = np.mean(inter_vals)

    # 4) intra-class: average pairwise distance within each class
    intra_vals = []
    for cls, cnt in zip(classes, counts):
        if cnt > 1:
            pts = X[y == cls]
            # pdist returns the flattened upper triangle
            dists = pdist(pts, metric='euclidean')
            intra_vals.append(np.mean(dists))
    avg_intra = np.mean(intra_vals)

    # 5) separability ratio
    separability_ratio = avg_inter / avg_intra

    return avg_inter, avg_intra, separability_ratio


def fisher_discriminant_ratio(X, y):
    """
    Fisher Discriminant Ratio:
      (between‐class variance) / (within‐class variance)
    """
    classes, counts = np.unique(y, return_counts=True)
    overall_mean = X.mean(axis=0)

    # Between‐class scatter (trace)
    sb = 0.0
    for cls, nk in zip(classes, counts):
        mu_k = X[y == cls].mean(axis=0)
        sb += nk * np.linalg.norm(mu_k - overall_mean) ** 2

    # Within‐class scatter (trace)
    sw = 0.0
    for cls in classes:
        pts = X[y == cls]
        sw += np.sum(np.linalg.norm(pts - pts.mean(axis=0), axis=1) ** 2)

    return sb / sw


def silhouette(X, y, **kwargs):
    try:
        return silhouette_score(X, y, **kwargs)
    except ValueError:
        return np.nan


def davies_bouldin(X, y, **kwargs):
    """ Davies‐Bouldin Index: lower is better clustering. """
    try:
        return davies_bouldin_score(X, y, **kwargs)
    except ValueError:
        return np.nan


def calinski_harabasz(X, y, **kwargs):
    """ Calinski‐Harabasz Index: higher is better clustering.  """
    try:
        return calinski_harabasz_score(X, y, **kwargs)
    except ValueError:
        return np.nan