import sys

if sys.version_info < (3, 8):
    from typing_extensions import Literal, Optional, Union
else:
    from typing import Literal, Optional, Union

from copy import deepcopy

import numpy as np
from scipy import linalg
from scipy.sparse import issparse
from scipy.sparse.linalg import eigs

from reservoirpy.model import Model
from reservoirpy.type import Weights
from reservoirpy.utils.random import rand_generator


def memory_capacity(
    model: Model,
    k_max: int,
    as_list: bool = False,
    series: Optional[np.ndarray] = None,
    test_size: Union[int, float] = 0.2,
    seed: Optional[Union[int, np.random.RandomState, np.random.Generator]] = None,
):
    from numpy.lib.stride_tricks import sliding_window_view

    # ----------------------------------
    # 1) Handle default or user-provided data
    # ----------------------------------
    if series is None:
        # If user did not provide a series, generate one
        rng = rand_generator(seed)
        # By default, generate a single-feature timeseries (n_features=1)
        # of length 10*k_max. For multi-dim, adjust shape as needed.
        series = rng.uniform(low=-0.8, high=0.8, size=(10 * k_max, 1))

    timesteps, n_features = series.shape

    # ----------------------------------
    # 2) Determine train/test split size
    # ----------------------------------
    # We'll have (timesteps - k_max) valid samples after sliding windows,
    # because we need k_max past steps for each sample.
    if isinstance(test_size, float) and 0 <= test_size < 1:
        test_len = round((timesteps - k_max) * test_size)
    elif isinstance(test_size, int):
        test_len = test_size
    else:
        raise ValueError(
            "Invalid test_size argument. test_size must be an integer or "
            f"a float in [0,1). Got {test_size}."
        )

    # ----------------------------------
    # 3) Build sliding window dataset
    # ----------------------------------
    # sliding_window_view(..., window_shape=k_max+1, axis=0)
    # will create a 3D array of shape:
    # (timesteps-k_max, k_max+1, n_features)
    #   - dataset[i, 0, :]   => input at time i
    #   - dataset[i, 1, :]   => input at time i+1
    #   ...
    #   - dataset[i, k_max, :] => input at time i+k_max
    # We interpret dataset[i, 0, :] as x(t), and the next k_max steps
    # as [x(t+1), x(t+2), ..., x(t+k_max)] for target.
    dataset = sliding_window_view(series, window_shape=(k_max + 1), axis=0)
    # dataset shape = (timesteps - k_max, k_max+1, n_features)

    # X = current inputs = dataset[:, 0, :], shape (timesteps - k_max, n_features)
    X = dataset[:, 0, :]

    # Y = the next k_max steps of the inputs, shape (timesteps - k_max, k_max, n_features)
    Y = dataset[:, 1:, :]

    # Flatten the target so each (lag, feature) becomes its own output dimension:
    # final shape = (timesteps - k_max, k_max*n_features)
    #   - e.g. the order is lag_1_feature_1, lag_1_feature_2, ..., lag_2_feature_1, ...
    Y = Y.reshape(Y.shape[0], k_max * n_features)

    # ----------------------------------
    # 4) Split into train/test sets
    # ----------------------------------
    train_end = (timesteps - k_max) - test_len  # index where test starts
    X_train, X_test = X[:train_end], X[train_end:]
    Y_train, Y_test = Y[:train_end], Y[train_end:]

    # ----------------------------------
    # 5) Fit a (cloned) model on the training set
    # ----------------------------------
    model_clone = deepcopy(model)
    # We warm up for k_max steps (optional). If your model requires a different
    # warmup, adapt it. If your model can handle zero warmup, set warmup=0.
    # Or if your model typically has separate warmup, you can remove "warmup=k_max".
    model_clone.fit(X_train, Y_train, warmup=k_max)

    # ----------------------------------
    # 6) Predict on the test set
    # ----------------------------------
    Y_pred = model_clone.run(X_test)
    # Y_pred has shape (test_len, k_max*n_features)

    # Reshape back to (test_len, k_max, n_features) to compute correlation
    Y_pred = Y_pred.reshape(test_len, k_max, n_features)
    Y_test = Y_test.reshape(test_len, k_max, n_features)

    # ----------------------------------
    # 7) Compute memory capacity for each (lag, feature) = MC_{k,d}
    # ----------------------------------
    # MC_{k,d} = correlation^2( actual[t - k, d], predicted[t, k, d] )
    # We'll do: correlation(Y_pred[:, k, d], Y_test[:, k, d]) for each k, d.
    capacities = np.zeros((k_max, n_features))

    for k in range(k_max):
        for d in range(n_features):
            # Extract the predicted vs true for the given lag k and dimension d
            pred = Y_pred[:, k, d]
            true = Y_test[:, k, d]

            # If there's too little variance, correlation can be NaN or fail.
            # We'll guard with a try/except or check for constant signals:
            if np.std(true) < 1e-12 or np.std(pred) < 1e-12:
                corr = 0.0
            else:
                corr = np.corrcoef(pred, true, rowvar=False)[0, 1]

            capacities[k, d] = corr**2

    # ----------------------------------
    # 8) Return result
    # ----------------------------------
    # - If as_list=False, sum over all lags and all features => scalar
    # - If as_list=True, return 2D array of shape (k_max, n_features)
    if as_list:
        return capacities
    else:
        return np.sum(capacities)