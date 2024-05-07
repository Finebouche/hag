import numpy as np
from tqdm import tqdm


def flexible_indexing(data, indices):
    if isinstance(data, list):
        return [data[i] for i in indices]
    elif isinstance(data, np.ndarray):
        return data[indices]
    else:
        raise TypeError("Unsupported data type for indexing")


def scale_data(X_train, X_val, X_test, scaler, is_instances_classification):
    if is_instances_classification:
        scaler.fit(np.concatenate(X_train, axis=0))

        # Transform the training, validation and test data using the same scaler
        X_train = np.array([scaler.transform(time_series) for time_series in X_train], dtype=object)
        X_val = np.array([scaler.transform(time_series) for time_series in X_val], dtype=object)
        X_test = np.array([scaler.transform(time_series) for time_series in X_test], dtype=object)
    else:
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
    return X_train, X_val, X_test


def add_noise(data, noise_std):
    """Adds Gaussian noise to each instance in the data."""
    return np.random.normal(0, noise_std, data.shape) + data


def duplicate_data(data_list, K):
    """Duplicates each instance in the data K times along the specified axis."""
    return [np.repeat(instance, K, axis=1) for instance in data_list]
