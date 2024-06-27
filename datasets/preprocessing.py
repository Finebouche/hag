import numpy as np
import matplotlib.pyplot as plt


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
        X_train = [scaler.transform(time_series) for time_series in X_train]
        X_val = [scaler.transform(time_series) for time_series in X_val]
        if X_test is not None:
            X_test = [scaler.transform(time_series) for time_series in X_test]
    else:
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        if X_test is not None:
            X_test = scaler.transform(X_test)
    return X_train, X_val, X_test


def add_noise(data, noise_std):
    """Adds Gaussian noise to each instance in the data."""
    return np.random.normal(0, noise_std, data.shape) + data


def duplicate_data(data_list, K):
    """Duplicates each instance in the data K times along the specified axis."""
    return [np.repeat(instance, K, axis=1) for instance in data_list]


def plot_data_distribution(y_train_encoded, y_test_encoded, val=False):
    # Sum across rows to get the count of each class (each column represents a class)
    train_counts = np.sum(y_train_encoded, axis=0)
    test_counts = np.sum(y_test_encoded, axis=0)

    # Get the number of classes (assuming all classes are represented in the training set)
    classes = np.arange(len(train_counts))

    plt.figure(figsize=(12, 6))

    # Plot histogram for Y_train
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
    color_train = "teal" if val else "blue"
    plt.bar(classes, train_counts, color=color_train, alpha=0.7)
    plt.title('Distribution of Y_train')
    plt.xlabel('Classes')
    plt.ylabel('Frequency')

    # Plot histogram for Y_test
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
    color_test = "orange" if val else "red"
    plt.bar(classes, test_counts, color=color_test, alpha=0.7)
    if val:
        plt.title('Distribution of Y_val')
    else:
        plt.title('Distribution of Y_test')
    plt.xlabel('Classes')
    plt.ylabel('Frequency')

    # Display the plots
    plt.tight_layout()
