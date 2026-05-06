from hag.datasets.load_classification import load_dataset_classification
from hag.datasets.load_forecasting import load_dataset_forecasting

FORECASTING_DATASETS = {"Lorenz", "MackeyGlass", "Sunspot_daily", "NARMA", "Henon"}
CLASSIFICATION_DATASETS = {"CatsDogs", "FSDD", "JapaneseVowels", "SPEECHCOMMANDS", "SpokenArabicDigits"}


def load_data(dataset_name, step_ahead=5, visualize=False):
    if dataset_name in FORECASTING_DATASETS:
        is_instances_classification = False
        use_spectral_representation = False
        is_multivariate, sampling_rate, X_train_raw, X_test_raw, Y_train_raw, Y_test = load_dataset_forecasting(dataset_name, step_ahead, visualize=visualize)
        groups = None
    elif dataset_name in CLASSIFICATION_DATASETS:
        is_instances_classification = True
        use_spectral_representation, is_multivariate, sampling_rate, X_train_raw, X_test_raw, Y_train_raw, Y_test, groups = load_dataset_classification(dataset_name, visualize=visualize)
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

    if use_spectral_representation and not is_multivariate:
        raise ValueError("Cannot use spectral representation if it's not multivariate.")

    return is_instances_classification, is_multivariate, sampling_rate, X_train_raw, X_test_raw, Y_train_raw, Y_test, use_spectral_representation, groups
