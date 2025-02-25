from datasets.load_classification import load_dataset_classification
from datasets.load_forecasting import load_dataset_forecasting

def load_data(dataset_name, step_ahead=5, visualize=False):
    if dataset_name in ["Lorenz", "MackeyGlass", "Sunspot", "NARMA", "Henon"]:
        is_instances_classification = False
        is_multivariate, sampling_rate, X_train_raw, X_test_raw, Y_train_raw, Y_test = load_dataset_forecasting(dataset_name, step_ahead, visualize=visualize)
        use_spectral_representation = False
        spectral_representation = None # Can be None, "stft" or "mfcc"
        groups = None
    elif dataset_name in ["CatsDogs", "FSDD", "JapaneseVowels", "SPEECHCOMMANDS", "SpokenArabicDigits"]:
        is_instances_classification = True
        spectral_representation = "mfcc"  # Can be None, "stft" or "mfcc"

        use_spectral_representation, is_multivariate, sampling_rate, X_train_raw, X_test_raw, Y_train_raw, Y_test, groups = load_dataset_classification(dataset_name, visualize=visualize)
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

    return is_instances_classification, is_multivariate, sampling_rate, X_train_raw, X_test_raw, Y_train_raw, Y_test, use_spectral_representation, spectral_representation, groups