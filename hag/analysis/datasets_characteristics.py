from hag.datasets.load_data import load_data as load_dataset
from hag.analysis.commons import load_data as load_processed_data
import math
import pandas as pd


print("###############")
print("#")
print("# DATASET STATISTICS")
print("#")
print("##############")


for dataset_name in ["CatsDogs", "FSDD", "JapaneseVowels", "SPEECHCOMMANDS", "SpokenArabicDigits"]:
    (
        is_instances_classification,
        is_multivariate,
        sampling_rate,
        X_train_raw,
        X_test_raw,
        Y_train_raw,
        Y_test,
        use_spectral_representation,
        groups,
    ) = load_dataset(dataset_name, 5, visualize=False)

    lengths = [len(x) for x in X_train_raw]
    avg_length = sum(lengths) / len(lengths)

    print(f"Dataset: {dataset_name}  -->  Average length of X_train_raw: {avg_length:.2f}")


datasets = [
    "MackeyGlass", "Lorenz", "Sunspot_daily", "Henon", "NARMA", "CatsDogs",
    "FSDD", "JapaneseVowels", "SPEECHCOMMANDS", "SpokenArabicDigits"
]


def dataset_characteristics(dataset, data_type="normal", noise_std=0.001):
    print(f"Dataset: {dataset}")
    spectral_representation = (
        "mfcc"
        if dataset in ["CatsDogs", "FSDD", "JapaneseVowels", "SPEECHCOMMANDS", "SpokenArabicDigits"]
        else "stft"
    )

    (
        pretrain_data,
        train_data,
        test_data,
        Y_train,
        Y_test,
        is_multivariate,
        is_instances_classification,
    ) = load_processed_data(
        dataset,
        spectral_representation,
        data_type,
        noise_std,
        visualize=False,
    )

    if dataset == "Sunspot_daily":
        start_step = 30
        end_step = 500
    else:
        start_step = 500
        end_step = 1500

    common_index = 1
    if is_instances_classification:
        common_size = pretrain_data[0].shape[common_index]
    else:
        common_size = pretrain_data.shape[common_index]

    network_size = 500
    K = math.ceil(network_size / common_size)
    n = common_size * K

    if is_instances_classification:
        total_pretrain_length = sum(instance.shape[0] for instance in pretrain_data)
        total_train_length = sum(instance.shape[0] for instance in train_data)
        total_test_length = sum(instance.shape[0] for instance in test_data)
        original_dimension = train_data[0].shape[1]
        number_class = Y_test.shape[1]

        sample_sizes = [instance.shape[0] for instance in train_data] + [instance.shape[0] for instance in test_data]
        min_sample_size = min(sample_sizes)
        max_sample_size = max(sample_sizes)
        avg_sample_size = sum(sample_sizes) / len(sample_sizes)
    else:
        total_pretrain_length = pretrain_data.shape[0]
        total_train_length = train_data.shape[0]
        total_test_length = test_data.shape[0]
        original_dimension = train_data.shape[1]
        number_class = 0
        min_sample_size = None
        max_sample_size = None
        avg_sample_size = None

    return {
        "dataset_name": dataset,
        "tot_pretrain_length": total_pretrain_length,
        "tot_train_length": total_train_length,
        "tot_test_length": total_test_length,
        "origin_dim": original_dimension,
        "nb_dupication": K,
        "reservoir_size": n,
        "final_dim": common_size,
        "nb_class": number_class,
        "min_sample_size": min_sample_size,
        "max_sample_size": max_sample_size,
        "avg_sample_size": avg_sample_size,
    }


results = []
for dataset in datasets:
    characteristics = dataset_characteristics(dataset)
    results.append(characteristics)

df = pd.DataFrame(results)
csv_filename = "../../outputs/dataset_characteristics.csv"
df.to_csv(csv_filename, index=False)

print(f"Dataset characteristics saved to {csv_filename}")
