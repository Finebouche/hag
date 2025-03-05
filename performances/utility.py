import re
import optuna


def camel_to_snake(name):
    str1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', str1).lower()

import optuna
import re

def camel_to_snake(name):
    str1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', str1).lower()

def retrieve_best_model(function_name, dataset_name, is_multivariate, variate_type="multi", data_type="normal", sampler_name="tpe"):
    if function_name not in ["desp", "hadsp", "random_ee", "ip_correct", "anti-oja", "random_ei", "ip-anti-oja"]:
        raise ValueError(f"Invalid function name: {function_name}")
    if variate_type not in ["multi", "uni"]:
        raise ValueError(f"Invalid variate type: {variate_type}")
    if data_type not in ["normal", "noisy"]:
        raise ValueError(f"Invalid data type: {data_type}")

    if variate_type == "uni" and is_multivariate:
        raise ValueError(f"Invalid variable type: {variate_type}")

    # Build the study name
    study_name = function_name + "_" + dataset_name + "_" + data_type + "_" + variate_type
    # Build the URL
    if sampler_name == "tpe":
        url = f"sqlite:///tpe_{camel_to_snake(dataset_name)}_db.sqlite3"
    elif sampler_name == "cmaes":
        url = f"sqlite:///cmaes_{camel_to_snake(dataset_name)}_db.sqlite3"
    else:
        raise ValueError(f"Unknown sampler_name: {sampler_name}")

    # Load the study
    study = optuna.load_study(study_name=study_name, storage=url)
    return study