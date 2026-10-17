"""
Test scores of the best hyperparameters of the HPO studies (hag/hpo/corrected_studies by default), as in the "Test
scores" cell of 2.test_analysis.ipynb: for each dataset and function, nb_trials reservoirs built with the best trial of
its study, trained on the train set and scored on the test set. The recurrent networks (RNN_FUNCTIONS: LSTM, GRU, RNN,
studies of hag.hpo.hpo_rnn) are trained nb_trials times with the best trial of their study
(hag.performances.evaluation_rnn), seeds 0 to nb_trials - 1.

One row per dataset and function (mean and standard deviation of the nb_trials scores), added to
outputs/hag/test_results/test_results_<classification|prediction>_corrected.csv.

HAG_TEST_RESULTS (environment variable): another file to add the rows to, e.g. one file per job when several jobs run at
the same time (slurm/submit_test_evaluation.sh), concatenated afterwards.

Usage: python -m hag.performances.run_test_evaluation [datasets] [functions] [nb_trials]
  e.g. python -m hag.performances.run_test_evaluation JapaneseVowels,CatsDogs mean_hag,var_hag 8
       python -m hag.performances.run_test_evaluation FSDD lstm_last,gru
"""
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
# (Triton, loaded by torch._dynamo when the LSTM and GRU are trained, crashes if TensorFlow is loaded before it: it
# reads the WAV files of FSDD and SPEECHCOMMANDS)
import torch._dynamo  # noqa: F401

from hag.datasets.pipeline import prepare_data
from hag.hpo.utility import retrieve_best_model, study_db_dir
from hag.performances.evaluation_esn import evaluate_dataset_on_test
from hag.performances.evaluation_rnn import evaluate_dataset_on_test_rnn

DATASETS = ["JapaneseVowels", "CatsDogs"]
FUNCTIONS = ["random_ee", "random_ei", "ip_correct", "anti-oja", "ip-anti-oja", "mean_hag", "var_hag"]
RNN_FUNCTIONS = ("lstm_last", "gru", "rnn")
SPECTRAL_DATASETS = {"CatsDogs", "JapaneseVowels", "FSDD", "SpokenArabicDigits", "SPEECHCOMMANDS", "Canary"}
SEED = 0
TEST_RESULTS = Path(__file__).resolve().parents[2] / "outputs" / "hag" / "test_results"


def main():
    datasets = sys.argv[1].split(",") if len(sys.argv) > 1 else DATASETS
    functions = sys.argv[2].split(",") if len(sys.argv) > 2 else FUNCTIONS
    nb_trials = int(sys.argv[3]) if len(sys.argv) > 3 else 8

    for dataset_name in datasets:
        np.random.seed(SEED)  # pretraining instances drawn by prepare_data
        spectral_representation = "mfcc" if dataset_name in SPECTRAL_DATASETS else "stft"
        (pretrain_data, train_data, test_data, Y_train, Y_test, is_multivariate,
         is_instances_classification) = prepare_data(dataset_name, spectral_representation)

        rows = []
        for function_name in functions:
            print(f"\n=== {dataset_name} / {function_name}", flush=True)
            study = retrieve_best_model(function_name, dataset_name, is_multivariate, variate_type="multi",
                                        data_type="normal", prefix="tpe", db_dir=study_db_dir(dataset_name))
            print(f"study {study.study_name}: {len(study.trials)} trials, best validation score {study.best_value:.4f}")
            if function_name in RNN_FUNCTIONS:
                scores = evaluate_dataset_on_test_rnn(study, dataset_name, function_name, pretrain_data, train_data,
                                                      test_data, Y_train, Y_test, is_instances_classification,
                                                      nb_trials=nb_trials)
            else:
                scores = evaluate_dataset_on_test(study, dataset_name, function_name, pretrain_data, train_data,
                                                  test_data, Y_train, Y_test, is_instances_classification,
                                                  nb_trials=nb_trials, seed=SEED)
            mean, std = np.mean(scores), np.std(scores)
            if is_instances_classification:
                average, deviation = f"{round(mean * 100, 5)} %", f"± {round(std * 100, 5)} %"
            else:
                average, deviation = f"{round(mean, 5)}", f"± {round(std, 5)}"
            print(f"{dataset_name} / {function_name}: {average} {deviation}", flush=True)
            rows.append({"Dataset": dataset_name, "Function": function_name, "Average Score": average,
                         "Standard Deviation": deviation, "Validation Score": study.best_value,
                         "Nb Trials": nb_trials, "Date": datetime.now().strftime("%Y-%m-%d")})

        kind = "classification" if is_instances_classification else "prediction"
        file_name = Path(os.environ.get("HAG_TEST_RESULTS", TEST_RESULTS / f"test_results_{kind}_corrected.csv"))
        file_name.parent.mkdir(parents=True, exist_ok=True)
        results = pd.DataFrame(rows)
        if file_name.exists():
            results = pd.concat([results, pd.read_csv(file_name)], ignore_index=True)
        results.to_csv(file_name, index=False)
        print(pd.DataFrame(rows).drop(columns=["Date"]).to_string(index=False))
        print(f"Results added to {file_name}")


if __name__ == "__main__":
    main()
