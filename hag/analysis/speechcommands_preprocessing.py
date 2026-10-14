"""
Why does var_hag lose 10 points on SPEECHCOMMANDS with the MFCC fix (30.8 -> 20.1 % test accuracy)? Test accuracy of
var_hag with the best hyperparameters of the earlier study (hag/hpo/legacy_studies, earlier preprocessing), the
dataset prepared with the earlier preprocessing (HAG_MFCC_LEGACY: 20 MFCC, 128 mel bands, sampling rate assumed at
22050 Hz) or with the corrected one (13 MFCC, 32 mel bands, real sampling rate). Same hyperparameters, seeds and
pretraining draw: the difference comes from the features only.

Results: outputs/hag/analysis_results/speechcommands_preprocessing_<preprocessing>.csv (one row).
Usage: python -m hag.analysis.speechcommands_preprocessing <legacy|corrected> [nb_trials]
"""
import os
import sys

preprocessing = sys.argv[1]
if preprocessing not in ("legacy", "corrected"):
    raise ValueError("preprocessing: legacy or corrected")
if preprocessing == "legacy":
    os.environ["HAG_MFCC_LEGACY"] = "1"  # read by hag.datasets.spectral_decomposition

import numpy as np
import pandas as pd

from hag.analysis.utils import ANALYSIS_RESULTS
from hag.datasets.pipeline import prepare_data
from hag.hpo.utility import LEGACY_STUDIES, retrieve_best_model
from hag.performances.evaluation_esn import evaluate_dataset_on_test

DATASET, FUNCTION, SEED = "SPEECHCOMMANDS", "var_hag", 0
nb_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 8

np.random.seed(SEED)  # pretraining instances drawn by prepare_data
pretrain, train, test, Y_train, Y_test, is_multivariate, is_classification = prepare_data(DATASET, "mfcc")
print(f"{preprocessing} preprocessing: {train[0].shape[1]} features per frame", flush=True)
study = retrieve_best_model(FUNCTION, DATASET, is_multivariate, db_dir=LEGACY_STUDIES, verbosity=0)
scores = evaluate_dataset_on_test(study, DATASET, FUNCTION, pretrain, train, test, Y_train, Y_test, is_classification,
                                  nb_trials=nb_trials, seed=SEED)
row = dict(dataset=DATASET, function=FUNCTION, hyperparameters="legacy study", preprocessing=preprocessing,
           features=train[0].shape[1], test_mean_percent=100 * np.mean(scores), test_std_percent=100 * np.std(scores),
           validation_score=study.best_value, nb_trials=nb_trials)
print(row, flush=True)
ANALYSIS_RESULTS.mkdir(parents=True, exist_ok=True)
pd.DataFrame([row]).to_csv(ANALYSIS_RESULTS / f"speechcommands_preprocessing_{preprocessing}.csv", index=False)
