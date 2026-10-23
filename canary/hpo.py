"""
Shared parts of the hyperparameter optimizations of the canary tasks (canary.next_phrase_prediction.hpo,
canary.syllables_classification.hpo): folds of the cross-validation by song, number of trials, storage of the studies
(one SQLite database each in STUDIES_DIR: HAG_CANARY_STUDIES, default outputs/canary/studies).
"""
import os
from pathlib import Path

from optuna.trial import TrialState

from canary.common import OUTPUTS

STUDIES_DIR = Path(os.environ.get("HAG_CANARY_STUDIES", OUTPUTS / "studies"))
N_FOLDS = 3
FOLD_SEED = 0
CV_SEED = 1000  # matrices of fold k drawn with np.random.seed(CV_SEED + k), the same for all the trials
N_TRIALS = 400


def storage(study_name):
    STUDIES_DIR.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{STUDIES_DIR / study_name}.sqlite3"


def study_exists(study_name):
    """Whether the database of the study exists (opening a missing SQLite database creates it, empty)."""
    return (STUDIES_DIR / f"{study_name}.sqlite3").exists()


def stop_after_failures(max_failures=10):
    """Callback stopping the optimization after max_failures failed trials in a row (systematic error)."""
    def callback(study, trial):
        last = [t.state for t in study.get_trials(deepcopy=False)][-max_failures:]
        if len(last) == max_failures and all(state == TrialState.FAIL for state in last):
            study.stop()
    return callback
