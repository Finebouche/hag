"""
Conditions of the syllable classification (canary.syllables_classification.classification): canary.evaluate.conditions
of the syllable studies (canary.syllables_classification.hpo.study_name_of), for each SIZE.
"""
from canary.evaluate import conditions as task_conditions
from canary.syllables_classification.hpo import study_name_of

SIZES = [250, 500, 1000]


def conditions(size):
    """canary.evaluate.conditions of the studies of the task (canary.syllables_classification.hpo.study_name_of)."""
    return task_conditions(size, study_name_of)
