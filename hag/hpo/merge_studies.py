"""
Merge the databases of the studies optimized by separate jobs (one database per function, see
slurm/hpo_classification.sbatch: hag/hpo/corrected_studies/parts/<dataset>/<function>/) into the database of the dataset
(hag/hpo/corrected_studies/tpe_<dataset>_db.sqlite3). A study already in the database of the dataset is replaced by the
one of the part if the part has more completed trials (e.g. a study started locally and continued on PlaFRIM).

Usage: python -m hag.hpo.merge_studies <dataset>
"""
import sys

import optuna
from optuna.trial import TrialState

from hag.hpo.utility import study_db_dir, study_db_filename


def completed(study):
    return sum(t.state == TrialState.COMPLETE for t in study.trials)


def main(dataset):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    target = f"sqlite:///{study_db_dir() / study_db_filename('tpe', dataset)}"
    parts = sorted((study_db_dir() / "parts" / dataset).glob("*/*.sqlite3"))
    if not parts:
        raise FileNotFoundError(f"No database in {study_db_dir() / 'parts' / dataset}")
    existing = {s.study_name for s in optuna.get_all_study_summaries(target)} if (
        study_db_dir() / study_db_filename('tpe', dataset)).exists() else set()
    for part in parts:
        url = f"sqlite:///{part}"
        for summary in optuna.get_all_study_summaries(url):
            name = summary.study_name
            source = optuna.load_study(study_name=name, storage=url)
            if name in existing:
                current = optuna.load_study(study_name=name, storage=target)
                if completed(current) >= completed(source):
                    print(f"{name}: kept ({completed(current)} completed trials >= {completed(source)} in {part.parent.name})")
                    continue
                optuna.delete_study(study_name=name, storage=target)
            optuna.copy_study(from_study_name=name, from_storage=url, to_storage=target)
            print(f"{name}: {completed(source)} completed trials copied from {part.parent.name}")


if __name__ == "__main__":
    main(sys.argv[1])
