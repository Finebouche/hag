import re
import optuna
from pathlib import Path

VALID_FUNCTION_NAMES = {
    "desp", "hadsp", "mean_hag", "var_hag", "random_ee", "random_ei", "diag_ee", "diag_ei",
    "ip_correct", "anti-oja_fast", "ip-anti-oja_fast",
    "lstm_last", "rnn", "rnn-mean_hag", "gru", "short-hag", "hsp"
}
VALID_PREFIXES = {
    "tpe", "new_tpe", "cmaes", "lstm_tpe",
    "rdn-proj_tpe_mfcc", "rdn-proj_tpe_custom", "rdn-proj_tpe_none", "rdn-proj_tpe_stft",
    "mod-proj_tpe_mfcc", "mod-proj_tpe_custom", "mod-proj_tpe_none", "mod-proj_tpe_stft",
}


def camel_to_snake(name):
    str1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', str1).lower()

def retrieve_best_model(
    function_name,
    dataset_name,
    is_multivariate,
    variate_type="multi",
    data_type="normal",
    prefix="tpe",
    db_dir: str | Path | None = None,
    verbosity=1,
):
    if function_name not in VALID_FUNCTION_NAMES:
        raise ValueError(f"Invalid function name: {function_name}")
    if variate_type not in ["multi", "uni"]:
        raise ValueError(f"Invalid variate type: {variate_type}")
    if data_type not in ["normal", "noisy"]:
        raise ValueError(f"Invalid data type: {data_type}")
    if variate_type == "uni" and is_multivariate:
        raise ValueError(f"Invalid variable type: {variate_type}")
    if prefix not in VALID_PREFIXES:
        raise ValueError(f"Unknown prefix: {prefix}")

    study_name = f"{function_name}_{dataset_name}_{data_type}_{variate_type}"

    # If db_dir is not provided, use the folder containing this utility.py file.
    if db_dir is None:
        db_dir = Path(__file__).resolve().parent
    else:
        db_dir = Path(db_dir).expanduser().resolve()
    db_filename = f"{prefix}_{camel_to_snake(dataset_name)}_db.sqlite3"
    db_path = db_dir / db_filename

    if not db_path.exists():
        available_dbs = sorted(p.name for p in db_dir.glob("*.sqlite3")) if db_dir.exists() else []
        raise FileNotFoundError(
            f"Optuna database not found:\n"
            f"  {db_path}\n\n"
            f"db_dir exists: {db_dir.exists()}\n"
            f"Expected filename: {db_filename}\n"
            f"Available .sqlite3 files in db_dir:\n"
            f"  {available_dbs}"
        )

    url = f"sqlite:///{db_path.resolve()}"

    if verbosity > 0:
        print("Loading study from URL:", url)
    study = optuna.load_study(study_name=study_name, storage=url)
    return study
