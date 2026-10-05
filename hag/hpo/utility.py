import re
import optuna
from pathlib import Path

VALID_FUNCTION_NAMES = {
    "mean_hag", "var_hag", "random_ee", "random_ei", "diag_ee", "diag_ei",
    "ip_correct", "anti-oja", "ip-anti-oja",
    "lstm_last", "rnn", "rnn-mean_hag", "gru", "short-hag", "hsp"
}
VALID_PREFIXES = {
    "tpe", "new_tpe", "cmaes", "lstm_tpe",
    "rdn-proj_tpe_mfcc", "rdn-proj_tpe_custom", "rdn-proj_tpe_none", "rdn-proj_tpe_stft",
    "mod-proj_tpe_mfcc", "mod-proj_tpe_custom", "mod-proj_tpe_none", "mod-proj_tpe_stft",
}


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGACY_STUDIES = PROJECT_ROOT / "hag" / "hpo" / "legacy_studies"


def study_db_dir(dataset_name=None):
    """Folder of the Optuna databases (the same for all datasets): hag/hpo/legacy_studies."""
    return LEGACY_STUDIES


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

    # If db_dir is not provided, use the folder of this dataset's studies.
    if db_dir is None:
        db_dir = study_db_dir(dataset_name)
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


def hag_reservoir_from_hyperparameters(params, function, input_dim, seed=None, **kwargs):
    """
    reservoirpy's HAGReservoir of HAG's hyperparameter optimization (mean_hag or var_hag, parameter names of
    hag/hpo/hpo_esn.py, e.g. the best trial of a study): network_size rounded up to a multiple of input_dim, as in
    HAG's evaluation. If W, Win and bias are not given, they are HAG's init_matrices with this seed (int), as in its
    evaluation. kwargs: other HAGReservoir arguments (W, Win, bias, ...).
    """
    import math
    from scipy import stats
    from reservoirpy.nodes import HAGReservoir
    from hag.models.reservoir import init_matrices

    K = math.ceil(params['network_size'] / input_dim)
    if not {"W", "Win", "bias"} <= kwargs.keys():
        Win, W, bias = init_matrices(input_dim * K, params['input_connectivity'], params['connectivity'], K,
                                     w_distribution=stats.uniform(loc=-1, scale=2), seed=seed)
        kwargs = {"W": W, "Win": Win * params['input_scaling'], "bias": bias * params['bias_scaling'], **kwargs}
    if function == "mean_hag":
        homeostasis, target, spread, extra = "mean", params['target_rate'], params['rate_spread'], {}
    elif function == "var_hag":
        homeostasis = "variance"
        target = params['variance_target']
        spread = params['variance_spread']
        extra = dict(intrinsic_saturation=params['intrinsic_saturation'], intrinsic_coef=params['intrinsic_coef'])
    else:
        raise ValueError(f"Not mean_hag or var_hag: {function!r}")
    return HAGReservoir(units=input_dim * K, homeostasis=homeostasis, target=target, spread=spread,
                        weight_increment=params['weight_increment'], min_window=params['min_increment'],
                        max_window=params.get('max_increment'), use_full_instance=params.get('use_full_instance', False),
                        lr=params['leaky_rate'], input_dim=input_dim, seed=seed, **extra, **kwargs)
