"""Model evaluation, losses, plotting, and utility helpers."""

from .esn_model_evaluation import (
    init_readout,
    init_nvar_model,
    init_ip_reservoir,
    init_reservoir,
    init_local_rule_reservoir,
    init_ip_local_rule_reservoir,
    train_model_for_prediction,
    train_model_for_classification,
    predict_model_for_classification,
    compute_score,
)

from .losses import (
    mse,
    nmse,
    nrmse,
    nrmse_multivariate,
)

from .plots import (
    plot_prediction_vs_actual,
)

from hag.hpo.utility import (
    camel_to_snake,
    retrieve_best_model,
)

__all__ = [
    # model initialization
    "init_readout",
    "init_nvar_model",
    "init_ip_reservoir",
    "init_reservoir",
    "init_local_rule_reservoir",
    "init_ip_local_rule_reservoir",

    # model training / inference / evaluation
    "train_model_for_prediction",
    "train_model_for_classification",
    "predict_model_for_classification",
    "compute_score",

    # losses
    "mse",
    "nmse",
    "nrmse",
    "nrmse_multivariate",

    # plotting
    "plot_prediction_vs_actual",

    # utilities
    "camel_to_snake",
]