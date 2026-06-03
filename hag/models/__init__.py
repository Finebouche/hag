"""Reservoir and recurrent model implementations."""

from .activation_functions import (
    softmax,
    softplus,
    sigmoid,
    tanh,
    identity,
    relu,
    heaviside,
)

from .reservoir import (
    update_reservoir,
    init_matrices,
)

from .reservoir_ei import (
    update_ei_reservoir,
    ridge_regression,
    train_ei,
    run_ei,
)

from .intrinsicSynapticPlasticityReservoir import (
    IPLocalPlasticityReservoir,
)

from .rnn import (
    DEVICE,
    pad_collate,
    BucketBatchSampler,
    SequenceDataset,
    PrecomputedForecastDataset,
    LSTMModel,
    GRUModel,
    RNNModel,
    make_sliding_windows,
    train,
    evaluate,
)

__all__ = [
    # activation functions
    "softmax",
    "softplus",
    "sigmoid",
    "tanh",
    "identity",
    "relu",
    "heaviside",

    # reservoir
    "update_reservoir",
    "init_matrices",

    # excitatory/inhibitory reservoir
    "update_ei_reservoir",
    "ridge_regression",
    "train_ei",
    "run_ei",

    # intrinsic synaptic plasticity reservoir
    "IPLocalPlasticityReservoir",

    # RNN utilities/models
    "LSTMModel",
    "GRUModel",
    "RNNModel",
    "SequenceDataset",
    "train",
    "evaluate",
    "pad_collate",
    "BucketBatchSampler",
    "PrecomputedForecastDataset",
    "make_sliding_windows",
]