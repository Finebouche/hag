import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix

from functools import partial
from typing import Callable, Dict, Optional, Sequence, Union

from reservoirpy._base import check_xy
from reservoirpy.activationsfunc import get_function, identity, tanh
from reservoirpy.mat_gen import bernoulli, uniform, zeros
from reservoirpy.node import Unsupervised, _init_with_sequences
from reservoirpy.nodes.reservoirs.base import (
    initialize_feedback,
    forward_external,
    initialize as initialize_base,
)
from reservoirpy.type import Weights
from reservoirpy.utils.random import noise, rand_generator
from reservoirpy.utils.validation import is_array

#########################
#  Local learning rules #
#########################


def local_plasticity_rule(reservoir, pre_state, post_state):
    """
    Apply the local learning rule (Oja, Anti-Oja, Hebbian, Anti-Hebbian, BCM)
    to update the recurrent weight matrix W.

    If `synapse_normalization=True`, then each row of W is L2-normalized
    immediately after the local rule update.

    This version supports both dense and sparse matrices. For sparse matrices,
    the weight matrix is converted to LIL format for efficient row modifications.
    """
    W = reservoir.W
    eta = reservoir.eta
    rule = reservoir.local_rule.lower()
    bcm_theta = reservoir.bcm_theta
    do_norm = reservoir.synapse_normalization

    # pre_state, post_state shape: (1, units) => extract the vectors
    x = pre_state[0]  # shape (units,) - 'presynaptic'
    y = post_state[0]  # shape (units,) - 'postsynaptic'

    # Convert W to COO format for vectorized operations.
    if not sp.isspmatrix_coo(W):
        W = W.tocoo()

    # Get row and column indices and the nonzero data array.
    rows = W.row
    cols = W.col
    data = W.data.copy()  # work on a copy to avoid modifying in place

    # For each nonzero element W[i, j], update according to the selected rule:
    if rule == "oja":
        # new_data = old + eta * y[i] * ( x[j] - y[i] * old )
        data = data + eta * y[rows] * (x[cols] - y[rows] * data)
    elif rule == "anti-oja":
        data = data - eta * y[rows] * (x[cols] - y[rows] * data)
    elif rule == "hebbian":
        data = data + eta * y[rows] * x[cols]
    elif rule == "anti-hebbian":
        data = data - eta * y[rows] * x[cols]
    elif rule == "bcm":
        data = data + eta * y[rows] * (y[rows] - bcm_theta) * x[cols]
    else:
        raise ValueError(
            f"Unknown learning rule '{rule}'. Choose from: "
            "['oja', 'anti-oja', 'hebbian', 'anti-hebbian', 'bcm']."
        )

    # If row normalization is enabled, compute the L2 norm for each row in a vectorized way.
    if do_norm:
        # Compute sum of squares per row using bincount.
        row_sums = np.bincount(rows, weights=data**2, minlength=reservoir.output_dim)
        row_norms = np.sqrt(row_sums)
        # To avoid division by zero, replace zeros with one.
        safe_norms = np.where(row_norms > 0, row_norms, 1)
        data = data / safe_norms[rows]

    # Construct the updated sparse matrix in COO format, then convert to CSR.
    W_new = coo_matrix((data, (rows, cols)), shape=W.shape).tocsr()
    return W_new

def local_backward(reservoir, X=None, *args, **kwargs):
    """
    Offline learning method for the local-rule-based reservoir.
    """
    for epoch in range(reservoir.epochs):
        for seq in X:
            for u in seq:
                pre_state = reservoir.internal_state  # shape (1, units)
                post_state = reservoir.call(u.reshape(1, -1))  # shape (1, units)

                # Update W with the chosen local rule
                W_new = local_plasticity_rule(reservoir, pre_state, post_state)
                reservoir.set_param("W", W_new)



def initialize_local_rule(reservoir, *args, **kwargs):
    """
    Custom initializer for the LocalRuleReservoir.
    Reuses the ESN-like initialization and sets the reservoir internal state to zeros.
    """
    initialize_base(reservoir, *args, **kwargs)


################################
#  LocalRuleReservoir  Class   #
################################

class LocalRuleReservoir(Unsupervised):
    """
    A reservoir that learns its recurrent weights W through a local
    learning rule selected by the 'learning_rule' hyperparameter.

    Supported rules:
      - "oja"
      - "anti-oja"
      - "hebbian"
      - "anti-hebbian"
      - "bcm"

    By default, "oja".

    For "bcm", you can set a threshold 'bcm_theta' (default 0.0).

    If `synapse_normalization=True`, then after each local-rule update
    on a row i of W, the row is rescaled to unit L2 norm.

    Reservoir states are updated with a standard Echo-State style:
      r[t+1] = (1 - lr)*r[t] + lr*(W r[t] + Win u[t+1] + Wfb fb[t] + bias)
      x[t+1] = activation(r[t+1])

    Then the local rule is applied each timestep to update W.

    Parameters
    ----------
    local_rule : str, optional
        One of ["oja", "anti-oja", "hebbian", "anti-hebbian", "bcm"].
        Default = "oja".
    bcm_theta : float, optional
        The threshold used in the "bcm" rule. Default = 0.0.
    eta : float, optional
        Local learning rate for the weight update. Default = 1e-3.
    synapse_normalization : bool, optional
        If True, L2-normalize each row of W after its update. Default = False.

    Other standard reservoir parameters:
      - units, sr, lr, epochs, ...
      - input_bias, noise_in, noise_rc, ...
      - input_scaling, rc_connectivity, ...
      - W, Win, Wfb initializers, etc.

    Example
    -------
    >>> reservoir = LocalRuleReservoir(
    ...     units=100, sr=0.9, local_rule="hebbian",
    ...     eta=1e-3, epochs=5, synapse_normalization=True
    ... )
    >>> # Fit on data timeseries
    >>> reservoir.fit(X_data, warmup=10)
    >>> # Then run
    >>> states = reservoir.run(X_data)
    """

    def __init__(
        self,
        # local rule choice
        local_rule: str = "oja",
        eta: float = 1e-3,
        synapse_normalization: bool = False,
        bcm_theta: float = 0.0,
        # standard reservoir params
        units: int = None,
        sr: Optional[float] = None,
        lr: float = 1.0,
        epochs: int = 1,
        input_bias: bool = True,
        noise_rc: float = 0.0,
        noise_in: float = 0.0,
        noise_fb: float = 0.0,
        noise_type: str = "normal",
        noise_kwargs: Dict = None,
        input_scaling: Union[float, Sequence] = 1.0,
        bias_scaling: float = 1.0,
        fb_scaling: Union[float, Sequence] = 1.0,
        input_connectivity: Optional[float] = 0.1,
        rc_connectivity: Optional[float] = 0.1,
        fb_connectivity: Optional[float] = 0.1,
        Win: Union[Weights, Callable] = bernoulli,
        W: Union[Weights, Callable] = uniform,
        Wfb: Union[Weights, Callable] = bernoulli,
        bias: Union[Weights, Callable] = bernoulli,
        feedback_dim: int = None,
        fb_activation: Union[str, Callable] = identity,
        activation: Union[str, Callable] = tanh,
        name=None,
        seed=None,
        **kwargs,
    ):
        if units is None and not is_array(W):
            raise ValueError(
                "'units' parameter must not be None if 'W' parameter is not a matrix."
            )

        rng = rand_generator(seed=seed)
        noise_kwargs = dict() if noise_kwargs is None else noise_kwargs

        # Validate local rule name
        valid_rules = ["oja", "anti-oja", "hebbian", "anti-hebbian", "bcm"]
        if local_rule.lower() not in valid_rules:
            raise ValueError(
                f"learning_rule must be one of {valid_rules}, got {local_rule}."
            )

        super(LocalRuleReservoir, self).__init__(
            fb_initializer=partial(
                initialize_feedback,
                Wfb_init=Wfb,
                fb_scaling=fb_scaling,
                fb_connectivity=fb_connectivity,
                seed=seed,
            ),
            params={
                "W": None,
                "Win": None,
                "Wfb": None,
                "bias": None,
                "internal_state": None,
            },
            hypers={
                "learning_rule": local_rule.lower(),
                "bcm_theta": bcm_theta,
                "eta": eta,
                "synapse_normalization": synapse_normalization,
                "sr": sr,
                "lr": lr,
                "epochs": epochs,
                "input_bias": input_bias,
                "input_scaling": input_scaling,
                "fb_scaling": fb_scaling,
                "rc_connectivity": rc_connectivity,
                "input_connectivity": input_connectivity,
                "fb_connectivity": fb_connectivity,
                "noise_in": noise_in,
                "noise_rc": noise_rc,
                "noise_out": noise_fb,
                "noise_type": noise_type,
                "activation": get_function(activation) if isinstance(activation, str) else activation,
                "fb_activation": get_function(fb_activation) if isinstance(fb_activation, str) else fb_activation,
                "units": units,
                "noise_generator": partial(noise, rng=rng, **noise_kwargs),
            },
            forward=forward_external,
            initializer=partial(
                initialize_local_rule,
                input_bias=input_bias,
                bias_scaling=bias_scaling,
                sr=sr,
                input_scaling=input_scaling,
                input_connectivity=input_connectivity,
                rc_connectivity=rc_connectivity,
                W_init=W,
                Win_init=Win,
                bias_init=bias,
                seed=seed,
            ),
            backward=local_backward,
            output_dim=units,
            feedback_dim=feedback_dim,
            name=name,
            **kwargs,
        )

    ##############
    # Properties #
    ##############

    @property
    def local_rule(self) -> str:
        return self.hypers["local_rule"]

    @property
    def bcm_theta(self) -> float:
        return self.hypers["bcm_theta"]

    @property
    def eta(self) -> float:
        return self.hypers["eta"]

    @property
    def synapse_normalization(self) -> bool:
        return self.hypers["synapse_normalization"]

    @property
    def fitted(self) -> bool:
        # For an unsupervised node that can always be updated,
        # we set `fitted = True` after first initialization/training.
        return True

    ############################
    # partial_fit => local rule
    ############################

    def partial_fit(self, X_batch, Y_batch=None, warmup=0, **kwargs) -> "LocalRuleReservoir":
        """Partial offline fitting method (for batch training)."""
        X, _ = check_xy(self, X_batch, allow_n_inputs=False)
        X, _ = _init_with_sequences(self, X)

        self.initialize_buffers()

        for i in range(len(X)):
            X_seq = X[i]

            if X_seq.shape[0] <= warmup:
                raise ValueError(
                    f"Warmup set to {warmup} timesteps, "
                    f"but one timeseries is only {X_seq.shape[0]} long."
                )

            # Run warmup if specified
            if warmup > 0:
                self.run(X_seq[:warmup])

            self._partial_backward(self, X_seq[warmup:])


        return self