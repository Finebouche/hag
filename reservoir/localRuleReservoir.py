import numpy as np

from functools import partial
from typing import Callable, Dict, Optional, Sequence, Union

from reservoirpy._base import check_xy
from reservoirpy.activationsfunc import get_function, identity, tanh
from reservoirpy.mat_gen import bernoulli, uniform, zeros
from reservoirpy.node import Unsupervised, _init_with_sequences
from reservoirpy.nodes.reservoirs.base import (
    initialize_feedback,
    reservoir_kernel,
    forward_external,
)
from reservoirpy.type import Weights
from reservoirpy.utils.random import noise, rand_generator
from reservoirpy.utils.validation import is_array

#########################
#  Local learning rules #
#########################

def local_update(reservoir, pre_state, post_state):
    """
    Apply the local learning rule (Oja, Anti-Oja, Hebbian, Anti-Hebbian, BCM)
    to update the recurrent weight matrix W.
    """
    W = reservoir.W
    eta = reservoir.learning_rate
    rule = reservoir.learning_rule.lower()

    x = pre_state[0]  # shape (units,)
    y = post_state[0] # shape (units,)

    bcm_theta = reservoir.bcm_theta  # default 0.0 if not used

    for i in range(reservoir.output_dim):

        if rule == "oja":
            # Oja's rule
            W[i, :] += eta * y[i] * (x - y[i] * W[i, :])

        elif rule == "anti-oja":
            # sign-flipped Oja
            W[i, :] -= eta * y[i] * (x - y[i] * W[i, :])

        elif rule == "hebbian":
            # Hebbian
            W[i, :] += eta * y[i] * x

        elif rule == "anti-hebbian":
            # sign-flipped Hebbian
            W[i, :] -= eta * y[i] * x

        elif rule == "bcm":
            # BCM
            # W[i,:] += eta * y_i ( y_i - theta ) x
            W[i, :] += eta * y[i] * (y[i] - bcm_theta) * x

        else:
            raise ValueError(f"Unknown learning rule '{rule}'. "
                             "Choose from: ['oja', 'anti-oja', 'hebbian', "
                             "'anti-hebbian', 'bcm'].")

    return W


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
                W_new = local_update(reservoir, pre_state, post_state)
                reservoir.set_param("W", W_new)


########################
#  Initialization base #
########################

def initialize_base(
    reservoir,
    x=None,
    y=None,
    sr=None,
    input_scaling=None,
    bias_scaling=None,
    input_connectivity=None,
    rc_connectivity=None,
    W_init=None,
    Win_init=None,
    bias_init=None,
    input_bias=None,
    seed=None,
):
    # Same code as in your original snippet.
    if x is not None:
        reservoir.set_input_dim(x.shape[1])

        dtype = reservoir.dtype
        dtype_msg = (
            "Data type {} not understood in {}. {} should be an array or a "
            "callable returning an array."
        )

        # Initialize W
        if is_array(W_init):
            W = W_init
            if W.shape[0] != W.shape[1]:
                raise ValueError(
                    "Dimension mismatch inside W: "
                    f"W is {W.shape} but should be a square matrix."
                )
            if W.shape[0] != reservoir.output_dim:
                reservoir._output_dim = W.shape[0]
                reservoir.hypers["units"] = W.shape[0]
        elif callable(W_init):
            W = W_init(
                reservoir.output_dim,
                reservoir.output_dim,
                sr=sr,
                connectivity=rc_connectivity,
                dtype=dtype,
                seed=seed,
            )
        else:
            raise ValueError(dtype_msg.format(str(type(W_init)), reservoir.name, "W"))
        reservoir.set_param("units", W.shape[0])
        reservoir.set_param("W", W.astype(dtype))

        out_dim = reservoir.output_dim

        # Initialize Win
        Win_has_bias = False
        if is_array(Win_init):
            Win = Win_init
            msg = (
                f"Dimension mismatch in {reservoir.name}: Win input dimension is "
                f"{Win.shape[1]} but input dimension is {x.shape[1]}."
            )
            if Win.shape[1] == x.shape[1] + 1:
                if input_bias:
                    Win_has_bias = True
                else:
                    bias_msg = (
                        " It seems Win has a bias column, but 'input_bias' is False."
                    )
                    raise ValueError(msg + bias_msg)
            elif Win.shape[1] != x.shape[1]:
                raise ValueError(msg)
            if Win.shape[0] != out_dim:
                raise ValueError(
                    f"Dimension mismatch in {reservoir.name}: Win internal dimension "
                    f"is {Win.shape[0]} but reservoir dimension is {out_dim}"
                )
        elif callable(Win_init):
            Win = Win_init(
                reservoir.output_dim,
                x.shape[1],
                input_scaling=input_scaling,
                connectivity=input_connectivity,
                dtype=dtype,
                seed=seed,
            )
        else:
            raise ValueError(
                dtype_msg.format(str(type(Win_init)), reservoir.name, "Win")
            )

        # Initialize bias
        if input_bias:
            if not Win_has_bias:
                if callable(bias_init):
                    bias = bias_init(
                        reservoir.output_dim,
                        1,
                        input_scaling=bias_scaling,
                        connectivity=input_connectivity,
                        dtype=dtype,
                        seed=seed,
                    )
                elif is_array(bias_init):
                    bias = bias_init
                    if bias.shape[0] != reservoir.output_dim or (
                        bias.ndim > 1 and bias.shape[1] != 1
                    ):
                        raise ValueError(
                            f"Dimension mismatch in {reservoir.name}: bias shape is "
                            f"{bias.shape} but should be {(reservoir.output_dim, 1)}"
                        )
                else:
                    raise ValueError(
                        dtype_msg.format(str(type(bias_init)), reservoir.name, "bias")
                    )
            else:
                bias = Win[:, :1]
                Win = Win[:, 1:]
        else:
            bias = zeros(reservoir.output_dim, 1, dtype=dtype)

        reservoir.set_param("Win", Win.astype(dtype))
        reservoir.set_param("bias", bias.astype(dtype))
        reservoir.set_param("internal_state", reservoir.zero_state())


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

    Reservoir states are updated with a standard Echo-State style:
      r[t+1] = (1 - lr)*r[t] + lr*(W r[t] + Win u[t+1] + Wfb fb[t] + bias)
      x[t+1] = activation(r[t+1])

    Then the local rule is applied each timestep to update W.

    Parameters
    ----------
    learning_rule : str, optional
        One of ["oja", "anti-oja", "hebbian", "anti-hebbian", "bcm"].
        Default = "oja".
    bcm_theta : float, optional
        The threshold used in the "bcm" rule. Default = 0.0.

    Other parameters:
      - units, sr, lr, learning_rate, epochs, ...
      - input_bias, noise_in, noise_rc, ...
      - input_scaling, rc_connectivity, ...
      - W, Win, Wfb initializers, etc.

    Example
    -------
    >>> reservoir = LocalRuleReservoir(
    ...     units=100, sr=0.9, learning_rule="hebbiAN",
    ...     learning_rate=1e-3, epochs=5
    ... )
    >>> # Fit on data timeseries
    >>> reservoir.fit(X_data, warmup=10)
    >>> # Then run
    >>> states = reservoir.run(X_data)
    """

    def __init__(
        self,
        # local rule choice
        learning_rule: str = "oja",
        bcm_theta: float = 0.0,
        # standard reservoir params
        units: int = None,
        sr: Optional[float] = None,
        lr: float = 1.0,
        learning_rate: float = 1e-3,
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
        dtype=np.float64,
        **kwargs,
    ):
        if units is None and not is_array(W):
            raise ValueError(
                "'units' parameter must not be None if 'W' parameter is not a matrix."
            )

        rng = rand_generator(seed=seed)
        noise_kwargs = dict() if noise_kwargs is None else noise_kwargs

        # Make sure the chosen rule is recognized:
        valid_rules = ["oja", "anti-oja", "hebbian", "anti-hebbian", "bcm"]
        if learning_rule.lower() not in valid_rules:
            raise ValueError(
                f"learning_rule must be one of {valid_rules}, got {learning_rule}."
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
                "learning_rule": learning_rule.lower(),  # store normalized
                "bcm_theta": bcm_theta,
                "sr": sr,
                "lr": lr,
                "learning_rate": learning_rate,
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
            dtype=dtype,
            **kwargs,
        )

    ##############
    # Properties #
    ##############

    @property
    def learning_rule(self) -> str:
        return self.hypers["learning_rule"]

    @property
    def bcm_theta(self) -> float:
        return self.hypers["bcm_theta"]

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

            # Apply local learning rule on the rest
            for e in range(self.epochs):
                for u in X_seq[warmup:]:
                    pre_state = self.internal_state
                    post_state = self.call(u.reshape(1, -1))

                    # Update W
                    W_new = local_update(self, pre_state, post_state)
                    self.set_param("W", W_new)

        return self