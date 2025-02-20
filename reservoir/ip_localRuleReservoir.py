import numpy as np
from functools import partial
from typing import Callable, Dict, Optional, Sequence, Union

from reservoirpy._base import check_xy
from reservoirpy.activationsfunc import get_function, identity
from reservoirpy.mat_gen import bernoulli, uniform
from reservoirpy.node import Unsupervised, _init_with_sequences
from reservoirpy.type import Weights
from reservoirpy.utils.random import noise, rand_generator
from reservoirpy.utils.validation import is_array

# ReservoirPy or your custom code
from reservoirpy.nodes.reservoirs.base import (
    initialize_feedback,
    forward_external
)
# IP update function:
from reservoirpy.nodes.reservoirs.intrinsic_plasticity import ip

from localRuleReservoir import local_update



###############################
#  IP + Local Rule Backward   #
###############################

def ip_local_backward(reservoir: "IP_LocalRuleReservoir", X=None, *args, **kwargs):
    """
    Offline learning method combining IP (intrinsic plasticity for a,b)
    and a local synaptic rule for W.

    Steps for each sample in X:
      1. Forward pass: post_state = reservoir.call(u)
      2. IP update for reservoir.a, reservoir.b
      3. local update (oja, anti-oja, hebbian, anti-hebbian, bcm) for reservoir.W
    """
    for e in range(reservoir.epochs):
        for seq in X:
            for u in seq:
                post_state = reservoir.call(u.reshape(1, -1))
                pre_state = reservoir.internal_state

                # 1) Intrinsic Plasticity update
                a_new, b_new = ip(reservoir, pre_state, post_state)
                reservoir.set_param("a", a_new)
                reservoir.set_param("b", b_new)

                # 2) Local rule update
                W_new = local_update(reservoir, pre_state, post_state)
                reservoir.set_param("W", W_new)


def initialize_ip_local(reservoir, *args, **kwargs):
    """
    Initialization for an IP + local-rule reservoir:
    - usual reservoir init for W, Win, bias, etc.
    - init IP parameters a, b
    """
    # If you have a base init function (like IPReservoir) use it,
    # or define your own. For simplicity, let's just do:
    reservoir.initialize_base(reservoir, *args, **kwargs)

    # Initialize IP params
    a = np.ones((reservoir.output_dim, 1), dtype=reservoir.dtype)
    b = np.zeros((reservoir.output_dim, 1), dtype=reservoir.dtype)
    reservoir.set_param("a", a)
    reservoir.set_param("b", b)


#####################################
#  The IP + LocalRule Reservoir     #
#####################################

class IP_LocalRuleReservoir(Unsupervised):
    """
    Reservoir implementing BOTH Intrinsic Plasticity (per-neuron a,b updates)
    AND a local rule update for W (e.g. Oja, anti-Oja, Hebbian, anti-Hebbian, BCM).

    Steps at each time step:
      1) r[t+1] = (1-lr)*r[t] + lr*(W*r[t] + Win*u[t] + Wfb*fb[t] + bias)
      2) x[t+1] = f(a*r[t+1] + b)
      3) IP update: a,b <- ip(...).
      4) local rule update: W <- local_update(...).

    Parameters
    ----------
    learning_rule : str
        Which local rule to apply for W.
        One of ["oja", "anti-oja", "hebbian", "anti-hebbian", "bcm"].
    bcm_theta : float
        Threshold used if learning_rule="bcm".
    activation_type : {"tanh", "sigmoid"}
        Used by IP to choose the target distribution.
    mu, sigma : floats
        Used by IP if activation_type="tanh".
    ... etc. (like in IP or localRule).
    """

    def __init__(
        self,
        # local rule
        learning_rule: str = "oja",
        bcm_theta: float = 0.0,
        # standard IP params
        units: int = None,
        sr: Optional[float] = None,
        lr: float = 1.0,
        mu: float = 0.0,
        sigma: float = 1.0,
        learning_rate: float = 1e-3,
        epochs: int = 1,
        activation_type: str = "tanh",
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
        # IP requires "tanh" or "sigmoid" for distribution:
        activation: Union[str, Callable] = "tanh",
        name=None,
        seed=None,
        dtype=np.float64,
        **kwargs,
    ):
        if units is None and not is_array(W):
            raise ValueError(
                "'units' parameter must not be None if 'W' parameter is not a matrix."
            )
        if activation_type not in ["tanh", "sigmoid"]:
            raise ValueError(
                f"activation_type must be 'tanh' or 'sigmoid' for IP. Got {activation_type}."
            )

        # Check the local rule
        valid_rules = ["oja", "anti-oja", "hebbian", "anti-hebbian", "bcm"]
        if learning_rule.lower() not in valid_rules:
            raise ValueError(
                f"learning_rule must be one of {valid_rules}, got {learning_rule}."
            )

        rng = rand_generator(seed=seed)
        noise_kwargs = dict() if noise_kwargs is None else noise_kwargs

        # We'll define a custom activation that includes IP's a,b:
        def _ip_activation(state, *, reservoir, base_f):
            a, b = reservoir.a, reservoir.b
            return base_f(a * state + b)

        if isinstance(activation, str):
            base_f = get_function(activation)
        else:
            base_f = activation

        final_activation = partial(_ip_activation, reservoir=self, base_f=base_f)

        super(IP_LocalRuleReservoir, self).__init__(
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
                "a": None,
                "b": None,
            },
            hypers={
                # local rule
                "learning_rule": learning_rule.lower(),
                "bcm_theta": bcm_theta,
                # IP
                "sr": sr,
                "lr": lr,
                "mu": mu,
                "sigma": sigma,
                "learning_rate": learning_rate,
                "epochs": epochs,
                "activation_type": activation_type,
                # final activation
                "activation": final_activation,
                "fb_activation": get_function(fb_activation) if isinstance(fb_activation, str) else fb_activation,
                # noise, scaling, connectivity
                "noise_in": noise_in,
                "noise_rc": noise_rc,
                "noise_out": noise_fb,
                "noise_type": noise_type,
                "units": units,
                "input_bias": input_bias,
                "input_scaling": input_scaling,
                "bias_scaling": bias_scaling,
                "fb_scaling": fb_scaling,
                "rc_connectivity": rc_connectivity,
                "input_connectivity": input_connectivity,
                "fb_connectivity": fb_connectivity,
                "noise_generator": partial(noise, rng=rng, **noise_kwargs),
            },
            forward=forward_external,
            initializer=partial(
                initialize_ip_local,
                sr=sr,
                input_bias=input_bias,
                bias_scaling=bias_scaling,
                input_scaling=input_scaling,
                input_connectivity=input_connectivity,
                rc_connectivity=rc_connectivity,
                W_init=W,
                Win_init=Win,
                bias_init=bias,
                seed=seed,
            ),
            backward=ip_local_backward,  # combine IP + local rule
            output_dim=units,
            feedback_dim=feedback_dim,
            name=name,
            dtype=dtype,
            **kwargs,
        )

    # IP shortcuts
    @property
    def a(self):
        return self.params["a"]

    @property
    def b(self):
        return self.params["b"]

    @property
    def activation_type(self) -> str:
        return self.hypers["activation_type"]

    @property
    def mu(self) -> float:
        return self.hypers["mu"]

    @property
    def sigma(self) -> float:
        return self.hypers["sigma"]

    # Local rule shortcuts
    @property
    def learning_rule(self) -> str:
        return self.hypers["learning_rule"]

    @property
    def bcm_theta(self) -> float:
        return self.hypers["bcm_theta"]

    # Fitted status
    @property
    def fitted(self) -> bool:
        return True

    # partial_fit => same logic as in `ip_local_backward`
    def partial_fit(self, X_batch, Y_batch=None, warmup=0, **kwargs) -> "IP_LocalRuleReservoir":
        X, _ = check_xy(self, X_batch, allow_n_inputs=False)
        X, _ = _init_with_sequences(self, X)

        self.initialize_buffers()

        for i in range(len(X)):
            X_seq = X[i]
            if X_seq.shape[0] <= warmup:
                raise ValueError(
                    f"Warmup set to {warmup}, but one timeseries has length {X_seq.shape[0]}."
                )
            if warmup > 0:
                self.run(X_seq[:warmup])

            seq_to_train = X_seq[warmup:]
            for e in range(self.epochs):
                for u in seq_to_train:
                    post_state = self.call(u.reshape(1, -1))
                    pre_state = self.internal_state

                    # 1) IP update
                    a_new, b_new = ip(self, pre_state, post_state)
                    self.set_param("a", a_new)
                    self.set_param("b", b_new)

                    # 2) Local rule update
                    W_new = local_update(self, pre_state, post_state)
                    self.set_param("W", W_new)

        return self