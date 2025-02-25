import sys
from tqdm import tqdm
if sys.version_info < (3, 8):
    from typing_extensions import Literal
else:
    from typing import Literal

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
    forward_external,
    initialize as initialize_base,
)
# IP update function:
from reservoirpy.nodes.reservoirs.intrinsic_plasticity import (
    ip,
    ip_activation
)

# Local update function:
from reservoir.synapticPlasticityReservoir import synaptic_plasticity


##########################################
#  1) IP + Local Rule "backward" method  #
##########################################

def ip_sp_backward(reservoir: "IPSPReservoir", X=None, *args, **kwargs):
    """
    Offline learning method for a reservoir combining intrinsic plasticity (IP)
    and a local synaptic rule (Oja, Hebb, etc.). The logic is:
      - For each epoch and each sequence:
        - For each time step:
          1) Update reservoir state by calling `reservoir.call(input)`.
          2) Perform IP update on parameters 'a', 'b'.
          3) Perform local rule update on recurrent weights 'W'.
    """
    for _ in range(reservoir.epochs):
        for seq in X:
            for u in seq:
                # 1) Forward pass: store old state, then compute new post_state
                post_state = reservoir.call(u.reshape(1, -1))
                pre_state = reservoir.internal_state

                # 2) Intrinsic Plasticity update
                a_new, b_new = ip(reservoir, pre_state, post_state)
                reservoir.set_param("a", a_new)
                reservoir.set_param("b", b_new)

                # 3) Synaptic Plasticity update
                W_new = synaptic_plasticity(reservoir, pre_state, post_state)
                reservoir.set_param("W", W_new)


###################################################
#  2) Initialization for the IP + Synaptic Plasticity ESN  #
###################################################

def initialize_ip_sp_reservoir(reservoir, *args, **kwargs):
    """
    Custom initializer for an IP + Synaptic Plasticity reservoir.
    1) Calls the usual ESN-like initialization (W, Win, bias, etc.).
    2) Initializes IP parameters (a, b).
    """
    # Use your base ESN-like init:
    initialize_base(reservoir, *args, **kwargs)

    # Initialize IP params
    a = np.ones((reservoir.output_dim, 1), dtype=reservoir.dtype)
    b = np.zeros((reservoir.output_dim, 1), dtype=reservoir.dtype)
    reservoir.set_param("a", a)
    reservoir.set_param("b", b)


########################################
#  3) IP + LocalRule Reservoir  Class  #
########################################

class IPSPReservoir(Unsupervised):
    """
    A reservoir implementing:
      - Intrinsic Plasticity (neuron-wise parameters 'a' and 'b'), and
      - A local synaptic learning rule (e.g., Oja, Hebbian, BCM, etc.) for the
        recurrent weight matrix 'W'.

    At each timestep:
      1) r[t+1] = (1-lr)*r[t] + lr*(W*r[t] + Win*u[t] + Wfb*fb[t] + bias)
      2) x[t+1] = activation( a * r[t+1] + b )   # i.e., IP-adjusted activation
      3) IP update: a, b <- ip(...)
      4) Local rule update: W <- local_update(...)

    Parameters
    ----------
    local_rule : str
        Local synaptic rule to apply. One of ["oja", "anti-oja", "hebbian",
        "anti-hebbian", "bcm"].
    eta : float
        Local learning rate for W update.
    synapse_normalization : bool
        If True, each row of W is L2-normalized after the local rule update.
    bcm_theta : float
        Threshold used for the "bcm" rule only.
    activation_type : {"tanh", "sigmoid"}
        Which IP distribution to aim for: 'tanh' => normal, 'sigmoid' => exponential.
    mu, sigma : float
        IP distribution parameters if activation_type="tanh" (i.e. mean, std).
        For "sigmoid", mu is 1/lambda, sigma is not used.
    learning_rate : float
        Intrinsic Plasticity learning rate for a,b updates.
    epochs : int
        How many passes of IP+Local learning to perform.
    (… plus typical reservoir hyperparameters: units, sr, lr, etc.)

    Notes
    -----
    - The code will initialize random W, Win, Wfb, bias, plus IP parameters a,b.
    - The IP update is done using the `ip(...)` function from reservoirpy.
    - The local rule update is done by the `local_update(...)` function from your code.
    """

    def __init__(
        self,
        # local rule choice
        local_rule: str = "oja",
        eta: float = 1e-3,
        synapse_normalization: bool = False,
        bcm_theta: float = 0.0,
        # IP params
        units: int = None,
        sr: Optional[float] = None,
        lr: float = 1.0,   # leaking rate
        mu: float = 0.0,
        sigma: float = 1.0,
        learning_rate: float = 1e-3,  # IP learning rate
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
        activation: Literal["tanh", "sigmoid"] = "tanh",
        name=None,
        seed=None,
        **kwargs,
    ):
        # Check for mandatory reservoir size or custom W
        if units is None and not is_array(W):
            raise ValueError(
                "'units' must be provided if 'W' is not an explicit matrix."
            )

        # Check IP activation
        if activation_type not in ["tanh", "sigmoid"]:
            raise ValueError(
                f"activation_type must be 'tanh' or 'sigmoid'. Got {activation_type}."
            )

        # Check local rule
        valid_rules = ["oja", "anti-oja", "hebbian", "anti-hebbian", "bcm"]
        if local_rule.lower() not in valid_rules:
            raise ValueError(
                f"learning_rule must be one of {valid_rules}, got '{local_rule}'."
            )

        # Prepare random generator, noise
        rng = rand_generator(seed=seed)
        noise_kwargs = {} if noise_kwargs is None else noise_kwargs

        super().__init__(
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
                "b": None,  # for IP
            },
            hypers={
                # Local rule hyperparams
                "local_rule": local_rule.lower(),
                "bcm_theta": bcm_theta,
                "eta": eta,
                "synapse_normalization": synapse_normalization,
                # IP hyperparams
                "sr": sr,
                "lr": lr,
                "mu": mu,
                "sigma": sigma,
                "learning_rate": learning_rate,
                "epochs": epochs,
                "activation_type": activation_type,
                # final activation
                "activation": partial(
                    ip_activation, reservoir=self, f=get_function(activation)
                ),
                "fb_activation": fb_activation,
                # standard reservoir stuff
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
                initialize_ip_sp_reservoir,
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
            backward=ip_sp_backward,
            output_dim=units,
            feedback_dim=feedback_dim,
            name=name,
            **kwargs,
        )

    #############
    #  IP props #
    #############

    @property
    def a(self):
        """Gain parameter (vector) for Intrinsic Plasticity."""
        return self.params["a"]

    @property
    def b(self):
        """Bias parameter (vector) for Intrinsic Plasticity."""
        return self.params["b"]

    @property
    def activation_type(self) -> str:
        """'tanh' or 'sigmoid'."""
        return self.hypers["activation_type"]

    @property
    def mu(self) -> float:
        return self.hypers["mu"]

    @property
    def sigma(self) -> float:
        return self.hypers["sigma"]

    #######################
    #  Local rule props   #
    #######################

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

    #######################
    #  Fitted status etc. #
    #######################

    @property
    def fitted(self) -> bool:
        """
        For unsupervised nodes that can always be updated,
        we typically mark them as 'fitted' after initialization.
        """
        return True

    ################################
    #  partial_fit => same logic   #
    ################################

    def partial_fit(self, X_batch, Y_batch=None, warmup=0, **kwargs) -> "IPSPReservoir":
        """
        Partial offline fitting method:
          - Warmup the reservoir for `warmup` steps (no learning).
          - Then, for each sequence, run IP + local rule updates.

        Parameters
        ----------
        X_batch : array-like of shape (n_sequences, timesteps, n_features)
            A batch of sequences. Can be a single sequence.
        Y_batch : ignored (for unsupervised).
        warmup : int
            Number of timesteps at the start of each sequence to skip training.

        Returns
        -------
        IPSPReservoir
            The reservoir itself (fitted).
        """
        X, _ = check_xy(self, X_batch, allow_n_inputs=False)
        X, _ = _init_with_sequences(self, X)

        self.initialize_buffers()

        for i in range(len(X)):
            X_seq = X[i]
            if X_seq.shape[0] <= warmup:
                raise ValueError(
                    f"Warmup={warmup}, but sequence length={X_seq.shape[0]}."
                )

            # Warmup phase
            if warmup > 0:
                self.run(X_seq[:warmup])

            self._partial_backward(self, X_seq[warmup:])

        return self