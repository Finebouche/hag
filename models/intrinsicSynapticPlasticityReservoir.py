"""
IPSPReservoir — Intrinsic Plasticity + Local Synaptic Plasticity reservoir.

Inherits from ``LocalPlasticityReservoir`` and adds per-neuron IP gain (``a``)
and bias (``b``) parameters that are updated each timestep to shape the
activation distribution toward a target (Gaussian for tanh, exponential for
sigmoid).
"""

from typing import Literal, Optional, Sequence, Union, Callable

import numpy as np
import scipy.sparse as sp

from reservoirpy.mat_gen import bernoulli, uniform
from reservoirpy.type import NodeInput, State, Timeseries, Timestep, Weights, is_array, is_multiseries
from reservoirpy.utils.data_validation import check_node_input
from reservoirpy.nodes import LocalPlasticityReservoir


class IPLocalPlasticityReservoir(LocalPlasticityReservoir):
    """
    A reservoir combining Intrinsic Plasticity (IP) with a local synaptic
    learning rule, built on top of :class:`LocalPlasticityReservoir`.

    The forward equation becomes:

    .. math::

        r[t+1] &= (1-lr) r[t] + lr (W r[t] + W_{in} u[t+1] + bias) \\\\
        x[t+1] &= f(a \\cdot r[t+1] + b)

    where ``a`` and ``b`` are per-neuron IP parameters updated each timestep.

    All local synaptic rule parameters (``local_rule``, ``eta``,
    ``synapse_normalization``, ``bcm_theta``, …) are inherited unchanged.

    Additional Parameters
    ---------------------
    ip_learning_rate : float, default 1e-3
        Learning rate for IP updates.
    mu : float, default 0.0
        Target mean (tanh/Gaussian) or 1/λ (sigmoid/exponential).
    sigma : float, default 1.0
        Target std (tanh/Gaussian only).

    Example
    -------
    >>> reservoir = IPSPReservoir(
    ...     units=100, sr=0.9, local_rule="oja",
    ...     eta=1e-3, ip_learning_rate=1e-3,
    ...     mu=0.0, sigma=1.0, epochs=5,
    ... )
    >>> reservoir.fit(X_data, warmup=10)
    >>> states = reservoir.run(X_data)
    """

    ip_learning_rate: float
    mu: float
    sigma: float
    a: np.ndarray
    b: np.ndarray

    def __init__(
        self,
        *,
        # IP-specific
        ip_learning_rate: float = 1e-3,
        mu: float = 0.0,
        sigma: float = 1.0,
        # everything else forwarded to parent
        units: Optional[int] = None,
        local_rule: Literal["oja", "anti-oja", "hebbian", "anti-hebbian", "bcm"] = "oja",
        eta: float = 1e-3,
        bcm_theta: float = 0.0,
        synapse_normalization: bool = False,
        epochs: int = 1,
        sr: float = 1.0,
        lr: float = 1.0,
        input_scaling: Union[float, Sequence] = 1.0,
        input_connectivity: float = 0.1,
        rc_connectivity: float = 0.1,
        Win: Union[Weights, Callable] = bernoulli,
        W: Union[Weights, Callable] = uniform,
        bias: Union[Weights, Callable] = bernoulli,
        activation: Literal["tanh", "sigmoid"] = "tanh",
        input_dim: Optional[int] = None,
        seed=None,
        dtype: type = np.float64,
        name: Optional[str] = None,
    ):
        super().__init__(
            units=units,
            local_rule=local_rule,
            eta=eta,
            bcm_theta=bcm_theta,
            synapse_normalization=synapse_normalization,
            epochs=epochs,
            sr=sr,
            lr=lr,
            input_scaling=input_scaling,
            input_connectivity=input_connectivity,
            rc_connectivity=rc_connectivity,
            Win=Win,
            W=W,
            bias=bias,
            activation=activation,
            input_dim=input_dim,
            seed=seed,
            dtype=dtype,
            name=name,
        )

        if activation not in ("tanh", "sigmoid"):
            raise ValueError(f"activation must be 'tanh' or 'sigmoid', got '{activation}'.")

        self.activation_type = activation
        self.ip_learning_rate = ip_learning_rate
        self.mu = mu
        self.sigma = sigma
        self.a = None
        self.b = None

    # ------------------------------------------------------------------
    #  Initialization — extend parent to add IP params
    # ------------------------------------------------------------------

    def initialize(self, x=None):
        super().initialize(x)
        self.a = np.ones((self.units,), dtype=self.dtype)
        self.b = np.zeros((self.units,), dtype=self.dtype)

    # ------------------------------------------------------------------
    #  Forward step — override to inject IP (a, b) into activation
    # ------------------------------------------------------------------

    def _step(self, state: State, x: Timestep) -> State:
        W = self.W
        Win = self.Win
        bias = self.bias
        f = self.activation
        lr = self.lr
        s = state["out"]

        # Leaky integration (pre-activation)
        r = W @ s + Win @ x + bias
        r = (1 - lr) * s + lr * r

        # IP-adjusted activation
        y = f(self.a * r + self.b)

        return {"internal": r, "out": y}

    # ------------------------------------------------------------------
    #  IP update
    # ------------------------------------------------------------------

    def _ip_update(self, r: np.ndarray, y: np.ndarray):
        """Update IP parameters a and b for one timestep."""
        eta = self.ip_learning_rate
        mu = self.mu
        sigma = self.sigma

        if self.activation_type == "tanh":
            db = eta * (
                -mu / (sigma ** 2)
                + y / (sigma ** 2) * (2 * sigma ** 2 + 1 - y ** 2 + mu * y)
            )
        else:  # sigmoid
            db = eta * (1.0 - (2.0 + 1.0 / mu) * y + (y ** 2) / mu)

        da = 1.0 / self.a + db * r

        self.a += da
        self.b += db

    # ------------------------------------------------------------------
    #  Fit — extend parent loop to include IP update each timestep
    # ------------------------------------------------------------------

    def fit(self, x: NodeInput, y=None, warmup: int = 0) -> "IPSPReservoir":
        check_node_input(x, expected_dim=self.input_dim)

        if not self.initialized:
            self.initialize(x)

        increment = self.increment
        do_norm = self.synapse_normalization

        def _train_sequence(seq: Timeseries):
            for u in seq:
                pre_state = self.state["internal"]

                new_state = self._step(self.state, u)
                self.state = new_state

                post_internal = new_state["internal"]
                post_output = new_state["out"]

                # IP update
                self._ip_update(post_internal, post_output)

                # Local synaptic plasticity update
                rows, cols, data = sp.find(self.W)
                self.W[rows, cols] += increment(
                    data, pre_state[cols], post_internal[rows]
                )

                if do_norm:
                    row_norms = np.sqrt(np.sum(self.W ** 2, axis=1)).reshape(-1, 1)
                    safe_norms = np.where(row_norms > 0, row_norms, 1)
                    self.W[:] /= safe_norms[:]

        for _epoch in range(self.epochs):
            if is_multiseries(x):
                for seq in x:
                    _train_sequence(seq[warmup:])
            else:
                _train_sequence(x[warmup:])

        return self