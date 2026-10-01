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

        r[t+1] &= (1-lr) r[t] + lr (W x[t] + W_{in} u[t+1] + bias) \\\\
        x[t+1] &= f(a \\cdot r[t+1] + b)

    where ``a`` and ``b`` are per-neuron IP parameters updated each timestep (same equation as reservoirpy's
    IPReservoir, and as this reservoir in 2025). Fix: from 2026-04-06 to 2026-10 the forward step returned f(r[t+1])
    without a, b, so that the IP parameters had no effect at all (IP + local rule was the local rule alone).

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
    rule_states : {"0.3", "0.4"}, default "0.3"
        States seen by the local rule, and warmup.
        "0.3": as this reservoir with reservoirpy 0.3 (IPSPReservoir, results of 2025 - March 2026): pre- and
        post-synaptic states are the internal state and the output after the step, and the warmup steps are run
        (state updated, no learning) before learning.
        "0.4": as reservoirpy >= 0.4.3's LocalPlasticityReservoir: internal states before (pre) and after (post) the
        step, warmup steps skipped.

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
        rule_states: Literal["0.3", "0.4"] = "0.3",
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
        if rule_states not in ("0.3", "0.4"):
            raise ValueError(f"rule_states must be '0.3' or '0.4', got {rule_states!r}.")
        self.rule_states = rule_states
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
        f = self.activation
        lr = self.lr

        s = np.asarray(state["out"]).ravel()
        x = np.asarray(x).ravel()

        ws = W @ s
        wx = Win @ x

        if hasattr(ws, "toarray"):
            ws = ws.toarray().ravel()
        else:
            ws = np.asarray(ws).ravel()

        if hasattr(wx, "toarray"):
            wx = wx.toarray().ravel()
        else:
            wx = np.asarray(wx).ravel()

        if hasattr(self.bias, "toarray"):
            bias = self.bias.toarray().ravel()
        else:
            bias = np.asarray(self.bias).ravel()

        internal = np.asarray(state["internal"]).ravel()
        next_state = ws + wx + bias
        next_state = (1 - lr) * internal + lr * next_state

        # IP-adjusted activation
        return {"internal": next_state, "out": f(self.a * next_state + self.b)}

    # ------------------------------------------------------------------
    #  IP update
    # ------------------------------------------------------------------

    def _ip_update(self, r: np.ndarray, y: np.ndarray):
        """Update IP parameters a and b for one timestep (r: internal state, y: output), with reservoirpy's IP
        gradients (reservoirpy.nodes.IPReservoir, same in 0.3 and 0.4). Fix: from 2026-02 to 2026-10, db had the
        opposite sign and da was 1 / a + db * r (no learning rate on 1 / a)."""
        eta = self.ip_learning_rate
        mu = self.mu
        sigma = self.sigma

        if self.activation_type == "tanh":
            db = -eta * (
                -mu / (sigma ** 2)
                + y / (sigma ** 2) * (2 * sigma ** 2 + 1 - y ** 2 + mu * y)
            )
        else:  # sigmoid
            db = eta * (1.0 - (2.0 + 1.0 / mu) * y + (y ** 2) / mu)

        da = eta / self.a + db * r

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
                if self.rule_states == "0.3":
                    pre, post = post_internal, post_output
                else:
                    pre, post = pre_state, post_internal
                rows, cols, data = sp.find(self.W)
                self.W[rows, cols] += increment(data, pre[cols], post[rows])

                if do_norm:
                    row_norms = np.sqrt(np.sum(self.W ** 2, axis=1)).reshape(-1, 1)
                    safe_norms = np.where(row_norms > 0, row_norms, 1)
                    self.W[:] /= safe_norms[:]

        sequences = list(x) if is_multiseries(x) else [x]
        if self.rule_states == "0.3":
            # reservoirpy 0.3: the warmup steps of every sequence are run first (no learning), then the epochs
            for seq in sequences:
                for u in seq[:warmup]:
                    self.state = self._step(self.state, u)

        for _epoch in range(self.epochs):
            for seq in sequences:
                _train_sequence(seq[warmup:])

        return self