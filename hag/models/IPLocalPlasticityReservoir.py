"""
IPLocalPlasticityReservoir: intrinsic plasticity (IP) and a local synaptic plasticity rule learned together.

Built from reservoirpy's two nodes: the forward step, the IP parameters (a, b) and their gradients come from
IPReservoir, the local rule (Oja, anti-Oja, Hebbian, anti-Hebbian, BCM) from LocalPlasticityReservoir; only the
learning loop, which applies both at each time step, is defined here.
"""
from typing import Literal

import numpy as np
import scipy.sparse as sp
from reservoirpy.nodes import IPReservoir, LocalPlasticityReservoir
from reservoirpy.type import NodeInput, is_multiseries
from reservoirpy.utils.data_validation import check_node_input


class IPLocalPlasticityReservoir(IPReservoir, LocalPlasticityReservoir):
    """
    Reservoir learning intrinsic plasticity and a local synaptic rule at the same time. Forward step of IPReservoir:

    .. math::

        r[t+1] &= (1-lr) r[t] + lr (W x[t] + W_{in} u[t+1] + bias) \\\\
        x[t+1] &= f(a \\cdot r[t+1] + b)

    At each time step of ``fit``: IP update of ``a`` and ``b`` (IPReservoir's gradients), then local rule update of
    the nonzero weights of ``W`` (LocalPlasticityReservoir's rule).

    Parameters: those of LocalPlasticityReservoir (``local_rule``, ``eta``, ``bcm_theta``, ``synapse_normalization``,
    ``epochs``, reservoir parameters) and of IPReservoir (``mu``, ``sigma``, ``activation`` "tanh" or "sigmoid"), plus:

    ip_learning_rate : float, default 1e-3
        Learning rate of the IP updates (IPReservoir's ``learning_rate``).
    rule_states : {"0.3", "0.4"}, default "0.3"
        States seen by the local rule, and warmup.
        "0.3": as this reservoir with reservoirpy 0.3 (IPSPReservoir, results of 2025 - March 2026): pre- and
        post-synaptic states are the internal state and the output after the step, and the warmup steps are run
        (state updated, no learning) before learning.
        "0.4": as reservoirpy >= 0.4.3's LocalPlasticityReservoir: internal states before (pre) and after (post) the
        step, warmup steps skipped.
    """

    def __init__(
        self,
        units: int = None,
        local_rule: Literal["oja", "anti-oja", "hebbian", "anti-hebbian", "bcm"] = "oja",
        eta: float = 1e-3,
        bcm_theta: float = 0.0,
        synapse_normalization: bool = False,
        mu: float = 0.0,
        sigma: float = 1.0,
        ip_learning_rate: float = 1e-3,
        rule_states: Literal["0.3", "0.4"] = "0.3",
        epochs: int = 1,
        activation: Literal["tanh", "sigmoid"] = "tanh",
        **kwargs,
    ):
        if rule_states not in ("0.3", "0.4"):
            raise ValueError(f"rule_states must be '0.3' or '0.4', got {rule_states!r}.")
        LocalPlasticityReservoir.__init__(self, units=units, local_rule=local_rule, eta=eta, bcm_theta=bcm_theta,
                                          synapse_normalization=synapse_normalization, epochs=epochs,
                                          activation=activation, **kwargs)
        IPReservoir.__init__(self, units=units, mu=mu, sigma=sigma, learning_rate=ip_learning_rate, epochs=epochs,
                             activation=activation, **kwargs)
        self.rule_states = rule_states
        self.activation_type = activation  # activation assumed by the IP rule

    @property
    def ip_learning_rate(self) -> float:
        return self.learning_rate

    @ip_learning_rate.setter
    def ip_learning_rate(self, value: float):
        self.learning_rate = value

    def fit(self, x: NodeInput, y=None, warmup: int = 0) -> "IPLocalPlasticityReservoir":
        check_node_input(x, expected_dim=self.input_dim)
        if not self.initialized:
            self.initialize(x)
        sequences = list(x) if is_multiseries(x) else [x]
        if self.rule_states == "0.3" and warmup:
            # reservoirpy 0.3: the warmup steps of every sequence are run first (no learning), then the epochs
            for seq in sequences:
                self.run(seq[:warmup])

        for _epoch in range(self.epochs):
            for seq in sequences:
                for u in seq[warmup:]:
                    pre_state = self.state["internal"]
                    output = self.step(u)
                    internal = self.state["internal"]
                    # IP update, as IPReservoir.partial_fit
                    delta_a, delta_b = self.gradient(x=internal, y=output, a=self.a)
                    self.a += self.learning_rate * delta_a
                    self.b += self.learning_rate * delta_b
                    # local rule update of the nonzero weights, as LocalPlasticityReservoir.fit
                    pre, post = (internal, output) if self.rule_states == "0.3" else (pre_state, internal)
                    rows, cols, data = sp.find(self.W)
                    self.W[rows, cols] += self.increment(data, pre[cols], post[rows])
                    if self.synapse_normalization:
                        norms = np.sqrt(np.sum(self.W ** 2, axis=1)).reshape(-1, 1)
                        self.W[:] /= np.where(norms > 0, norms, 1)
        return self
