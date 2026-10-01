"""
JAX versions of the local plasticity reservoirs, in the style of the reservoirpy.jax nodes (the reservoirpy.jax
LocalPlasticityReservoir is not implemented in reservoirpy 0.4.2):

- LocalPlasticityReservoir: reservoirpy.nodes.LocalPlasticityReservoir.
- IPLocalPlasticityReservoir: hag.models.IPLocalPlasticityReservoir.IPLocalPlasticityReservoir.

Both learn with the same local rules and give the same weights as the NumPy versions, the time loop of the learning
being compiled with jax.lax.scan.
"""
from functools import partial
from typing import Callable, Literal, Optional, Union

import jax
import jax.numpy as jnp
from reservoirpy.jax.activationsfunc import get_function
from reservoirpy.jax.node import TrainableNode
from reservoirpy.type import NodeInput, State, Timestep, is_multiseries

# weight increment of each local rule (w: weight, pre / post: pre- and post-synaptic states), as in reservoirpy
LOCAL_RULES = {
    "oja": lambda w, pre, post, eta, theta: eta * post * (pre - post * w),
    "anti-oja": lambda w, pre, post, eta, theta: -eta * post * (pre - post * w),
    "hebbian": lambda w, pre, post, eta, theta: eta * post * pre,
    "anti-hebbian": lambda w, pre, post, eta, theta: -eta * post * pre,
    "bcm": lambda w, pre, post, eta, theta: eta * post * (post - theta) * pre,
}


def _ip_update(a, b, r, y, mu, sigma, learning_rate, ip_activation):
    # reservoirpy's IP gradients, as IPLocalPlasticityReservoir._ip_update (r: internal state, y: output)
    if ip_activation == "tanh":
        db = -learning_rate * (-mu / sigma ** 2 + y / sigma ** 2 * (2 * sigma ** 2 + 1 - y ** 2 + mu * y))
    else:  # sigmoid
        db = learning_rate * (1.0 - (2.0 + 1.0 / mu) * y + y ** 2 / mu)
    return a + (learning_rate / a + db * r), b + db


def _forward(W, Win, bias, lr, internal, out, a, b, u, activation, ip, leak_internal):
    """Next (internal, out): leak on the internal state (reservoirpy 0.3, IP) or on the output (reservoirpy 0.4's
    LocalPlasticityReservoir), activation f(a * internal + b) with IP, f(internal) otherwise."""
    post = (1 - lr) * (internal if leak_internal else out) + lr * (W @ out + Win @ u + bias)
    return post, activation(a * post + b) if ip else activation(post)


@partial(jax.jit, static_argnames=("activation", "ip", "leak_internal"))
def _run_states(W, Win, bias, lr, a, b, internal, out, x, activation, ip, leak_internal):
    """(internal, out) after running x, without learning."""
    def step(carry, u):
        return _forward(W, Win, bias, lr, *carry, a, b, u, activation, ip, leak_internal), None

    (internal, out), _ = jax.lax.scan(step, (internal, out), x)
    return internal, out


@partial(jax.jit, static_argnames=("activation", "rule", "normalize", "update_state", "ip_activation", "rule_states"))
def _learn(Win, bias, lr, eta, theta, ip_parameters, carry, x, activation, rule, normalize, update_state,
           ip_activation, rule_states):
    increment = LOCAL_RULES[rule]
    ip = ip_activation is not None

    def step(carry, u):
        W, internal, out, a, b = carry
        # a and b before this step's IP update
        post, post_out = _forward(W, Win, bias, lr, internal, out, a, b, u, activation, ip,
                                  leak_internal=ip or rule_states == "0.3")
        if rule_states == "0.3":
            # reservoirpy 0.3: post-synaptic = output after the step; pre-synaptic = internal state before the step
            # (local rule alone) or after it (IP + local rule, IPSPReservoir)
            rule_pre, rule_post = (post if ip else internal), post_out
        else:
            rule_pre, rule_post = internal, post
        if update_state:
            internal, out = post, post_out
        if ip:
            a, b = _ip_update(a, b, post, post_out, *ip_parameters, ip_activation)
        # only the existing (nonzero) connexions are updated
        W = jnp.where(W != 0, W + increment(W, rule_pre[None, :], rule_post[:, None], eta, theta), W)
        if normalize:
            norms = jnp.sqrt(jnp.sum(W ** 2, axis=1, keepdims=True))
            W = W / jnp.where(norms > 0, norms, 1)
        return (W, internal, out, a, b), None

    carry, _ = jax.lax.scan(step, carry, x)
    return carry


class LocalPlasticityReservoir(TrainableNode):
    """
    Reservoir whose recurrent weights are adapted by a local plasticity rule (Oja, anti-Oja, Hebbian, anti-Hebbian,
    BCM), as reservoirpy.nodes.LocalPlasticityReservoir.

    Forward step (the leak is applied before the activation):
        internal <- (1 - lr) * out + lr * (W @ out + Win @ x + bias)       ("0.3": (1 - lr) * internal + ...)
        out <- f(internal)
    Learning, at each time step, for the nonzero weights:
        W[i, j] += increment(W[i, j], pre=internal_before[j], post=internal_after[i])   ("0.4")
        W[i, j] += increment(W[i, j], pre=internal_before[j], post=out_after[i])        ("0.3")

    rule_states: "0.4" (default) as reservoirpy >= 0.4.3, warmup steps skipped; "0.3" as reservoirpy 0.3 (and HAG's
    results of 2025 - March 2026): post-synaptic state = output, warmup steps run (no learning) before learning.

    fit_updates_state: True (default) runs the reservoir during learning, as reservoirpy >= 0.4.3. In reservoirpy <= 0.4.2,
    LocalPlasticityReservoir.fit never updated the reservoir state (fixed by reservoirpy PR #250): the pre-synaptic
    state stayed the initial one (zeros) and the post-synaptic state only depended on the input. False reproduces this
    behaviour, e.g. to compare with results obtained with reservoirpy <= 0.4.2.
    """

    def __init__(
        self,
        units: Optional[int] = None,
        W: jax.Array = None,
        Win: jax.Array = None,
        bias: Union[jax.Array, float] = 0.0,
        lr: Union[float, jax.Array] = 1.0,
        activation: Union[str, Callable] = "tanh",
        local_rule: Literal["oja", "anti-oja", "hebbian", "anti-hebbian", "bcm"] = "oja",
        eta: float = 1e-3,
        bcm_theta: float = 0.0,
        synapse_normalization: bool = False,
        epochs: int = 1,
        fit_updates_state: bool = True,
        rule_states: Literal["0.3", "0.4"] = "0.4",
        name: Optional[str] = None,
    ):
        self.W = jnp.asarray(W)
        self.Win = jnp.asarray(Win)
        self.bias = jnp.asarray(bias)
        self.lr = lr
        self.activation = get_function(activation)
        if local_rule not in LOCAL_RULES:
            raise ValueError(f"Unknown learning rule '{local_rule}'. Choose from: {list(LOCAL_RULES)}.")
        self.local_rule = local_rule
        self.eta = eta
        self.bcm_theta = 0.0 if bcm_theta is None else bcm_theta
        self.synapse_normalization = synapse_normalization
        self.epochs = epochs
        self.fit_updates_state = fit_updates_state
        if rule_states not in ("0.3", "0.4"):
            raise ValueError(f"rule_states must be '0.3' or '0.4', got {rule_states!r}.")
        self.rule_states = rule_states
        self.name = name
        if units is not None and units != self.W.shape[-1]:
            raise ValueError(f"Both 'units' and 'W' are set but their dimensions don't match: "
                             f"{units} != {self.W.shape[-1]}.")
        self.units = self.output_dim = self.W.shape[-1]
        self.input_dim = self.Win.shape[-1]

    def initialize(self, x: Optional[Union[NodeInput, Timestep]] = None, y: None = None):
        if x is not None:
            self._set_input_dim(x)
        self.state = {"internal": jnp.zeros((self.units,)), "out": jnp.zeros((self.units,))}
        self.initialized = True

    @partial(jax.jit, static_argnums=(0,))
    def _step(self, state: State, x: Timestep) -> State:
        W = self.W  # NxN
        Win = self.Win  # NxI
        bias = self.bias  # N or float
        f = self.activation
        lr = self.lr
        s = state["internal"] if self.rule_states == "0.3" else state["out"]

        next_state = W @ state["out"] + Win @ x + bias
        next_state = (1 - lr) * s + lr * next_state

        return {"internal": next_state, "out": f(next_state)}

    def _ip(self):
        return None, None, None  # no intrinsic plasticity: (a, b, (mu, sigma, learning rate))

    def fit(self, x: NodeInput, y: None = None, warmup: int = 0) -> "LocalPlasticityReservoir":
        if not self.initialized:
            self.initialize(x)
        a, b, ip_parameters = self._ip()
        ip_activation = getattr(self, "activation_type", None)
        zeros = jnp.zeros((self.units,))
        a, b = zeros if a is None else a, zeros if b is None else b
        sequences = list(x) if is_multiseries(x) else [x]
        internal, out = self.state["internal"], self.state["out"]
        if self.rule_states == "0.3" and warmup:
            # reservoirpy 0.3: the warmup steps of every sequence are run first (no learning), then the epochs
            for seq in sequences:
                internal, out = _run_states(self.W, self.Win, self.bias, self.lr, a, b, internal, out,
                                            jnp.asarray(seq[:warmup]), activation=self.activation,
                                            ip=ip_activation is not None, leak_internal=True)
        carry = (self.W, internal, out, a, b)
        for _epoch in range(self.epochs):
            for seq in sequences:
                carry = _learn(self.Win, self.bias, self.lr, self.eta, self.bcm_theta, ip_parameters, carry,
                               jnp.asarray(seq[warmup:]), activation=self.activation, rule=self.local_rule,
                               normalize=self.synapse_normalization, update_state=self.fit_updates_state,
                               ip_activation=ip_activation, rule_states=self.rule_states)
        self.W, internal, out, a, b = carry
        self.state = {"internal": internal, "out": out}
        if ip_activation is not None:
            self.a, self.b = a, b
        return self


class IPLocalPlasticityReservoir(LocalPlasticityReservoir):
    """
    Local plasticity with intrinsic plasticity (IP) learned at the same time, as
    hag.models.IPLocalPlasticityReservoir.IPLocalPlasticityReservoir (rule_states "0.3" by default, as it):
        internal <- (1 - lr) * internal + lr * (W @ out + Win @ x + bias)
        out <- f(a * internal + b)
    """

    def __init__(self, *args, mu: float = 0.0, sigma: float = 1.0, ip_learning_rate: float = 5e-4,
                 activation_type: Literal["tanh", "sigmoid"] = "tanh", a: jax.Array = None, b: jax.Array = None,
                 rule_states: Literal["0.3", "0.4"] = "0.3", **kwargs):
        super().__init__(*args, rule_states=rule_states, **kwargs)
        # activation_type: activation assumed by the IP rule, as in the NumPy version
        self.mu, self.sigma, self.ip_learning_rate, self.activation_type = mu, sigma, ip_learning_rate, activation_type
        self.a = jnp.ones((self.units,)) if a is None else jnp.asarray(a)
        self.b = jnp.zeros((self.units,)) if b is None else jnp.asarray(b)

    def _ip(self):
        return self.a, self.b, (self.mu, self.sigma, self.ip_learning_rate)

    @partial(jax.jit, static_argnums=(0,))
    def _step(self, state: State, x: Timestep) -> State:
        next_state = self.W @ state["out"] + self.Win @ x + self.bias
        next_state = (1 - self.lr) * state["internal"] + self.lr * next_state
        return {"internal": next_state, "out": self.activation(self.a * next_state + self.b)}
