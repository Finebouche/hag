"""
JAX version of reservoirpy.nodes.HAGReservoir (same algorithm and parameters), the plasticity steps of fit being
compiled with jax.lax.scan. Statistically identical to the NumPy version, but not equal for a given seed (the random
choices are drawn with Jax's random generator).
"""
from functools import partial
from typing import Callable, Literal, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np
from reservoirpy import mat_gen as np_mat_gen
from reservoirpy.jax.activationsfunc import get_function, tanh
from reservoirpy.jax.mat_gen import JaxInitializer, zeros
from reservoirpy.jax.node import TrainableNode
from reservoirpy.jax.type import NodeInput, State, Timestep, Weights, is_array, is_multiseries
from reservoirpy.jax.utils import rand_generator
from reservoirpy.utils.data_validation import check_node_input

block_input = JaxInitializer(np_mat_gen._block_input, allow_sr=False)

# the number of windows is padded (empty windows) to a multiple of WINDOW_BUCKET: fit is compiled again only when it
# changes of bucket, not for each draw of the windows lengths
WINDOW_BUCKET = 64


@partial(jax.jit, static_argnames=("activation",))
def _forward(W, Win, bias, lr, state, x, activation):
    """Next state of the reservoir (matrices passed as arguments: they change during fit)."""
    return (1 - lr) * state + lr * activation(W @ state + Win @ x + bias)


@partial(jax.jit, static_argnames=("activation",))
def _run_inputs(W, Win, bias, lr, state, inputs, activation):
    """Last state of the reservoir after running inputs (timesteps, input_dim)."""

    def step(s, u):
        return _forward(W, Win, bias, lr, s, u, activation), None

    state, _ = jax.lax.scan(step, state, inputs)
    return state


def _random_choice(key, mask):
    """Column drawn uniformly among the True elements of each row of mask (0 for the empty rows): the k-th True
    element, k drawn at random (one draw per row instead of one per element)."""
    counts = jnp.sum(mask, axis=1)
    k = jnp.minimum(jnp.floor(jax.random.uniform(key, (mask.shape[0],)) * counts), counts - 1)
    return jnp.argmax(jnp.cumsum(mask, axis=1) > k[:, None], axis=1)


@partial(jax.jit, static_argnames=("activation", "homeostasis"))
def _fit_windows(
    W,
    Win,
    bias,
    lr,
    state,
    key,
    windows,
    lengths,
    target,
    spread,
    weight_increment,
    max_partners,
    intrinsic_saturation,
    intrinsic_coef,
    activation,
    homeostasis,
):
    """HAG on windows (n_windows, max_length, input_dim) of lengths (n_windows,), zero-padded: one plasticity step per
    window (none for the empty windows). Returns W, the last state, and the numbers of added and removed (weakened) connections."""
    units = W.shape[0]
    steps = jnp.arange(windows.shape[1])
    rows = jnp.arange(units)
    not_self = ~jnp.eye(units, dtype=bool)

    def window_step(carry, window_and_length):
        W, state, key, n_added, n_pruned = carry
        window, length = window_and_length
        key, prune_key, add_key = jax.random.split(key, 3)

        # run the reservoir on the window (the padding steps leave the state unchanged)
        def run(s, t_u):
            t, u = t_u
            s = jnp.where(t < length, _forward(W, Win, bias, lr, s, u, activation), s)
            return s, s

        state, states = jax.lax.scan(run, state, (steps, window))
        valid = (steps < length)[:, None]
        n = length.astype(states.dtype)
        mean = jnp.sum(jnp.where(valid, states, 0.0), axis=0) / n
        if homeostasis == "mean":
            activity = mean
        else:
            activity = jnp.sqrt(jnp.sum(jnp.where(valid, (states - mean) ** 2, 0.0), axis=0) / n)
        delta_z = (activity - target) / spread

        # too active: one incoming connection, drawn at random, is weakened
        connected = W != 0
        prune = (delta_z >= 1) & jnp.any(connected, axis=1)
        partner = _random_choice(prune_key, connected)
        old = W[rows, partner]
        W = W.at[rows, partner].set(jnp.where(prune, jnp.maximum(old - weight_increment, 0.0), old))

        # not active enough: one incoming connection from the most correlated neuron (Pearson correlation over the
        # window without its first timestep, among the other neurons not active enough, ties drawn at random)
        pool = delta_z <= -1
        corr_valid = ((steps >= 1) & (steps < length))[:, None]
        corr_mean = jnp.sum(jnp.where(corr_valid, states, 0.0), axis=0) / jnp.sum(corr_valid)
        centered = jnp.where(corr_valid, states - corr_mean, 0.0)
        cov = centered.T @ centered
        std = jnp.sqrt(jnp.diag(cov))
        correlations = jnp.clip(cov / jnp.outer(std, std), -1.0, 1.0)  # NaN where a neuron has a constant activity

        connected = W != 0
        # beyond max_partners, only the existing connections are strengthened
        available = jnp.where(
            (jnp.sum(connected, axis=1) >= max_partners)[:, None], connected, pool[None, :] & not_self
        )
        scores = jnp.where(available, correlations, jnp.nan)
        best = jnp.nanmax(scores, axis=1)
        ties = available & jnp.isclose(scores, best[:, None])
        # undefined correlations: no new connection
        add = pool & jnp.any(ties, axis=1) & (jnp.sum(pool) > 1)
        partner = _random_choice(add_key, ties)
        W = W.at[rows, partner].add(jnp.where(add, weight_increment, 0.0))

        if homeostasis == "variance":
            # intrinsic homeostatic plasticity: weaker inputs for the neurons saturated on the whole window
            saturated = jnp.all(jnp.where(valid, states >= intrinsic_saturation, True), axis=0) & (length > 0)
            W = jnp.where(saturated[:, None], W * intrinsic_coef, W)

        return (W, state, key, n_added + jnp.sum(add), n_pruned + jnp.sum(prune)), None

    carry = (W, state, key, jnp.array(0), jnp.array(0))
    (W, state, _, n_added, n_pruned), _ = jax.lax.scan(window_step, carry, (windows, lengths))
    return W, state, n_added, n_pruned


class HAGReservoir(TrainableNode):
    """
    Jax version of :py:class:`reservoirpy.nodes.HAGReservoir`: a reservoir that
    learns its recurrent connections through HAG [1]_, a homeostatic structural
    plasticity rule [2]_.

    Same algorithm and parameters as the NumPy version, whose documentation
    describes the rule. The plasticity steps of ``fit`` are compiled with
    :py:func:`jax.lax.scan`: the windows of timesteps are built as in the NumPy
    version, zero-padded to the same length, and the random choices (connection to
    weaken, ties between the most correlated neurons) are drawn with Jax's random
    generator. The results are therefore statistically identical to those of the
    NumPy version, but not equal for a given seed.

    Reservoir states are updated with the equation:

    .. math::

        x[t+1] = (1 - lr) x[t] + lr f(W x[t] + W_{in} u[t+1] + b)

    Parameters
    ----------
    units : int, optional
        Number of reservoir units. If None, the number of units will be inferred from
        the ``W`` matrix shape.
    homeostasis : {"mean", "variance"}, default to "mean"
        Activity regulated by the plasticity: mean (mean HAG) or standard deviation
        (variance HAG) of the states.
    target : float
        Target activity (mean or standard deviation of the states).
    spread : float
        Tolerance around the target: connections change when the activity is more
        than ``spread`` away from it.
    weight_increment : float
        Weight added to (or removed from) a connection at each change. Must be
        positive: the weights of HAG are never negative.
    min_window : int
        Minimal number of timesteps between two plasticity steps. Must be at least 3:
        the correlations are computed over the window without its first timestep.
    max_window : int, optional
        Maximal number of timesteps between two plasticity steps. If None,
        ``min_window`` is used.
    use_full_instance : bool, default to False
        If True and the data is a list of sequences, one plasticity step per
        sequence instead of windows of random lengths.
    max_partners : int, default to infinity
        Maximal number of incoming connections of a neuron: beyond it, only its
        existing connections are strengthened.
    intrinsic_saturation : float, default to 0.9
        Saturation threshold of the states (``homeostasis="variance"`` only).
    intrinsic_coef : float, default to 0.9
        Factor applied to the incoming weights of saturated neurons
        (``homeostasis="variance"`` only).
    lr : float or array-like of shape (units,), default to 1.0
        Neurons leak rate. Must be in :math:`[0, 1]`.
    input_scaling : float or array-like of shape (features,), default to 1.0.
        Input gain, used by the default ``Win`` initializer.
    bias_scaling : float, default to 1.0
        Bias gain, used by the default ``bias`` initializer.
    W : callable or array-like of shape (units, units), default to :py:func:`~reservoirpy.jax.mat_gen.zeros`
        Initial recurrent weights matrix or initializer (HAG grows the connections
        from an empty matrix by default).
    Win : callable or array-like of shape (units, features), default to :py:func:`~reservoirpy.mat_gen.block_input`
        Input weights matrix or initializer. By default, each input feature drives
        its own block of ``units // features`` neurons.
    bias : callable or array-like of shape (units,), default to ``random_sparse(dist="foldnorm", c=1.0, scale=0.1)``
        Bias weights vector or initializer. By default, :math:`|\\mathcal{N}(0.1, 0.1)|`.
    activation : str or callable, default to :py:func:`~reservoirpy.jax.activationsfunc.tanh`
        Reservoir units activation function.
    input_dim : int, optional
        Input dimension. Can be inferred at first call.
    seed : int or :py:class:`numpy.random.Generator`, optional
        A random state seed, for the initializers and the random choices of ``fit``.
    dtype : Numpy dtype, default to jnp.float64
        Numerical type for node parameters.
    name : str, optional
        Node name.

    References
    ----------

    .. [1] Cazalets, T., & Dambre, J. (2026). Reshaping reservoirs with
           unsupervised Hebbian adaptation. Nature Communications, 17, 450.
           https://doi.org/10.1038/s41467-025-67137-1

    .. [2] Cazalets, T., & Dambre, J. (2023). An homeostatic activity-dependent
           structural plasticity algorithm for richer input combination.
           In 2023 International Joint Conference on Neural Networks (IJCNN)
           (pp. 1-8). IEEE. https://doi.org/10.1109/IJCNN54540.2023.10191230

    Example
    -------
    >>> from hag.models.jax_hag_reservoir import HAGReservoir
    >>> reservoir = HAGReservoir(
    ...     units=120, homeostasis="mean", target=0.8, spread=0.1,
    ...     weight_increment=0.05, min_window=5, max_window=50,
    ...     input_scaling=0.1, bias_scaling=0.1, seed=0,
    ... )
    >>> # Grow the connections on input timeseries (unsupervised)
    >>> reservoir.fit(X_data)
    >>> # Then run, each sequence starting from a zero state
    >>> reservoir.reset()
    >>> states = reservoir.run(X_train)
    """

    #: Number of neuronal units in the reservoir.
    units: int
    #: Activity regulated by the plasticity ("mean" or "variance").
    homeostasis: str
    #: Target activity (mean or standard deviation of the states).
    target: float
    #: Tolerance around the target activity.
    spread: float
    #: Weight added to (or removed from) a connection at each change (positive).
    weight_increment: float
    #: Minimal number of timesteps between two plasticity steps.
    min_window: int
    #: Maximal number of timesteps between two plasticity steps.
    max_window: int
    #: If True, one plasticity step per sequence.
    use_full_instance: bool
    #: Maximal number of incoming connections of a neuron.
    max_partners: float
    #: Saturation threshold of the states (variance HAG).
    intrinsic_saturation: float
    #: Factor applied to the incoming weights of saturated neurons (variance HAG).
    intrinsic_coef: float
    #: Leaking rate (1.0 by default) (:math:`\mathrm{lr}`).
    lr: float
    #: Input scaling (float or array) (1.0 by default).
    input_scaling: Union[float, Sequence]
    #: Bias scaling (1.0 by default).
    bias_scaling: float
    #: Input weights matrix (:math:`\mathbf{W}_{in}`).
    Win: Weights
    #: Recurrent weights matrix (:math:`\mathbf{W}`).
    W: Weights
    #: Bias vector (:math:`\mathbf{b}`).
    bias: Weights
    #: Activation of the reservoir units (tanh by default) (:math:`f`).
    activation: Callable
    #: Type of matrices elements. By default, ``jnp.float64``.
    dtype: type
    #: A random state generator. Used for generating Win, W, bias and the random choices of ``fit``.
    rng: np.random.Generator
    #: Number of connections added by the last ``fit``.
    n_added: int
    #: Number of connections removed (weakened) by the last ``fit``.
    n_pruned: int

    def __init__(
        self,
        units: Optional[int] = None,
        # HAG
        homeostasis: Literal["mean", "variance"] = "mean",
        target: float = None,
        spread: float = None,
        weight_increment: float = None,
        min_window: int = None,
        max_window: Optional[int] = None,
        use_full_instance: bool = False,
        max_partners: float = np.inf,
        intrinsic_saturation: float = 0.9,
        intrinsic_coef: float = 0.9,
        # standard reservoir params
        lr: Union[float, jax.Array] = 1.0,
        input_scaling: Union[float, Sequence] = 1.0,
        bias_scaling: float = 1.0,
        W: Union[Weights, Callable] = zeros,
        Win: Union[Weights, Callable] = block_input,
        bias: Union[Weights, Callable] = JaxInitializer(np_mat_gen._random_sparse)(dist="foldnorm", c=1.0, scale=0.1),
        activation: Union[str, Callable] = tanh,
        input_dim: Optional[int] = None,
        seed: Optional[Union[int, np.random.Generator]] = None,
        dtype: type = jnp.float64,
        name: Optional[str] = None,
    ):
        if homeostasis not in ("mean", "variance"):
            raise ValueError(f"Unknown homeostasis '{homeostasis}'. Choose from: ['mean', 'variance'].")
        missing = [
            name_
            for name_, value in dict(
                target=target, spread=spread, weight_increment=weight_increment, min_window=min_window
            ).items()
            if value is None
        ]
        if missing:
            raise ValueError(f"HAGReservoir needs {', '.join(missing)}.")
        if weight_increment <= 0:
            raise ValueError(f"'weight_increment' must be positive, got {weight_increment}.")
        if min_window < 3:
            raise ValueError(f"'min_window' must be at least 3, got {min_window}.")
        if max_window is not None and max_window < min_window:
            raise ValueError(
                f"'max_window' ({max_window}) must be greater than or equal to 'min_window' ({min_window})."
            )

        self.homeostasis = homeostasis
        self.target = target
        self.spread = spread
        self.weight_increment = weight_increment
        self.min_window = min_window
        self.max_window = min_window if max_window is None else max_window
        self.use_full_instance = use_full_instance
        self.max_partners = max_partners
        self.intrinsic_saturation = intrinsic_saturation
        self.intrinsic_coef = intrinsic_coef
        self.lr = lr
        self.input_scaling = input_scaling
        self.bias_scaling = bias_scaling
        self.Win = Win
        self.W = W
        self.bias = bias
        self.activation = get_function(activation) if isinstance(activation, str) else activation
        self.dtype = dtype
        self.rng = rand_generator(seed=seed)
        self.name = name
        self.n_added = 0
        self.n_pruned = 0

        # set units / output_dim
        if units is None and not is_array(W):
            raise ValueError("'units' parameter must not be None if 'W' parameter is not a matrix.")
        if units is not None and is_array(W) and W.shape[-1] != units:
            raise ValueError(
                f"Both 'units' and 'W' are set but their dimensions doesn't match: " f"{units} != {W.shape[-1]}."
            )
        self.units = units if units is not None else W.shape[-1]
        self.output_dim = self.units

        # set input_dim (if possible)
        if input_dim is not None and is_array(Win) and Win.shape[-1] != input_dim:
            raise ValueError(
                f"Both 'input_dim' and 'Win' are set but their dimensions doesn't "
                f"match: {input_dim} != {Win.shape[-1]}."
            )
        self.input_dim = Win.shape[-1] if is_array(Win) else input_dim

    def initialize(self, x: Optional[Union[NodeInput, Timestep]], y: None = None):

        # set input_dim
        self._set_input_dim(x)

        [Win_rng, W_rng, bias_rng, plasticity_rng] = self.rng.spawn(4)

        if callable(self.Win):
            self.Win = self.Win(
                self.units,
                self.input_dim,
                input_scaling=self.input_scaling,
                dtype=self.dtype,
                seed=Win_rng,
            )

        if callable(self.W):
            self.W = self.W(
                self.units,
                self.units,
                dtype=self.dtype,
                seed=W_rng,
            )

        if callable(self.bias):
            self.bias = self.bias(
                self.units,
                input_scaling=self.bias_scaling,
                dtype=self.dtype,
                seed=bias_rng,
            )

        # dense matrices: the plasticity changes W
        self.W, self.Win, self.bias = (
            jnp.asarray(M.todense() if hasattr(M, "todense") else M) for M in (self.W, self.Win, self.bias)
        )
        self.bias = jnp.ravel(self.bias)

        # random choices of fit: windows lengths (NumPy) and plasticity (Jax)
        self._plasticity_rng = plasticity_rng
        self._key = jax.random.PRNGKey(int(plasticity_rng.integers(2**31)))

        self.state = {"out": jnp.zeros((self.units,))}

        self.initialized = True

    def _step(self, state: State, x: Timestep) -> State:
        return {"out": _forward(self.W, self.Win, self.bias, self.lr, state["out"], x, activation=self.activation)}

    def _windows(self, sequences: list, multiple: bool) -> tuple:
        """(initialization inputs, zero-padded windows (n_windows, max_length, input_dim), lengths (n_windows,)), as
        the NumPy version: one window per sequence (use_full_instance), or windows of random lengths of the
        concatenated sequences."""
        if self.use_full_instance and multiple:
            init, windows = np.concatenate(sequences[:3]), sequences[3:]
        else:
            inputs = np.concatenate(sequences)
            init_length = 5 * self.min_window
            init, inputs = inputs[:init_length], inputs[init_length:]
            lengths = np.unique(
                np.round(np.logspace(np.log10(self.min_window), np.log10(self.max_window), num=10)).astype(int)
            )
            windows = []
            while len(inputs) > self.max_window:
                T = self._plasticity_rng.choice(lengths)
                windows.append(inputs[:T])
                inputs = inputs[T:]
        lengths = np.zeros(WINDOW_BUCKET * -(-len(windows) // WINDOW_BUCKET), dtype=int)
        lengths[: len(windows)] = [len(window) for window in windows]
        padded = np.zeros((len(lengths), max(lengths, default=1), self.input_dim))
        for k, window in enumerate(windows):
            padded[k, : len(window)] = window
        return init, padded, lengths

    def fit(self, x: NodeInput, y: None = None, warmup: int = 0) -> "HAGReservoir":
        """Offline fitting method of the HAG reservoir: grows and prunes the recurrent connections on the input
        timeseries (unsupervised). Leaves the node in its last state.

        Parameters
        ----------
        x : list or array-like of shape ([series, ] timesteps, input_dim)
            Input sequences dataset.
        y : None
            Not used, HAG is unsupervised.
        warmup : int, default to 0
            Number of timesteps to discard at the beginning of each timeseries before training.

        Returns
        -------
        HAGReservoir
            Node trained offline.
        """
        check_node_input(x, expected_dim=self.input_dim)

        if not self.initialized:
            self.initialize(x)

        multiple = is_multiseries(x)
        sequences = [np.asarray(seq, dtype=float)[warmup:] for seq in (x if multiple else [x])]
        init, windows, lengths = self._windows(sequences, multiple)

        self._key, init_key, fit_key = jax.random.split(self._key, 3)
        state = jax.random.uniform(init_key, (self.units,), dtype=self.W.dtype)
        if len(init) > 0:
            state = _run_inputs(self.W, self.Win, self.bias, self.lr, state, jnp.asarray(init, dtype=self.W.dtype), activation=self.activation)
        self.n_added = self.n_pruned = 0
        if len(lengths) > 0:
            self.W, state, n_added, n_pruned = _fit_windows(
                self.W, self.Win, self.bias, self.lr, state, fit_key, jnp.asarray(windows, dtype=self.W.dtype),
                jnp.asarray(lengths), self.target, self.spread, self.weight_increment, self.max_partners,
                self.intrinsic_saturation, self.intrinsic_coef, activation=self.activation,
                homeostasis=self.homeostasis,
            )
            self.n_added, self.n_pruned = int(n_added), int(n_pruned)

        self.state = {"out": state}
        return self
