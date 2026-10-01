"""
HAG reservoir: a reservoirpy node whose recurrent connections are grown and pruned by homeostatic structural
plasticity (HAG) when it is fitted, then run as a usual reservoir.

Self-contained (numpy and reservoirpy only), to be proposed to reservoirpy. With the same matrices, data and random
draws, fit gives exactly the same W as HAG's reference implementation, hag.hag.hag.run_algorithm.

    >>> from reservoirpy.nodes import Ridge
    >>> reservoir = HAGReservoir(units=12 * 42, target=0.8, spread=0.1, weight_increment=0.05, min_window=5,
    ...                          max_window=50, input_scaling=0.1, bias_scaling=0.1, seed=0)
    >>> reservoir.fit(pretrain_sequences)       # unsupervised, on the inputs only
    >>> reservoir.reset()                       # run starts every sequence from the current state
    >>> readout = Ridge(ridge=1e-6).fit(reservoir.run(X_train), Y_train)
    >>> reservoir.reset()
    >>> Y_pred = readout.run(reservoir.run(X_test))
"""
from typing import Callable, Literal, Optional, Sequence, Union

import numpy as np
from reservoirpy.activationsfunc import get_function, tanh
from reservoirpy.mat_gen import zeros
from reservoirpy.node import TrainableNode
from reservoirpy.type import NodeInput, State, Timeseries, Timestep, Weights, is_multiseries
from reservoirpy.utils.data_validation import check_node_input
from reservoirpy.utils.random import rand_generator

Rng = Union[np.random.Generator, np.random.RandomState]


def block_input(units: int, input_dim: int, input_scaling: Union[float, Sequence] = 1.0, dtype=np.float64,
                seed: Optional[Union[int, Rng]] = None) -> np.ndarray:
    """Input matrix of HAG: input feature k drives its own block of units // input_dim neurons, with weights drawn
    uniformly in [0, 1) (times input_scaling)."""
    if units % input_dim:
        raise ValueError(f"units ({units}) must be a multiple of the input dimension ({input_dim}): each input feature "
                         f"drives a block of units // input_dim neurons.")
    rng = rand_generator(seed) if not isinstance(seed, np.random.RandomState) else seed
    block = units // input_dim
    Win = np.zeros((units, input_dim), dtype=dtype)
    for k in range(input_dim):
        Win[k * block:(k + 1) * block, k] = rng.uniform(0, 1, block)
    return Win * np.asarray(input_scaling)


def positive_normal_bias(units: int, bias_scaling: float = 1.0, dtype=np.float64,
                         seed: Optional[Union[int, Rng]] = None) -> np.ndarray:
    """Bias of HAG: |N(0.1, 0.1)| (times bias_scaling)."""
    rng = rand_generator(seed) if not isinstance(seed, np.random.RandomState) else seed
    return (np.abs(rng.normal(0.1, 0.1, units)) * bias_scaling).astype(dtype)


class HAGReservoir(TrainableNode):
    """
    Reservoir whose recurrent connections are learned by HAG, a homeostatic structural plasticity rule: during
    ``fit``, the neurons whose activity is too high lose incoming connections and the ones whose activity is too low
    gain incoming connections from the neurons most correlated with them. The reservoir is then run with the usual
    equation, its weights being fixed:

    .. math::

        x[t+1] = (1 - lr) x[t] + lr f(W x[t] + W_{in} u[t+1] + b)

    **Fit.** The reservoir is run on the inputs from a random state (uniform in :math:`[0, 1]`). After each window of
    :math:`T` time steps (``T`` drawn among ``min_window`` ... ``max_window``, log-spaced, or one whole sequence
    if ``use_full_instance``), each neuron's activity on the window is compared to the target:

    .. math::

        \\Delta z_i = (a_i - \\mathrm{target}) / \\mathrm{spread}

    where :math:`a_i` is the mean (``homeostasis="mean"``) or the standard deviation (``homeostasis="variance"``) of
    the state of neuron :math:`i` over the window. Then:

    - :math:`\\Delta z_i \\geq 1` (too active): one incoming connection of :math:`i`, drawn at random, is decreased by
      ``weight_increment`` (and removed when it reaches 0),
    - :math:`\\Delta z_i \\leq -1` (not active enough): the connection from the neuron :math:`j` most correlated with
      :math:`i` over the window (Pearson correlation), among the other neurons not active enough, is increased by
      ``weight_increment`` (created if absent).

    With ``homeostasis="variance"``, the incoming weights of the neurons saturated (state :math:`\\geq`
    ``intrinsic_saturation``) on the whole window are also multiplied by ``intrinsic_coef``.

    Parameters
    ----------
    units : int, optional
        Number of reservoir units. If None, inferred from ``W``.
    homeostasis : {"mean", "variance"}, default to "mean"
        Activity regulated by the plasticity: mean (mean HAG) or standard deviation (variance HAG) of the states.
    target : float
        Target activity (mean or standard deviation of the states).
    spread : float
        Tolerance around the target: connections change when the activity is more than ``spread`` away from it.
    weight_increment : float
        Weight added to (or removed from) a connection at each change.
    min_window : int
        Minimal number of time steps between two plasticity steps.
    max_window : int, optional
        Maximal number of time steps between two plasticity steps (default: ``min_window``).
    use_full_instance : bool, default to False
        If True and the data is a list of sequences, one plasticity step per sequence instead of windows of random
        lengths.
    max_partners : int, default to infinity
        Maximal number of incoming connections of a neuron: beyond it, only existing connections are strengthened.
    intrinsic_saturation : float, default to 0.9
        Saturation threshold of the states (``homeostasis="variance"`` only).
    intrinsic_coef : float, default to 0.9
        Factor applied to the incoming weights of saturated neurons (``homeostasis="variance"`` only).
    lr : float or array-like of shape (units,), default to 1.0
        Neurons leak rate. Must be in :math:`[0, 1]`.
    input_scaling : float or array-like of shape (features,), default to 1.0
        Input gain, used by the default ``Win`` initializer.
    bias_scaling : float, default to 1.0
        Bias gain, used by the default ``bias`` initializer.
    W : callable or array-like of shape (units, units), default to :py:func:`~reservoirpy.mat_gen.zeros`
        Initial recurrent weights (HAG grows them from an empty matrix by default).
    Win : callable or array-like of shape (units, features), default to :py:func:`block_input`
        Input weights: by default, each input feature drives its own block of ``units // features`` neurons.
    bias : callable or array-like of shape (units,), default to :py:func:`positive_normal_bias`
        Bias weights.
    activation : str or callable, default to tanh
        Activation function of the units.
    input_dim : int, optional
        Input dimension. Can be inferred at first call.
    seed : int or :py:class:`numpy.random.Generator`, optional
        Seed of the initializers and of the random choices of ``fit``.
    rng : :py:class:`numpy.random.Generator` or :py:class:`numpy.random.RandomState`, optional
        Random generator of the random choices of ``fit`` (default: derived from ``seed``).
        ``np.random.mtrand._rand`` uses numpy's global random state.
    dtype : Numpy dtype, default to np.float64
        Numerical type for node parameters.
    name : str, optional
        Node name.
    """

    def __init__(
        self,
        units: Optional[int] = None,
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
        lr: Union[float, np.ndarray] = 1.0,
        input_scaling: Union[float, Sequence] = 1.0,
        bias_scaling: float = 1.0,
        W: Union[Weights, Callable] = zeros,
        Win: Union[Weights, Callable] = block_input,
        bias: Union[Weights, Callable] = positive_normal_bias,
        activation: Union[str, Callable] = tanh,
        input_dim: Optional[int] = None,
        seed: Optional[Union[int, np.random.Generator]] = None,
        rng: Optional[Rng] = None,
        dtype: type = np.float64,
        name: Optional[str] = None,
    ):
        if homeostasis not in ("mean", "variance"):
            raise ValueError(f"Unknown homeostasis {homeostasis!r}. Choose from: ['mean', 'variance'].")
        missing = [k for k, v in dict(target=target, spread=spread, weight_increment=weight_increment,
                                      min_window=min_window).items() if v is None]
        if missing:
            raise ValueError(f"HAGReservoir needs {', '.join(missing)}.")
        if units is None and callable(W):
            raise ValueError("'units' parameter must not be None if 'W' parameter is not a matrix.")
        if units is not None and not callable(W) and np.shape(W)[-1] != units:
            raise ValueError(f"Both 'units' and 'W' are set but their dimensions doesn't match: "
                             f"{units} != {np.shape(W)[-1]}.")

        self.homeostasis = homeostasis
        self.target, self.spread, self.weight_increment = target, spread, weight_increment
        self.min_window = min_window
        self.max_window = min_window if max_window is None else max_window
        self.use_full_instance = use_full_instance
        self.max_partners = max_partners
        self.intrinsic_saturation, self.intrinsic_coef = intrinsic_saturation, intrinsic_coef
        self.lr = lr
        self.input_scaling, self.bias_scaling = input_scaling, bias_scaling
        self.W, self.Win, self.bias = W, Win, bias
        self.activation = get_function(activation) if isinstance(activation, str) else activation
        self.dtype = dtype
        self.seed = rand_generator(seed)
        self.rng = rng
        self.name = name
        self.units = self.output_dim = units if units is not None else np.shape(W)[-1]
        self.input_dim = np.shape(Win)[-1] if not callable(Win) else input_dim
        # numbers of connections added / removed by the last fit
        self.n_added = self.n_pruned = 0

    def initialize(self, x: Optional[Union[NodeInput, Timestep]] = None, y: None = None):
        self._set_input_dim(x)
        [Win_rng, W_rng, bias_rng, fit_rng] = self.seed.spawn(4)
        if callable(self.Win):
            self.Win = self.Win(self.units, self.input_dim, input_scaling=self.input_scaling, dtype=self.dtype,
                                seed=Win_rng)
        if callable(self.W):
            self.W = self.W(self.units, self.units, dtype=self.dtype, seed=W_rng)
        if callable(self.bias):
            self.bias = self.bias(self.units, bias_scaling=self.bias_scaling, dtype=self.dtype, seed=bias_rng)
        # dense copy: the plasticity changes W in place
        self.W = np.array(self.W.toarray() if hasattr(self.W, "toarray") else self.W, dtype=self.dtype)
        self.Win = np.asarray(self.Win.toarray() if hasattr(self.Win, "toarray") else self.Win, dtype=self.dtype)
        self.bias = np.ravel(self.bias.toarray() if hasattr(self.bias, "toarray") else self.bias).astype(self.dtype)
        if self.rng is None:
            self.rng = fit_rng
        self.state = {"out": np.zeros((self.units,))}
        self.initialized = True

    def _step(self, state: State, x: Timestep) -> State:
        s = state["out"]
        next_state = self.activation(self.W @ s + self.Win @ x + self.bias)
        return {"out": (1 - self.lr) * s + self.lr * next_state}

    def _run_window(self, state: State, inputs: Timeseries) -> tuple[State, np.ndarray]:
        states = np.empty((len(inputs), self.units))
        for t, u in enumerate(inputs):
            state = self._step(state, u)
            states[t] = state["out"]
        return state, states

    # ------------------------------------------------------------------
    #  Plasticity
    # ------------------------------------------------------------------

    def _activity_error(self, states: np.ndarray) -> np.ndarray:
        """Delta z of each neuron: (activity - target) / spread, activity = mean or standard deviation over the
        window."""
        if self.homeostasis == "mean":
            return np.mean((states - self.target) / self.spread, axis=0)
        return (np.std(states, axis=0) - self.target) / self.spread

    def _change(self, i: int, j: int, value: float):
        # connection j -> i, never negative
        self.W[i, j] = max(self.W[i, j] + value, 0)

    def _new_partners(self, neurons: np.ndarray, states: np.ndarray) -> list:
        """(neuron, presynaptic neuron) of the new connections of the neurons not active enough, chosen among the
        other neurons not active enough (states: (units, T)). Computed for all the neurons before any change."""
        pool = list(neurons)
        if len(pool) <= 1:
            return []
        with np.errstate(divide="ignore", invalid="ignore"):
            correlations = np.corrcoef(states[:, 1:])
        pairs = []
        for neuron in neurons:
            partners = self.W[neuron].nonzero()[0]
            # beyond max_partners, only the existing connections are strengthened
            available = partners if len(partners) >= self.max_partners else [n for n in pool if n != neuron]
            # the most correlated neurons (ties drawn at random)
            scores = correlations[neuron, available]
            candidates = np.array(available)[np.isclose(scores, np.nanmax(scores))]
            if candidates.size == 0:
                raise ValueError(f"No candidate presynaptic neuron for neuron {neuron}: undefined correlations.")
            pairs.append((neuron, self.rng.choice(candidates)))
        return pairs

    def _plasticity(self, states: np.ndarray):
        """One plasticity step on the states (T, units) of a window."""
        delta_z = self._activity_error(states)
        neurons = np.arange(self.units)
        # too active: one incoming connection, drawn at random, is weakened
        for neuron in neurons[delta_z >= 1]:
            partners = self.W[neuron].nonzero()[0]
            if len(partners) > 0:
                self._change(neuron, self.rng.choice(partners), -self.weight_increment)
                self.n_pruned += 1
        # not active enough: one incoming connection from a correlated neuron is strengthened
        for neuron, partner in self._new_partners(neurons[delta_z <= -1], states.T):
            self._change(neuron, partner, self.weight_increment)
            self.n_added += 1
        if self.homeostasis == "variance":
            # intrinsic homeostatic plasticity: weaker inputs for the neurons saturated on the whole window
            self.W[np.all(states >= self.intrinsic_saturation, axis=0)] *= self.intrinsic_coef

    def fit(self, x: NodeInput, y: None = None, warmup: int = 0) -> "HAGReservoir":
        """HAG on x (one sequence (T, input_dim) or a list of sequences), the first warmup steps of each sequence
        being dropped. Leaves the node in its last state."""
        check_node_input(x, expected_dim=self.input_dim)
        if not self.initialized:
            self.initialize(x)
        multiple = is_multiseries(x)
        sequences = [np.asarray(seq, dtype=self.dtype)[warmup:] for seq in (x if multiple else [x])]
        self.n_added = self.n_pruned = 0

        state = {"out": self.rng.uniform(0, 1, self.units)}
        if self.use_full_instance and multiple:
            # the first 3 sequences initialize the state, then one plasticity step per sequence
            state, _ = self._run_window(state, np.concatenate(sequences[:3]))
            for seq in sequences[3:]:
                state, states = self._run_window(state, seq)
                self._plasticity(states)
        else:
            # sequences concatenated, initialization on 5 * min_window steps, then windows of random lengths
            inputs = np.concatenate(sequences)
            init_length = 5 * self.min_window
            state, _ = self._run_window(state, inputs[:init_length])
            inputs = inputs[init_length:]
            lengths = np.unique(np.round(np.logspace(np.log10(self.min_window), np.log10(self.max_window),
                                                     num=10)).astype(int))
            while len(inputs) > self.max_window:
                T = self.rng.choice(lengths)
                state, states = self._run_window(state, inputs[:T])
                inputs = inputs[T:]
                self._plasticity(states)
        self.state = state
        return self
