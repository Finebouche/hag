"""
Online (causal) features of the observations of several environments at once, with a state per environment reset at
the start of each episode:

  observation (D,) -> MinMax scaling to [0, 1] -> causal filter bank (D * F features) -> [reservoir state (units,)]

The filter bank turns each observation component into F signals at different time scales, the multivariate input that
HAG is made for (each feature driving its own block of the reservoir), as the frequency bands of audio signals. Filter
banks (make_filter_bank), linear systems s <- A s + b u applied to each scaled observation component u:
  - "ema": exponential moving averages (EMA) with log-spaced rates, the first one being the observation itself, and the
    differences of successive EMAs (band-pass signals, signed: they keep the direction of the changes),
  - "resonator": damped complex resonators z <- r e^(iω) z + (1 - r) u with log-spaced frequencies ω and a constant
    quality factor (constant-Q spectral decomposition, as the cochlea and the filters behind MFCCs): the observation
    itself, then the real and imaginary parts of the resonators (signed, they keep the phase),
  - "resonator_energy": the observation itself, then the energies |z| of the same resonators (non-negative, as the
    spectral bands of audio signals),
  - "legendre": the observation itself, then the coefficients of the projection of a sliding window of its past on
    Legendre polynomials (Legendre memory unit, Voelker et al., 2019).
Their time scales go from 1 step to 1 / slowest steps: rates of the EMAs from 1 to slowest, frequencies of the
resonators from π/2 (period of 4 steps) to slowest rad/step, window of 1 / slowest steps for the Legendre projection.
As the bands of the audio datasets of HAG, each filter bank feature is then MinMax scaled (bounds fitted on pretraining
episodes, see fit_feature_scaling): otherwise the band-pass signals, much smaller than the observations, leave their
blocks silent.
"""
from typing import Optional

import numpy as np
from scipy.linalg import expm

DECOMPOSITIONS = ("ema", "resonator", "resonator_energy", "legendre")
RESONATOR_Q = 2.0  # quality factor of the resonators: they decay over Q / ω steps


class FilterBank:
    """
    Causal linear filters applied to each observation component: state s (..., S) <- s A^T + u b, features
    (..., n_features) of the state and of the observation component u. At the start of an episode, the state is the
    steady state of the first observation (the response to a constant input).
    """

    def __init__(self, kind: str, A: np.ndarray, b: np.ndarray):
        self.kind = kind
        self.A, self.b = np.asarray(A, dtype=float), np.asarray(b, dtype=float)
        self.steady = np.linalg.solve(np.eye(len(self.b)) - self.A, self.b)  # state for a constant input 1
        self.n_features = self.features(np.zeros((1, len(self.b))), np.zeros(1)).shape[-1]

    def init(self, u: np.ndarray) -> np.ndarray:
        """Steady state (..., S) of the inputs u (...)."""
        return u[..., None] * self.steady

    def step(self, s: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Next state (..., S) of the states s (..., S) with the inputs u (...)."""
        return s @ self.A.T + u[..., None] * self.b

    def features(self, s: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Features (..., n_features) of the states s (..., S) and inputs u (...)."""
        if self.kind == "ema":
            # low-pass (first EMA = the observation) and band-pass signals
            return np.concatenate([s[..., :1], s[..., :-1] - s[..., 1:]], axis=-1)
        if self.kind == "resonator_energy":
            s = np.hypot(s[..., 0::2], s[..., 1::2])
        return np.concatenate([u[..., None], s], axis=-1)


def make_filter_bank(decomposition: str = "ema", n_filters: int = 8, slowest: float = 0.02) -> FilterBank:
    """Filter bank of n_filters filters (EMAs, resonators or Legendre coefficients), time scales from 1 to 1 / slowest
    steps."""
    if decomposition == "ema":
        rates = np.geomspace(1.0, slowest, n_filters)
        return FilterBank("ema", np.diag(1 - rates), rates)
    if decomposition in ("resonator", "resonator_energy"):
        omegas = np.geomspace(np.pi / 2, slowest, n_filters)
        radii = np.exp(-omegas / RESONATOR_Q)
        A, b = np.zeros((2 * n_filters, 2 * n_filters)), np.zeros(2 * n_filters)
        for k, (omega, r) in enumerate(zip(omegas, radii)):
            # (real, imaginary) parts of z <- r e^(iω) z + (1 - r) u
            A[2 * k:2 * k + 2, 2 * k:2 * k + 2] = r * np.array([[np.cos(omega), -np.sin(omega)],
                                                                 [np.sin(omega), np.cos(omega)]])
            b[2 * k] = 1 - r
        return FilterBank(decomposition, A, b)
    if decomposition == "legendre":
        # Legendre memory unit: continuous-time system of order n_filters over a window of theta steps, discretized
        # (zero-order hold, one step)
        theta, i = 1 / slowest, np.arange(n_filters)
        A = (2 * i[:, None] + 1) / theta * np.where(i[:, None] < i[None, :], -1.0,
                                                    (-1.0) ** (i[:, None] - i[None, :] + 1))
        B = (2 * i + 1) * (-1.0) ** i / theta
        Ad = expm(A)
        return FilterBank("legendre", Ad, np.linalg.solve(A, (Ad - np.eye(n_filters)) @ B))
    raise ValueError(f"Unknown decomposition {decomposition!r}: {DECOMPOSITIONS}")


class FeaturePipeline:
    """
    Online features of n_envs environments.

    Parameters
    ----------
    low, high : array of shape (D,)
        Bounds of the MinMax scaling of the observations (the scaled observations are clipped to [0, 1]).
    bank : FilterBank, optional
        Filter bank applied to each scaled observation component (see make_filter_bank). If None, the scaled
        observations are the features.
    reservoir : dict, optional
        Reservoir matrices {"W", "Win", "bias"} and leak rate "lr": the features are then the states of the reservoir,
        x <- (1 - lr) x + lr tanh(W x + Win u + bias), u being the filter bank (or scaled observation) features.
    include_input : bool, default to False
        With a reservoir, also return its input features (concatenated to its states).
    n_envs : int, default to 1
        Number of environments.
    """

    def __init__(self, low, high, bank: Optional[FilterBank] = None, reservoir: Optional[dict] = None,
                 include_input: bool = False, n_envs: int = 1):
        self.low = np.asarray(low, dtype=float)
        self.scale = np.maximum(np.asarray(high, dtype=float) - self.low, 1e-12)
        self.bank = bank
        self.reservoir = reservoir
        self.include_input = include_input
        self.feature_low = self.feature_scale = None  # MinMax scaling of the filter bank features
        self.n_inputs = self.low.size * (1 if bank is None else bank.n_features)
        self.n_features = self.n_inputs if reservoir is None else reservoir["W"].shape[0] + include_input * self.n_inputs
        self.set_n_envs(n_envs)

    def _initial_filters(self, n: int) -> np.ndarray:
        return np.zeros((n, self.low.size, 1 if self.bank is None else len(self.bank.b)))

    def set_n_envs(self, n_envs: int):
        self.n_envs = n_envs
        self.filters = self._initial_filters(n_envs)
        self.fresh = np.ones(n_envs, dtype=bool)  # first observation of the episode: filters initialized to it
        self.state = None if self.reservoir is None else np.zeros((n_envs, self.reservoir["W"].shape[0]))

    def reset(self, indices=None):
        """Reset the states of the environments of indices (all by default) for a new episode."""
        indices = slice(None) if indices is None else indices
        self.filters[indices] = 0.0
        self.fresh[indices] = True
        if self.state is not None:
            self.state[indices] = 0.0

    def inputs(self, observations: np.ndarray, filters: np.ndarray, fresh: np.ndarray) -> tuple:
        """(input features, new filter states) of observations (n, D) from the filter states (n, D, S)."""
        u = np.clip((observations - self.low) / self.scale, 0.0, 1.0)
        if self.bank is None:
            return u, filters
        filters = np.where(fresh[:, None, None], self.bank.init(u), self.bank.step(filters, u))
        features = self.bank.features(filters, u).reshape(len(u), -1)
        if self.feature_low is not None:
            features = (features - self.feature_low) / self.feature_scale
        return features, filters

    def step(self, observations: np.ndarray, indices=None, commit: bool = True) -> np.ndarray:
        """Features of observations (n, D) of the environments of indices (all by default); commit=False computes
        them without updating the states (e.g. for the terminal observation of an episode)."""
        indices = np.arange(self.n_envs) if indices is None else np.asarray(indices)
        u, filters = self.inputs(np.asarray(observations, dtype=float), self.filters[indices], self.fresh[indices])
        if self.reservoir is None:
            features, state = u, None
        else:
            r = self.reservoir
            x = self.state[indices]
            state = (1 - r["lr"]) * x + r["lr"] * np.tanh(x @ r["W"].T + u @ r["Win"].T + r["bias"])
            features = np.concatenate([state, u], axis=1) if self.include_input else state
        if commit:
            self.filters[indices], self.fresh[indices] = filters, False
            if state is not None:
                self.state[indices] = state
        return features.astype(np.float32)

    def fit_feature_scaling(self, episodes: list):
        """MinMax scaling of the filter bank features, fitted on the features of episodes (list of (T, D))."""
        if self.bank is None:
            return self
        self.feature_low = self.feature_scale = None
        features = np.concatenate([self.transform_episode(episode) for episode in episodes])
        self.feature_low = features.min(axis=0)
        self.feature_scale = np.maximum(features.max(axis=0) - self.feature_low, 1e-12)
        return self

    def transform_episode(self, observations: np.ndarray) -> np.ndarray:
        """Input features (T, D * F) of the observations (T, D) of one episode (e.g. to fit HAG offline)."""
        filters, fresh, out = self._initial_filters(1), np.ones(1, bool), []
        for obs in observations:
            u, filters = self.inputs(obs[None, :], filters, fresh)
            fresh = np.zeros(1, bool)
            out.append(u[0])
        return np.asarray(out)
