"""
Batched reservoir runs for sequence-to-vector classification.

Running reservoirpy's Reservoir one sequence at a time costs a Python loop iteration per time step. Here, sequences
are sorted by length and run in batches, so each time step becomes one matrix product for the whole batch. With JAX
installed, the time loop is also compiled (jax.lax.scan); otherwise a NumPy loop is used. Both compute exactly
reservoirpy's Reservoir update, starting from a zero state for every sequence:
    s <- (1 - lr) * s + lr * f(W @ s + Win @ u + bias)
"""
import math

import numpy as np

from hag.models.activation_functions import tanh

try:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)  # float64, to match reservoirpy
    JAX_ACTIVATIONS = {tanh: jnp.tanh, np.tanh: jnp.tanh}
except ImportError:
    jax = None
    JAX_ACTIVATIONS = {}


def _pad_batch(sequences, indices, lengths, n_rows, n_steps):
    x = np.zeros((n_rows, n_steps, sequences[indices[0]].shape[1]))
    for k, i in enumerate(indices):
        x[k, :lengths[i]] = sequences[i]
    return x


if jax is not None:
    def _make_jax_run(activation):
        @jax.jit
        def run(Wt, Wint, bias, lr, x, lengths):
            u = x @ Wint  # input drive for all time steps at once, (B, T, N)

            def step(s, inputs):
                u_t, t = inputs
                new = (1 - lr) * s + lr * activation(s @ Wt + u_t + bias)
                # sequences that are already finished keep their last state
                return jnp.where((t < lengths)[:, None], new, s), None

            s0 = jnp.zeros((x.shape[0], Wt.shape[0]), dtype=x.dtype)
            s, _ = jax.lax.scan(step, s0, (jnp.swapaxes(u, 0, 1), jnp.arange(x.shape[1])))
            return s

        return run

    _JAX_RUNS = {f: _make_jax_run(jax_f) for f, jax_f in JAX_ACTIVATIONS.items()}


def last_states(W, Win, bias, lr, activation, sequences, batch_size=256, length_bucket=64, use_jax=True):
    """
    Last reservoir state of each sequence (each one starting from a zero state).

    Parameters:
    - W (N, N), Win (N, D), bias (N,), lr: reservoir parameters, as in reservoirpy's Reservoir.
    - activation: activation function. The JAX path is used only for the activations in JAX_ACTIVATIONS.
    - sequences: list of (T_i, D) arrays.
    - batch_size: number of sequences run together.
    - length_bucket: with JAX, batches are padded to a multiple of this length (and to batch_size rows), so that
      only a few distinct shapes are compiled.

    Returns an array of shape (len(sequences), N).
    """
    lengths = np.array([len(sequence) for sequence in sequences])
    order = np.argsort(lengths)  # similar lengths in the same batch -> little padding
    Wt, Wint = np.asarray(W).T, np.asarray(Win).T
    bias = np.broadcast_to(np.ravel(bias), (Wt.shape[0],))
    states = np.empty((len(sequences), Wt.shape[0]))

    if use_jax and activation in JAX_ACTIVATIONS:
        run = _JAX_RUNS[activation]
        Wt, Wint, bias = jnp.asarray(Wt), jnp.asarray(Wint), jnp.asarray(bias)
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            batch_lengths = np.zeros(batch_size, dtype=int)
            batch_lengths[:len(indices)] = lengths[indices]
            n_steps = math.ceil(batch_lengths.max() / length_bucket) * length_bucket
            x = _pad_batch(sequences, indices, lengths, batch_size, n_steps)
            s = run(Wt, Wint, bias, lr, jnp.asarray(x), jnp.asarray(batch_lengths))
            states[indices] = np.asarray(s)[:len(indices)]
    else:
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            batch_lengths = lengths[indices]
            x = _pad_batch(sequences, indices, lengths, len(indices), batch_lengths.max())
            u = x @ Wint
            s = np.zeros((len(indices), Wt.shape[0]))
            for t in range(x.shape[1]):
                new = (1 - lr) * s + lr * activation(s @ Wt + u[:, t] + bias)
                s = np.where((t < batch_lengths)[:, None], new, s)
            states[indices] = s

    return states
