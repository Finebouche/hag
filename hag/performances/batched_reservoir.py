"""
Batched runs of reservoirpy.jax reservoirs.

reservoirpy.jax nodes run one series at a time (a multiseries input goes through lax.map) and each run call retraces
the time loop: on many variable-length sequences (classification), this is no faster than NumPy reservoirpy.
Here the node's own step (the reservoirpy.jax `_step` equations) is vectorised over a batch of sequences with
jax.vmap and the whole time loop is compiled with jax.lax.scan. Sequences are sorted by length and padded at the end,
and a mask keeps the state of the finished ones, so each sequence gets exactly the states reservoirpy gives it.

The node parameters are passed to the compiled function as arguments: one compilation per node type, activation and
batch shape is reused for every node (e.g. across HPO trials).
"""
import math
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)  # float64, to match reservoirpy

# node attributes read by the reservoir `_step` functions (a, b: intrinsic plasticity gain and bias)
PARAMETERS = ("W", "Win", "bias", "lr", "a", "b")


@partial(jax.jit, static_argnames=("node_type", "activation", "return_all"))
def _run(parameters, x, lengths, state0, node_type, activation, return_all):
    # instance holding only the (traced) parameters its `_step` reads
    node = object.__new__(node_type)
    node.__dict__.update(parameters, activation=activation)
    # the node's step, without its own jit (it is compiled here with the rest of the loop), over the whole batch
    step = jax.vmap(partial(node_type._step.__wrapped__, node))

    def scan_step(state, inputs):
        x_t, t = inputs
        new_state = step(state, x_t)
        active = (t < lengths)[:, None]  # sequences that are already finished keep their last state
        state = jax.tree_util.tree_map(lambda new, old: jnp.where(active, new, old), new_state, state)
        return state, (state["out"] if return_all else None)

    state, outs = jax.lax.scan(scan_step, state0, (jnp.swapaxes(x, 0, 1), jnp.arange(x.shape[1])))
    return jnp.swapaxes(outs, 0, 1) if return_all else state["out"]


def run_batched(node, sequences, return_all=False, initial_state=None, batch_size=256, length_bucket=64):
    """
    Run an initialized reservoirpy.jax node on each sequence, every sequence starting from the same state.

    Parameters:
    - node: initialized reservoirpy.jax node (Reservoir, IPReservoir, or a Node with the same kind of `_step`).
    - sequences: list of (T_i, D) arrays.
    - return_all: return all the states of each sequence instead of the last one.
    - initial_state: dict of (N,) arrays with the keys of node.state, zeros by default.
    - batch_size: number of sequences run together.
    - length_bucket: batches are padded to a multiple of this length, so that only a few shapes are compiled.

    Returns an array (len(sequences), N) of last states, or a list of (T_i, N) arrays if return_all.
    """
    n = node.output_dim
    lengths = np.array([len(sequence) for sequence in sequences])
    order = np.argsort(lengths)  # similar lengths in the same batch -> little padding
    parameters = {name: jnp.asarray(getattr(node, name)) for name in PARAMETERS if getattr(node, name, None) is not None}
    if initial_state is None:
        initial_state = {key: np.zeros(n) for key in node.state}
    results = [None] * len(sequences) if return_all else np.empty((len(sequences), n))

    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        # few sequences: pad to a power of two rows instead of batch_size (e.g. one long forecasting series)
        n_rows = batch_size if len(indices) == batch_size else 2 ** math.ceil(math.log2(len(indices)))
        batch_lengths = np.zeros(n_rows, dtype=int)
        batch_lengths[:len(indices)] = lengths[indices]
        n_steps = math.ceil(batch_lengths.max() / length_bucket) * length_bucket
        x = np.zeros((n_rows, n_steps, node.input_dim))
        for k, i in enumerate(indices):
            x[k, :lengths[i]] = sequences[i]
        state0 = {key: jnp.broadcast_to(jnp.ravel(jnp.asarray(value, dtype=float)), (n_rows, n))
                  for key, value in initial_state.items()}

        states = np.asarray(_run(parameters, jnp.asarray(x), jnp.asarray(batch_lengths), state0,
                                 node_type=type(node), activation=node.activation, return_all=return_all))
        for k, i in enumerate(indices):
            results[i] = states[k, :lengths[i]] if return_all else states[k]

    return results
