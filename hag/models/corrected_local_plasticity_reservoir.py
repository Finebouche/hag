"""
reservoirpy's LocalPlasticityReservoir with a corrected fit.

In reservoirpy 0.4.2, LocalPlasticityReservoir.fit computes the next reservoir state at each time step but never
stores it: the pre-synaptic state stays the initial one (zeros) during the whole learning, and the post-synaptic
state only depends on the current input. As a result, Oja / anti-Oja reduce to an input-driven rescaling of the rows
of W, and Hebbian / anti-Hebbian / BCM do not change W at all.
"""
import numpy as np
import scipy.sparse as sp
from reservoirpy.nodes import LocalPlasticityReservoir
from reservoirpy.type import NodeInput, Timeseries, is_multiseries
from reservoirpy.utils.data_validation import check_node_input, filter_nan_targets


class CorrectedLocalPlasticityReservoir(LocalPlasticityReservoir):
    """
    reservoirpy.nodes.LocalPlasticityReservoir whose fit runs the reservoir: same fit as reservoirpy's, except that
    the state is updated at each time step, so that the local rule sees the reservoir dynamics.
    """

    # read by hag.performances.esn_model_evaluation.fit_reservoir, which learns the rule with JAX
    fit_updates_state = True

    def fit(self, x: NodeInput, y: None = None, warmup: int = 0) -> "CorrectedLocalPlasticityReservoir":
        check_node_input(x, expected_dim=self.input_dim)
        x, y = filter_nan_targets(x, y)

        if not self.initialized:
            self.initialize(x)

        increment = self.increment
        do_norm = self.synapse_normalization

        def _local_synaptic_plasticity(seq: Timeseries):
            """
            Apply the local learning rule (Oja, Anti-Oja, Hebbian, Anti-Hebbian, BCM)
            to update the recurrent weight matrix W.

            If `synapse_normalization=True`, then each row of W is L2-normalized
            immediately after the local rule update.
            """
            for u in seq:
                pre_state = self.state["internal"]  # (units,)
                self.state = self._step(self.state, u)  # the correction: reservoirpy does not store the new state
                post_state = self.state["internal"]  # (units,)
                # Vectorized update of nonzero elements based on the chosen rule.
                (rows, cols, data) = sp.find(self.W)
                self.W[rows, cols] += increment(data, pre_state[cols], post_state[rows])
                # Optionally normalize each row.
                if do_norm:
                    # Compute the L2 norm per row for the updated data.
                    row_norms = np.sqrt(np.sum(self.W**2, axis=1)).reshape(-1, 1)
                    safe_norms = np.where(row_norms > 0, row_norms, 1)
                    self.W[:] /= safe_norms[:]

        for _epoch in range(self.epochs):
            if is_multiseries(x):
                for seq in x:
                    _local_synaptic_plasticity(seq[warmup:])
            else:
                _local_synaptic_plasticity(x[warmup:])

        return self
