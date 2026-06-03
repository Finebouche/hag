"""Hebbian adaptive growth algorithms for reservoir computing."""

__version__ = "0.1.0"

from .hag import (
    hag_step,
    run_algorithm,
    compute_synaptic_change,
)

__all__ = [
    "hag_step",
    "run_algorithm",
    "compute_synaptic_change",
]