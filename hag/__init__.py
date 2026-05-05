"""Hebbian adaptive growth algorithms for reservoir computing."""

__version__ = "0.1.0"

__all__ = [
    "compute_synaptic_change",
    "compute_variance",
    "hag_step",
    "run_algorithm",
]


def __getattr__(name):
    if name in __all__:
        from hag import hag as _hag

        return getattr(_hag, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
