"""Dataset loading and preprocessing helpers for HAG experiments."""

from .load_data import load_data
from .load_classification import load_dataset_classification, visualize_groups_distribution
from .load_forecasting import load_dataset_forecasting
from .load_categorical_forecasting import load_dataset_categorical_forecasting, load_canary_dataset, build_phrase_sequences
from .preprocessing import flexible_indexing, plot_classes_distribution, scale_data
from .peak_centered_decomposition import extract_peak_frequencies
from .spectral_decomposition import generate_multivariate_dataset

__all__ = [
    "load_data",
    "load_dataset_classification",
    "load_dataset_forecasting",
    "load_dataset_categorical_forecasting",
    "load_canary_dataset",
    "build_phrase_sequences",
    "visualize_groups_distribution",
    "flexible_indexing",
    "plot_classes_distribution",
    "extract_peak_frequencies",
    "generate_multivariate_dataset"
]