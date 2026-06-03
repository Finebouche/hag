"""Analysis, capacity, topology, plotting, and network utility functions."""

from .ip_capacities import (
    test_loading,
    polyval,
    extract_mean,
    legendre_step,
    legendre_incremental,
    generate_task,
    cov_capacity,
    capacity_iterator,
)

from .m_capacities import (
    memory_capacity,
)

from .network import (
    state_to_color,
    create_network,
    draw_network,
    animate_network,
)

from .plots import (
    plot_readout,
    show_matrice,
    show_ei_matrix,
)

from .richness import (
    spectral_radius,
    pearson,
    squared_uncoupled_dynamics,
    squared_uncoupled_dynamics_alternative,
    linear_uncoupled_dynamics,
    condition_number,
    distance_correlation,
)

from .separability import (
    inter_intra_class_distance,
    fisher_discriminant_ratio,
    silhouette,
    davies_bouldin,
    calinski_harabasz,
)

from .topology import (
    motif_distribution,
    draw_motifs_distribution,
)

__all__ = [
    # intrinsic plasticity / information-processing capacities
    "test_loading",
    "polyval",
    "extract_mean",
    "legendre_step",
    "legendre_incremental",
    "generate_task",
    "cov_capacity",
    "capacity_iterator",

    # memory capacities
    "memory_capacity",

    # network visualization
    "state_to_color",
    "create_network",
    "draw_network",
    "animate_network",

    # plotting
    "plot_readout",
    "show_matrice",
    "show_ei_matrix",

    # richness metrics
    "spectral_radius",
    "pearson",
    "squared_uncoupled_dynamics",
    "squared_uncoupled_dynamics_alternative",
    "linear_uncoupled_dynamics",
    "condition_number",
    "distance_correlation",

    # separability metrics
    "inter_intra_class_distance",
    "fisher_discriminant_ratio",
    "silhouette",
    "davies_bouldin",
    "calinski_harabasz",

    # topology
    "motif_distribution",
    "draw_motifs_distribution",
]