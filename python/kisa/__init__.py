"""
K-ISA (K-Inspired Instruction Set Architecture) Python Implementation.

This package provides a Python implementation of the K-ISA framework,
focusing on vector operations, EDM (Empirical Dynamic Modeling), and
market analysis tools.
"""

__version__ = "0.1.0"

from .vector_ops import (
    vector_add,
    vector_sub,
    vector_mul,
    vector_div,
    vector_fft,
    vector_ifft,
    vector_sort,
    vector_reduce,
    vector_scan,
    vector_compare,
    vector_select,
)

from .edm import (
    time_delay_embedding,
    euclidean_distance,
    find_nearest_neighbors,
    edm_predict,
)

from .market import (
    generate_simulated_events,
    calculate_event_rates,
    analyze_autocorrelation,
    analyze_cross_correlation,
    analyze_event_rates_with_edm,
) 