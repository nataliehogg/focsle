"""
focsle - Fisher Forecasting for Cosmological Strong Lensing Experiments

A JAX-accelerated package for computing Fisher matrix forecasts from
LOS-LOS, LOS-shear, and LOS-position correlation functions.

Example usage:
    >>> from focsle import FisherForecast
    >>>
    >>> # List available datasets
    >>> datasets = FisherForecast.list_datasets('/path/to/data')
    >>> print(datasets)
    >>>
    >>> # Initialize forecast with a specific dataset
    >>> forecast = FisherForecast(
    ...     data_dir='/path/to/data/Nlens=1e5_sigL=0.2_Nbin_z=6_...',
    ...     lens_file='/path/to/Euclid_lenses.txt'
    ... )
    >>>
    >>> # Setup (expensive P(k) grid computation)
    >>> forecast.setup(nOm=5, nAs=5)
    >>>
    >>> # Compute Fisher matrices
    >>> results = forecast.compute_fisher()
    >>>
    >>> # Plot results
    >>> from focsle.plotting import plot_constraints
    >>> fig = plot_constraints(results)
    >>>
    >>> # Save results
    >>> forecast.save_results('results.pkl')
"""

from .fisher import FisherForecast
from .theory import TheoryJAX
from .data_loader import (
    list_available_datasets,
    load_covariance,
    build_full_covariance,
    detect_nbins,
    parse_dataset_name,
    load_redshift_distributions,
    load_angular_distributions,
    load_lens_catalog,
)
from .plotting import (
    plot_fisher_ellipse,
    plot_constraints,
    plot_constraints_overlay,
    plot_comparison,
    print_constraints_table,
    plot_fom_comparison,
)

__version__ = '0.1.0'
__author__ = 'focsle developers'

__all__ = [
    # Main class
    'FisherForecast',

    # Theory
    'TheoryJAX',

    # Data loading
    'list_available_datasets',
    'load_covariance',
    'build_full_covariance',
    'detect_nbins',
    'parse_dataset_name',
    'load_redshift_distributions',
    'load_angular_distributions',
    'load_lens_catalog',

    # Plotting
    'plot_fisher_ellipse',
    'plot_constraints',
    'plot_constraints_overlay',
    'plot_comparison',
    'print_constraints_table',
    'plot_fom_comparison',
]
