#!/usr/bin/env python
"""
Command-line script for Fisher forecast computation.

This script provides a simple CLI interface to the focsle package.
For interactive use, prefer the Jupyter notebook in notebooks/.

Usage:
    python compute_fisher.py --data-dir /path/to/dataset --lens-file /path/to/lenses.txt

Example:
    python compute_fisher.py \
        --data-dir /path/to/data/Nlens=1e5_sigL=0.2_Nbin_z=6_SNR_goal=8_Nbin_max=20_nsamp=1e6 \
        --lens-file data/Euclid_lenses.txt \
        --output results/my_forecast.pkl
"""

import argparse
from pathlib import Path

from focsle import FisherForecast, list_available_datasets
from focsle.plotting import plot_constraints, print_constraints_table


def main():
    parser = argparse.ArgumentParser(
        description='Compute Fisher matrix forecasts for cosmological parameters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--data-dir', '-d',
        type=str,
        required=True,
        help='Path to the dataset directory containing covariance matrices'
    )

    parser.add_argument(
        '--lens-file', '-l',
        type=str,
        default=None,
        help='Path to lens catalog file (e.g., Euclid_lenses.txt)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output file path for results (default: results/fisher_results_<dataset>.pkl)'
    )

    parser.add_argument(
        '--nOm',
        type=int,
        default=5,
        help='Number of Omega_m grid points (default: 5)'
    )

    parser.add_argument(
        '--nAs',
        type=int,
        default=5,
        help='Number of A_s grid points (default: 5)'
    )

    parser.add_argument(
        '--As-range',
        type=float,
        nargs=2,
        metavar=('As_min', 'As_max'),
        default=[1.5e-9, 2.7e-9],
        help='Range of A_s values for the grid (default: 1.5e-9 2.7e-9)'
    )

    parser.add_argument(
        '--parameters',
        nargs='+',
        choices=FisherForecast.SUPPORTED_PARAMETERS,
        default=list(FisherForecast.DEFAULT_PARAMETERS),
        help=(
            'Ordered parameters to vary (default: Omega_m sigma_8). '
            'Example: --parameters Omega_m sigma_8 w0 wa'
        )
    )

    parser.add_argument(
        '--w0-step',
        type=float,
        default=0.05,
        help='Central CAMB response step for w0 (default: 0.05)'
    )

    parser.add_argument(
        '--wa-step',
        type=float,
        default=0.1,
        help='Central CAMB response step for wa (default: 0.1)'
    )

    parser.add_argument(
        '--theta-min-arcmin',
        type=float,
        default=None,
        help='Minimum theta in arcmin to include (default: no cut)'
    )

    parser.add_argument(
        '--custom-probes',
        nargs='+',
        choices=('LL', 'LE', 'LP', 'EE', 'EP', 'PP'),
        default=None,
        help=(
            'Also compute and save one full-covariance custom probe '
            'combination, e.g. --custom-probes EE EP PP'
        )
    )

    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate and save constraint plots'
    )

    parser.add_argument(
        '--plot-parameters',
        nargs=2,
        choices=FisherForecast.SUPPORTED_PARAMETERS,
        default=None,
        metavar=('X_PARAMETER', 'Y_PARAMETER'),
        help=(
            'Marginalized parameter pair to plot. Defaults to the '
            'first two entries in --parameters.'
        )
    )

    parser.add_argument(
        '--list-datasets',
        type=str,
        metavar='DATA_ROOT',
        help='List available datasets in the given directory and exit'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress messages'
    )

    args = parser.parse_args()

    if args.plot:
        plot_parameters = (
            args.plot_parameters if args.plot_parameters is not None
            else args.parameters[:2]
        )
        if len(plot_parameters) != 2:
            parser.error(
                "--plot requires at least two --parameters, or an explicit "
                "two-name --plot-parameters selection"
            )
        missing_plot_parameters = [
            name for name in plot_parameters if name not in args.parameters
        ]
        if missing_plot_parameters:
            parser.error(
                "--plot-parameters must be included in --parameters; missing "
                f"{missing_plot_parameters}"
            )

    # List datasets mode
    if args.list_datasets:
        print("Available datasets:")
        print("=" * 70)
        for i, ds in enumerate(list_available_datasets(args.list_datasets), 1):
            print(f"{i}. {ds}")
        return

    # Main forecast computation
    verbose = not args.quiet

    if verbose:
        print("=" * 70)
        print("focsle Fisher Forecast")
        print("=" * 70)

    # Initialize forecast
    forecast = FisherForecast(
        data_dir=args.data_dir,
        lens_file=args.lens_file,
        verbose=verbose
    )

    # Setup (expensive P(k) computation)
    forecast.setup(
        nOm=args.nOm,
        nAs=args.nAs,
        As_range=tuple(args.As_range),
        theta_min_arcmin=args.theta_min_arcmin,
        param_names=args.parameters,
        w0_step=args.w0_step,
        wa_step=args.wa_step,
    )

    # Compute Fisher matrices
    results = forecast.compute_fisher()

    if args.custom_probes:
        custom = forecast.compute_custom_fisher(args.custom_probes)
        label = custom['probe_combination']
        forecast.fisher_matrices[label] = custom['fisher_matrix']
        forecast.constraints[label] = custom['constraints']

    # Print summary
    if verbose:
        print_constraints_table(results)

    # Save results
    if args.output:
        output_file = args.output
    else:
        dataset_name = Path(args.data_dir).name
        parameter_tag = '-'.join(args.parameters)
        default_tag = '-'.join(FisherForecast.DEFAULT_PARAMETERS)
        suffix = '' if parameter_tag == default_tag else f'_{parameter_tag}'
        output_file = (
            f'results/fisher_results_{dataset_name}{suffix}.pkl'
        )

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    forecast.save_results(output_file)

    # Generate plots if requested
    if args.plot:
        plot_file = Path(output_file).with_suffix('.png')
        parameter_pair = (
            tuple(args.plot_parameters) if args.plot_parameters
            else tuple(args.parameters[:2])
        )
        fig = plot_constraints(
            results,
            probes=['LL', 'LE', 'LP', 'Combined'],
            parameter_pair=parameter_pair,
            output_file=str(plot_file)
        )

    if verbose:
        print("\n" + "=" * 70)
        print("Done!")
        print("=" * 70)


if __name__ == "__main__":
    main()
