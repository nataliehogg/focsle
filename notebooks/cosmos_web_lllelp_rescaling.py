#!/usr/bin/env python
"""Quick, deliberately approximate COSMOS-Web LLLELP Fisher rescaling.

This script uses the existing optimistic Euclid LL+LE+LP Fisher matrices as
its reference.  It reports two limiting estimates:

1. ``noise-dominated``: rescale the separate LL, LE, and LP Fisher matrices
   using their tracer-noise powers, then add them as if the probes were
   independent;
2. ``area/cosmic-variance``: retain the reference covariance structure and
   scale every parameter error as the inverse square root of survey area.

Neither estimate is a replacement for a COSMOS-Web covariance calculation.
In particular, the noise-dominated estimate ignores cross-covariances and
retains the Euclid signal derivatives and redshift binning.  The range between
the two limits is intended only as a quick feasibility diagnostic.

Examples
--------
Use the default single-band F150W assumptions::

    python notebooks/cosmos_web_lllelp_rescaling.py

Use the combined JWST effective source density of 129 arcmin^-2::

    python notebooks/cosmos_web_lllelp_rescaling.py \
        --shape-galaxies 250776 --position-galaxies 250776 \
        --shape-noise 0.30
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_FILE = (
    ROOT / 'results' /
    'EUCLID-ENV-OPT_fine_qmc1pct_scaled_inverse_LLLELP_w0wa.pkl'
)

PROBES = ('LL', 'LE', 'LP')
COMBINED_KEY = 'LL+LE+LP'


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--results-file',
        type=Path,
        default=DEFAULT_RESULTS_FILE,
        help='Reference optimistic Euclid LLLELP Fisher result',
    )

    target = parser.add_argument_group('COSMOS-Web assumptions')
    target.add_argument('--area-deg2', type=float, default=0.54)
    target.add_argument('--lenses', type=float, default=100)
    target.add_argument('--los-noise', type=float, default=0.01)
    target.add_argument('--shape-galaxies', type=float, default=209_893)
    target.add_argument('--position-galaxies', type=float, default=209_893)
    target.add_argument(
        '--shape-noise',
        type=float,
        default=0.31,
        help='RMS shear scatter per component (default: F150W value 0.31)',
    )

    reference = parser.add_argument_group('reference Euclid assumptions')
    reference.add_argument('--reference-area-deg2', type=float, default=15_000)
    reference.add_argument('--reference-lenses', type=float, default=100_000)
    reference.add_argument('--reference-los-noise', type=float, default=0.05)
    reference.add_argument(
        '--reference-shape-galaxies', type=float, default=2e9
    )
    reference.add_argument(
        '--reference-position-galaxies', type=float, default=2e9
    )
    reference.add_argument('--reference-shape-noise', type=float, default=0.30)
    return parser.parse_args()


def require_positive(name, value):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive, got {value!r}')


def fisher_errors(fisher):
    """Return marginalized 1-sigma errors from a symmetric Fisher matrix."""
    fisher = np.asarray(fisher, dtype=float)
    fisher = 0.5 * (fisher + fisher.T)
    covariance = np.linalg.inv(fisher)
    diagonal = np.diag(covariance)
    if np.any(diagonal <= 0):
        raise ValueError('Fisher inverse has a non-positive diagonal')
    return np.sqrt(diagonal)


def fixed_dark_energy_fisher(fisher, parameter_names):
    """Select Omega_m and sigma_8 while holding all other parameters fixed."""
    required = ('Omega_m', 'sigma_8')
    missing = [name for name in required if name not in parameter_names]
    if missing:
        raise KeyError(f'reference Fisher result lacks parameters {missing}')
    indices = [parameter_names.index(name) for name in required]
    return np.asarray(fisher)[np.ix_(indices, indices)]


def print_error_table(parameter_names, rows):
    labels = ['approximation', *parameter_names]
    widths = [max(len(labels[0]), *(len(name) for name, _ in rows))]
    for index, label in enumerate(parameter_names):
        widths.append(
            max(len(label), *(len(f'{errors[index]:.5g}') for _, errors in rows))
        )

    print('  '.join(label.ljust(width) for label, width in zip(labels, widths)))
    print('  '.join('-' * width for width in widths))
    for name, errors in rows:
        values = [name.ljust(widths[0])]
        values.extend(
            f'{value:.5g}'.rjust(width)
            for value, width in zip(errors, widths[1:])
        )
        print('  '.join(values))


def main():
    args = parse_args()
    positive_values = {
        'area_deg2': args.area_deg2,
        'lenses': args.lenses,
        'los_noise': args.los_noise,
        'shape_galaxies': args.shape_galaxies,
        'position_galaxies': args.position_galaxies,
        'shape_noise': args.shape_noise,
        'reference_area_deg2': args.reference_area_deg2,
        'reference_lenses': args.reference_lenses,
        'reference_los_noise': args.reference_los_noise,
        'reference_shape_galaxies': args.reference_shape_galaxies,
        'reference_position_galaxies': args.reference_position_galaxies,
        'reference_shape_noise': args.reference_shape_noise,
    }
    for name, value in positive_values.items():
        require_positive(name, value)

    with args.results_file.open('rb') as source:
        results = pickle.load(source)

    matrices = results['fisher_matrices']
    missing = [key for key in (*PROBES, COMBINED_KEY) if key not in matrices]
    if missing:
        raise KeyError(f'reference Fisher result lacks matrices {missing}')
    parameter_names = list(results['param_names'])

    area_covariance_ratio = args.reference_area_deg2 / args.area_deg2
    area_error_ratio = np.sqrt(area_covariance_ratio)

    # Number densities can use any common area unit because only their ratios
    # enter below.  The printed values are converted from deg^-2 to arcmin^-2.
    reference_lens_density = (
        args.reference_lenses / args.reference_area_deg2
    )
    target_lens_density = args.lenses / args.area_deg2
    reference_shape_density = (
        args.reference_shape_galaxies / args.reference_area_deg2
    )
    target_shape_density = args.shape_galaxies / args.area_deg2
    reference_position_density = (
        args.reference_position_galaxies / args.reference_area_deg2
    )
    target_position_density = args.position_galaxies / args.area_deg2

    # For shear-like fields N ~ sigma^2 / number density.  For the position
    # field the Poisson shot-noise power is N ~ 1 / number density.
    lens_noise_power_ratio = (
        (args.los_noise / args.reference_los_noise) ** 2
        * reference_lens_density / target_lens_density
    )
    shape_noise_power_ratio = (
        (args.shape_noise / args.reference_shape_noise) ** 2
        * reference_shape_density / target_shape_density
    )
    position_noise_power_ratio = (
        reference_position_density / target_position_density
    )

    # In the pure-noise Gaussian limit, the standard deviation of an XY
    # correlation scales as sqrt(A_ref/A) * sqrt(q_X q_Y), where q_X is the
    # target/reference noise-power ratio for tracer X.
    probe_error_ratios = {
        'LL': area_error_ratio * lens_noise_power_ratio,
        'LE': area_error_ratio * np.sqrt(
            lens_noise_power_ratio * shape_noise_power_ratio
        ),
        'LP': area_error_ratio * np.sqrt(
            lens_noise_power_ratio * position_noise_power_ratio
        ),
    }

    reference_fisher = np.asarray(matrices[COMBINED_KEY], dtype=float)
    noise_fisher = sum(
        np.asarray(matrices[probe], dtype=float)
        / probe_error_ratios[probe] ** 2
        for probe in PROBES
    )
    area_fisher = reference_fisher / area_covariance_ratio

    four_parameter_rows = [
        ('Euclid reference', fisher_errors(reference_fisher)),
        ('noise dominated', fisher_errors(noise_fisher)),
        ('area/cosmic variance', fisher_errors(area_fisher)),
    ]

    fixed_names = ['Omega_m', 'sigma_8']
    fixed_rows = [
        (
            'Euclid reference',
            fisher_errors(
                fixed_dark_energy_fisher(reference_fisher, parameter_names)
            ),
        ),
        (
            'noise dominated',
            fisher_errors(fixed_dark_energy_fisher(noise_fisher, parameter_names)),
        ),
        (
            'area/cosmic variance',
            fisher_errors(fixed_dark_energy_fisher(area_fisher, parameter_names)),
        ),
    ]

    print('Survey densities [arcmin^-2]')
    print(
        f'  lenses:   reference={reference_lens_density / 3600:.6g}, '
        f'target={target_lens_density / 3600:.6g}'
    )
    print(
        f'  shapes:   reference={reference_shape_density / 3600:.6g}, '
        f'target={target_shape_density / 3600:.6g}'
    )
    print(
        f'  positions: reference={reference_position_density / 3600:.6g}, '
        f'target={target_position_density / 3600:.6g}'
    )

    print('\nTarget/reference noise-power ratios')
    print(f'  L: {lens_noise_power_ratio:.6g}')
    print(f'  E: {shape_noise_power_ratio:.6g}')
    print(f'  P: {position_noise_power_ratio:.6g}')
    print(f'  area error factor: {area_error_ratio:.6g}')

    print('\nPure-noise probe error factors')
    for probe in PROBES:
        print(f'  {probe}: {probe_error_ratios[probe]:.6g}')

    print('\nMarginalized over all reference parameters')
    print_error_table(parameter_names, four_parameter_rows)

    print('\nHolding dark-energy parameters fixed')
    print_error_table(fixed_names, fixed_rows)

    print(
        '\nCaution: these are limiting rescalings, not a COSMOS-Web Fisher '
        'forecast. The noise-dominated line ignores LL/LE/LP '
        'cross-covariances and retains the Euclid redshift kernels and signal '
        'derivatives.'
    )


if __name__ == '__main__':
    main()
