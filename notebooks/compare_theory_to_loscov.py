"""
Compare focsle theory predictions against loscov's binned correlation data.

This is the end-to-end sanity check of the theory pipeline: focsle's
xi_LL(theta) at the fiducial cosmology should track the loscov simulated
signal it is forecasting for (it will not match exactly - different P(k)
sources and integration schemes - but must agree in shape and normalisation).

Usage:
    python notebooks/compare_theory_to_loscov.py [--data-dir DIR] [--out FILE]
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from focsle.fisher import FisherForecast

DEFAULT_DATA_DIR = ('/home/nataliehogg/Documents/Projects/6x2pt/loscov_fork/data/'
                    'Nlens=1e5_sigL=0.05_Nbin_z=6_SNR_goal=8_Nbin_max=20_nsamp=1e6')
DEFAULT_LENS_FILE = str(Path(__file__).resolve().parents[1] / 'data' / 'Euclid_lenses.txt')

RAD_TO_ARCMIN = 180 * 60 / np.pi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default=DEFAULT_DATA_DIR)
    parser.add_argument('--lens-file', default=DEFAULT_LENS_FILE)
    parser.add_argument('--out', default='theory_vs_loscov_LL.pdf')
    args = parser.parse_args()

    # loscov binned signal
    with open(Path(args.data_dir) / 'binned_correlations' / 'LL', 'rb') as f:
        ll_data = pickle.load(f)
    xi_plus_data = np.asarray(ll_data['plus_correlation'])
    xi_minus_data = np.asarray(ll_data['minus_correlation'])

    # focsle prediction at the fiducial (no theta cut: compare the full range)
    forecast = FisherForecast(data_dir=args.data_dir, lens_file=args.lens_file,
                              verbose=False)
    forecast.setup(nOm=5, nAs=5)
    theory = forecast.theory

    pred = np.array(theory.predict_data_vector_jax(
        theory.cosmo_fid['Omega_m'], theory.cosmo_fid['sigma8']))

    # Data vector is minus-first within each probe section (loscov convention)
    n_plus = len(theory.theta_LL_plus)
    n_minus = len(theory.theta_LL_minus)
    xi_minus_theory = pred[:n_minus]
    xi_plus_theory = pred[n_minus:n_minus + n_plus]

    theta_plus = np.array(theory.theta_LL_plus) * RAD_TO_ARCMIN
    theta_minus = np.array(theory.theta_LL_minus) * RAD_TO_ARCMIN

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=False)
    for ax, th, d, t, label in [
        (axes[0], theta_plus, xi_plus_data, xi_plus_theory, r'$\xi_+^{LL}$'),
        (axes[1], theta_minus, xi_minus_data, xi_minus_theory, r'$\xi_-^{LL}$'),
    ]:
        ax.loglog(th, np.abs(d), 'ko', ms=4, label='loscov (binned data)')
        ax.loglog(th, np.abs(t), 'C3-', lw=2, label='focsle theory')
        ax.set_xlabel(r'$\theta$ [arcmin]')
        ax.set_ylabel(label)
        ax.legend()

    plt.tight_layout()
    plt.savefig(args.out, bbox_inches='tight')
    print(f'Saved {args.out}')

    # simple quantitative summary
    for name, d, t in [('plus', xi_plus_data, xi_plus_theory),
                       ('minus', xi_minus_data, xi_minus_theory)]:
        ratio = t / d
        print(f'xi_{name}: theory/data ratio min={ratio.min():.3f} '
              f'max={ratio.max():.3f} median={np.median(ratio):.3f}')


if __name__ == '__main__':
    main()
