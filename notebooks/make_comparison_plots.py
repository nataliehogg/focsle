"""
Reproduce the conservative vs optimistic comparison plots from the notebook
without re-running the Fisher forecast.

Outputs:
  3panel_comparison.pdf         -- plot_comparison across LL, LE, LP
  conservative_components.pdf  -- plot_constraints_overlay for conservative
  optimistic_components.pdf    -- plot_constraints_overlay for optimistic
"""

import numpy as np
import matplotlib.pyplot as plt

from focsle.fisher import FisherForecast
from focsle.plotting import plot_comparison, plot_constraints_overlay

CONSERVATIVE = (
    'results/fisher_results_Nlens=1e4_sigL=0.1_Nbin_z=1_SNR_goal=8_Nbin_max=20_nsamp=1e6'
    '_scalecut=0.5_gridding=10.pkl'
)
OPTIMISTIC = (
    'results/fisher_results_Nlens=1e5_sigL=0.05_Nbin_z=6_SNR_goal=8_Nbin_max=20_nsamp=1e6'
    '_scalecut=0.5_gridding=10.pkl'
)

conservative = FisherForecast.load_results(CONSERVATIVE)
optimistic   = FisherForecast.load_results(OPTIMISTIC)


def recompute_combined(results):
    """Replace Combined with block-diagonal sum of individual Fisher matrices.

    The full-covariance-inverse approach stored in the pkl can be numerically
    unstable for some datasets (ill-conditioned due to shared Q_L kernel
    across LL/LE/LP), so we recompute it here.
    """
    F = (results['fisher_matrices']['LL'] +
         results['fisher_matrices']['LE'] +
         results['fisher_matrices']['LP'])
    results['fisher_matrices']['Combined'] = F
    C = np.linalg.inv(F)
    errors = np.sqrt(np.diag(C))
    fiducial = np.array(results['fiducial'])
    results['constraints']['Combined'] = {
        'errors': errors,
        'covariance': C,
        'correlation': C[0, 1] / (errors[0] * errors[1]),
        'fractional_errors': errors / fiducial,
    }
    return results


conservative = recompute_combined(conservative)
optimistic   = recompute_combined(optimistic)

# --- 3-panel comparison ---
fig = plot_comparison(
    [conservative, optimistic],
    labels=['Conservative', 'Optimistic'],
    probes=['LL', 'LE', 'LP'],
    title=None,
    show_fiducial=False,
    output_file='3panel_comparison.pdf',
)
plt.show()

# --- Conservative overlay ---
fig = plot_constraints_overlay(
    conservative,
    probes=['LL', 'LE', 'LP'],
    title='Conservative',
    show_2sigma=True,
    output_file='conservative_components.pdf',
)
plt.show()

# --- Optimistic overlay ---
fig = plot_constraints_overlay(
    optimistic,
    probes=['LL', 'LE', 'LP'],
    title='Optimistic',
    show_2sigma=True,
    output_file='optimistic_components.pdf',
)
plt.show()
