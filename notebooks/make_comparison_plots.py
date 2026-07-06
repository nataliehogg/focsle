"""
Reproduce the conservative vs optimistic comparison plots from the notebook
without re-running the Fisher forecast.

Outputs:
  3panel_comparison.pdf         -- plot_comparison across LL, LE, LP
  conservative_components.pdf  -- plot_constraints_overlay for conservative
  optimistic_components.pdf    -- plot_constraints_overlay for optimistic
"""

import matplotlib.pyplot as plt

from focsle.fisher import FisherForecast
from focsle.plotting import plot_comparison, plot_constraints_overlay

# The pickles' 'Combined' is the full-covariance Fisher - use it as stored.
# The old block-diagonal recompute that used to live here double-counted
# shared information (audit B1).
CONSERVATIVE = (
    'results/fisher_results_Nlens=1e4_sigL=0.1_Nbin_z=1_SNR_goal=8_Nbin_max=20_nsamp=1e6'
    '_audited_060726_As_matched.pkl'
)
OPTIMISTIC = (
    'results/fisher_results_Nlens=1e5_sigL=0.05_Nbin_z=6_SNR_goal=8_Nbin_max=20_nsamp=1e6'
    '_audited_060726_As_matched.pkl'
)

conservative = FisherForecast.load_results(CONSERVATIVE)
optimistic   = FisherForecast.load_results(OPTIMISTIC)

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
