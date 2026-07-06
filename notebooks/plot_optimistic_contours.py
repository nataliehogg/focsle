"""Plot combined LL, LE, LP contours for the optimistic case."""

from focsle.fisher import FisherForecast
from focsle.plotting import plot_constraints_overlay
import matplotlib.pyplot as plt

# The pickle's 'Combined' is the full-covariance Fisher (J^T C^+ J with
# cross-probe covariances and noise-floor modes projected out) - use it
# as stored. The old block-diagonal recompute that used to live here
# double-counted shared information (audit B1).
results = FisherForecast.load_results(
    'results/fisher_results_Nlens=1e5_sigL=0.05_Nbin_z=6_SNR_goal=8_Nbin_max=20_nsamp=1e6_audited_060726_As_matched.pkl'
)

fig = plot_constraints_overlay(
    results,
    probes=['LL', 'LE', 'LP'],
    # probes=['Combined'],
    title=None,
)

ax = fig.axes[0]
# ax.set_xlim(0.2, 0.4)
# ax.set_ylim(0.6, 1.1)

ax.set_box_aspect(1)  # square axes
ax.lines[0].remove()  # remove fiducial cross
# ax.get_legend().get_texts()[0].set_text(r'Euclid DR3 $3\times2$ pt')

# plt.savefig('../optimistic_contours_contributions_030326.pdf', bbox_inches='tight')

plt.show()
