"""Plot combined LL, LE, LP contours for the optimistic case."""

from focsle.fisher import FisherForecast
from focsle.plotting import plot_constraints_overlay
import matplotlib.pyplot as plt

results = FisherForecast.load_results(
    'results/fisher_results_Nlens=1e5_sigL=0.05_Nbin_z=6_SNR_goal=8_Nbin_max=20_nsamp=1e6_scalecut=0.5.pkl'
)

fig = plot_constraints_overlay(
    results,
    probes=['Combined'],
    title=None,
)

ax = fig.axes[0]
ax.set_xlim(0.2, 0.4)
ax.set_ylim(0.6, 1.1)

ax.set_box_aspect(1)  # square axes
ax.lines[0].remove()  # remove fiducial cross
ax.get_legend().get_texts()[0].set_text(r'Euclid DR3 $6\times2$ pt')

plt.savefig('optimistic_contours.pdf', bbox_inches='tight')

plt.show()
