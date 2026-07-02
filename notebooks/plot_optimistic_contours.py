"""Plot combined LL, LE, LP contours for the optimistic case."""

import numpy as np
from focsle.fisher import FisherForecast
from focsle.plotting import plot_constraints_overlay
import matplotlib.pyplot as plt

results = FisherForecast.load_results(
    'results/fisher_results_Nlens=1e5_sigL=0.05_Nbin_z=6_SNR_goal=8_Nbin_max=20_nsamp=1e6_scalecut=0.5_gridding=10.pkl'
)

# Recompute Combined as block-diagonal sum of individual Fisher matrices.
# The full-covariance-inverse approach stored in the pkl is numerically unstable
# for this dataset (ill-conditioned due to shared Q_L kernel across LL/LE/LP).
F_combined = (results['fisher_matrices']['LL'] +
              results['fisher_matrices']['LE'] +
              results['fisher_matrices']['LP'])
results['fisher_matrices']['Combined'] = F_combined
C_combined = np.linalg.inv(F_combined)
errors = np.sqrt(np.diag(C_combined))
fiducial = np.array(results['fiducial'])
results['constraints']['Combined'] = {
    'errors': errors,
    'covariance': C_combined,
    'correlation': C_combined[0, 1] / (errors[0] * errors[1]),
    'fractional_errors': errors / fiducial,
}

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
