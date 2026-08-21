#!/usr/bin/env python
"""Plot conditional COSMOS w0-wa forecasts for 3, 6, and 9 bins."""

import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / 'src'))

from focsle.fisher import FisherForecast
from focsle.plotting import plot_fisher_ellipse
from sanglier.palettes import analogous, green, warm


try:
    plt.style.use('sanglier')
except OSError:
    pass


RESULTS_FILES = {
    3: (
        ROOT / 'results' /
        'fisher_results_cosmos_3bin_lllelp_w0wa_fixed_omegam_sigma8.pkl'
    ),
    6: (
        ROOT / 'results' /
        'fisher_results_cosmos_6bin_lllelp_w0wa_fixed_omegam_sigma8.pkl'
    ),
    9: (
        ROOT / 'results' /
        'fisher_results_cosmos_9bin_lllelp_w0wa_fixed_omegam_sigma8.pkl'
    ),
}
OUTPUT_FILE = (
    PROJECT_ROOT
    / 'figures'
    / 'COSMOS_tomography_fixed_omegam_sigma8_w0wa_contours.pdf'
)

FISHER_KEY = 'Combined'
STYLES = {
    3: (green[2], r'Three source bins'),
    6: (analogous[0], r'Six source bins'),
    9: (warm[2], r'Nine source bins'),
}


forecasts = {
    bin_count: FisherForecast.load_results(path)
    for bin_count, path in RESULTS_FILES.items()
}

fig, ax = plt.subplots(figsize=(4.8, 4.6))

limits = [[], []]
for bin_count, results in forecasts.items():
    if results['param_names'] != ['w0', 'wa']:
        raise ValueError(
            f'{RESULTS_FILES[bin_count]} is not a conditional w0-wa forecast'
        )

    fisher = results['fisher_matrices'][FISHER_KEY]
    fiducial = tuple(results['fiducial'])
    errors = results['constraints'][FISHER_KEY]['errors']
    covariance = np.linalg.inv(fisher)
    correlation = covariance[0, 1] / np.sqrt(
        covariance[0, 0] * covariance[1, 1]
    )
    fom = np.sqrt(np.linalg.det(fisher))
    color, label = STYLES[bin_count]

    plot_fisher_ellipse(
        fisher,
        fiducial,
        ax,
        color=color,
        label=label,
        show_2sigma=True,
        zorder=10 + bin_count,
    )

    for index in range(2):
        limits[index].extend((
            fiducial[index] - 2.8 * errors[index],
            fiducial[index] + 2.8 * errors[index],
        ))

    print(
        f'{bin_count} bins: sigma(w0)={errors[0]:.6g}, '
        f'sigma(wa)={errors[1]:.6g}, correlation={correlation:.4f}, '
        f'w0-wa FoM={fom:.3f}'
    )

for index, values in enumerate(limits):
    lower, upper = min(values), max(values)
    padding = 0.05 * (upper - lower)
    limits[index] = (lower - padding, upper + padding)

ax.plot(
    -1.0,
    0.0,
    marker='+',
    color='black',
    markersize=7,
    markeredgewidth=1.3,
    zorder=30,
)
ax.set_xlim(*limits[0])
ax.set_ylim(*limits[1])
ax.set_xlabel(r'$w_0$')
ax.set_ylabel(r'$w_a$')
ax.tick_params(labelsize=9)
ax.set_box_aspect(1)
ax.set_title(r'COSMOS LLLELP; fixed $\Omega_{\rm m}$ and $\sigma_8$', fontsize=10)

legend_handles = [
    mpatches.Patch(color=color, alpha=0.6, label=label)
    for color, label in STYLES.values()
]
ax.legend(handles=legend_handles, loc='best', fontsize=8, frameon=False)

fig.tight_layout()
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_FILE, bbox_inches='tight')
plt.close(fig)

print(f'Saved to {OUTPUT_FILE}')
