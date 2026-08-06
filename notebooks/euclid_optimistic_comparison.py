#!/usr/bin/env python
"""
Compare the optimistic Euclid EEEPPP and 6x2pt Fisher constraints.

The five panels show:
    (a) EEEPPP alone
    (b) LL overlaid on EEEPPP
    (c) LE overlaid on EEEPPP
    (d) LP overlaid on EEEPPP
    (e) the full LLLELPEEEPPP combination overlaid on EEEPPP

Each combination uses its full covariance sub-block. In particular, EEEPPP
is not formed by adding the independent EE, EP and PP Fisher matrices.
"""

import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / 'src'))

from focsle.fisher import FisherForecast
from focsle.plotting import plot_fisher_ellipse

try:
    plt.style.use('sanglier')
except OSError:
    pass

from matplotlib import rc

rc('text', usetex=True)
rc('font', family='serif', size=11)

cool = ['#41b6c4', '#2c7fb8', '#253494']
warm = ['#fdcc8a', '#fc8d59', '#d7301f']

RESULTS_FILE = (
    ROOT / 'results' / 'EUCLID-ENV-OPT_fine_qmc1pct_scaled_inverse.pkl'
)
OUTPUT_FILE = PROJECT_ROOT / 'figures' / 'euclid_optimistic_comparison.pdf'

BASELINE = 'EE+EP+PP'
PANELS = [
    (r'(a) EEEPPP', None, None),
    (r'(b) LL over EEEPPP', 'LL', r'LL'),
    (r'(c) LE over EEEPPP', 'LE', r'LE'),
    (r'(d) LP over EEEPPP', 'LP', r'LP'),
    (
        r'(e) $6\times2$pt over EEEPPP',
        'Combined',
        r'LLLELPEEEPPP ($6\times2$pt)',
    ),
]


def axis_limits(fisher_matrices, fiducial, margin=2.8):
    """Return limits containing the joint 95 per cent contours."""
    covariances = [np.linalg.inv(matrix) for matrix in fisher_matrices]
    sigma_omega_m = max(np.sqrt(covariance[0, 0])
                        for covariance in covariances)
    sigma_sigma_8 = max(np.sqrt(covariance[1, 1])
                        for covariance in covariances)

    return (
        (fiducial[0] - margin * sigma_omega_m,
         fiducial[0] + margin * sigma_omega_m),
        (fiducial[1] - margin * sigma_sigma_8,
         fiducial[1] + margin * sigma_sigma_8),
    )


results = FisherForecast.load_results(RESULTS_FILE)
fisher_matrices = results['fisher_matrices']
fiducial = tuple(results['fiducial'])

if BASELINE not in fisher_matrices:
    raise KeyError(
        f'{BASELINE} is missing from {RESULTS_FILE}; rerun compute_fisher.py '
        'with --custom-probes EE EP PP'
    )

fig, axes = plt.subplots(1, 5, figsize=(18, 3.8))

for ax, (title, overlay_key, overlay_label) in zip(axes, PANELS):
    matrices = [fisher_matrices[BASELINE]]
    if overlay_key is not None:
        matrices.append(fisher_matrices[overlay_key])

    plot_fisher_ellipse(
        fisher_matrices[BASELINE],
        fiducial,
        ax,
        color=cool[1],
        label=r'EEEPPP ($3\times2$pt)',
        show_2sigma=True,
        zorder=20,
    )

    legend_handles = [
        mpatches.Patch(
            color=cool[1],
            alpha=0.6,
            label=r'EEEPPP ($3\times2$pt)',
        )
    ]

    if overlay_key is not None:
        plot_fisher_ellipse(
            fisher_matrices[overlay_key],
            fiducial,
            ax,
            color=warm[2],
            label=overlay_label,
            show_2sigma=True,
            zorder=10,
        )
        legend_handles.append(
            mpatches.Patch(
                color=warm[2],
                alpha=0.6,
                label=overlay_label,
            )
        )

    xlim, ylim = axis_limits(matrices, fiducial)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    # EEEPPP is genuinely sub-pixel on the LL/LE/LP scales. Keep the true
    # common-scale overlay above, and show its shape in a clearly labelled
    # inset rather than artificially enlarging it.
    if overlay_key in {'LL', 'LE', 'LP'}:
        axins = ax.inset_axes([0.05, 0.05, 0.31, 0.31])
        plot_fisher_ellipse(
            fisher_matrices[BASELINE],
            fiducial,
            axins,
            color=cool[1],
            show_2sigma=True,
        )
        inset_xlim, inset_ylim = axis_limits(
            [fisher_matrices[BASELINE]],
            fiducial,
        )
        axins.set_xlim(*inset_xlim)
        axins.set_ylim(*inset_ylim)
        axins.set_xticks([])
        axins.set_yticks([])
        axins.set_title(r'EEEPPP zoom', fontsize=6, pad=1)
        axins.set_box_aspect(1)

    ax.plot(
        *fiducial,
        marker='+',
        color='black',
        markersize=6,
        markeredgewidth=1.2,
        zorder=30,
    )

    ax.set_title(title, fontsize=11)
    ax.set_xlabel(r'$\Omega_{\rm m}$')
    ax.tick_params(labelsize=8)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_box_aspect(1)
    ax.legend(
        handles=legend_handles,
        loc='best',
        fontsize=7.5,
        frameon=False,
    )

axes[0].set_ylabel(r'$\sigma_8$')

fig.tight_layout()
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_FILE, bbox_inches='tight')
plt.close(fig)

print(f'Saved to {OUTPUT_FILE}')
