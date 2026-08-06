#!/usr/bin/env python
"""Compare optimistic Euclid constraints after removing the PP observable.

The five panels show:
    (a) EE+EP alone
    (b) LL overlaid on EE+EP
    (c) LE overlaid on EE+EP
    (d) LP overlaid on EE+EP
    (e) LL+LE+LP+EE+EP overlaid on EE+EP

PP is absent from both plotted combinations in every panel. Each combined
constraint uses the corresponding full covariance sub-block, including all
cross-covariances between the retained probes.

The first run computes and caches the two required no-PP Fisher matrices.
Subsequent runs load that cache and only regenerate the figure.
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

DATA_DIR = (
    PROJECT_ROOT / 'loscov_dev_fork' / 'data' /
    'Nlens=1e5_sigL=0.05_Nbin_z=6_SNR_goal=8_Nbin_max=20_'
    'nsamp=5e3_all_grid=fine'
)
LENS_FILE = ROOT / 'data' / 'Euclid_lenses.txt'
RESULTS_FILE = (
    ROOT / 'results' /
    'EUCLID-ENV-OPT_fine_qmc1pct_scaled_inverse_no_pp.pkl'
)
OUTPUT_FILE = (
    PROJECT_ROOT / 'figures' /
    'euclid_optimistic_comparison_no_pp.pdf'
)

BASELINE = 'EE+EP'
COMBINED = 'LL+LE+LP+EE+EP'
PANELS = [
    (r'(a) EE+EP', None, None),
    (r'(b) LL over EE+EP', 'LL', r'LL'),
    (r'(c) LE over EE+EP', 'LE', r'LE'),
    (r'(d) LP over EE+EP', 'LP', r'LP'),
    (
        r'(e) All except PP over EE+EP',
        COMBINED,
        r'LL+LE+LP+EE+EP',
    ),
]


def compute_no_pp_results():
    """Compute and cache Fisher matrices that cannot be formed by subtraction."""
    forecast = FisherForecast(
        data_dir=str(DATA_DIR),
        lens_file=str(LENS_FILE),
        verbose=True,
    )
    forecast.setup(nOm=5, nAs=5)
    forecast.compute_fisher()

    for probes in (['EE', 'EP'], ['LL', 'LE', 'LP', 'EE', 'EP']):
        custom = forecast.compute_custom_fisher(probes)
        label = custom['probe_combination']
        forecast.fisher_matrices[label] = custom['fisher_matrix']
        forecast.constraints[label] = custom['constraints']

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    forecast.save_results(RESULTS_FILE)


def load_no_pp_results():
    """Load a complete cache, recomputing it if it is absent or incomplete."""
    if RESULTS_FILE.exists():
        results = FisherForecast.load_results(RESULTS_FILE)
        required = {BASELINE, COMBINED, 'LL', 'LE', 'LP'}
        if required.issubset(results['fisher_matrices']):
            return results

    compute_no_pp_results()
    return FisherForecast.load_results(RESULTS_FILE)


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


results = load_no_pp_results()
fisher_matrices = results['fisher_matrices']
fiducial = tuple(results['fiducial'])

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
        label=r'EE+EP (no PP)',
        show_2sigma=True,
        zorder=20,
    )

    legend_handles = [
        mpatches.Patch(
            color=cool[1],
            alpha=0.6,
            label=r'EE+EP (no PP)',
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
        axins.set_title(r'EE+EP zoom', fontsize=6, pad=1)
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
