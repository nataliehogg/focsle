#!/usr/bin/env python
"""Overlay DESI DR2 and Euclid LLLELP at the full DESI fiducial."""

import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from chainconsumer import Chain
from chainconsumer.plotting import plot_contour


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / 'src'))

from focsle.fisher import FisherForecast
from focsle.plotting import (
    marginalize_fisher_to_pair,
    plot_fisher_ellipse,
)
from sanglier.palettes import cool, warm


try:
    plt.style.use('sanglier')
except OSError:
    pass


CHAIN_DIR = ROOT / 'data' / 'desi_dr2_planck_pantheon'
RESULTS_FILE = (
    ROOT / 'results' /
    'EUCLID-ENV-OPT_fine_qmc1pct_scaled_inverse_'
    'LLLELP_w0wa_DESI_full_fiducial.pkl'
)
OUTPUT_FILE = (
    PROJECT_ROOT / 'figures' /
    'LLLELP_desi_full_fiducial_w0wa_contours.pdf'
)

FISHER_KEY = 'LL+LE+LP'
W0_LABEL = r'$w_0$'
WA_LABEL = r'$w_a$'


def load_desi_chain(directory):
    """Load and combine the four weighted Cobaya w0-wa chains."""
    paths = [directory / f'chain.{index}.txt' for index in range(1, 5)]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f'Missing DESI chain files: {missing}')

    header = paths[0].read_text().splitlines()[0].lstrip('#').split()
    required = ('weight', 'w', 'wa')
    absent = [name for name in required if name not in header]
    if absent:
        raise ValueError(f'Missing DESI chain columns: {absent}')

    columns = tuple(header.index(name) for name in required)
    samples = np.concatenate([
        np.loadtxt(path, comments='#', usecols=columns)
        for path in paths
    ])
    weights, w0, wa = samples.T
    mask = (
        np.isfinite(weights)
        & np.isfinite(w0)
        & np.isfinite(wa)
        & (weights > 0)
    )
    return w0[mask], wa[mask], weights[mask]


def make_desi_chain(w0, wa, weights):
    """Build the ChainConsumer representation of the DESI posterior."""
    samples = pd.DataFrame({
        W0_LABEL: w0,
        WA_LABEL: wa,
        'weight': weights,
    })
    return Chain(
        samples=samples,
        weight_column='weight',
        name=r'DESI DR2 BAO + CMB + Pantheon+',
        shade=True,
        color=cool[1],
        smooth=10,
        bins=20,
        shade_gradient=0.8,
        linewidth=2.0,
        zorder=20,
    )


def load_euclid_forecast(path):
    """Load and marginalize the four-parameter Euclid Fisher forecast."""
    results = FisherForecast.load_results(path)
    if FISHER_KEY not in results['fisher_matrices']:
        raise KeyError(f'{FISHER_KEY} is missing from {path}')

    fisher, fiducial = marginalize_fisher_to_pair(
        results['fisher_matrices'][FISHER_KEY],
        results['fiducial'],
        results['param_names'],
        ('w0', 'wa'),
    )
    return fisher, fiducial


def weighted_quantile(values, weights, probabilities):
    """Return weighted quantiles for one-dimensional samples."""
    order = np.argsort(values)
    values = np.asarray(values)[order]
    weights = np.asarray(weights)[order]
    cumulative = np.cumsum(weights) - 0.5 * weights
    cumulative /= weights.sum()
    return np.interp(probabilities, cumulative, values)


def combined_limits(w0, wa, weights, fisher, fiducial):
    """Set limits containing both the posterior and Euclid 95% contour."""
    covariance = np.linalg.inv(fisher)
    errors = np.sqrt(np.diag(covariance))
    quantiles = (
        weighted_quantile(w0, weights, (0.0025, 0.9975)),
        weighted_quantile(wa, weights, (0.0025, 0.9975)),
    )

    limits = []
    for index, bounds in enumerate(quantiles):
        lower = min(bounds[0], fiducial[index] - 2.8 * errors[index])
        upper = max(bounds[1], fiducial[index] + 2.8 * errors[index])
        padding = 0.05 * (upper - lower)
        limits.append((lower - padding, upper + padding))
    return limits


desi_w0, desi_wa, desi_weights = load_desi_chain(CHAIN_DIR)
desi_chain = make_desi_chain(desi_w0, desi_wa, desi_weights)
euclid_fisher, euclid_fiducial = load_euclid_forecast(RESULTS_FILE)

fig, ax = plt.subplots(figsize=(4.4, 4.4))

plot_contour(
    ax,
    desi_chain,
    px=W0_LABEL,
    py=WA_LABEL,
)
plot_fisher_ellipse(
    euclid_fisher,
    euclid_fiducial,
    ax,
    color=warm[2],
    label=r'Euclid DR3 LLLELP',
    show_2sigma=True,
    zorder=100,
)
ax.plot(
    *euclid_fiducial,
    marker='+',
    color='black',
    markersize=7,
    markeredgewidth=1.3,
    zorder=110,
)

xlim, ylim = combined_limits(
    desi_w0,
    desi_wa,
    desi_weights,
    euclid_fisher,
    euclid_fiducial,
)
ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
ax.set_xlabel(W0_LABEL)
ax.set_ylabel(WA_LABEL)
ax.tick_params(labelsize=9)
ax.set_box_aspect(1)

legend_handles = [
    mpatches.Patch(
        color=cool[1],
        alpha=0.6,
        label=r'DESI DR2 BAO + CMB + Pantheon+',
    ),
    mpatches.Patch(
        color=warm[2],
        alpha=0.6,
        label=r'Euclid DR3 LLLELP',
    ),
]
ax.legend(handles=legend_handles, loc='best', fontsize=8, frameon=False)

fig.tight_layout()
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_FILE, bbox_inches='tight')
plt.close(fig)

print(f'DESI weighted samples: {len(desi_w0)}')
print(
    'DESI weighted mean: '
    f'w0={np.average(desi_w0, weights=desi_weights):.6f}, '
    f'wa={np.average(desi_wa, weights=desi_weights):.6f}'
)
print(
    'Euclid fiducial: '
    f'w0={euclid_fiducial[0]:.6f}, wa={euclid_fiducial[1]:.6f}'
)
print(f'Saved to {OUTPUT_FILE}')
