#!/usr/bin/env python
"""Compare COSMOS three- and six-bin w0-wa forecasts with DESI DR2."""

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
from focsle.plotting import marginalize_fisher_to_pair, plot_fisher_ellipse
from sanglier.palettes import analogous, cool, green


try:
    plt.style.use('sanglier')
except OSError:
    pass


CHAIN_DIR = ROOT / 'data' / 'desi_dr2_planck_pantheon'
RESULTS_FILES = {
    'three bins': (
        ROOT / 'results' / 'fisher_results_cosmos_3bin_lllelp_w0wa.pkl'
    ),
    'six bins': (
        ROOT / 'results' / 'fisher_results_cosmos_6bin_lllelp_w0wa.pkl'
    ),
}
OUTPUT_FILE = (
    PROJECT_ROOT / 'figures' / 'COSMOS_tomography_desi_w0wa_contours.pdf'
)

FISHER_KEY = 'Combined'
PARAMETER_PAIR = ('w0', 'wa')
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
        zorder=10,
    )


def load_forecast(path):
    """Load one four-parameter forecast and marginalize it to w0-wa."""
    results = FisherForecast.load_results(path)
    if FISHER_KEY not in results['fisher_matrices']:
        raise KeyError(f'{FISHER_KEY} is missing from {path}')
    fisher, fiducial = marginalize_fisher_to_pair(
        results['fisher_matrices'][FISHER_KEY],
        results['fiducial'],
        results['param_names'],
        PARAMETER_PAIR,
    )
    errors = dict(zip(
        results['param_names'],
        results['constraints'][FISHER_KEY]['errors'],
    ))
    return fisher, fiducial, errors


def weighted_quantile(values, weights, probabilities):
    """Return weighted quantiles for one-dimensional samples."""
    order = np.argsort(values)
    values = np.asarray(values)[order]
    weights = np.asarray(weights)[order]
    cumulative = np.cumsum(weights) - 0.5 * weights
    cumulative /= weights.sum()
    return np.interp(probabilities, cumulative, values)


def combined_limits(w0, wa, weights, forecasts):
    """Return limits containing DESI and both forecast 95% contours."""
    bounds = [
        list(weighted_quantile(w0, weights, (0.0025, 0.9975))),
        list(weighted_quantile(wa, weights, (0.0025, 0.9975))),
    ]
    for fisher, fiducial, _ in forecasts.values():
        errors = np.sqrt(np.diag(np.linalg.inv(fisher)))
        for index in range(2):
            bounds[index][0] = min(
                bounds[index][0],
                fiducial[index] - 2.8 * errors[index],
            )
            bounds[index][1] = max(
                bounds[index][1],
                fiducial[index] + 2.8 * errors[index],
            )

    limits = []
    for lower, upper in bounds:
        padding = 0.05 * (upper - lower)
        limits.append((lower - padding, upper + padding))
    return limits


desi_w0, desi_wa, desi_weights = load_desi_chain(CHAIN_DIR)
desi_chain = make_desi_chain(desi_w0, desi_wa, desi_weights)
forecasts = {
    name: load_forecast(path)
    for name, path in RESULTS_FILES.items()
}

styles = {
    'three bins': (green[2], r'COSMOS, three source bins'),
    'six bins': (analogous[0], r'COSMOS, six source bins'),
}

fig, ax = plt.subplots(figsize=(4.8, 4.6))

plot_contour(ax, desi_chain, px=W0_LABEL, py=WA_LABEL)
for name, (fisher, fiducial, errors) in forecasts.items():
    color, label = styles[name]
    plot_fisher_ellipse(
        fisher,
        fiducial,
        ax,
        color=color,
        label=label,
        show_2sigma=True,
        zorder=20,
    )
    print(
        f'{name}: sigma(w0)={errors["w0"]:.6g}, '
        f'sigma(wa)={errors["wa"]:.6g}, '
        f'w0-wa FoM={np.sqrt(np.linalg.det(fisher)):.3f}'
    )

xlim, ylim = combined_limits(desi_w0, desi_wa, desi_weights, forecasts)
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
]
legend_handles.extend(
    mpatches.Patch(color=color, alpha=0.6, label=label)
    for color, label in styles.values()
)
ax.legend(handles=legend_handles, loc='best', fontsize=8, frameon=False)

fig.tight_layout()
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_FILE, bbox_inches='tight')
plt.close(fig)

print(f'Saved to {OUTPUT_FILE}')
