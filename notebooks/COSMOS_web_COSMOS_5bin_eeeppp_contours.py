#!/usr/bin/env python
"""Plot five-bin EEEPPP COSMOS-Web and full-COSMOS forecasts."""

import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / 'src'))

from focsle.fisher import FisherForecast
from focsle.plotting import marginalize_fisher_to_pair, plot_fisher_ellipse
from sanglier.palettes import green, purple


try:
    plt.style.use('sanglier')
except OSError:
    pass

plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
})

RESULTS = {
    'COSMOS-Web (0.54 deg$^2$)': {
        'color': green[2],
        'two_parameter': (
            ROOT / 'results'
            / 'fisher_results_cosmos_web_5bin_eeeppp.pkl'
        ),
        'four_parameter': (
            ROOT / 'results'
            / 'fisher_results_cosmos_web_5bin_eeeppp_w0wa.pkl'
        ),
        'zorder': 10,
    },
    'COSMOS (2.0 deg$^2$)': {
        'color': purple[2],
        'two_parameter': (
            ROOT / 'results'
            / 'fisher_results_cosmos_5bin_eeeppp.pkl'
        ),
        'four_parameter': (
            ROOT / 'results'
            / 'fisher_results_cosmos_5bin_eeeppp_w0wa.pkl'
        ),
        'zorder': 20,
    },
}
OUTPUT_FILE = (
    PROJECT_ROOT / 'figures'
    / 'COSMOS_web_COSMOS_5bin_eeeppp_forecasts.pdf'
)
FISHER_KEY = 'Combined'


def load_pair(path, pair):
    """Load a Fisher result and marginalize it to a named parameter pair."""
    results = FisherForecast.load_results(path)
    fisher, fiducial = marginalize_fisher_to_pair(
        results['fisher_matrices'][FISHER_KEY],
        results['fiducial'],
        results['param_names'],
        pair,
    )
    covariance = np.linalg.inv(fisher)
    return fisher, fiducial, np.sqrt(np.diag(covariance))


def limits_from_forecasts(forecasts):
    """Return padded limits containing both forecast 95 per cent ellipses."""
    bounds = [[np.inf, -np.inf], [np.inf, -np.inf]]
    for _, fiducial, errors in forecasts.values():
        for index in range(2):
            bounds[index][0] = min(
                bounds[index][0], fiducial[index] - 2.8 * errors[index]
            )
            bounds[index][1] = max(
                bounds[index][1], fiducial[index] + 2.8 * errors[index]
            )
    output = []
    for lower, upper in bounds:
        padding = 0.06 * (upper - lower)
        output.append((lower - padding, upper + padding))
    return output


sigma_forecasts = {
    name: load_pair(settings['two_parameter'], ('Omega_m', 'sigma_8'))
    for name, settings in RESULTS.items()
}
dark_energy_forecasts = {
    name: load_pair(settings['four_parameter'], ('w0', 'wa'))
    for name, settings in RESULTS.items()
}

fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8))
panels = (
    (axes[0], sigma_forecasts, r'$\Omega_{\rm m}$', r'$\sigma_8$'),
    (axes[1], dark_energy_forecasts, r'$w_0$', r'$w_a$'),
)

for axis, forecasts, xlabel, ylabel in panels:
    for name, (fisher, fiducial, _) in forecasts.items():
        settings = RESULTS[name]
        plot_fisher_ellipse(
            fisher,
            fiducial,
            axis,
            color=settings['color'],
            label=name,
            show_2sigma=True,
            zorder=settings['zorder'],
        )
    xlim, ylim = limits_from_forecasts(forecasts)
    axis.set_xlim(*xlim)
    axis.set_ylim(*ylim)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.tick_params(labelsize=8)
    axis.set_box_aspect(1)

legend_handles = [
    mpatches.Patch(
        color=settings['color'], alpha=0.6, label=name
    )
    for name, settings in RESULTS.items()
]
fig.legend(
    handles=legend_handles,
    loc='upper center',
    ncol=2,
    frameon=False,
    bbox_to_anchor=(0.5, 1.01),
)
fig.tight_layout(rect=(0, 0, 1, 0.92))
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_FILE, bbox_inches='tight')
plt.close(fig)

for label, forecasts in (
    ('Omega_m-sigma_8', sigma_forecasts),
    ('w0-wa (marginalized over Omega_m and sigma_8)', dark_energy_forecasts),
):
    web_errors = forecasts['COSMOS-Web (0.54 deg$^2$)'][2]
    cosmos_errors = forecasts['COSMOS (2.0 deg$^2$)'][2]
    improvements = 100.0 * (1.0 - cosmos_errors / web_errors)
    print(f'{label}:')
    print(f'  COSMOS-Web errors: {web_errors}')
    print(f'  COSMOS errors:     {cosmos_errors}')
    print(f'  improvements:      {improvements}%')

print(f'Saved to {OUTPUT_FILE}')
