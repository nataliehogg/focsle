#!/usr/bin/env python
"""Compare fixed- and free-galaxy-bias five-bin EEEPPP forecasts."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
    'legend.fontsize': 7.5,
})

SURVEYS = {
    'COSMOS-Web (0.54 deg$^2$)': {
        'color': green[2],
        'prefix': 'cosmos_web',
        'zorder': 10,
    },
    'COSMOS (2.0 deg$^2$)': {
        'color': purple[2],
        'prefix': 'cosmos',
        'zorder': 20,
    },
}
OUTPUT_FILE = (
    PROJECT_ROOT / 'figures'
    / 'COSMOS_web_COSMOS_5bin_eeeppp_bias_comparison.pdf'
)


def result_path(prefix, dark_energy, free_bias):
    """Construct one saved-result path from its forecast choices."""
    suffix = '_w0wa' if dark_energy else ''
    suffix += '_free_bias' if free_bias else ''
    return (
        ROOT / 'results'
        / f'fisher_results_{prefix}_5bin_eeeppp{suffix}.pkl'
    )


def load_pair(path, pair):
    """Load and marginalize a forecast to the requested two parameters."""
    results = FisherForecast.load_results(path)
    fisher, fiducial = marginalize_fisher_to_pair(
        results['fisher_matrices']['Combined'],
        results['fiducial'],
        results['param_names'],
        pair,
    )
    errors = np.sqrt(np.diag(np.linalg.inv(fisher)))
    return fisher, fiducial, errors


def load_forecasts(pair, dark_energy):
    """Load fixed- and free-bias results for both footprints."""
    return {
        (survey, free_bias): load_pair(
            result_path(settings['prefix'], dark_energy, free_bias), pair
        )
        for survey, settings in SURVEYS.items()
        for free_bias in (False, True)
    }


def limits_from_free_bias(forecasts):
    """Use the free-bias contours, which enclose the fixed-bias contours."""
    bounds = [[np.inf, -np.inf], [np.inf, -np.inf]]
    for (survey, free_bias), (_, fiducial, errors) in forecasts.items():
        if not free_bias:
            continue
        for index in range(2):
            bounds[index][0] = min(
                bounds[index][0], fiducial[index] - 2.8 * errors[index]
            )
            bounds[index][1] = max(
                bounds[index][1], fiducial[index] + 2.8 * errors[index]
            )
    limits = []
    for lower, upper in bounds:
        padding = 0.06 * (upper - lower)
        limits.append((lower - padding, upper + padding))
    return limits


sigma_forecasts = load_forecasts(('Omega_m', 'sigma_8'), False)
dark_energy_forecasts = load_forecasts(('w0', 'wa'), True)

fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8))
for axis, forecasts, xlabel, ylabel in (
    (axes[0], sigma_forecasts, r'$\Omega_{\rm m}$', r'$\sigma_8$'),
    (axes[1], dark_energy_forecasts, r'$w_0$', r'$w_a$'),
):
    # Draw the broad marginalized regions first and the compact fixed-bias
    # dashed outlines last, so the filled regions cannot obscure them.
    for free_bias in (True, False):
        for survey, settings in SURVEYS.items():
            fisher, fiducial, _ = forecasts[(survey, free_bias)]
            plot_fisher_ellipse(
                fisher,
                fiducial,
                axis,
                color=settings['color'],
                linestyle='-' if free_bias else '--',
                alpha_1sig=0.28 if free_bias else 0.0,
                alpha_2sig=0.12 if free_bias else 0.0,
                show_2sigma=True,
                zorder=settings['zorder'] if free_bias else 40 + settings['zorder'],
            )
    xlim, ylim = limits_from_free_bias(forecasts)
    axis.set_xlim(*xlim)
    axis.set_ylim(*ylim)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.tick_params(labelsize=8)
    axis.set_box_aspect(1)

legend_handles = []
for survey, settings in SURVEYS.items():
    legend_handles.extend((
        Line2D(
            [0], [0], color=settings['color'], linestyle='-', linewidth=2,
            label=survey + r', free $b_i$',
        ),
        Line2D(
            [0], [0], color=settings['color'], linestyle='--', linewidth=1.5,
            label=survey + r', fixed $b_i$',
        ),
    ))
fig.legend(
    handles=legend_handles,
    loc='upper center',
    ncol=2,
    frameon=False,
    bbox_to_anchor=(0.5, 1.02),
)
fig.tight_layout(rect=(0, 0, 1, 0.86))
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_FILE, bbox_inches='tight')
plt.close(fig)

for label, forecasts in (
    ('Omega_m-sigma_8', sigma_forecasts),
    ('w0-wa', dark_energy_forecasts),
):
    print(label)
    for survey in SURVEYS:
        fixed = forecasts[(survey, False)][2]
        free = forecasts[(survey, True)][2]
        print(f'  {survey}')
        print(f'    fixed bias: {fixed}')
        print(f'    free bias:  {free}')
        print(f'    degradation factors: {free / fixed}')

print(f'Saved to {OUTPUT_FILE}')
