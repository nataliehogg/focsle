#!/usr/bin/env python
"""Isolate the information added by strong-lens observables to EE.

The two panels compare the nested combinations

    EE -> EE+LE -> EE+LE+LL+LP

for the optimistic and conservative Euclid scenarios. EP and PP are excluded
from every combination. Each Fisher matrix uses the full covariance sub-block
of the probes it contains, including their cross-covariances.
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

LENS_FILE = ROOT / 'data' / 'Euclid_lenses.txt'
OUTPUT_FILE = PROJECT_ROOT / 'figures' / 'euclid_l_increment_no_ep_pp.pdf'

SCENARIOS = {
    'optimistic': {
        'title': r'(a) Optimistic',
        'data_dir': (
            PROJECT_ROOT / 'loscov_dev_fork' / 'data' /
            'Nlens=1e5_sigL=0.05_Nbin_z=6_SNR_goal=8_Nbin_max=20_'
            'nsamp=5e3_all_grid=fine'
        ),
        'results_file': (
            ROOT / 'results' /
            'EUCLID-ENV-OPT_fine_qmc1pct_scaled_inverse_l_increment.pkl'
        ),
    },
    'conservative': {
        'title': r'(b) Conservative',
        'data_dir': (
            PROJECT_ROOT / 'loscov_dev_fork' / 'data' /
            'Nlens=1e4_sigL=0.1_Nbin_z=1_SNR_goal=8_Nbin_max=20_'
            'nsamp=5e3_all_grid=fine'
        ),
        'results_file': (
            ROOT / 'results' /
            'EUCLID-ENV-CON_fine_qmc1pct_scaled_inverse_l_increment.pkl'
        ),
    },
}

COMBINATIONS = [
    ('EE', ['EE'], '#2c7fb8', '-'),
    ('EE+LE', ['EE', 'LE'], '#fdae61', '--'),
    ('EE+LE+LL+LP', ['EE', 'LE', 'LL', 'LP'], '#d7301f', ':'),
]


def compute_results(config):
    """Compute and cache all nested combinations for one scenario."""
    if not config['data_dir'].exists():
        raise FileNotFoundError(
            f"Production dataset not found: {config['data_dir']}"
        )

    forecast = FisherForecast(
        data_dir=str(config['data_dir']),
        lens_file=str(LENS_FILE),
        verbose=True,
    )
    forecast.setup(nOm=5, nAs=5)
    forecast.compute_fisher()

    for label, probes, _, _ in COMBINATIONS[1:]:
        custom = forecast.compute_custom_fisher(probes)
        forecast.fisher_matrices[label] = custom['fisher_matrix']
        forecast.constraints[label] = custom['constraints']

    config['results_file'].parent.mkdir(parents=True, exist_ok=True)
    forecast.save_results(config['results_file'])


def load_results(config):
    """Load a complete cache, recomputing it when necessary."""
    required = {item[0] for item in COMBINATIONS}
    if config['results_file'].exists():
        results = FisherForecast.load_results(config['results_file'])
        if required.issubset(results['fisher_matrices']):
            return results

    compute_results(config)
    return FisherForecast.load_results(config['results_file'])


def axis_limits(matrices, fiducial, margin=2.8):
    """Return limits containing all joint 95 per cent contours."""
    covariances = [np.linalg.inv(matrix) for matrix in matrices]
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


def improvement_text(constraints):
    """Summarise marginal-error improvements relative to EE."""
    baseline = constraints['EE']['errors']
    lines = []
    for label in ('EE+LE', 'EE+LE+LL+LP'):
        gain = 100.0 * (baseline - constraints[label]['errors']) / baseline
        lines.append(
            rf'{label}: '
            rf'$\Delta\sigma(\Omega_{{\rm m}})={gain[0]:.2f}\%$, '
            rf'$\Delta\sigma(\sigma_8)={gain[1]:.2f}\%$'
        )
    return '\n'.join(lines)


scenario_results = {
    name: load_results(config)
    for name, config in SCENARIOS.items()
}

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.3))

for ax, (scenario, config) in zip(axes, SCENARIOS.items()):
    results = scenario_results[scenario]
    matrices = results['fisher_matrices']
    fiducial = tuple(results['fiducial'])

    for zorder, (label, _, color, linestyle) in enumerate(
            COMBINATIONS, start=10):
        plot_fisher_ellipse(
            matrices[label],
            fiducial,
            ax,
            color=color,
            label=label,
            linestyle=linestyle,
            alpha_1sig=0.20,
            alpha_2sig=0.08,
            show_2sigma=True,
            zorder=zorder,
        )

    xlim, ylim = axis_limits(
        [matrices[label] for label, _, _, _ in COMBINATIONS],
        fiducial,
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.plot(
        *fiducial,
        marker='+',
        color='black',
        markersize=7,
        markeredgewidth=1.3,
        zorder=30,
    )

    handles = [
        mpatches.Patch(color=color, alpha=0.45, label=label)
        for label, _, color, _ in COMBINATIONS
    ]
    ax.legend(handles=handles, loc='upper right', fontsize=9, frameon=False)
    ax.text(
        0.03,
        0.03,
        improvement_text(results['constraints']),
        transform=ax.transAxes,
        ha='left',
        va='bottom',
        fontsize=7.5,
        bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.8},
    )
    ax.set_title(config['title'])
    ax.set_xlabel(r'$\Omega_{\rm m}$')
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_box_aspect(1)

axes[0].set_ylabel(r'$\sigma_8$')

fig.tight_layout()
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_FILE, bbox_inches='tight')
plt.close(fig)

print(f'Saved to {OUTPUT_FILE}')
for scenario, results in scenario_results.items():
    print(f'\n{scenario}:')
    print(improvement_text(results['constraints']).replace('$', ''))
