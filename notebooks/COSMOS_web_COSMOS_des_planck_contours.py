#!/usr/bin/env python
"""Plot DES, Planck, COSMOS-Web, and 2 deg^2 COSMOS constraints."""

import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from chainconsumer import Chain
from chainconsumer.plotting import plot_contour
from sanglier.palettes import cool, green, purple, warm


ROOT = Path(__file__).resolve().parents[1]
FIGURE_ROOT = Path('/home/nataliehogg/Documents/Projects/6x2pt/figures')
sys.path.insert(0, str(ROOT / 'src'))

from focsle.fisher import FisherForecast
from focsle.plotting import plot_fisher_ellipse


try:
    plt.style.use('sanglier')
except OSError:
    pass


DES_CHAIN_FILE = ROOT / 'data' / 'chain_3x2pt_lcdm_SR_maglim.txt'

PLANCK_DIR = (
    ROOT
    / 'data'
    / 'COM_CosmoParams_base-plikHM-TTTEEE-lowl-lowE_R3.00'
    / 'base'
    / 'plikHM_TTTEEE_lowl_lowE'
)
PLANCK_PREFIX = 'base_plikHM_TTTEEE_lowl_lowE'

COSMOS_WEB_RESULTS_FILE = (
    ROOT / 'results' / 'fisher_results_cosmos_web_lllelp.pkl'
)
COSMOS_RESULTS_FILE = (
    ROOT / 'results' / 'fisher_results_cosmos_lllelp.pkl'
)

OUTPUT_FILE = FIGURE_ROOT / 'COSMOS_web_COSMOS_des_planck_contours.pdf'

OMEGA_M_LABEL = r'$\Omega_{\rm m}$'
SIGMA_8_LABEL = r'$\sigma_8$'


def filter_samples(omega_m, sigma_8, weights):
    """Remove non-finite or zero-weight posterior samples."""
    mask = (
        np.isfinite(omega_m)
        & np.isfinite(sigma_8)
        & np.isfinite(weights)
        & (weights > 0)
    )
    return omega_m[mask], sigma_8[mask], weights[mask]


def load_des_chain(path):
    """Load Omega_m, sigma_8, and weights from the DES CosmoSIS chain."""
    with path.open() as source:
        header = None
        for line in source:
            if line.startswith('#') and '\t' in line:
                header = line.lstrip('#').strip().split('\t')
                break

    if header is None:
        raise ValueError(f'No tab-separated header found in {path}')

    data = np.loadtxt(path, comments='#')
    omega_m = data[:, header.index('cosmological_parameters--omega_m')]
    sigma_8 = data[:, header.index('COSMOLOGICAL_PARAMETERS--SIGMA_8')]
    weights = data[:, header.index('weight')]
    return filter_samples(omega_m, sigma_8, weights)


def load_planck_chain(directory, prefix):
    """Load the four baseline Planck TTTEEE+lowl+lowE CosmoMC chains."""
    paramnames_path = directory / f'{prefix}.paramnames'
    parameter_names = [
        line.split()[0].rstrip('*')
        for line in paramnames_path.read_text().splitlines()
        if line.strip()
    ]

    chain_paths = [directory / f'{prefix}_{index}.txt' for index in range(1, 5)]
    missing = [str(path) for path in chain_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f'Missing Planck chain files: {missing}')

    data = np.concatenate([np.loadtxt(path) for path in chain_paths], axis=0)
    omega_m = data[:, parameter_names.index('omegam') + 2]
    sigma_8 = data[:, parameter_names.index('sigma8') + 2]
    weights = data[:, 0]
    return filter_samples(omega_m, sigma_8, weights)


def make_chain(omega_m, sigma_8, weights, **settings):
    """Construct a two-dimensional ChainConsumer chain."""
    samples = pd.DataFrame({
        OMEGA_M_LABEL: omega_m,
        SIGMA_8_LABEL: sigma_8,
        'weight': weights,
    })
    return Chain(samples=samples, weight_column='weight', **settings)


def figure_of_merit(omega_m, sigma_8, weights):
    """Return the inverse square root of the weighted covariance determinant."""
    normalised_weights = weights / weights.sum()
    covariance = np.cov(
        np.stack([omega_m, sigma_8]),
        aweights=normalised_weights,
    )
    return 1.0 / np.sqrt(np.linalg.det(covariance))


des_omega_m, des_sigma_8, des_weights = load_des_chain(DES_CHAIN_FILE)
planck_omega_m, planck_sigma_8, planck_weights = load_planck_chain(
    PLANCK_DIR,
    PLANCK_PREFIX,
)

des_chain = make_chain(
    des_omega_m,
    des_sigma_8,
    des_weights,
    name=r'DES Y3 $3\times2$pt',
    shade=True,
    color=cool[1],
    smooth=10,
    bins=20,
    shade_gradient=0.8,
    linewidth=2.0,
    zorder=20,
)

planck_chain = make_chain(
    planck_omega_m,
    planck_sigma_8,
    planck_weights,
    name=r'Planck 2018 TTTEEE+lowE',
    shade=True,
    color=purple[2],
    smooth=10,
    bins=20,
    shade_gradient=0.8,
    linewidth=2.0,
    zorder=30,
)

cosmos_web_results = FisherForecast.load_results(COSMOS_WEB_RESULTS_FILE)
cosmos_web_fisher = cosmos_web_results['fisher_matrices']['Combined']
cosmos_web_fiducial = tuple(cosmos_web_results['fiducial'])

cosmos_results = FisherForecast.load_results(COSMOS_RESULTS_FILE)
cosmos_fisher = cosmos_results['fisher_matrices']['Combined']
cosmos_fiducial = tuple(cosmos_results['fiducial'])

print(f'FoM DES Y3 3x2pt:        {figure_of_merit(des_omega_m, des_sigma_8, des_weights):.1f}')
print(f'FoM Planck 2018 TTTEEE:  {figure_of_merit(planck_omega_m, planck_sigma_8, planck_weights):.1f}')
print(f'FoM COSMOS-Web LLLELP:   {np.sqrt(np.linalg.det(cosmos_web_fisher)):.1f}')
print(f'FoM COSMOS LLLELP:       {np.sqrt(np.linalg.det(cosmos_fisher)):.1f}')

fig, ax = plt.subplots(figsize=(5.2, 4.6))

plot_contour(ax, des_chain, px=OMEGA_M_LABEL, py=SIGMA_8_LABEL)
# plot_contour(ax, planck_chain, px=OMEGA_M_LABEL, py=SIGMA_8_LABEL)

plot_fisher_ellipse(
    cosmos_web_fisher,
    cosmos_web_fiducial,
    ax,
    color=green[2],
    label=r'COSMOS-Web LLLELP',
    show_2sigma=True,
    zorder=10,
)
plot_fisher_ellipse(
    cosmos_fisher,
    cosmos_fiducial,
    ax,
    color=purple[2],
    label=r'COSMOS $2\,\mathrm{deg}^2$ LLLELP',
    show_2sigma=True,
    zorder=15,
)

ax.set_xlim(0.05, 0.58)
ax.set_ylim(0.50, 1.08)
ax.set_xlabel(OMEGA_M_LABEL, fontsize=9)
ax.set_ylabel(SIGMA_8_LABEL, fontsize=9)
ax.tick_params(labelsize=9)
ax.set_box_aspect(1)

legend_handles = [
    mpatches.Patch(
        color=cool[1],
        alpha=0.6,
        label=r'DES Y3',
    ),
    # mpatches.Patch(
    #     color=purple[2],
    #     alpha=0.6,
    #     label=r'Planck 2018 TTTEEE+lowE',
    # ),
    mpatches.Patch(
        color=green[2],
        alpha=0.6,
        label=r'COSMOS-Web (0.54 deg$^2$)',
    ),
    mpatches.Patch(
        color=purple[2],
        alpha=0.6,
        label=r'COSMOS-Web (2.0 deg$^2$)',
    ),
]
ax.legend(handles=legend_handles, loc='upper right', fontsize=8, frameon=False)

fig.tight_layout()
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_FILE, bbox_inches='tight')
plt.close(fig)

print(f'Saved to {OUTPUT_FILE}')
