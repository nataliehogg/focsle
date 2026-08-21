#!/usr/bin/env python
"""Figure 1 (left panel) for the URF proposal.

Proposal-only copy of LLLELP_des_planck_contours.py. Differences:

  * writes into the URF proposal figures directory
  * DES Y3 and Planck 2018 are backgrounded in grey (dashed / dot-dashed)
  * the Euclid DR3 LOS shear forecast is opaque, bold and drawn last
  * relabelled "(this work)" so the key contour is obviously the applicant's
  * no COSMOS-Web contour; axis limits zoomed back to the proposal range

Do not repoint this at the 6x2pt figures directory: the other script owns
that output.
"""

import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from chainconsumer import Chain
from chainconsumer.plotting import plot_contour


ROOT = Path(__file__).resolve().parents[1]
URF_ROOT = Path('/home/nataliehogg/Documents/Applications/2026/URF')
sys.path.insert(0, str(ROOT / 'src'))

from focsle.fisher import FisherForecast
from focsle.plotting import plot_fisher_ellipse


try:
    plt.style.use('sanglier')
except OSError:
    pass

from sanglier.palettes import warm, green, cool, purple


DES_CHAIN_FILE = ROOT / 'data' / 'chain_3x2pt_lcdm_SR_maglim.txt'

PLANCK_DIR = (
    ROOT
    / 'data'
    / 'COM_CosmoParams_base-plikHM-TTTEEE-lowl-lowE_R3.00'
    / 'base'
    / 'plikHM_TTTEEE_lowl_lowE'
)
PLANCK_PREFIX = 'base_plikHM_TTTEEE_lowl_lowE'

RESULTS_FILE = (
    ROOT
    / 'results'
    / 'fisher_results_Nlens=1e5_sigL=0.05_Nbin_z=6_SNR_goal=8_'
      'Nbin_max=20_nsamp=1e6_audited_060726_As_matched.pkl'
)

OUTPUT_FILE = (
    URF_ROOT
    / 'application'
    / 'proposal'
    / 'figures'
    / 'LLLELP_des_planck_contours.pdf'
)

OMEGA_M_LABEL = r'$\Omega_{\rm m}$'
SIGMA_8_LABEL = r'$\sigma_8$'

# Backgrounded probes: two greys so they stay separable in greyscale print,
# reinforced by the dash pattern.
DES_GREY = 'k'
PLANCK_GREY = 'k'
DES_LINESTYLE = '--'
PLANCK_LINESTYLE = '-'

# Key result. text.usetex is True (set in the sanglier style), so \textbf is
# the only way to get bold here: matplotlib ignores fontweight for usetex text.
# Split over two lines to fit the panel width.
LOS_COLOUR = purple[2]
LOS_LABEL = r'Euclid with LOS shear \textbf{(this proposal)}'


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

    # CosmoMC columns are: multiplicity weight, -log(likelihood), parameters.
    data = np.concatenate([np.loadtxt(path) for path in chain_paths], axis=0)
    omega_m = data[:, parameter_names.index('omegam') + 2]
    sigma_8 = data[:, parameter_names.index('sigma8') + 2]
    weights = data[:, 0]
    return filter_samples(omega_m, sigma_8, weights)


def filter_samples(omega_m, sigma_8, weights):
    """Remove non-finite or zero-weight posterior samples."""
    mask = (
        np.isfinite(omega_m)
        & np.isfinite(sigma_8)
        & np.isfinite(weights)
        & (weights > 0)
    )
    return omega_m[mask], sigma_8[mask], weights[mask]


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
    shade_alpha=0.05,
    shade_gradient=0.2,
    color=DES_GREY,
    linestyle=DES_LINESTYLE,
    smooth=10,
    bins=20,
    linewidth=1.6,
    zorder=20,
)

planck_chain = make_chain(
    planck_omega_m,
    planck_sigma_8,
    planck_weights,
    name=r'Planck 2018 TTTEEE+lowE',
    shade=True,
    shade_alpha=0.05,
    shade_gradient=0.2,
    color=PLANCK_GREY,
    linestyle=PLANCK_LINESTYLE,
    smooth=10,
    bins=20,
    linewidth=1.6,
    zorder=30,
)

results = FisherForecast.load_results(RESULTS_FILE)
fisher_lllelp = results['fisher_matrices']['Combined']
fiducial = tuple(results['fiducial'])

fom_des = figure_of_merit(des_omega_m, des_sigma_8, des_weights)
fom_planck = figure_of_merit(planck_omega_m, planck_sigma_8, planck_weights)
fom_forecast = np.sqrt(np.linalg.det(fisher_lllelp))

print(f'FoM DES Y3 3x2pt:          {fom_des:.1f}')
print(f'FoM Planck 2018 TTTEEE:    {fom_planck:.1f}')
print(f'FoM Euclid LOS (forecast): {fom_forecast:.1f}')
print(f'  improvement over DES Y3: {fom_forecast / fom_des:.1f}x')
print(f'  improvement over Planck: {fom_forecast / fom_planck:.1f}x')

# \textwidth is 517.84pt = 7.165in, and the proposal includes this at
# width=0.44\textwidth, so the panel is drawn at exactly its final size and
# saved without a tight bbox: LaTeX then scales it by 1.0 and every font size
# set below is the size it prints at.
FIGURE_WIDTH_IN = 0.44 * 517.84 / 72.27

fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, FIGURE_WIDTH_IN))

plot_contour(ax, des_chain, px=OMEGA_M_LABEL, py=SIGMA_8_LABEL)
plot_contour(ax, planck_chain, px=OMEGA_M_LABEL, py=SIGMA_8_LABEL)

# Drawn last, opaque, and with a high zorder so nothing washes over it.
plot_fisher_ellipse(
    fisher_lllelp,
    fiducial,
    ax,
    color=LOS_COLOUR,
    label=LOS_LABEL,
    alpha_1sig=0.85,
    alpha_2sig=0.45,
    show_2sigma=True,
    zorder=100,
)

# Centre the panel on the forecast fiducial so the key contour sits in the
# middle of the frame rather than off to one side. Half-widths keep the
# original spans (0.04 in Omega_m, 0.08 in sigma_8).
OMEGA_M_HALF_WIDTH = 0.03
SIGMA_8_HALF_WIDTH = 0.06 #.04

ax.set_xlim(fiducial[0] - OMEGA_M_HALF_WIDTH, fiducial[0] + OMEGA_M_HALF_WIDTH)
ax.set_ylim(fiducial[1] - SIGMA_8_HALF_WIDTH, fiducial[1] + SIGMA_8_HALF_WIDTH)
ax.set_xlabel(OMEGA_M_LABEL, fontsize=10)
ax.set_ylabel(SIGMA_8_LABEL, fontsize=10)
ax.tick_params(labelsize=8)
ax.set_box_aspect(1)

# Key result listed first: reading order carries importance.
legend_handles = [
    mpatches.Patch(
        facecolor=LOS_COLOUR,
        edgecolor=LOS_COLOUR,
        alpha=0.85,
        label=LOS_LABEL,
    ),
    mlines.Line2D(
        [], [],
        color=DES_GREY,
        linestyle=DES_LINESTYLE,
        linewidth=1.6,
        label=r'DES Y3 $3\times2$pt',
    ),
    mlines.Line2D(
        [], [],
        color=PLANCK_GREY,
        linestyle=PLANCK_LINESTYLE,
        linewidth=1.6,
        label=r'Planck 2018 TTTEEE+lowE',
    ),
]
# Centring on the fiducial pushes the Planck/DES contours through the upper
# corners, so the legend moves to the empty lower left and gets an opaque
# backing box to stay readable wherever the contours run.
legend = ax.legend(
    handles=legend_handles,
    # loc='lower left',
    loc='lower center',
    fontsize=7.5,
    frameon=True,
    facecolor='white',
    edgecolor='none',
    framealpha=0.9,
    labelspacing=0.5,
    handlelength=1.8,
    borderaxespad=0.4,
).set_zorder(300)
legend = ax.get_legend()

# set_color is a matplotlib property, so it still applies under usetex: the
# key entry is bold *and* carries the contour colour.
# legend.get_texts()[0].set_color(LOS_COLOUR)

fig.tight_layout(pad=0.3)
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_FILE)
plt.close(fig)

print(f'Saved to {OUTPUT_FILE}')
