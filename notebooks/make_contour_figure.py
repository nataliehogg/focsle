"""
Plot DES Y3 + KiDS-1000 posterior contours in Omega_m vs sigma_8,
overlaid with the focsle optimistic Fisher forecast (Combined probe).

Run from /home/nataliehogg/Documents/Applications/2026/URF/ with the
focsle venv active.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from chainconsumer import Chain
from chainconsumer.plotting import plot_contour

sys.path.insert(0, '/home/nataliehogg/Documents/Projects/6x2pt/focsle/src')
from focsle.fisher import FisherForecast
from focsle.plotting import plot_fisher_ellipse

try:
    plt.style.use('sanglier')
except:
    pass

from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif', size=11)

cool = ['#41b6c4', '#2c7fb8', '#253494']
warm = ['#fdcc8a', '#fc8d59', '#d7301f']

CHAIN_FILE = (
    '/home/nataliehogg/Documents/Projects/6x2pt/focsle/data/'
    'chain_3x2pt_lcdm_SR_maglim.txt'
)
RESULTS_FILE = (
    '/home/nataliehogg/Documents/Projects/6x2pt/focsle/results/'
    'fisher_results_Nlens=1e5_sigL=0.05_Nbin_z=6_SNR_goal=8_Nbin_max=20_nsamp=1e6_audited_060726_As_matched.pkl'
)

# --- Load chain ---
with open(CHAIN_FILE) as f:
    header = None
    for line in f:
        if line.startswith('#') and '\t' in line:
            header = line.lstrip('#').strip().split('\t')
            break

data = np.loadtxt(CHAIN_FILE, comments='#')

om_idx = header.index('cosmological_parameters--omega_m')
s8_idx = header.index('COSMOLOGICAL_PARAMETERS--SIGMA_8')
w_idx  = header.index('weight')

omega_m = data[:, om_idx]
sigma8  = data[:, s8_idx]
weights = data[:, w_idx]

mask    = weights > 0
omega_m = omega_m[mask]
sigma8  = sigma8[mask]
weights = weights[mask]

# --- Load Fisher forecast ---
# 'Combined' in the pickle is the full-covariance Fisher; the old
# block-diagonal sum double-counted shared information (audit B1).
results    = FisherForecast.load_results(RESULTS_FILE)
F_combined = results['fisher_matrices']['Combined']
fiducial   = tuple(results['fiducial'])

# --- Build chain object ---
df = pd.DataFrame({
    r'$\Omega_{\rm m}$': omega_m,
    r'$\sigma_8$':       sigma8,
    'weight':            weights,
})

chain = Chain(
    samples=df,
    weight_column='weight',
    name=r'DES Y3 $3\times2$pt',
    shade=True,
    color=cool[1],
    smooth=10,
    bins=20,
    shade_gradient=0.8,
    linewidth=2.0,
)

# --- Figure of merit ---
weights_norm = weights / weights.sum()
om_mean = np.average(omega_m, weights=weights_norm)
s8_mean = np.average(sigma8,  weights=weights_norm)
C_des = np.cov(np.stack([omega_m, sigma8]), aweights=weights_norm)

fom_des      = 1.0 / np.sqrt(np.linalg.det(C_des))
fom_forecast = np.sqrt(np.linalg.det(F_combined))
print(f'FoM DES Y3 3x2pt:        {fom_des:.1f}')
print(f'FoM Euclid LOS (forecast): {fom_forecast:.1f}')
print(f'Improvement factor:        {fom_forecast / fom_des:.0f}x')

# --- Plot ---
fig, ax = plt.subplots(figsize=(4, 4))

plot_contour(ax, chain,
             px=r'$\Omega_{\rm m}$',
             py=r'$\sigma_8$')

plot_fisher_ellipse(
    F_combined, fiducial, ax,
    color=warm[2],
    label=r'Euclid LOS shear (forecast)',
    show_2sigma=True,
    zorder=100
)

# Zoom in enough to show the Euclid ellipse alongside the DES contour.
# Adjust these if the ellipse is too small or the DES contour is clipped.
# ax.set_xlim(0.15, 0.55)
# ax.set_ylim(0.55, 1.10)

ax.set_xlim(0.29, 0.33)
ax.set_ylim(0.77, 0.85)

ax.set_xlabel(r'$\Omega_{\rm m}$')
ax.set_ylabel(r'$\sigma_8$')
ax.tick_params(labelsize=9)
ax.set_box_aspect(1)

legend_handles = [
    mpatches.Patch(color=cool[1], alpha=0.6, label=r'DES Y3'),
    mpatches.Patch(color=warm[2], alpha=0.6, label=r'Euclid DR3'),
]
ax.legend(handles=legend_handles, loc='upper right', fontsize=9, frameon=False)

plt.tight_layout()
out = 'proposal/figures/contour_comparison_postbugfix.pdf'
plt.savefig(out, bbox_inches='tight')
print(f'Saved to {out}')
plt.show()
