"""
Six-panel figure: the correct full-covariance Fisher contour for each
forecast case, over the DES Y3 3x2pt posterior.

Panels:
    (a) conservative EEEPPP                 - loscov_dev _all data, nsamp=5e3
    (b) optimistic  EEEPPP                  - loscov_dev _all data, nsamp=5e3
        (apples-to-apples: the same probes as DES 3x2pt)
    (c) conservative LLLELP (LOS 3x2pt)     - elrond published data, nsamp=1e6
    (d) optimistic  LLLELP (LOS 3x2pt)      - elrond published data, nsamp=1e6
    (e) conservative 6x2pt (all six)        - loscov_dev _all data, nsamp=5e3
    (f) optimistic  6x2pt (all six)         - loscov_dev _all data, nsamp=5e3

'Combined' is used exactly as stored in the pickles: F = J^T C^+ J with the
full cross-probe covariance (audit B1/B6). The DES Y3 chain is plotted as
in make_contour_figure.py.

Caveats: the 6x2pt covariances are MC-noisy (nsamp=5e3) and their EE data
carry the loscov EE kernel bug (audit D1) until regenerated - treat the
6x2pt panels as previews, not science.
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
except OSError:
    pass

from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif', size=11)

cool = ['#41b6c4', '#2c7fb8', '#253494']
warm = ['#fdcc8a', '#fc8d59', '#d7301f']

ROOT = '/home/nataliehogg/Documents/Projects/6x2pt/focsle'
CHAIN_FILE = f'{ROOT}/data/chain_3x2pt_lcdm_SR_maglim.txt'

PANELS = [
    # EE+EP+PP: computed with compute_custom_fisher(['EE','EP','PP']) - the
    # full-covariance sub-block for the probes DES Y3 3x2pt actually uses
    # (r'conservative EEEPPP',
    #  f'{ROOT}/results/fisher_results_Nlens=1e4_sigL=0.1_Nbin_z=1_SNR_goal=8_'
    #  f'Nbin_max=20_nsamp=5e3_all_EEEPPP_060726.pkl'),
    # (r'optimistic EEEPPP',
    #  f'{ROOT}/results/fisher_results_Nlens=1e5_sigL=0.05_Nbin_z=6_SNR_goal=8_'
    #  f'Nbin_max=20_nsamp=5e3_all_EEEPPP_060726.pkl'),
    # (r'conservative LLLELP',
    #  f'{ROOT}/results/fisher_results_Nlens=1e4_sigL=0.1_Nbin_z=1_SNR_goal=8_'
    #  f'Nbin_max=20_nsamp=1e6_audited_060726_As_matched.pkl'),
    (r'optimistic LLLELP',
     f'{ROOT}/results/fisher_results_Nlens=1e5_sigL=0.05_Nbin_z=6_SNR_goal=8_'
     f'Nbin_max=20_nsamp=1e6_audited_060726_As_matched.pkl'),
    # (r'conservative $6\times2$pt',
    #  f'{ROOT}/results/fisher_results_Nlens=1e4_sigL=0.1_Nbin_z=1_SNR_goal=8_'
    #  f'Nbin_max=20_nsamp=5e3_all_6x2pt_060726.pkl'),
    # (r'optimistic $6\times2$pt',
    #  f'{ROOT}/results/fisher_results_Nlens=1e5_sigL=0.05_Nbin_z=6_SNR_goal=8_'
    #  f'Nbin_max=20_nsamp=5e3_all_6x2pt_060726.pkl'),
]

# --- Load DES Y3 chain (as in make_contour_figure.py) ---
with open(CHAIN_FILE) as f:
    header = None
    for line in f:
        if line.startswith('#') and '\t' in line:
            header = line.lstrip('#').strip().split('\t')
            break

data = np.loadtxt(CHAIN_FILE, comments='#')
omega_m = data[:, header.index('cosmological_parameters--omega_m')]
sigma8 = data[:, header.index('COSMOLOGICAL_PARAMETERS--SIGMA_8')]
weights = data[:, header.index('weight')]

mask = weights > 0
omega_m, sigma8, weights = omega_m[mask], sigma8[mask], weights[mask]

df = pd.DataFrame({
    r'$\Omega_{\rm m}$': omega_m,
    r'$\sigma_8$': sigma8,
    'weight': weights,
})
des_chain = Chain(
    samples=df,
    weight_column='weight',
    name=r'DES Y3 EEEPPP',
    shade=True,
    color=cool[1],#cool[1],
    smooth=10,
    bins=20,
    shade_gradient=0.8,
    linewidth=2.0,
)

# DES FoM for the printed comparison
w_norm = weights / weights.sum()
C_des = np.cov(np.stack([omega_m, sigma8]), aweights=w_norm)
fom_des = 1.0 / np.sqrt(np.linalg.det(C_des))
print(f'FoM DES Y3 3x2pt: {fom_des:.1f}')

# Common axis limits: cover the DES contour with a small margin.
om_mean = np.average(omega_m, weights=w_norm)
s8_mean = np.average(sigma8, weights=w_norm)

# xlim = (0.29, 0.33)
# ylim = (0.77, 0.85)
xlim= (om_mean - 4 * np.sqrt(C_des[0, 0]), om_mean + 4 * np.sqrt(C_des[0, 0]))
ylim=(s8_mean - 4 * np.sqrt(C_des[1, 1]), s8_mean + 4 * np.sqrt(C_des[1, 1]))

# --- Figure ---
fig, ax = plt.subplots(1, 1, figsize=(4, 4), sharex=True, sharey=True)

# for ax, (title, pkl) in zip(axes.ravel(), PANELS):
results = FisherForecast.load_results(PANELS[0][1])
F_combined = results['fisher_matrices']['Combined']
fiducial = tuple(results['fiducial'])

plot_contour(ax, des_chain, px=r'$\Omega_{\rm m}$', py=r'$\sigma_8$')
plot_fisher_ellipse(
    F_combined, fiducial, ax,
    color=warm[2],
    show_2sigma=True,
    zorder=100,
)


ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
ax.tick_params(labelsize=9)
ax.set_box_aspect(1)

ax.set_xlabel(r'$\Omega_{\rm m}$')
ax.set_ylabel(r'$\sigma_8$')

legend_handles = [
    mpatches.Patch(color=cool[1], alpha=0.6, label=r'DES Y3'),
    mpatches.Patch(color=warm[2], alpha=0.6, label=r'Euclid forecast'),
]
# fig.legend(handles=legend_handles, loc='upper center',
        #    bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)#, fontsize=10)

plt.legend(handles=legend_handles, loc='upper right')

plt.tight_layout()#rect=[0, 0, 1, 0.96])

# plt.savefig('/home/nataliehogg/Documents/Projects/6x2pt/figures/six_panel_contours_zoomed_060726.pdf', bbox_inches='tight')
plt.show()
