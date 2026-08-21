#!/usr/bin/env python
"""Compute the 2 deg^2 COSMOS LLLELP Fisher forecast."""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from focsle import FisherForecast
from focsle.plotting import print_constraints_table


DATA_DIR = (
    ROOT.parent
    / 'loscov_dev_fork'
    / 'data'
    / 'Nlens=4e2_sigL=0.01_Nbin_z=1_SNR_goal=8_Nbin_max=20_'
      'nsamp=5e3_new_scenario=cosmos_grid=fine'
)
LENS_FILE = ROOT.parent / 'loscov_dev_fork' / 'lenses_COSMOS-Web.txt'
OUTPUT_FILE = ROOT / 'results' / 'fisher_results_cosmos_lllelp.pkl'


forecast = FisherForecast(
    data_dir=str(DATA_DIR),
    lens_file=str(LENS_FILE),
    verbose=True,
)
forecast.setup()

# LOSCOV preprocessing retains angular-bin metadata for all six probes even
# when a targeted production run contains only LLLELP covariance blocks. The
# covariance defines the Fisher data vector, so activate exactly those probes.
covariance_probes = [key[2:] for key in forecast.sizes]
missing = [
    probe for probe in covariance_probes
    if probe not in forecast.theory.probes
]
if missing:
    raise ValueError(
        'Covariance probes lack angular-distribution metadata: '
        f'{missing}'
    )
forecast.theory.probes = [
    probe for probe in forecast.theory.probes
    if probe in set(covariance_probes)
]
print(f'Active covariance probes: {forecast.theory.probes}')

results = forecast.compute_fisher()
print_constraints_table(results)
forecast.save_results(OUTPUT_FILE)
print(f'Saved to {OUTPUT_FILE}')
