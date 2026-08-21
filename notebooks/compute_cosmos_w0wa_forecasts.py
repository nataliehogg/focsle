#!/usr/bin/env python
"""Compute four-parameter COSMOS-Web and full-COSMOS forecasts."""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOSCOV_ROOT = ROOT.parent / 'loscov_dev_fork'
sys.path.insert(0, str(ROOT / 'src'))

from focsle import FisherForecast
from focsle.plotting import print_constraints_table


PARAMETERS = ['Omega_m', 'sigma_8', 'w0', 'wa']
LENS_FILE = LOSCOV_ROOT / 'lenses_COSMOS-Web.txt'
FORECASTS = {
    'cosmos_web': {
        'data_dir': (
            LOSCOV_ROOT
            / 'data'
            / 'Nlens=1e2_sigL=0.01_Nbin_z=1_SNR_goal=8_Nbin_max=20_'
              'nsamp=5e3_new_scenario=cosmos-web_grid=fine'
        ),
        'output': (
            ROOT / 'results' / 'fisher_results_cosmos_web_lllelp_w0wa.pkl'
        ),
    },
    'cosmos': {
        'data_dir': (
            LOSCOV_ROOT
            / 'data'
            / 'Nlens=4e2_sigL=0.01_Nbin_z=1_SNR_goal=8_Nbin_max=20_'
              'nsamp=5e3_new_scenario=cosmos_grid=fine'
        ),
        'output': (
            ROOT / 'results' / 'fisher_results_cosmos_lllelp_w0wa.pkl'
        ),
    },
}


for name, paths in FORECASTS.items():
    print('\n' + '=' * 70)
    print(f'Computing {name} LLLELP Omega_m/sigma_8/w0/wa forecast')
    print('=' * 70)

    forecast = FisherForecast(
        data_dir=str(paths['data_dir']),
        lens_file=str(LENS_FILE),
        verbose=True,
    )
    forecast.setup(param_names=PARAMETERS)

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
    covariance_probe_set = set(covariance_probes)
    forecast.theory.probes = [
        probe for probe in forecast.theory.probes
        if probe in covariance_probe_set
    ]
    print(f'Active covariance probes: {forecast.theory.probes}')

    results = forecast.compute_fisher()
    print_constraints_table(results)
    forecast.save_results(paths['output'])
    print(f"Saved to {paths['output']}")
