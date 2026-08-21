#!/usr/bin/env python
"""Compute conditional COSMOS w0-wa forecasts for 3, 6, and 9 bins."""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOSCOV_ROOT = ROOT.parent / 'loscov_dev_fork'
sys.path.insert(0, str(ROOT / 'src'))

from focsle import FisherForecast
from focsle.plotting import print_constraints_table


PARAMETERS = ['w0', 'wa']
LENS_FILE = LOSCOV_ROOT / 'lenses_COSMOS-Web.txt'
FORECASTS = {
    bin_count: {
        'data_dir': (
            LOSCOV_ROOT
            / 'data'
            / (
                f'Nlens=4e2_sigL=0.01_Nbin_z={bin_count}_SNR_goal=8_'
                f'Nbin_max=20_nsamp=5e3_new_scenario=cosmos-{bin_count}bin_'
                'grid=fine'
            )
        ),
        'output': (
            ROOT
            / 'results'
            / (
                f'fisher_results_cosmos_{bin_count}bin_lllelp_'
                'w0wa_fixed_omegam_sigma8.pkl'
            )
        ),
    }
    for bin_count in (3, 6, 9)
}


for bin_count, paths in FORECASTS.items():
    print('\n' + '=' * 70)
    print(
        f'Computing COSMOS {bin_count}-bin conditional w0/wa forecast '
        '(Omega_m and sigma_8 fixed)'
    )
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
