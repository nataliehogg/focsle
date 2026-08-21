#!/usr/bin/env python
"""Compute five-bin EEEPPP forecasts for COSMOS-Web and full COSMOS."""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOSCOV_ROOT = ROOT.parent / 'loscov_dev_fork'
sys.path.insert(0, str(ROOT / 'src'))

from focsle import FisherForecast
from focsle.plotting import print_constraints_table


LENS_FILE = LOSCOV_ROOT / 'lenses_COSMOS-Web.txt'
FOUR_PARAMETERS = ['Omega_m', 'sigma_8', 'w0', 'wa']
TWO_PARAMETERS = ['Omega_m', 'sigma_8']
BIAS_PARAMETERS = [f'bias_P_{index}' for index in range(5)]
FORECASTS = {
    'cosmos-web': {
        'data_dir': (
            LOSCOV_ROOT
            / 'data'
            / 'Nlens=1e2_sigL=0.01_Nbin_z=5_SNR_goal=8_Nbin_max=20_'
              'nsamp=5e3_old_scenario=cosmos-web-5bin_grid=fine'
        ),
        'two_parameter_output': (
            ROOT / 'results'
            / 'fisher_results_cosmos_web_5bin_eeeppp.pkl'
        ),
        'four_parameter_output': (
            ROOT / 'results'
            / 'fisher_results_cosmos_web_5bin_eeeppp_w0wa.pkl'
        ),
        'two_parameter_bias_output': (
            ROOT / 'results'
            / 'fisher_results_cosmos_web_5bin_eeeppp_free_bias.pkl'
        ),
        'four_parameter_bias_output': (
            ROOT / 'results'
            / 'fisher_results_cosmos_web_5bin_eeeppp_w0wa_free_bias.pkl'
        ),
    },
    'cosmos': {
        'data_dir': (
            LOSCOV_ROOT
            / 'data'
            / 'Nlens=4e2_sigL=0.01_Nbin_z=5_SNR_goal=8_Nbin_max=20_'
              'nsamp=5e3_old_scenario=cosmos-5bin_grid=fine'
        ),
        'two_parameter_output': (
            ROOT / 'results'
            / 'fisher_results_cosmos_5bin_eeeppp.pkl'
        ),
        'four_parameter_output': (
            ROOT / 'results'
            / 'fisher_results_cosmos_5bin_eeeppp_w0wa.pkl'
        ),
        'two_parameter_bias_output': (
            ROOT / 'results'
            / 'fisher_results_cosmos_5bin_eeeppp_free_bias.pkl'
        ),
        'four_parameter_bias_output': (
            ROOT / 'results'
            / 'fisher_results_cosmos_5bin_eeeppp_w0wa_free_bias.pkl'
        ),
    },
}


def retain_covariance_probes(forecast):
    """Make the theory data vector match the probes present on disk."""
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


def compute_and_save(forecast, parameters, output):
    """Compute one parameter selection and save its Fisher results."""
    results = forecast.compute_fisher(param_names=parameters)
    print_constraints_table(results)
    forecast.save_results(output)
    print(f'Saved to {output}')


for name, paths in FORECASTS.items():
    print('\n' + '=' * 70)
    print(f'Preparing {name} five-bin EEEPPP forecasts')
    print('=' * 70)

    forecast = FisherForecast(
        data_dir=str(paths['data_dir']),
        lens_file=str(LENS_FILE),
        verbose=True,
    )
    forecast.theory.requested_probes = ['EE', 'EP', 'PP']
    # The four-parameter setup includes the CAMB response tables needed for
    # w0-wa. The same setup can then be reused for the two-parameter forecast.
    forecast.setup(param_names=FOUR_PARAMETERS)
    retain_covariance_probes(forecast)

    compute_and_save(
        forecast,
        TWO_PARAMETERS,
        paths['two_parameter_output'],
    )
    compute_and_save(
        forecast,
        FOUR_PARAMETERS,
        paths['four_parameter_output'],
    )
    compute_and_save(
        forecast,
        TWO_PARAMETERS + BIAS_PARAMETERS,
        paths['two_parameter_bias_output'],
    )
    compute_and_save(
        forecast,
        FOUR_PARAMETERS + BIAS_PARAMETERS,
        paths['four_parameter_bias_output'],
    )
