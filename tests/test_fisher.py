"""Tests for FisherForecast API guards."""

import pytest


def _bare_forecast():
    """FisherForecast instance without running the heavy __init__."""
    from focsle.fisher import FisherForecast
    forecast = FisherForecast.__new__(FisherForecast)
    forecast._is_setup = False
    forecast.jacobian = None
    return forecast


def test_compute_custom_fisher_requires_setup():
    forecast = _bare_forecast()

    with pytest.raises(RuntimeError, match=r'setup\(\)'):
        forecast.compute_custom_fisher(['LL'])


def test_compute_custom_fisher_requires_jacobian():
    """After setup() but before compute_fisher(), the error must say what to run.

    Regression test: the old guard only checked _is_setup, so this call
    crashed later with an opaque TypeError on self.jacobian[...].
    """
    forecast = _bare_forecast()
    forecast._is_setup = True

    with pytest.raises(RuntimeError, match=r'compute_fisher\(\)'):
        forecast.compute_custom_fisher(['LL'])
