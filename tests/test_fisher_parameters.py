"""Tests for general Fisher parameter selection and marginalized plotting."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import pytest

from focsle.fisher import FisherForecast


class _LinearTheory:
    """Four-output linear model with an exactly known Jacobian."""

    cosmo_fid = {
        "Omega_m": 0.3,
        "sigma8": 0.8,
        "w0": -1.0,
        "wa": 0.0,
    }
    dlnPk_dw0 = object()
    dlnPk_dwa = object()

    def predict_data_vector_jax(
        self, Omega_m, sigma_8, ell_grid=None, w0=None, wa=None
    ):
        if w0 is None:
            w0 = self.cosmo_fid["w0"]
        if wa is None:
            wa = self.cosmo_fid["wa"]
        return jnp.stack((Omega_m, sigma_8, w0, wa))

    @staticmethod
    def prediction_sizes():
        return {"LL": 2, "LE": 1, "LP": 1}


def _linear_forecast():
    forecast = FisherForecast.__new__(FisherForecast)
    forecast._is_setup = True
    forecast.verbose = False
    forecast.theory = _LinearTheory()
    forecast.param_names = list(FisherForecast.SUPPORTED_PARAMETERS)
    forecast.sizes = {"n_LL": 2, "n_LE": 1, "n_LP": 1}
    forecast.C_full = np.eye(4)
    forecast.C_inv = np.eye(4)
    forecast.C_LL_inv = np.eye(2)
    forecast.C_LE_inv = np.eye(1)
    forecast.C_LP_inv = np.eye(1)
    forecast.fisher_matrices = {}
    forecast.constraints = {}
    forecast.jacobian = None
    return forecast


@pytest.mark.parametrize(
    "parameters",
    [
        [],
        ["Omega_m", "not_a_parameter"],
        ["w0", "w0"],
    ],
)
def test_parameter_selection_rejects_invalid_lists(parameters):
    with pytest.raises(ValueError):
        FisherForecast._normalise_param_names(parameters)


def test_four_parameter_fisher_uses_requested_order():
    forecast = _linear_forecast()
    names = ["Omega_m", "sigma_8", "w0", "wa"]

    results = forecast.compute_fisher(names)

    assert results["param_names"] == names
    np.testing.assert_allclose(results["fiducial"], [0.3, 0.8, -1.0, 0.0])
    np.testing.assert_allclose(forecast.jacobian, np.eye(4), atol=1e-7)
    np.testing.assert_allclose(
        results["fisher_matrices"]["Combined"], np.eye(4), atol=1e-7
    )
    combined = results["constraints"]["Combined"]
    np.testing.assert_allclose(combined["errors"], np.ones(4))
    assert np.isnan(combined["fractional_errors"][-1])
    np.testing.assert_allclose(combined["correlation_matrix"], np.eye(4))


def test_parameter_subset_holds_omitted_values_fixed():
    forecast = _linear_forecast()
    names = ["wa", "Omega_m"]

    results = forecast.compute_fisher(names)

    assert results["param_names"] == names
    np.testing.assert_allclose(results["fiducial"], [0.0, 0.3])
    expected_jacobian = np.array([
        [0.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, 0.0],
    ])
    np.testing.assert_allclose(forecast.jacobian, expected_jacobian, atol=1e-7)
    np.testing.assert_allclose(
        results["fisher_matrices"]["Combined"], np.eye(2), atol=1e-7
    )


def test_dark_energy_cannot_be_added_after_two_parameter_setup():
    forecast = _linear_forecast()
    forecast.theory = object()

    with pytest.raises(RuntimeError, match=r"rerun setup"):
        forecast.compute_fisher(["Omega_m", "w0"])


@pytest.fixture(autouse=True)
def _headless_plotting():
    import focsle.plotting  # noqa: F401

    matplotlib.rcParams["text.usetex"] = False
    yield
    plt.close("all")


def test_marginalize_fisher_extracts_pair_from_full_covariance():
    from focsle.plotting import marginalize_fisher_to_pair

    covariance = np.array([
        [4.0, 0.3, 0.5, 0.2],
        [0.3, 3.0, 0.1, 0.4],
        [0.5, 0.1, 2.0, -0.6],
        [0.2, 0.4, -0.6, 1.5],
    ])
    fisher = np.linalg.inv(covariance)
    names = ["Omega_m", "sigma_8", "w0", "wa"]

    pair_fisher, pair_fiducial = marginalize_fisher_to_pair(
        fisher, [0.3, 0.8, -1.0, 0.0], names, ("w0", "wa")
    )

    np.testing.assert_allclose(
        np.linalg.inv(pair_fisher), covariance[np.ix_([2, 3], [2, 3])]
    )
    assert pair_fiducial == (-1.0, 0.0)


def test_four_parameter_constraint_plot_uses_selected_pair():
    from focsle.plotting import plot_constraints

    results = {
        "fisher_matrices": {"Combined": np.diag([4.0, 5.0, 6.0, 7.0])},
        "fiducial": [0.3, 0.8, -1.0, 0.0],
        "param_names": ["Omega_m", "sigma_8", "w0", "wa"],
    }

    fig = plot_constraints(
        results, probes=["Combined"], parameter_pair=("w0", "wa")
    )

    assert len(fig.axes[0].patches) == 2
    assert fig.axes[0].get_xlabel() == r"$w_0$"
    assert fig.axes[0].get_ylabel() == r"$w_a$"


def test_plot_pair_must_be_varied_and_distinct():
    from focsle.plotting import marginalize_fisher_to_pair

    with pytest.raises(ValueError, match="distinct"):
        marginalize_fisher_to_pair(
            np.eye(2), [0.3, 0.8], ["Omega_m", "sigma_8"],
            ("Omega_m", "Omega_m"),
        )
    with pytest.raises(ValueError, match="were not varied"):
        marginalize_fisher_to_pair(
            np.eye(2), [0.3, 0.8], ["Omega_m", "sigma_8"],
            ("Omega_m", "w0"),
        )
