"""Tests for propagating cached w0/wa responses through the JAX theory."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from focsle.theory import TheoryJAX


def _response_theory():
    """Small synthetic TheoryJAX instance with known response derivatives."""
    theory = TheoryJAX.__new__(TheoryJAX)
    theory.c_km_s = 299792.458
    theory.cosmo_fid = {
        "H0": 67.37,
        "Omega_m": 0.3,
        "sigma8": 0.8,
        "w0": -1.0,
        "wa": 0.0,
    }

    theory.Om_bg = jnp.array([0.3, 0.4])
    theory.z_bg = jnp.array([0.0, 1.0, 2.0])
    theory.chi_bg_table = jnp.array([
        [0.0, 1000.0, 2000.0],
        [0.0, 900.0, 1800.0],
    ])
    theory.dchi_dw0 = jnp.array([0.0, 100.0, 200.0])
    theory.dchi_dwa = jnp.array([0.0, -50.0, -100.0])
    theory.w0_step = 0.05
    theory.wa_step = 0.1

    theory.Om_grid = jnp.array([0.3, 0.4])
    theory.As_grid = jnp.array([1.0, 2.0])
    theory.z_grid = jnp.array([0.0, 1.0, 2.0])
    theory.k_grid = jnp.array([0.1, 1.0])
    theory.sigma8_grid = jnp.array([[0.7, 0.9], [0.7, 0.9]])
    theory.Pk_grid = jnp.full((2, 2, 3, 2), 4.0)
    theory.dlnPk_dw0 = jnp.full((3, 2), 2.0)
    theory.dlnPk_dwa = jnp.full((3, 2), -3.0)

    theory.z_pdf_grid = jnp.linspace(0.0, 2.0, 17)
    theory.P_pdf_table = jnp.ones((1, 17)) / 2.0
    return theory


def test_distance_response_and_inverse_are_differentiable():
    theory = _response_theory()
    shifted_chi = theory.chi_of_z(0.3, 1.0, w0=-0.9, wa=0.1)

    np.testing.assert_allclose(np.asarray(shifted_chi), 1005.0, rtol=1e-6)
    np.testing.assert_allclose(
        np.asarray(theory.z_of_chi(0.3, shifted_chi, w0=-0.9, wa=0.1)),
        1.0,
        rtol=1e-6,
    )
    derivative = jax.grad(
        lambda w0: theory.chi_of_z(0.3, 1.0, w0=w0, wa=0.0)
    )(-1.0)
    np.testing.assert_allclose(np.asarray(derivative), 100.0, rtol=1e-6)


def test_power_response_is_log_linear_and_differentiable():
    theory = _response_theory()
    np.testing.assert_allclose(
        np.asarray(theory.Pk_interp(0.3, 0.8, 0.5, 0.3)),
        4.0,
    )

    def log_power(w0, wa):
        return jnp.log(theory.Pk_interp(
            0.3, 0.8, 0.5, 0.3, w0=w0, wa=wa
        ))

    actual = jnp.exp(log_power(-0.9, 0.2))
    expected = 4.0 * np.exp(0.1 * 2.0 + 0.2 * -3.0)
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-6)
    np.testing.assert_allclose(
        np.asarray(jax.grad(log_power, argnums=(0, 1))(-1.0, 0.0)),
        np.array([2.0, -3.0]),
        rtol=1e-6,
    )


def test_explicit_dark_energy_requires_cached_responses():
    theory = _response_theory()
    del theory.dlnPk_dw0
    del theory.dlnPk_dwa

    with pytest.raises(RuntimeError, match="setup_dark_energy_responses"):
        theory.Pk_interp(0.3, 0.8, 0.5, 0.3, w0=-0.9, wa=0.0)


def test_lensing_kernel_geometry_responses_are_precomputed():
    theory = _response_theory()
    theory.z_d_array = np.array([0.5])
    theory.z_s_array = np.array([1.5])
    theory.N_lenses = 1
    theory.Nbinz_E = 1
    theory.E_pdf_table = jnp.ones((1, 17)) / 2.0

    theory._precompute_QL_mean(
        chi_min=10.0, chi_max=1800.0, nchi=24, verbose=False
    )
    theory._precompute_QE_mean(
        chi_min=10.0, chi_max=1800.0, nchi=24, verbose=False
    )

    for name in (
        "dKL_mean_dw0_grid",
        "dKL_mean_dwa_grid",
        "dKE_mean_dw0_grid",
        "dKE_mean_dwa_grid",
    ):
        values = np.asarray(getattr(theory, name))
        assert np.all(np.isfinite(values))
        assert np.any(np.abs(values) > 0.0)

    ql_derivative = jax.grad(
        lambda w0: theory.QL_mean(500.0, 0.3, w0=w0, wa=0.0)
    )(-1.0)
    qe_derivative = jax.grad(
        lambda wa: theory.QE_mean(0.3, 500.0, 0, w0=-1.0, wa=wa)
    )(0.0)
    assert np.isfinite(np.asarray(ql_derivative))
    assert np.isfinite(np.asarray(qe_derivative))
    assert float(ql_derivative) != 0.0
    assert float(qe_derivative) != 0.0


def test_cpl_response_reaches_cl_and_legacy_fiducial_qp_is_unchanged():
    theory = _response_theory()
    z = theory.z_of_chi(0.3, 500.0)
    legacy_Ez = jnp.sqrt(0.3 * (1.0 + z) ** 3 + 0.7)
    expected_qp = (
        theory.cosmo_fid["H0"] / theory.c_km_s
        * legacy_Ez
        * 0.5
        * theory.galaxy_bias(z)
        / 500.0
    )
    np.testing.assert_allclose(
        np.asarray(theory.QP_mean(0.3, 500.0, 0)),
        np.asarray(expected_qp),
        rtol=1e-6,
    )

    def cl_from_dark_energy(w0, wa):
        return theory.compute_Cl_PP_jax(
            0.3, 0.8, 100.0, 0, w0=w0, wa=wa
        )

    fiducial_cl = cl_from_dark_energy(-1.0, 0.0)
    derivatives = jax.grad(cl_from_dark_energy, argnums=(0, 1))(-1.0, 0.0)
    assert np.isfinite(np.asarray(fiducial_cl))
    assert float(fiducial_cl) > 0.0
    assert np.all(np.isfinite(np.asarray(derivatives)))
    assert np.all(np.abs(np.asarray(derivatives)) > 0.0)
