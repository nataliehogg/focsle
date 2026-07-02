"""Tests for the Ogata-quadrature Hankel transforms.

The transforms turn C_ell into xi(theta); their accuracy at large theta is
what A7 was about, so these tests pin (a) an analytic transform pair and
(b) agreement with the `hankel` package that loscov uses to build the data.
"""

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


@pytest.fixture(scope='module')
def theory():
    from focsle.theory import TheoryJAX
    return TheoryJAX()  # fallback lens sample is fine; we only use transforms


def test_hankel_j0_analytic_gaussian(theory):
    """For C(ell) = exp(-a ell^2):
    xi(theta) = 1/(2 pi) int ell C J0(ell theta) d ell = exp(-theta^2/4a)/(4 pi a).
    """
    a = 1e-6
    ell = jnp.logspace(-1, 6, 300)
    Cl = jnp.exp(-a * ell ** 2)

    thetas = np.array([3e-4, 1e-3, 3e-3])  # radians; spans small to large x
    xi = np.array(theory.hankel_j0(Cl, ell, jnp.array(thetas)))
    expected = np.exp(-thetas ** 2 / (4 * a)) / (4 * np.pi * a)

    # Accuracy is limited by linear interpolation of C on the 300-point
    # ell grid (the Gaussian is a curvature worst case), not by the Ogata
    # quadrature itself; sub-percent suffices (data MC noise is ~1%).
    np.testing.assert_allclose(xi, expected, rtol=5e-3)


def test_hankel_matches_hankel_package(theory):
    """All three orders must agree with loscov's transform machinery."""
    hankel_pkg = pytest.importorskip('hankel')

    # smooth, realistically shaped spectrum with decaying tails
    def cl_func(ell):
        ell = np.asarray(ell, dtype=float)
        return ell / (1.0 + (ell / 60.0) ** 3.2)

    ell = jnp.logspace(-1, 6, 300)
    Cl = jnp.array(cl_func(np.array(ell)))

    thetas = np.array([3e-4, 3e-3, 3e-2, 8e-2])  # ~1 arcmin to ~4.5 degrees

    for nu, method in [(0, theory.hankel_j0), (2, theory.hankel_j2),
                       (4, theory.hankel_j4)]:
        ht = hankel_pkg.HankelTransform(nu=nu, N=10000, h=1e-2)
        ref = ht.transform(cl_func, thetas, ret_err=False) / (2 * np.pi)
        ours = np.array(method(Cl, ell, jnp.array(thetas)))
        # residual difference is our C-grid interpolation (the package
        # evaluates the function exactly); sub-percent is sufficient
        np.testing.assert_allclose(ours, ref, rtol=1e-2,
                                   err_msg=f'J{nu} mismatch vs hankel package')


def test_hankel_scalar_theta(theory):
    """Scalar theta input must return a scalar, consistent with array input."""
    ell = jnp.logspace(-1, 6, 300)
    Cl = jnp.exp(-1e-6 * ell ** 2)

    xi_scalar = theory.hankel_j0(Cl, ell, jnp.array(1e-3))
    xi_array = theory.hankel_j0(Cl, ell, jnp.array([1e-3]))

    assert np.ndim(xi_scalar) == 0
    np.testing.assert_allclose(float(xi_scalar), float(xi_array[0]), rtol=1e-12)


def test_hankel_differentiable(theory):
    """The transform must remain autodiff-able end to end (Fisher needs it)."""
    from jax import grad

    ell = jnp.logspace(-1, 6, 300)

    def xi_of_amp(A):
        Cl = A * jnp.exp(-1e-6 * ell ** 2)
        return theory.hankel_j0(Cl, ell, jnp.array(1e-3))

    g = grad(xi_of_amp)(1.0)
    assert np.isfinite(float(g))
    # xi is linear in the amplitude, so the gradient equals xi at A=1
    np.testing.assert_allclose(float(g), float(xi_of_amp(1.0)), rtol=1e-10)
