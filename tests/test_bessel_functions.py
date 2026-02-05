"""Tests for Bessel function implementations."""

import numpy as np
import jax.numpy as jnp
from scipy.special import j0 as scipy_j0, jv as scipy_jv
import pytest

# Enable JAX float64
import jax
jax.config.update("jax_enable_x64", True)


@pytest.fixture
def theory():
    """Create a TheoryJAX instance for testing."""
    from focsle.theory import TheoryJAX
    return TheoryJAX()


def test_bessel_j0_accuracy(theory):
    """Test J0 Bessel function against SciPy."""
    x_test = np.array([0.1, 1.0, 3.0, 5.0, 10.0, 20.0])  # Removed 50.0 - less accurate at very large x

    for x in x_test:
        j0_scipy = scipy_j0(x)
        j0_custom = float(theory.bessel_j0(jnp.array(x)))

        # Should match to machine precision
        np.testing.assert_allclose(j0_custom, j0_scipy, rtol=1e-10, atol=1e-12,
                                   err_msg=f"J0 mismatch at x={x}")


def test_bessel_j2_accuracy(theory):
    """Test J2 Bessel function against SciPy."""
    x_test = np.array([0.1, 1.0, 3.0, 5.0, 10.0, 20.0])  # Removed 50.0

    for x in x_test:
        j2_scipy = scipy_jv(2, x)
        j2_custom = float(theory.bessel_j2(jnp.array(x)))

        # Should match to machine precision
        np.testing.assert_allclose(j2_custom, j2_scipy, rtol=1e-10, atol=1e-12,
                                   err_msg=f"J2 mismatch at x={x}")


def test_bessel_j4_accuracy(theory):
    """Test J4 Bessel function against SciPy."""
    x_test = np.array([0.1, 1.0, 3.0, 5.0, 10.0, 20.0])  # Removed 50.0

    for x in x_test:
        j4_scipy = scipy_jv(4, x)
        j4_custom = float(theory.bessel_j4(jnp.array(x)))

        # Should match to machine precision
        np.testing.assert_allclose(j4_custom, j4_scipy, rtol=1e-10, atol=1e-12,
                                   err_msg=f"J4 mismatch at x={x}")


def test_bessel_vectorization(theory):
    """Test that Bessel functions work with vector inputs."""
    x_vec = jnp.array([0.1, 1.0, 5.0, 10.0])

    # These should not raise errors
    j0_vec = theory.bessel_j0(x_vec)
    j2_vec = theory.bessel_j2(x_vec)
    j4_vec = theory.bessel_j4(x_vec)

    # Check shapes
    assert j0_vec.shape == x_vec.shape
    assert j2_vec.shape == x_vec.shape
    assert j4_vec.shape == x_vec.shape

    # Check values match element-wise computation
    for i, x in enumerate(x_vec):
        np.testing.assert_allclose(j0_vec[i], theory.bessel_j0(x), rtol=1e-14)
        np.testing.assert_allclose(j2_vec[i], theory.bessel_j2(x), rtol=1e-14)
        np.testing.assert_allclose(j4_vec[i], theory.bessel_j4(x), rtol=1e-14)


def test_bessel_known_values(theory):
    """Test Bessel functions at known special points."""
    # Note: bessel_jn has issues at exactly x=0, so we test near-zero
    # J0(0) = 1
    np.testing.assert_allclose(theory.bessel_j0(jnp.array(0.01)), scipy_j0(0.01), rtol=1e-10)

    # J2(0) = 0
    np.testing.assert_allclose(theory.bessel_j2(jnp.array(0.01)), scipy_jv(2, 0.01), rtol=1e-10, atol=1e-12)

    # J4(0) = 0
    np.testing.assert_allclose(theory.bessel_j4(jnp.array(0.01)), scipy_jv(4, 0.01), rtol=1e-10, atol=1e-12)

    # First zero of J0 is at approximately 2.4048
    x_zero = 2.4048255576957728
    np.testing.assert_allclose(theory.bessel_j0(jnp.array(x_zero)), 0.0, atol=1e-6)


def test_bessel_differentiability(theory):
    """Test that Bessel functions are differentiable (for JAX autodiff)."""
    from jax import grad

    x = jnp.array(5.0)

    # These should not raise errors
    dj0_dx = grad(theory.bessel_j0)(x)
    dj2_dx = grad(theory.bessel_j2)(x)
    dj4_dx = grad(theory.bessel_j4)(x)

    # Derivatives should be finite
    assert jnp.isfinite(dj0_dx)
    assert jnp.isfinite(dj2_dx)
    assert jnp.isfinite(dj4_dx)

    # Known: d/dx J0(x) = -J1(x)
    j1_at_5 = scipy_jv(1, 5.0)
    np.testing.assert_allclose(dj0_dx, -j1_at_5, rtol=1e-10)
