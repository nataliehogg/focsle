"""Tests for matrix operations and numerical stability."""

import numpy as np
import pytest


# Mock classes for testing (must be at module level for pickle)
class MockRedshiftDist:
    """Mock redshift distribution."""
    Nbinz = 1
    limits = [0.0, 1.0]


class MockAngularDist:
    """Mock angular distribution."""
    Thetas = np.array([0.01])


def test_robust_symmetric_inverse():
    """Test robust matrix inversion with positive definiteness enforcement."""
    from focsle.fisher import FisherForecast
    import tempfile
    from pathlib import Path
    import pickle

    # Create a mock FisherForecast instance to access the method
    class MockFisher:
        def __init__(self):
            self.forecast = None

        def _robust_symmetric_inverse(self, M, verbose=False):
            """Import method from FisherForecast."""
            # Create a temporary directory with minimal structure
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                cov_dir = tmp_path / "covariance" / "LLLL"
                cov_dir.mkdir(parents=True)

                # Create minimal pickle files
                import pickle
                dummy_cov = np.eye(2)
                for name in ['ccov', 'ncov', 'scov']:
                    with open(cov_dir / name, 'wb') as f:
                        pickle.dump(dummy_cov, f)

                # Create redshift_distributions file
                with open(tmp_path / 'redshift_distributions', 'wb') as f:
                    pickle.dump({'E': MockRedshiftDist(), 'P': MockRedshiftDist()}, f)

                # Create angular_distributions file
                ang_data = {
                    'LL_plus': MockAngularDist(),
                    'LL_minus': MockAngularDist(),
                    'LE_plus': [MockAngularDist()],
                    'LE_minus': [MockAngularDist()],
                    'LP': [MockAngularDist()],
                }
                with open(tmp_path / 'angular_distributions', 'wb') as f:
                    pickle.dump(ang_data, f)

                forecast = FisherForecast(str(tmp_path), verbose=False)
                return forecast._robust_symmetric_inverse(M, verbose=verbose)

    mock = MockFisher()

    # Test 1: Well-conditioned positive definite matrix
    M1 = np.array([[4.0, 1.0], [1.0, 3.0]])
    M1_inv = mock._robust_symmetric_inverse(M1)

    # Check it's actually an inverse
    np.testing.assert_allclose(M1 @ M1_inv, np.eye(2), atol=1e-10)

    # Check it's symmetric
    np.testing.assert_allclose(M1_inv, M1_inv.T, atol=1e-14)

    # Check positive definiteness
    eigenvalues = np.linalg.eigvalsh(M1_inv)
    assert np.all(eigenvalues > 0)

    # Test 2: Matrix with small negative eigenvalue (numerical noise)
    M2 = np.array([[1.0, 0.9999], [0.9999, 1.0]])
    M2[0, 1] += 1e-15  # Break symmetry slightly
    M2_inv = mock._robust_symmetric_inverse(M2)

    # Should still be positive definite after regularization
    eigenvalues2 = np.linalg.eigvalsh(M2_inv)
    assert np.all(eigenvalues2 > 0)

    # Should be symmetric
    np.testing.assert_allclose(M2_inv, M2_inv.T, atol=1e-13)


def test_covariance_symmetry():
    """Test that block covariance construction maintains symmetry."""
    from focsle.data_loader import build_full_covariance

    n_LL, n_LE, n_LP = 10, 20, 15

    # Create symmetric blocks
    LLLL = np.random.randn(n_LL, n_LL)
    LLLL = (LLLL + LLLL.T) / 2

    LELE = np.random.randn(n_LE, n_LE)
    LELE = (LELE + LELE.T) / 2

    LPLP = np.random.randn(n_LP, n_LP)
    LPLP = (LPLP + LPLP.T) / 2

    # Cross-correlation blocks
    LLLE = np.random.randn(n_LL, n_LE)
    LELP = np.random.randn(n_LE, n_LP)
    LLLP = np.random.randn(n_LL, n_LP)

    cov_blocks = {
        'LLLL': LLLL,
        'LELE': LELE,
        'LPLP': LPLP,
        'LLLE': LLLE,
        'LELP': LELP,
        'LLLP': LLLP,
    }

    C_full, sizes = build_full_covariance(cov_blocks)

    # Check symmetry
    np.testing.assert_allclose(C_full, C_full.T, atol=1e-14)

    # Check sizes
    assert sizes['n_LL'] == n_LL
    assert sizes['n_LE'] == n_LE
    assert sizes['n_LP'] == n_LP
    assert C_full.shape == (n_LL + n_LE + n_LP, n_LL + n_LE + n_LP)


def test_fisher_matrix_properties():
    """Test basic properties of Fisher matrices."""
    # A valid 2x2 Fisher matrix should be:
    # - Symmetric
    # - Positive semi-definite
    # - Invertible (for non-degenerate cases)

    F = np.array([[100.0, -20.0],
                  [-20.0, 50.0]])

    # Symmetry
    np.testing.assert_allclose(F, F.T)

    # Positive definite
    eigenvalues = np.linalg.eigvalsh(F)
    assert np.all(eigenvalues >= 0)

    # Invertible
    C = np.linalg.inv(F)
    errors = np.sqrt(np.diag(C))

    # Errors should be positive
    assert np.all(errors > 0)

    # Check correlation coefficient is in [-1, 1]
    corr = C[0, 1] / (errors[0] * errors[1])
    assert -1 <= corr <= 1
