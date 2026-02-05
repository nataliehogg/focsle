"""Tests for data loading utilities."""

import numpy as np
import pytest
import tempfile
from pathlib import Path
import pickle


# Mock classes for pickle compatibility (must be at module level)
class MockDistribution:
    """Mock distribution class for testing."""
    def __init__(self, nbinz=4):
        self.Nbinz = nbinz


def test_parse_dataset_name():
    """Test parsing dataset directory names."""
    from focsle.data_loader import parse_dataset_name

    # Test standard format
    name1 = "Nlens=1e5_sigL=0.2"
    params1 = parse_dataset_name(name1)

    assert params1['Nlens'] == 1e5
    assert params1['sigL'] == 0.2

    # Test scientific notation
    name2 = "Nlens=1e4_sigL=0.1"
    params2 = parse_dataset_name(name2)

    assert params2['Nlens'] == 1e4
    assert params2['sigL'] == 0.1


def test_build_full_covariance():
    """Test full covariance matrix construction from blocks."""
    from focsle.data_loader import build_full_covariance

    # Create test blocks
    n_LL, n_LE, n_LP = 5, 10, 8

    cov_blocks = {
        'LLLL': np.eye(n_LL),
        'LELE': np.eye(n_LE),
        'LPLP': np.eye(n_LP),
        'LLLE': np.zeros((n_LL, n_LE)),
        'LELP': np.zeros((n_LE, n_LP)),
        'LLLP': np.zeros((n_LL, n_LP)),
    }

    C_full, sizes = build_full_covariance(cov_blocks)

    # Check sizes
    assert sizes['n_LL'] == n_LL
    assert sizes['n_LE'] == n_LE
    assert sizes['n_LP'] == n_LP

    # Check total size
    assert C_full.shape == (n_LL + n_LE + n_LP, n_LL + n_LE + n_LP)

    # Check diagonal blocks
    np.testing.assert_array_equal(C_full[:n_LL, :n_LL], cov_blocks['LLLL'])
    np.testing.assert_array_equal(C_full[n_LL:n_LL+n_LE, n_LL:n_LL+n_LE], cov_blocks['LELE'])
    np.testing.assert_array_equal(C_full[n_LL+n_LE:, n_LL+n_LE:], cov_blocks['LPLP'])

    # Check symmetry
    np.testing.assert_array_equal(C_full, C_full.T)


def test_detect_nbins_from_data():
    """Test automatic detection of tomographic bins."""
    from focsle.data_loader import detect_nbins

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create mock redshift distribution
        dist_data = {'E': MockDistribution(4), 'P': MockDistribution(4)}

        with open(tmp_path / 'redshift_distributions', 'wb') as f:
            pickle.dump(dist_data, f)

        nbins = detect_nbins(str(tmp_path))
        assert nbins == 4


def test_load_lens_catalog():
    """Test loading lens catalog from file."""
    from focsle.data_loader import load_lens_catalog

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        lens_file = tmp_path / 'test_lenses.txt'

        # Create test lens catalog
        test_data = np.array([
            [0.3, 1.2],
            [0.4, 1.5],
            [0.5, 1.8],
        ])

        np.savetxt(lens_file, test_data)

        z_d, z_s = load_lens_catalog(str(lens_file))

        np.testing.assert_array_equal(z_d, test_data[:, 0])
        np.testing.assert_array_equal(z_s, test_data[:, 1])
