"""Tests for the new-format (4-index, 6-probe) covariance loader."""

import numpy as np
import pickle
import pytest
import tempfile
from pathlib import Path


def test_get_probe_sections():
    """Section enumeration must mirror loscov's get_valid_pairs."""
    from focsle.data_loader import get_probe_sections

    assert get_probe_sections('LL', 1, 6, 6) == [(0, 0)]
    assert get_probe_sections('LE', 1, 3, 3) == [(0, 0), (0, 1), (0, 2)]
    assert get_probe_sections('LP', 1, 3, 2) == [(0, 0), (0, 1)]
    # EE/EP: lower triangle in row-major order
    assert get_probe_sections('EE', 1, 3, 3) == [
        (0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2)]
    assert get_probe_sections('EP', 1, 2, 2) == [(0, 0), (1, 0), (1, 1)]
    # PP: diagonal only
    assert get_probe_sections('PP', 1, 3, 3) == [(0, 0), (1, 1), (2, 2)]


def _write_block(path, mat):
    with open(path, 'wb') as f:
        pickle.dump(mat, f)


def _make_synthetic_dataset(tmp_path, rng):
    """
    Tiny new-format dataset: LL (1 section, size 2) and EE with 2 bins
    (3 sections, sizes 2/3/4). Within-probe folders store only the upper
    triangle in enumeration order; LLEE stores the full rectangle.
    Returns the expected assembled full covariance.
    """
    sizes = {'LL': [2], 'EE': [2, 3, 4]}
    sections = {'LL': [(0, 0)], 'EE': [(0, 0), (1, 0), (1, 1)]}

    n_total = sum(sum(v) for v in sizes.values())
    C_expected = rng.standard_normal((n_total, n_total))
    C_expected = C_expected + C_expected.T  # symmetric ground truth

    offsets = {'LL': [0], 'EE': [2, 4, 7]}

    def block(p1, a, p2, b):
        r = offsets[p1][a]
        c = offsets[p2][b]
        return C_expected[r:r + sizes[p1][a], c:c + sizes[p2][b]]

    for p1, p2 in [('LL', 'LL'), ('LL', 'EE'), ('EE', 'EE')]:
        d = tmp_path / 'covariance' / (p1 + p2)
        d.mkdir(parents=True)
        for a, s1 in enumerate(sections[p1]):
            for b, s2 in enumerate(sections[p2]):
                if p1 == p2 and a > b:
                    continue  # only the upper triangle is stored
                m = block(p1, a, p2, b)
                # split into ccov + ncov + scov thirds
                for kind in ('ccov', 'ncov', 'scov'):
                    _write_block(
                        d / f'{kind}_{s1[0]}_{s1[1]}_{s2[0]}_{s2[1]}', m / 3.0)

    return C_expected


def test_load_covariance_full_assembly():
    """Full-matrix assembly: triangle mirroring, cross blocks, sizes."""
    from focsle.data_loader import load_covariance_full, is_new_format

    rng = np.random.default_rng(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        C_expected = _make_synthetic_dataset(tmp_path, rng)

        assert is_new_format(str(tmp_path))

        C, sizes, layout = load_covariance_full(
            str(tmp_path), nbins_E=2, nbins_P=2, verbose=False)

        assert sizes == {'n_LL': 2, 'n_EE': 9}
        assert layout['LL']['section_sizes'] == [2]
        assert layout['EE']['section_sizes'] == [2, 3, 4]
        assert layout['EE']['start'] == 2
        np.testing.assert_allclose(C, C_expected, atol=1e-12)
        np.testing.assert_allclose(C, C.T, atol=1e-12)


def test_load_covariance_full_missing_file():
    """A missing per-section file must raise naming the file."""
    from focsle.data_loader import load_covariance_full

    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        _make_synthetic_dataset(tmp_path, rng)
        (tmp_path / 'covariance' / 'EEEE' / 'ncov_1_0_1_1').unlink()

        with pytest.raises(FileNotFoundError, match='ncov_1_0_1_1'):
            load_covariance_full(str(tmp_path), nbins_E=2, nbins_P=2,
                                 verbose=False)
