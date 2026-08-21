"""Tests for the persistent CAMB array cache."""

import numpy as np
import pytest

from focsle.camb_cache import CambArrayCache
from focsle.theory import TheoryJAX


def test_fingerprint_is_deterministic_and_configuration_sensitive(tmp_path):
    cache = CambArrayCache(tmp_path)
    first = {
        "camb_version": "test",
        "cosmology": {"w0": -1.0, "wa": 0.0},
        "z_grid": np.array([0.0, 1.0]),
    }
    reordered = {
        "z_grid": [0.0, 1.0],
        "cosmology": {"wa": 0.0, "w0": -1.0},
        "camb_version": "test",
    }
    changed = {**first, "cosmology": {"w0": -0.9, "wa": 0.0}}

    assert cache.fingerprint(first) == cache.fingerprint(reordered)
    assert cache.fingerprint(first) != cache.fingerprint(changed)
    assert cache.path_for(first) != cache.path_for(changed)


def test_cache_round_trip_preserves_arrays(tmp_path):
    cache = CambArrayCache(tmp_path)
    configuration = {"camb_version": "test", "grid": [2, 3]}
    expected = {
        "Pk_grid": np.arange(6, dtype=float).reshape(2, 3),
        "sigma8_grid": np.array([0.79, 0.81]),
    }

    assert cache.load(configuration) is None
    path = cache.store(configuration, expected)
    actual = cache.load(configuration)

    assert path.exists()
    assert actual is not None
    assert actual.keys() == expected.keys()
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])


def test_get_or_compute_reuses_existing_entry(tmp_path):
    cache = CambArrayCache(tmp_path)
    configuration = {"camb_version": "test", "w0": -1.0, "wa": 0.0}
    calls = 0

    def compute():
        nonlocal calls
        calls += 1
        return {"chi": np.array([0.0, 1000.0])}

    first, first_hit = cache.get_or_compute(configuration, compute)
    second, second_hit = cache.get_or_compute(configuration, compute)

    assert calls == 1
    assert first_hit is False
    assert second_hit is True
    np.testing.assert_array_equal(first["chi"], second["chi"])


def test_cache_rejects_object_arrays(tmp_path):
    cache = CambArrayCache(tmp_path)

    with pytest.raises(TypeError, match="object dtype"):
        cache.store(
            {"camb_version": "test"},
            {"unsafe": np.array([{"not": "portable"}], dtype=object)},
        )



def _bare_theory_with_cache(tmp_path):
    """Construct TheoryJAX without loading lenses or allocating Ogata nodes."""
    theory = TheoryJAX.__new__(TheoryJAX)
    theory.cosmo_fid = {
        'H0': 67.37,
        'ombh2': 0.0223,
        'mnu': 0.06,
        'omk': 0.0,
        'tau': 0.054,
        'ns': 0.965,
        'As': 2.0e-9,
        'Omega_m': 0.3138,
        'sigma8': 0.7913,
        'w0': -1.0,
        'wa': 0.0,
    }
    theory.camb_cache = CambArrayCache(tmp_path)
    return theory


def test_power_grid_setup_reuses_persistent_cache(tmp_path):
    theory = _bare_theory_with_cache(tmp_path)
    calls = 0

    def fake_compute(Om_grid, As_grid, z_grid, k_grid, verbose=True):
        nonlocal calls
        calls += 1
        return {
            'Om_grid': Om_grid,
            'As_grid': As_grid,
            'z_grid': z_grid,
            'k_grid': k_grid,
            'Pk_grid': np.ones((len(Om_grid), len(As_grid),
                                len(z_grid), len(k_grid))),
            'sigma8_grid': np.full((len(Om_grid), len(As_grid)), 0.8),
        }

    theory._compute_power_grid = fake_compute
    theory._setup_background = lambda Om_grid, verbose=True: None

    settings = dict(nOm=2, nAs=3, nz=4, nk=5, verbose=False)
    theory.setup_Pk_grid(**settings)
    theory.setup_Pk_grid(**settings)

    assert calls == 1
    np.testing.assert_array_equal(
        np.asarray(theory.Pk_grid), np.ones((2, 3, 4, 5))
    )


def test_background_setup_reuses_persistent_cache(tmp_path):
    theory = _bare_theory_with_cache(tmp_path)
    calls = 0

    def fake_compute(Om_grid, z_grid):
        nonlocal calls
        calls += 1
        return {
            'Om_bg': Om_grid,
            'z_bg': z_grid,
            'chi_bg_table': Om_grid[:, None] + z_grid[None, :],
        }

    theory._compute_background_grid = fake_compute
    Om_grid = np.array([0.25, 0.40])
    theory._setup_background(Om_grid, n_Om_min=3, verbose=False)
    theory._setup_background(Om_grid, n_Om_min=3, verbose=False)

    assert calls == 1
    assert np.asarray(theory.chi_bg_table).shape == (3, 500)


def test_dark_energy_stencil_is_cached_per_cosmology(tmp_path):
    theory = _bare_theory_with_cache(tmp_path)
    theory.z_grid = np.array([0.0, 1.0])
    theory.k_grid = np.array([0.01, 0.1, 1.0])
    theory.z_bg = np.array([0.0, 0.5, 1.0])
    calls = 0

    def fake_compute(w0, wa, target_sigma8, z_grid, k_grid, z_background):
        nonlocal calls
        calls += 1
        return {
            'Pk': np.full((len(z_grid), len(k_grid)),
                          np.exp(2.0 * w0 + 3.0 * wa)),
            'chi': np.full(len(z_background), 100.0 + 5.0 * w0 + 7.0 * wa),
            'As': np.asarray(2.0e-9),
            'sigma8': np.asarray(target_sigma8),
        }

    theory._compute_dark_energy_point = fake_compute
    theory.setup_dark_energy_responses(
        w0_step=0.05, wa_step=0.1, verbose=False
    )
    theory.setup_dark_energy_responses(
        w0_step=0.05, wa_step=0.1, verbose=False
    )

    assert calls == 4
    np.testing.assert_allclose(np.asarray(theory.dlnPk_dw0), 2.0)
    np.testing.assert_allclose(np.asarray(theory.dlnPk_dwa), 3.0)
    np.testing.assert_allclose(np.asarray(theory.dchi_dw0), 5.0)
    np.testing.assert_allclose(np.asarray(theory.dchi_dwa), 7.0)
    assert all(theory.dark_energy_response_metadata['cache_hits'].values())

    # Only the two changed w0 locations are new; the wa stencil is reused.
    theory.setup_dark_energy_responses(
        w0_step=0.025, wa_step=0.1, verbose=False
    )
    assert calls == 6


def test_baseline_cache_identity_includes_dark_energy_fiducial(tmp_path):
    theory = _bare_theory_with_cache(tmp_path)
    grid = np.array([0.25, 0.40])

    lcdm_power = theory._power_grid_cache_configuration(
        grid, grid, grid, grid
    )
    lcdm_background = theory._background_cache_configuration(grid, grid)

    theory.cosmo_fid['w0'] = -0.838
    theory.cosmo_fid['wa'] = -0.62
    cpl_power = theory._power_grid_cache_configuration(
        grid, grid, grid, grid
    )
    cpl_background = theory._background_cache_configuration(grid, grid)

    assert theory.camb_cache.fingerprint(lcdm_power) != (
        theory.camb_cache.fingerprint(cpl_power)
    )
    assert theory.camb_cache.fingerprint(lcdm_background) != (
        theory.camb_cache.fingerprint(cpl_background)
    )


def test_dark_energy_stencil_accepts_non_lcdm_fiducial(tmp_path):
    theory = _bare_theory_with_cache(tmp_path)
    theory.cosmo_fid['w0'] = -0.838
    theory.cosmo_fid['wa'] = -0.62
    theory.z_grid = np.array([0.0, 1.0])
    theory.k_grid = np.array([0.01, 0.1])
    theory.z_bg = np.array([0.0, 1.0])

    def fake_compute(w0, wa, target_sigma8, z_grid, k_grid, z_background):
        return {
            'Pk': np.full((len(z_grid), len(k_grid)), np.exp(w0 + wa)),
            'chi': np.full(len(z_background), 100.0 + w0 + wa),
            'As': np.asarray(2.0e-9),
            'sigma8': np.asarray(target_sigma8),
        }

    theory._compute_dark_energy_point = fake_compute
    theory.setup_dark_energy_responses(verbose=False)

    assert theory.dark_energy_response_metadata['fiducial'] == {
        'w0': -0.838,
        'wa': -0.62,
        'sigma8': theory.cosmo_fid['sigma8'],
    }


@pytest.mark.parametrize('name,value', [('w0_step', 0.0),
                                        ('wa_step', -0.1),
                                        ('w0_step', np.nan)])
def test_dark_energy_stencil_rejects_invalid_steps(tmp_path, name, value):
    theory = _bare_theory_with_cache(tmp_path)
    theory.z_grid = np.array([0.0, 1.0])
    theory.k_grid = np.array([0.01, 0.1])
    theory.z_bg = np.array([0.0, 1.0])
    kwargs = {'w0_step': 0.05, 'wa_step': 0.1, name: value}

    with pytest.raises(ValueError, match=name):
        theory.setup_dark_energy_responses(verbose=False, **kwargs)
