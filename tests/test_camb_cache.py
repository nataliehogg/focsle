"""Tests for the persistent CAMB array cache."""

import numpy as np
import pytest

from focsle.camb_cache import CambArrayCache


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

