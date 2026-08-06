"""Persistent, configuration-addressed cache for CAMB array products.

CAMB results are expensive to construct but the arrays consumed by FOCSLE are
plain NumPy data.  This module stores those arrays in a non-pickle ``.npz``
file whose name is the SHA-256 fingerprint of the complete calculation
configuration.  Callers are responsible for including every scientific and
numerical setting that can affect the arrays (including the CAMB version).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


CACHE_SCHEMA_VERSION = 1
_MANIFEST_KEY = "__focsle_cache_manifest__"


class CambCacheError(RuntimeError):
    """Base class for persistent CAMB-cache errors."""


class CambCacheCorruptionError(CambCacheError):
    """Raised when a cache file does not match its expected identity."""


def default_camb_cache_dir() -> Path:
    """Return the user-level cache directory, allowing an explicit override."""
    override = os.environ.get("FOCSLE_CAMB_CACHE_DIR")
    if override:
        return Path(override).expanduser()

    cache_root = Path(
        os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
    ).expanduser()
    return cache_root / "focsle" / "camb"


def _normalise_for_json(value: Any) -> Any:
    """Convert configuration values to a deterministic JSON representation."""
    if isinstance(value, Mapping):
        normalised = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("CAMB cache configuration keys must be strings")
            normalised[key] = _normalise_for_json(item)
        return normalised
    if isinstance(value, (list, tuple)):
        return [_normalise_for_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _normalise_for_json(value.tolist())
    if isinstance(value, np.generic):
        return _normalise_for_json(value.item())
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(
        f"Unsupported CAMB cache configuration value: {type(value).__name__}"
    )


class CambArrayCache:
    """Store and retrieve CAMB-derived NumPy arrays by configuration hash."""

    def __init__(self, cache_dir: str | Path | None = None):
        self.cache_dir = (
            default_camb_cache_dir() if cache_dir is None else Path(cache_dir)
        ).expanduser()

    @staticmethod
    def canonical_configuration(configuration: Mapping[str, Any]) -> dict[str, Any]:
        """Return the configuration in its deterministic serialisable form."""
        if not isinstance(configuration, Mapping):
            raise TypeError("CAMB cache configuration must be a mapping")
        return _normalise_for_json(configuration)

    @classmethod
    def fingerprint(cls, configuration: Mapping[str, Any]) -> str:
        """Return the cache identity for a complete CAMB configuration."""
        payload = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "configuration": cls.canonical_configuration(configuration),
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def path_for(self, configuration: Mapping[str, Any]) -> Path:
        """Return the file path associated with ``configuration``."""
        return self.cache_dir / f"{self.fingerprint(configuration)}.npz"

    def load(
        self, configuration: Mapping[str, Any]
    ) -> dict[str, np.ndarray] | None:
        """Load cached arrays, or return ``None`` when no entry exists."""
        expected_configuration = self.canonical_configuration(configuration)
        expected_fingerprint = self.fingerprint(configuration)
        path = self.path_for(configuration)
        if not path.exists():
            return None

        try:
            with np.load(path, allow_pickle=False) as source:
                if _MANIFEST_KEY not in source.files:
                    raise CambCacheCorruptionError(
                        f"CAMB cache entry has no manifest: {path}"
                    )
                manifest = json.loads(str(source[_MANIFEST_KEY].item()))
                arrays = {
                    name: np.array(source[name], copy=True)
                    for name in source.files
                    if name != _MANIFEST_KEY
                }
        except CambCacheCorruptionError:
            raise
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise CambCacheCorruptionError(
                f"Could not read CAMB cache entry {path}: {exc}"
            ) from exc

        if manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
            raise CambCacheCorruptionError(
                f"CAMB cache schema mismatch in {path}"
            )
        if manifest.get("fingerprint") != expected_fingerprint:
            raise CambCacheCorruptionError(
                f"CAMB cache fingerprint mismatch in {path}"
            )
        if manifest.get("configuration") != expected_configuration:
            raise CambCacheCorruptionError(
                f"CAMB cache configuration mismatch in {path}"
            )
        return arrays

    def store(
        self,
        configuration: Mapping[str, Any],
        arrays: Mapping[str, Any],
    ) -> Path:
        """Atomically store one cache entry and return its final path."""
        if not isinstance(arrays, Mapping) or not arrays:
            raise ValueError("CAMB cache arrays must be a non-empty mapping")

        prepared = {}
        for name, value in arrays.items():
            if not isinstance(name, str):
                raise TypeError("CAMB cache array names must be strings")
            if name == _MANIFEST_KEY:
                raise ValueError(f"Reserved CAMB cache array name: {name}")
            array = np.asarray(value)
            if array.dtype.hasobject:
                raise TypeError(
                    f"CAMB cache array {name!r} has object dtype; "
                    "only non-pickle NumPy data are supported"
                )
            prepared[name] = array

        canonical_configuration = self.canonical_configuration(configuration)
        fingerprint = self.fingerprint(configuration)
        manifest = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "configuration": canonical_configuration,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        final_path = self.path_for(configuration)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=self.cache_dir,
                prefix=f".{fingerprint}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                np.savez_compressed(
                    temporary,
                    **prepared,
                    **{_MANIFEST_KEY: json.dumps(manifest, sort_keys=True)},
                )
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, final_path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
        return final_path

    def get_or_compute(
        self,
        configuration: Mapping[str, Any],
        compute: Callable[[], Mapping[str, Any]],
    ) -> tuple[dict[str, np.ndarray], bool]:
        """Return arrays and a flag indicating whether they came from cache."""
        cached = self.load(configuration)
        if cached is not None:
            return cached, True

        computed = compute()
        self.store(configuration, computed)
        loaded = self.load(configuration)
        if loaded is None:  # pragma: no cover - store/load invariant
            raise CambCacheError("CAMB cache entry disappeared after being stored")
        return loaded, False

