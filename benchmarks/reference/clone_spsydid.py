"""On-demand clone of the Serenini & Masek (2024) Spatial-SDID reference repo.

The authors' replication repository
(https://github.com/serenini/spatial_SDID) carries **no licence**, so its code
is not vendored into mlsynth. Instead the SpSyDiD cross-validation benchmark
clones it at a pinned commit into a local cache and imports its
``functions_ssdid`` module from there. If git or the network is unavailable the
benchmark skips gracefully.

The pinned commit (`_COMMIT`) freezes the reference so the cross-check is
reproducible; bump it deliberately, never silently.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

from benchmarks.compare import BenchmarkSkipped
from benchmarks.reference._fetch import fetch_pinned_repo

_REPO = "https://github.com/serenini/spatial_SDID.git"
_COMMIT = "e43427d67c88d06a80db690416f7c61ef0143287"
_CACHE = Path(__file__).resolve().parent / ".cache" / "spatial_SDID"


def _ensure_clone() -> Path:
    """Clone (or reuse) the reference repo pinned at ``_COMMIT``. Returns its path."""
    funcs = _CACHE / "functions_ssdid.py"
    if funcs.exists():
        return _CACHE
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    fetch_pinned_repo(_REPO, _COMMIT, _CACHE)    # git clone, else codeload tarball
    if not funcs.exists():  # pragma: no cover - defensive
        raise BenchmarkSkipped("reference clone missing functions_ssdid.py")
    return _CACHE


def import_functions_ssdid() -> ModuleType:
    """Import the authors' ``functions_ssdid`` module from the pinned clone."""
    path = _ensure_clone()
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    try:
        return importlib.import_module("functions_ssdid")
    except ImportError as exc:  # pragma: no cover - e.g. cvxpy/toolz missing
        raise BenchmarkSkipped(
            f"reference functions_ssdid import failed: {exc}"
        ) from exc
