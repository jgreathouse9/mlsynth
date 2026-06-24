"""On-demand clone of the Rho et al. (2025) ClusterSC reference repo.

The authors' replication repository (https://github.com/srho1/ClusterSC) is
MIT-licensed but is not on PyPI, so the cross-validation benchmark clones it at a
pinned commit into a local cache and imports its ``syclib`` package from there
(rather than vendoring a copy into mlsynth). If git or the network is
unavailable the benchmark skips gracefully.

The pinned commit (`_COMMIT`) freezes the reference; bump it deliberately.
"""
from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path
from types import ModuleType

from benchmarks.compare import BenchmarkSkipped
from benchmarks.reference._fetch import fetch_pinned_repo

_REPO = "https://github.com/srho1/ClusterSC.git"
_COMMIT = "b223e1e2a2fd68aaff9da9feac8a5c08e1163ad7"
_CACHE = Path(__file__).resolve().parent / ".cache" / "ClusterSC"


def _ensure_clone() -> Path:
    """Clone (or reuse) the reference repo pinned at ``_COMMIT``. Returns its path."""
    marker = _CACHE / "syclib" / "__init__.py"
    if marker.exists():
        return _CACHE
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    fetch_pinned_repo(_REPO, _COMMIT, _CACHE)    # git clone, else codeload tarball
    if not marker.exists():  # pragma: no cover - defensive
        raise BenchmarkSkipped("reference clone missing syclib package")
    return _CACHE


def import_syclib() -> ModuleType:
    """Import the authors' ``syclib`` package from the pinned clone.

    Returns the top-level module; submodules (``syclib.gendata``,
    ``syclib.cluster``) are importable once this succeeds.
    """
    path = _ensure_clone()
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    try:
        syclib = importlib.import_module("syclib")
        importlib.import_module("syclib.gendata")
        importlib.import_module("syclib.cluster")  # needs kneed, sklearn
    except ImportError as exc:  # pragma: no cover - e.g. kneed missing
        raise BenchmarkSkipped(
            f"reference syclib import failed ({exc}); "
            f"install its deps (`pip install kneed scikit-learn`)"
        ) from exc
    return syclib
