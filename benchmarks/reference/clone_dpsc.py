"""On-demand clone of the Rho-Cummings-Misra (2023) DP synthetic-control repo.

The authors' code (https://github.com/srho1/dpsc) is cloned at a pinned commit
into a local cache; the benchmark imports its ``PrivateSC`` and
``SyntheticControl`` classes from there. If git and the codeload tarball are
both unavailable the benchmark skips gracefully.

Bump ``_COMMIT`` deliberately, never silently.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

from benchmarks.compare import BenchmarkSkipped
from benchmarks.reference._fetch import fetch_pinned_repo

_REPO = "https://github.com/srho1/dpsc.git"
_COMMIT = "0be4eba6a162d61fd8262ab06c4ab44d2b371817"
_CACHE = Path(__file__).resolve().parent / ".cache" / "dpsc"


def _ensure_clone() -> Path:
    """Clone (or reuse) the reference repo pinned at ``_COMMIT``. Returns its path."""
    marker = _CACHE / "src" / "private_synthetic_control.py"
    if marker.exists():
        return _CACHE
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    fetch_pinned_repo(_REPO, _COMMIT, _CACHE)
    if not marker.exists():  # pragma: no cover - defensive
        raise BenchmarkSkipped("reference clone missing private_synthetic_control.py")
    return _CACHE


def import_private_sc() -> ModuleType:
    """Import the authors' ``src.private_synthetic_control`` from the pinned clone."""
    path = _ensure_clone()
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    try:
        return importlib.import_module("src.private_synthetic_control")
    except ImportError as exc:  # pragma: no cover - e.g. sklearn/cvxpy missing
        raise BenchmarkSkipped(
            f"reference src.private_synthetic_control import failed: {exc}") from exc
