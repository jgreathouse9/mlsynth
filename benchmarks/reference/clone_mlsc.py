"""On-demand clone of the Bottmer (2024/2025) multi-level SC reference repo.

The author's implementation
(https://github.com/leabottmer/multi-level-sc-estimator) is a small pip
package (``multi-levelSC``), but to keep the cross-check reproducible and free
of an extra install we clone it at a pinned commit into a local cache and import
``multi_level_sc_estimator.mlSC`` -- exposing ``mlSC_estimator`` -- from there.
If git or the network is unavailable the benchmark skips gracefully.

The reference solves the penalized state-county program with ``cvxpy`` + ``SCS``
(open source -- no commercial solver required). Its module imports ``ray`` at
top level but never uses it, so we stub ``ray`` to avoid that optional, heavy
dependency.

The pinned commit (``_COMMIT``) freezes the reference so the cross-check is
reproducible; bump it deliberately, never silently.
"""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from types import ModuleType

from benchmarks.compare import BenchmarkSkipped
from benchmarks.reference._fetch import fetch_pinned_repo

_REPO = "https://github.com/leabottmer/multi-level-sc-estimator.git"
_COMMIT = "0fb26391a21840085d7cb4d7b1aa257e7360f9ea"
_CACHE = Path(__file__).resolve().parent / ".cache" / "multi-level-sc-estimator"


def _ensure_clone() -> Path:
    """Clone (or reuse) the reference repo pinned at ``_COMMIT``. Returns its path."""
    marker = _CACHE / "multi_level_sc_estimator" / "mlSC.py"
    if marker.exists():
        return _CACHE
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    fetch_pinned_repo(_REPO, _COMMIT, _CACHE)    # git clone, else codeload tarball
    if not marker.exists():  # pragma: no cover - defensive
        raise BenchmarkSkipped("reference clone missing multi_level_sc_estimator/mlSC.py")
    return _CACHE


def import_mlsc() -> ModuleType:
    """Import the author's ``mlSC`` module (exposes ``mlSC_estimator``)."""
    root = _ensure_clone()
    # ``ray`` is imported at module top but never used; stub it so the optional
    # dependency is not required.
    sys.modules.setdefault("ray", types.ModuleType("ray"))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        return importlib.import_module("multi_level_sc_estimator.mlSC")
    except Exception as exc:  # pragma: no cover - import-time deps (cvxpy/scs)
        raise BenchmarkSkipped(f"could not import reference mlSC module: {exc}")
