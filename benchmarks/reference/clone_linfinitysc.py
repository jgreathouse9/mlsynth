"""On-demand clone of the Wang, Xing & Ye (2025) L-infinity SC reference repo.

The authors' implementation
(https://github.com/BioAlgs/LinfinitySC) carries no packaging, so its code is
not vendored into mlsynth. The LINF cross-validation benchmark clones it at a
pinned commit into a local cache and imports its ``utils/synth.py`` module --
exposing ``our(Y1, Y0, alpha, lam, method, std, intercept)`` and
``param_selector`` -- from there. If git or the network is unavailable the
benchmark skips gracefully.

The reference solves the L-infinity / L1+L-infinity penalized regression with
``cvxopt`` (open source), so no commercial solver is required.

The pinned commit (``_COMMIT``) freezes the reference so the cross-check is
reproducible; bump it deliberately, never silently.
"""
from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path
from types import ModuleType

from benchmarks.compare import BenchmarkSkipped
from benchmarks.reference._fetch import fetch_pinned_repo

_REPO = "https://github.com/BioAlgs/LinfinitySC.git"
_COMMIT = "37499abcf3b3722cd6b9a42599e344b8653da4d4"
_CACHE = Path(__file__).resolve().parent / ".cache" / "LinfinitySC"


def _ensure_clone() -> Path:
    """Clone (or reuse) the reference repo pinned at ``_COMMIT``. Returns its path."""
    synth = _CACHE / "utils" / "synth.py"
    if synth.exists():
        return _CACHE
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    fetch_pinned_repo(_REPO, _COMMIT, _CACHE)    # git clone, else codeload tarball
    if not synth.exists():  # pragma: no cover - defensive
        raise BenchmarkSkipped("reference clone missing utils/synth.py")
    return _CACHE


def import_synth() -> ModuleType:
    """Import the authors' ``synth`` module (``our``, ``param_selector``)."""
    path = _ensure_clone() / "utils"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    try:
        return importlib.import_module("synth")
    except Exception as exc:  # pragma: no cover - import-time deps (cvxopt)
        raise BenchmarkSkipped(f"could not import reference synth module: {exc}")
