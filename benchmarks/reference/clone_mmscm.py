"""On-demand clone of the Kato & Ohda (2026) MMSCM reference repo.

The authors' replication repository (https://github.com/MasaKat0/mmscm) is not
on PyPI, so the replication study clones it at a pinned commit into a local
cache and imports its ``mmscm`` module from there. It is GPL-3.0, so it is
cloned, not vendored: mlsynth is MIT, and a copy in-tree would put the whole
distribution under copyleft. If git or the network is unavailable the
study skips the cross-validation and runs the port alone.

The pinned commit (`_COMMIT`) freezes the reference; bump it deliberately.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

from benchmarks.compare import BenchmarkSkipped
from benchmarks.reference._fetch import fetch_pinned_repo

_REPO = "https://github.com/MasaKat0/mmscm.git"
_COMMIT = "451ca13ace7141da29333d521fcf86c7590a29eb"
_CACHE = Path(__file__).resolve().parent / ".cache" / "mmscm"


def _ensure_clone() -> Path:
    """Clone (or reuse) the reference repo pinned at ``_COMMIT``. Returns its path."""
    marker = _CACHE / "mmscm.py"
    if marker.exists():
        return _CACHE
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    fetch_pinned_repo(_REPO, _COMMIT, _CACHE)    # git clone, else codeload tarball
    if not marker.exists():  # pragma: no cover - defensive
        raise BenchmarkSkipped("reference clone missing mmscm.py")
    return _CACHE


def import_mmscm() -> ModuleType:
    """Import the authors' ``mmscm`` module from the pinned clone.

    Exposes ``MMSCM`` (the estimator class, with ``method`` in
    ``{"MMSCM", "Abadie", "DiSCo"}``) and the objective functions
    ``distscm_obj_func`` / ``abadie_obj_func``.
    """
    path = _ensure_clone()
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    try:
        return importlib.import_module("mmscm")
    except ImportError as exc:  # pragma: no cover - defensive
        raise BenchmarkSkipped(f"could not import the reference mmscm module: {exc}")
