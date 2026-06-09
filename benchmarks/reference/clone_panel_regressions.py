"""On-demand clone of the Shen et al. panel-data-regressions reference repo.

The authors' replication repository (https://github.com/deshen24/panel-data-regressions)
carries no licence, so its code is not vendored into mlsynth. The Shen-CI
cross-validation benchmark clones it at a pinned commit into a local cache and
imports its ``var`` / ``regr`` / ``rank`` modules and bundled case-study data
from there. If git or the network is unavailable the benchmark skips gracefully.
"""
from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

from benchmarks.compare import BenchmarkSkipped

_REPO = "https://github.com/deshen24/panel-data-regressions.git"
_COMMIT = "51e2170d33463bbf403f23fe8a72cbf66bcc34ef"
_CACHE = Path(__file__).resolve().parent / ".cache" / "panel-data-regressions"


def ensure_clone() -> Path:
    """Clone (or reuse) the reference repo pinned at ``_COMMIT``. Returns its path."""
    marker = _CACHE / "var.py"
    if marker.exists():
        return _CACHE
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["git", "-c", "credential.helper=", "clone", "--quiet", _REPO, str(_CACHE)],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(_CACHE), "checkout", "--quiet", _COMMIT],
            check=True, capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:  # pragma: no cover
        detail = getattr(exc, "stderr", b"")
        msg = detail.decode(errors="ignore").strip() if detail else str(exc)
        raise BenchmarkSkipped(
            f"could not clone reference repo {_REPO} @ {_COMMIT[:7]}: {msg}"
        ) from exc
    if not marker.exists():  # pragma: no cover - defensive
        raise BenchmarkSkipped("reference clone missing var.py")
    return _CACHE


def import_reference():
    """Import and return the authors' ``(var, regr, rank)`` modules from the clone."""
    path = ensure_clone()
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    try:
        var = importlib.import_module("var")
        regr = importlib.import_module("regr")
        rank = importlib.import_module("rank")
    except ImportError as exc:  # pragma: no cover - e.g. toolz missing
        raise BenchmarkSkipped(
            f"reference modules failed to import ({exc}); "
            f"install their deps (`pip install toolz scipy scikit-learn`)"
        ) from exc
    return var, regr, rank
