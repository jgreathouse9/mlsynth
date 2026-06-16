"""On-demand clone of Abadie & L'Hour's ``pensynth`` reference repo.

The authors' replication repository (https://github.com/jeremylhour/pensynth,
MIT-licensed) carries the penalized synthetic-control solver ``wsoll1`` -- the
exact QP of Abadie & L'Hour (2021, JASA) -- as loose R functions rather than a
package. Rather than vendor them, the ``pensynth_prop99`` cross-validation
benchmark clones the repo at a pinned commit into a local cache and points the
reference R script (``benchmarks/R/pensynth_prop99.R``) at its ``functions``
directory. If git or the network is unavailable the benchmark skips gracefully.

The pinned commit (``_COMMIT``) freezes the reference so the cross-check runs the
same ``wsoll1``/``TZero`` source every time; bump it deliberately, never silently.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from benchmarks.compare import BenchmarkSkipped

_REPO = "https://github.com/jeremylhour/pensynth.git"
# pensynth master @ 2024 (the EXB_Lalonde / EXA_California solver functions).
_COMMIT = "3f2ad93a96acd558841275d07cd70576c78d451f"
_CACHE = Path(__file__).resolve().parent / ".cache" / "pensynth"


def ensure_clone() -> Path:
    """Clone (or reuse) the reference repo pinned at ``_COMMIT``. Returns its path."""
    wsoll1 = _CACHE / "functions" / "wsoll1.R"
    if wsoll1.exists():
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
    except (OSError, subprocess.CalledProcessError) as exc:  # pragma: no cover - env-dependent
        detail = getattr(exc, "stderr", b"")
        msg = detail.decode(errors="ignore").strip() if detail else str(exc)
        raise BenchmarkSkipped(
            f"could not clone reference repo {_REPO} @ {_COMMIT[:7]}: {msg}"
        ) from exc
    if not wsoll1.exists():  # pragma: no cover - defensive
        raise BenchmarkSkipped("reference clone missing functions/wsoll1.R")
    return _CACHE


def functions_dir() -> Path:
    """Path to the pinned clone's ``functions`` directory (has wsoll1.R, TZero.R)."""
    return ensure_clone() / "functions"
