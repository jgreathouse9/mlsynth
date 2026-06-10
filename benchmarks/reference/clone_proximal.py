"""On-demand clone of the freshtaste/proximal reference repo (Panic of 1907).

The proximal-inference SC papers (Liu, Tchetgen Tchetgen & Varjao; Shi et al.)
ship a Python reference implementation at https://github.com/freshtaste/proximal,
including the trust-company panel (``data/trust.dta``) and the committed
empirical output ``empirical_results.csv`` -- whose full-window row is the
paper's Table-3 estimate (PI / PI-surrogate / PI-surrogate-post). We clone it at
a pinned commit and read that committed reference output; if git or the network
is unavailable the benchmark skips gracefully.

The pinned commit (``_COMMIT``) freezes the reference so the cross-check is
reproducible; bump it deliberately, never silently.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from benchmarks.compare import BenchmarkSkipped

_REPO = "https://github.com/freshtaste/proximal.git"
_COMMIT = "a67d81e7abd33a491646db558afc0e0ffa120f28"
_CACHE = Path(__file__).resolve().parent / ".cache" / "freshtaste-proximal"


def _ensure_clone() -> Path:
    """Clone (or reuse) the reference repo pinned at ``_COMMIT``."""
    marker = _CACHE / "empirical_results.csv"
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
        raise BenchmarkSkipped("reference clone missing empirical_results.csv")
    return _CACHE


def reference_table3() -> dict:
    """Return the reference full-window ATTs ``{PI, PIS, PIPost}``.

    Reads the committed ``empirical_results.csv`` (columns
    ``[OLS, OLS-surrogate, PI, PI-surrogate, PI-surrogate-post]``; estimate rows
    interleaved with s.e. rows). The last estimate row is the full-window Table-3
    estimate.
    """
    arr = np.loadtxt(_ensure_clone() / "empirical_results.csv", delimiter=",")
    est = arr[-2]                      # last estimate row (full window)
    return {"PI": float(est[2]), "PIS": float(est[3]), "PIPost": float(est[4])}
