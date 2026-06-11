"""On-demand clone of the synthetic-control-with-spillover reference repo.

Cao & Dowd, *"Estimation and Inference for Synthetic Control Methods with
Spillover Effects,"* ship the CRAN package ``scmSpillover`` and the MATLAB
Proposition-99 replication at
https://github.com/jcao0/synthetic-control-spillover. The repo carries the
committed empirical output ``spillover.csv`` -- the per-state spillover-adjusted
effects ``alpha_hat`` for the post-treatment years 1989-2000; the ``CA`` row is
the spillover-adjusted ATT on California (the treated unit). We clone it at a
pinned commit and read that committed output; if git or the network is
unavailable the benchmark skips gracefully.

The pinned commit (``_COMMIT``) freezes the reference so the cross-check is
reproducible; bump it deliberately, never silently.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from benchmarks.compare import BenchmarkSkipped

_REPO = "https://github.com/jcao0/synthetic-control-spillover.git"
_COMMIT = "60bbebe73c73135110d48d5fe406e49cbc8e9b89"
_CACHE = Path(__file__).resolve().parent / ".cache" / "jcao0-synthetic-control-spillover"
_SPILLOVER = "replication files/proposition99_matlab/output/spillover.csv"


def _ensure_clone() -> Path:
    """Clone (or reuse) the reference repo pinned at ``_COMMIT``."""
    marker = _CACHE / _SPILLOVER
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
        raise BenchmarkSkipped("reference clone missing spillover.csv")
    return _CACHE


def reference_ca_alpha() -> np.ndarray:
    """Return the committed California spillover-adjusted ATT path (1989-2000)."""
    import pandas as pd

    df = pd.read_csv(_ensure_clone() / _SPILLOVER)
    ca = df[df["state"] == "CA"].iloc[0]
    cols = [c for c in df.columns if c.startswith("alpha_hat_")]
    return ca[cols].to_numpy(dtype=float)
