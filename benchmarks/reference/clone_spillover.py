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

from pathlib import Path

import numpy as np

from benchmarks.compare import BenchmarkSkipped
from benchmarks.reference._fetch import fetch_pinned_repo

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
    fetch_pinned_repo(_REPO, _COMMIT, _CACHE)    # git clone, else codeload tarball
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
