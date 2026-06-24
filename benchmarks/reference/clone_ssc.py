"""On-demand clone of the staggered-synthetic-control reference repo.

Cao, Lu & Wu, *"Synthetic Control Inference for Staggered Adoption,"* ship their
estimator and the empirical replication at
https://github.com/jcao0/staggered_synthetic_control. The repo carries the
committed Section-4 output (``results_ssc.csv`` -- the event-time ATT estimates
and confidence intervals for the seven "intergovernmental coordination and
criminality" outcomes) and the design-matrix diagnostics
(``Table1_eigenvalue.csv`` -- the smallest eigenvalue of the SSC Gram matrix per
outcome). We clone it at a pinned commit and read those committed outputs; if
git or the network is unavailable the benchmark skips gracefully.

The pinned commit (``_COMMIT``) freezes the reference so the cross-check is
reproducible; bump it deliberately, never silently.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from benchmarks.compare import BenchmarkSkipped
from benchmarks.reference._fetch import fetch_pinned_repo

_REPO = "https://github.com/jcao0/staggered_synthetic_control.git"
_COMMIT = "74e77d44d4fb1a5b73828c8ba7a52049d731f34c"
_CACHE = Path(__file__).resolve().parent / ".cache" / "jcao0-staggered_synthetic_control"

_RESULTS = ("replication package/Intergovernmental coordination and criminality/"
            "treatment_effect/output/results_ssc.csv")
_EIGEN = "stagsynth/inst/replication/output/Table1_eigenvalue.csv"

# Table-1 rows (a)-(g) in order map to the results_ssc.csv outcome labels.
EIGEN_OUTCOME_ORDER = [
    "hom_all_rate", "hom_ym_rate", "theft_violent_rate", "theft_nonviolent_rate",
    "presence_strength", "co_num", "war",
]


def _ensure_clone() -> Path:
    """Clone (or reuse) the reference repo pinned at ``_COMMIT``."""
    marker = _CACHE / _EIGEN
    if marker.exists():
        return _CACHE
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    fetch_pinned_repo(_REPO, _COMMIT, _CACHE)    # git clone, else codeload tarball
    if not marker.exists():  # pragma: no cover - defensive
        raise BenchmarkSkipped("reference clone missing Table1_eigenvalue.csv")
    return _CACHE


def reference_att():
    """Return the committed event-time ATT table (``results_ssc.csv``) as a DataFrame.

    Columns ``outcome``, ``event_time`` (1-based), ``ref_att``.
    """
    import pandas as pd

    path = _ensure_clone() / _RESULTS
    df = pd.read_csv(path).rename(
        columns={"event time": "event_time", "att estimate": "ref_att"})
    return df[["outcome", "event_time", "ref_att"]]


def reference_eigenvalues() -> dict:
    """Return the committed ``Table1_eigenvalue.csv`` min eigenvalues by outcome."""
    arr = np.loadtxt(
        _ensure_clone() / _EIGEN, delimiter=",", skiprows=1, usecols=1)
    return dict(zip(EIGEN_OUTCOME_ORDER, [float(x) for x in arr]))
