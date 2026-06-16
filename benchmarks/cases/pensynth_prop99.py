"""pensynth live cross-check vs Abadie & L'Hour's wsoll1 on Prop 99.

Cross-validation against the *running* reference. mlsynth implements the
penalized synthetic-control estimator of Abadie & L'Hour (2021, JASA) -- the
pairwise-penalised QP

    min_W  || X1 - X0 W ||^2  +  lambda * sum_j W_j || X1 - X_j ||^2
    s.t.   W >= 0,  sum_j W_j = 1

-- as ``mlsynth.utils.bilevel.penalized.penalized_weights`` (the ``penalized``
backend of ``VanillaSC``). This case feeds the *identical* predictor matrix to
mlsynth's solver and to the authors' own ``wsoll1`` (their ``functions/wsoll1.R``,
solved with LowRankQP) over a regularisation path, and checks they agree.

The predictor matrix is the canonical Abadie-Diamond-Hainmueller (2010) Prop 99
spec, built from mlsynth's vendored ``basedata/augmented_cali_long.csv`` through
``mlsynth.utils.datautils.dataprep`` and the same covariate-mean / unit-variance
machinery ``VanillaSC`` uses (no hand-pivoting): California vs 38 states matched
on the *original covariate averages* -- ln(income), retail price and percent aged
15-24 over 1980-1988, beer over 1984-1988 -- plus cigarette sales in 1975, 1980
and 1988, each scaled to unit variance (``Gamma = I``).

The penalised QP is deterministic in ``(X0, X1, lambda)`` and strictly convex for
``lambda > 0`` (their Theorem 1), so mlsynth's FISTA and the reference's
interior-point LowRankQP land on the same optimum: at ``lambda = 0.1`` the
synthetic California loads ~0.65 on Colorado and the post-period ATT is -23.4
packs, matched to 4 decimals. Across the path the largest weight gap is a
sub-1% residual at one active-set transition, where LowRankQP stops with a tiny
extra donor weight while mlsynth reaches the true (marginally lower-objective)
vertex -- the reference solver's tolerance, not a methodological difference.

The reference is **commit-pinned**: ``benchmarks/reference/clone_pensynth.py``
clones ``jeremylhour/pensynth`` at a fixed SHA, and ``install_pensynth.sh``
freezes LowRankQP, so the cross-check runs the same solver source every time.
Refresh by re-pinning those and updating ``EXPECTED``.

Skips itself (``BenchmarkSkipped``) when ``Rscript``, LowRankQP, or the clone is
unavailable, so it is a no-op in CI and runs only where the reference is present.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import warnings

import numpy as np
import pandas as pd

from benchmarks.compare import BenchmarkSkipped
from benchmarks.reference.clone_pensynth import functions_dir
from mlsynth.utils.bilevel.penalized import penalized_weights
from mlsynth.utils.datautils import dataprep
from mlsynth.utils.vanillasc_helpers.pipeline import (
    _covariate_means, _scale_unit_variance,
)

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DATA = os.path.join(_ROOT, "basedata", "augmented_cali_long.csv")
_RSCRIPT_REF = os.path.join(_ROOT, "benchmarks", "R", "pensynth_prop99.R")

_TREATED = "California"
_LAGS = (1975, 1980, 1988)
# Abadie-Diamond-Hainmueller (2010) predictor spec: covariate averages over their
# original windows plus the three special lagged outcomes.
_COVS = ["loginc", "p_cig", "pct15-24", "pc_beer", "cig1975", "cig1980", "cig1988"]
_WINDOWS = {
    "loginc": (1980, 1988), "p_cig": (1980, 1988), "pct15-24": (1980, 1988),
    "pc_beer": (1984, 1988),
    "cig1975": (1975, 1975), "cig1980": (1980, 1980), "cig1988": (1988, 1988),
}
_GRID = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0]


def _matrices():
    """Build the ADH predictor matrix via dataprep + mlsynth's covariate machinery."""
    d = pd.read_csv(_DATA)
    d["treated"] = ((d["state"] == _TREATED) & (d["year"] >= 1989)).astype(int)
    for L in _LAGS:                                   # special lagged-outcome predictors
        m = d[d["year"] == L].set_index("state")["cigsale"]
        d[f"cig{L}"] = d["state"].map(m)

    prep = dataprep(
        df=d, unit_id_column_name="state", time_period_column_name="year",
        outcome_column_name="cigsale", treatment_indicator_column_name="treated")
    pre = int(prep["pre_periods"])
    time_labels = list(np.asarray(prep["time_labels"]))
    pre_labels = time_labels[:pre]
    donors = [str(x) for x in prep["donor_names"]]
    units = [str(prep["treated_unit_name"])] + donors

    Xraw = _covariate_means(d, units, _COVS, _WINDOWS, pre_labels, "state", "year")
    Xs = _scale_unit_variance(Xraw)                   # unit-variance, Gamma = I
    X1, X0 = Xs[:, 0], Xs[:, 1:]                       # (K,), (K, J)

    y = np.asarray(prep["y"], dtype=float).ravel()    # treated outcome path
    Y0 = np.asarray(prep["donor_matrix"], dtype=float)  # (T, J)
    return X1, X0, y, Y0, pre, donors


def _tzero(W, tol=1e-6):
    """Threshold tiny weights and renormalise (the reference's TZero convention)."""
    Y = np.where(W < tol, 0.0, W)
    return Y / Y.sum(axis=1, keepdims=True)


def _reference_weights(X0, X1, funcs) -> np.ndarray:
    """Run the authors' wsoll1 over the grid via Rscript; return the weight matrix."""
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise BenchmarkSkipped("Rscript not on PATH (run benchmarks/R/install_pensynth.sh)")
    probe = subprocess.run(
        [rscript, "-e", "suppressMessages(library(LowRankQP))"],
        capture_output=True, text=True)
    if probe.returncode != 0:
        raise BenchmarkSkipped("R package 'LowRankQP' not installed")

    with tempfile.TemporaryDirectory() as tmp:
        x0p = os.path.join(tmp, "X0.csv")
        x1p = os.path.join(tmp, "X1.csv")
        gp = os.path.join(tmp, "grid.csv")
        outp = os.path.join(tmp, "Wref.csv")
        np.savetxt(x0p, X0, delimiter=",")
        np.savetxt(x1p, X1, delimiter=",")
        np.savetxt(gp, np.asarray(_GRID), delimiter=",")
        out = subprocess.run(
            [rscript, _RSCRIPT_REF, str(funcs), x0p, x1p, gp, outp],
            capture_output=True, text=True)
        if out.returncode != 0 or not os.path.exists(outp):
            raise BenchmarkSkipped(f"pensynth reference failed: {out.stderr.strip()[-200:]}")
        return np.loadtxt(outp, delimiter=",")


def run() -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X1, X0, y, Y0, pre, donors = _matrices()
    funcs = functions_dir()                       # clones the pinned repo (skips if absent)
    Wref = _reference_weights(X0, X1, funcs)      # (len(grid), J)

    Wml = np.array([penalized_weights(X1, X0, lam, max_iter=50000, tol=1e-13)
                    for lam in _GRID])
    Wml_t, Wref_t = _tzero(Wml), _tzero(Wref)

    # Largest donor-weight gap across the whole regularisation path.
    weight_max_abs_diff = float(np.max(np.abs(Wml_t - Wref_t)))

    # ATT (post-period mean gap) at each lambda, both implementations.
    att_ml = np.array([float(np.mean((y - Y0 @ w)[pre:])) for w in Wml])
    att_ref = np.array([float(np.mean((y - Y0 @ w)[pre:])) for w in Wref])
    att_max_abs_diff = float(np.max(np.abs(att_ml - att_ref)))

    i01 = _GRID.index(0.1)
    w_colorado = float(Wml_t[i01, donors.index("Colorado")])

    return {
        "weight_max_abs_diff": weight_max_abs_diff,
        "att_max_abs_diff": att_max_abs_diff,
        "att_lambda_0p1": float(att_ml[i01]),
        "w_colorado_lambda_0p1": w_colorado,
    }


# mlsynth reproduces the authors' wsoll1 on the identical ADH predictor matrix.
# The anchored cells fix the actual penalized fit (synthetic California loads
# ~0.65 on Colorado at lambda=0.1, a -23.4 packs post-period ATT); the diff
# tolerances pin a genuine match and absorb the reference's interior-point slack
# (a sub-1% residual weight at one active-set transition, where mlsynth's FISTA
# reaches the marginally lower-objective vertex).
EXPECTED = {
    "weight_max_abs_diff": (0.0, 1.5e-2),
    "att_max_abs_diff": (0.0, 0.1),
    "att_lambda_0p1": (-23.433, 0.05),
    "w_colorado_lambda_0p1": (0.653, 0.01),
}
