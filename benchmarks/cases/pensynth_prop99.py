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

The predictor matrix is the specification in the authors' own California example
(``examples/EXA_CaliforniaTobacco.R`` in ``jeremylhour/pensynth``): the four MLAB
covariate averages -- income, retail price, percent aged 15-24 and beer
consumption -- stacked with the full pre-treatment cigarette-sales path
1970-1988, matched with ``V = I`` (raw, no rescaling), exactly as that script
builds ``X`` and sets ``V = diag(ncol(X))``. It is constructed from mlsynth's
vendored ``basedata/augmented_cali_long.csv`` through
:func:`mlsynth.utils.datautils.dataprep` and the covariate-mean helper
``VanillaSC`` uses -- no hand-pivoting. California is the treated unit and the
remaining 38 states are the donors.

The penalised QP is deterministic in ``(X0, X1, lambda)`` and strictly convex for
``lambda > 0`` (their Theorem 1). Over the penalty path before the nearest-
neighbour collapse (``lambda`` up to 0.25) mlsynth's FISTA and the reference's
interior-point LowRankQP land on the same optimum: weights agree to ~2e-4 and the
post-period ATT to ~1e-3 packs. At small ``lambda`` the fit recovers the canonical
Abadie-Diamond-Hainmueller donor pool (Utah, Nevada, Montana, Colorado,
Connecticut); as ``lambda`` grows the weights concentrate toward the nearest
neighbour (Montana), reproducing the penalty's interpolation property.

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
from mlsynth.utils.vanillasc_helpers.pipeline import _covariate_means

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DATA = os.path.join(_ROOT, "basedata", "augmented_cali_long.csv")
_RSCRIPT_REF = os.path.join(_ROOT, "benchmarks", "R", "pensynth_prop99.R")

_TREATED = "California"
# EXA_CaliforniaTobacco.R predictors: four MLAB covariate averages over their
# original windows + the full pre-treatment outcome path; V = diag (raw, no scaling).
_COVS = ["loginc", "p_cig", "pct15-24", "pc_beer"]
_WINDOWS = {
    "loginc": (1980, 1988), "p_cig": (1980, 1988),
    "pct15-24": (1980, 1988), "pc_beer": (1984, 1988),
}
# Penalty path up to the nearest-neighbour collapse (beyond ~0.25 the solution
# jumps to a single donor; that discontinuity is not a solver-parity test).
_GRID = [0.001, 0.01, 0.05, 0.1, 0.25]


def _matrices():
    """Build the EXA predictor matrix via dataprep + mlsynth's covariate helper."""
    d = pd.read_csv(_DATA)
    d["treated"] = ((d["state"] == _TREATED) & (d["year"] >= 1989)).astype(int)

    prep = dataprep(
        df=d, unit_id_column_name="state", time_period_column_name="year",
        outcome_column_name="cigsale", treatment_indicator_column_name="treated")
    pre = int(prep["pre_periods"])
    time_labels = list(np.asarray(prep["time_labels"]))
    pre_labels = time_labels[:pre]
    donors = [str(x) for x in prep["donor_names"]]
    units = [str(prep["treated_unit_name"])] + donors

    y = np.asarray(prep["y"], dtype=float).ravel()        # treated outcome path
    Y0 = np.asarray(prep["donor_matrix"], dtype=float)    # (T, J)

    # Covariate-average block (K_cov, N), then the pre-treatment outcome path
    # (pre, N) with the treated unit first -- the EXA stack, raw (V = I).
    Xcov = _covariate_means(d, units, _COVS, _WINDOWS, pre_labels, "state", "year")
    path = np.column_stack([y[:pre], Y0[:pre]])           # (pre, N)
    X = np.vstack([Xcov, path])                           # (K_cov + pre, N)
    X1, X0 = X[:, 0], X[:, 1:]
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
    w_montana = float(Wml_t[i01, donors.index("Montana")])

    return {
        "weight_max_abs_diff": weight_max_abs_diff,
        "att_max_abs_diff": att_max_abs_diff,
        "att_lambda_0p1": float(att_ml[i01]),
        "w_montana_lambda_0p1": w_montana,
    }


# mlsynth reproduces the authors' wsoll1 on the identical EXA predictor matrix to
# solver precision (weights ~2e-4, ATT ~1e-3 across the interpolation path). The
# anchored cells fix the actual penalized fit: at lambda=0.1 the synthetic
# California loads ~0.43 on Montana for a -23.3 packs post-period ATT.
EXPECTED = {
    "weight_max_abs_diff": (0.0, 2e-3),
    "att_max_abs_diff": (0.0, 5e-3),
    "att_lambda_0p1": (-23.268, 0.05),
    "w_montana_lambda_0p1": (0.434, 0.01),
}
