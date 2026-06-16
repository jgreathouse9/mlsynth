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

The predictor matrix is the California-vs-38-states Prop 99 panel from mlsynth's
vendored ``basedata/P99data.csv`` (Abadie's smoking dataset, 1970-2000), matched
on the pre-treatment cigarette-sales path 1970-1988 with ``Gamma = I`` -- the
lagged-outcome predictor block of the authors' California example
(``EXA_CaliforniaTobacco.R``). The penalised QP is deterministic in
``(X0, X1, lambda)`` and strictly convex for ``lambda > 0`` (their Theorem 1), so
mlsynth's FISTA and the reference's interior-point LowRankQP land on the same
unique optimum: weights agree to ~1e-4 and the implied ATT to ~1e-3 across the
path. (The residual is convergence/threshold slack on sub-1e-6 weights, not a
methodological difference.)

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

import numpy as np
import pandas as pd

from benchmarks.compare import BenchmarkSkipped
from benchmarks.reference.clone_pensynth import functions_dir
from mlsynth.utils.bilevel.penalized import penalized_weights

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DATA = os.path.join(_ROOT, "basedata", "P99data.csv")
_RSCRIPT_REF = os.path.join(_ROOT, "benchmarks", "R", "pensynth_prop99.R")

_TREATED = "California"
_PRE = list(range(1970, 1989))          # pre-treatment matching window (predictors)
_ALL = list(range(1970, 2001))
_POST = list(range(1989, 2001))         # Prop 99 took effect in 1989
_GRID = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0]


def _matrices():
    """Build the California-vs-donors predictor and outcome matrices from P99data."""
    d = pd.read_csv(_DATA)
    wide = d.pivot(index="state", columns="year", values="cigsale")
    donors = [s for s in wide.index if s != _TREATED]
    X1 = wide.loc[_TREATED, _PRE].to_numpy(float)             # (p,)
    X0 = wide.loc[donors, _PRE].to_numpy(float).T             # (p, J) predictors x donors
    y1_all = wide.loc[_TREATED, _ALL].to_numpy(float)         # (T,)
    Y0_all = wide.loc[donors, _ALL].to_numpy(float)           # (J, T)
    return X1, X0, y1_all, Y0_all, donors


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
    X1, X0, y1_all, Y0_all, donors = _matrices()
    funcs = functions_dir()                       # clones the pinned repo (skips if absent)
    Wref = _reference_weights(X0, X1, funcs)      # (len(grid), J)

    Wml = np.array([penalized_weights(X1, X0, lam, max_iter=50000, tol=1e-13)
                    for lam in _GRID])
    Wml_t, Wref_t = _tzero(Wml), _tzero(Wref)

    # Largest donor-weight gap across the whole regularisation path.
    weight_max_abs_diff = float(np.max(np.abs(Wml_t - Wref_t)))

    # ATT (post-period mean gap) at each lambda, both implementations.
    post = np.array([_ALL.index(y) for y in _POST])
    att_ml = np.array([float(np.mean((y1_all - w @ Y0_all)[post])) for w in Wml])
    att_ref = np.array([float(np.mean((y1_all - w @ Y0_all)[post])) for w in Wref])
    att_max_abs_diff = float(np.max(np.abs(att_ml - att_ref)))

    i01 = _GRID.index(0.1)
    w_montana = float(Wml_t[i01, donors.index("Montana")])

    return {
        "weight_max_abs_diff": weight_max_abs_diff,
        "att_max_abs_diff": att_max_abs_diff,
        "att_lambda_0p1": float(att_ml[i01]),
        "w_montana_lambda_0p1": w_montana,
    }


# mlsynth reproduces the authors' wsoll1 to solver precision on the identical
# matrices. The diff tolerances pin a genuine match (weights ~1e-4, ATT ~1e-3);
# the two anchored cells fix the actual fit (California's synthetic at lambda=0.1
# loads ~0.48 on Montana, giving a -23.5 packs post-period ATT).
EXPECTED = {
    "weight_max_abs_diff": (0.0, 2e-3),
    "att_max_abs_diff": (0.0, 5e-3),
    "att_lambda_0p1": (-23.478, 0.05),
    "w_montana_lambda_0p1": (0.478, 0.01),
}
