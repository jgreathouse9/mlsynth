"""Cross-validation: mlsynth RESCM L2-relaxation vs the authors' ``scmrelax``.

Matches mlsynth's relaxed-SCM engine cell-by-cell against the reference
implementation of Liao, Shi & Zheng (2026) -- the ``scmrelax`` package
(https://github.com/metricshilab/scmrelax) -- on a shared panel at a matched
relaxation level ``tau``. At a fixed ``tau`` the L2-relaxation program is a
unique QP, so the two independent implementations should agree to solver
precision; this guards mlsynth's relaxed branch against drift from the source.

Provenance / scenario
---------------------
* Full repo (scenario 3): cross-validation is mandatory and done here.
* Reference: ``scmrelax.L2RelaxationCV`` at a single ``tau``; pinned package
  https://github.com/metricshilab/scmrelax (installable from
  https://github.com/PanJi-0/scmrelax).
* mlsynth side: ``RESCM(methods=["RELAX_L2"], tau=tau, standardize=False)`` --
  ``standardize=False`` matches the reference, which solves on the raw series.
* The case **skips gracefully** when ``scmrelax`` is not installed, or when no
  open conic solver can stand in for the reference's MOSEK dependency.

Notes
-----
``scmrelax`` hardcodes the commercial MOSEK solver; the L2 relaxation is a plain
QP, so we transparently route its solves to an open solver (CLARABEL) for the
duration of the case and restore cvxpy afterwards. The optimum is solver-
invariant for this convex program.
"""
from __future__ import annotations

import warnings

import numpy as np

T0, J, T1 = 30, 10, 8
SEED = 1


def _toy_panel():
    """Deterministic factor panel: donors ``Yc`` (J, T), treated ``y0`` (T,)."""
    rng = np.random.default_rng(SEED)
    T = T0 + T1
    F = rng.normal(size=(T, 3))
    load = rng.normal(size=(J, 3))
    Yc = (load @ F.T) + rng.normal(0, 0.3, size=(J, T))      # (J, T)
    y0 = (F @ np.array([0.4, 0.3, 0.3])) + rng.normal(0, 0.3, T)
    return Yc, y0


def run() -> dict:
    import pandas as pd

    from benchmarks.compare import BenchmarkSkipped
    try:
        import cvxpy as cp
        from scmrelax.scmrelax import L2RelaxationCV
    except Exception as exc:
        raise BenchmarkSkipped(f"scmrelax/cvxpy unavailable: {exc}")

    from mlsynth import RESCM
    from mlsynth.utils.laxscm_helpers.simulation import to_panel

    Yc, y0 = _toy_panel()
    X_pre = Yc[:, :T0].T            # (T0, J) for scmrelax
    y_pre = y0[:T0]

    # Relaxation level: half of eta-bar (where the equal-weight solution becomes
    # feasible), comfortably interior so the QP is well-conditioned.
    Sig = X_pre.T @ X_pre / T0
    Ups = X_pre.T @ y_pre / T0
    g = cp.Variable()
    prob = cp.Problem(cp.Minimize(
        cp.pnorm(Sig @ (np.ones(J) / J) - Ups + g * np.ones(J), "inf")))
    try:
        etabar = float(prob.solve(solver=cp.CLARABEL))
    except Exception as exc:
        raise BenchmarkSkipped(f"no open solver for the reference QP: {exc}")
    tau = 0.5 * etabar

    # Reference (scmrelax) with MOSEK transparently routed to CLARABEL.
    _orig = cp.Problem.solve

    def _patched(self, *a, **k):
        if k.get("solver") == cp.MOSEK:
            k["solver"] = cp.CLARABEL
            k.pop("mosek_params", None)
        return _orig(self, *a, **k)

    try:
        cp.Problem.solve = _patched
        ref = L2RelaxationCV(taus=np.array([tau]), cv=2, nonneg=True).fit(X_pre, y_pre)
    except Exception as exc:
        raise BenchmarkSkipped(f"scmrelax reference solve failed: {exc}")
    finally:
        cp.Problem.solve = _orig
    w_ref = np.asarray(ref.coef_, dtype=float)

    # mlsynth via the public RESCM at the same tau, raw scale.
    df = to_panel(Yc, y0, T0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = RESCM({"df": df, "outcome": "y", "treat": "treat", "unitid": "unit",
                     "time": "time", "methods": ["RELAX_L2"], "tau": tau,
                     "standardize": False, "display_graphs": False}).fit()
    dw = res.fits["RELAX_L2"].donor_weights
    w_ml = np.array([dw.get(f"c{j:03d}", 0.0) for j in range(J)], dtype=float)

    return {
        "weight_l1_diff": float(np.abs(w_ref - w_ml).sum()),
        "weight_max_abs_diff": float(np.abs(w_ref - w_ml).max()),
        "ref_on_simplex": float(abs(w_ref.sum() - 1.0) < 1e-4 and (w_ref >= -1e-6).all()),
        # RESCM rounds its public donor_weights to 3 sig figs, so the reported
        # vector sums to ~1 only up to that rounding.
        "ml_on_simplex": float(abs(w_ml.sum() - 1.0) < 1e-2 and (w_ml >= -1e-6).all()),
    }


# Two independent implementations of the same unique QP -> agreement to solver
# precision. Tolerances cover the CLARABEL-vs-CLARABEL numerical slack only.
EXPECTED = {
    "weight_l1_diff": (0.0, 5e-3),
    "weight_max_abs_diff": (0.0, 2e-3),
    "ref_on_simplex": (1.0, 0.0),
    "ml_on_simplex": (1.0, 0.0),
}
