#!/usr/bin/env python3
"""Reference runner: the authors' ``scmrelax`` L2 relaxation on the toy factor
panel of the ``rescm_relax_ref`` cross-validation case.

Generates the same deterministic factor panel the Python case uses, picks the
same interior relaxation level ``tau`` (half of ``eta-bar``, the level at which
the equal-weight solution becomes feasible), and solves the L2-relaxation QP with
``scmrelax.L2RelaxationCV`` at that single ``tau``. At a fixed ``tau`` the program
is a unique convex QP, so its weights are the verified optimum -- emitted under
``== REFERENCE VALUES ==`` for ``benchmarks/reference/generate.py`` to capture.

``scmrelax`` hardcodes the commercial MOSEK solver; the L2 relaxation is a plain
QP whose optimum is solver-invariant, so the run transparently routes MOSEK to
the open CLARABEL solver. The reference is Liao-Shi-Zheng's relaxed-balanced
synthetic control, available as BOTH github.com/YapengZheng/Relaxed_SC (original)
and github.com/metricshilab/scmrelax (the packaged code, used here).

Run via the bundle's manifest::

    python benchmarks/reference/generate.py rescm_relax_ref
"""
from __future__ import annotations

import numpy as np

# Mirror of benchmarks/cases/rescm_relax_ref.py's panel parameters.
T0, J, T1 = 30, 10, 8
SEED = 1


def _toy_panel():
    """Deterministic factor panel: donors ``Yc`` (J, T), treated ``y0`` (T,)."""
    rng = np.random.default_rng(SEED)
    T = T0 + T1
    F = rng.normal(size=(T, 3))
    load = rng.normal(size=(J, 3))
    Yc = (load @ F.T) + rng.normal(0, 0.3, size=(J, T))
    y0 = (F @ np.array([0.4, 0.3, 0.3])) + rng.normal(0, 0.3, T)
    return Yc, y0


def main() -> int:
    import cvxpy as cp
    from scmrelax.scmrelax import L2RelaxationCV

    Yc, y0 = _toy_panel()
    X_pre = Yc[:, :T0].T            # (T0, J)
    y_pre = y0[:T0]

    # Interior relaxation level: half of eta-bar (equal-weight feasibility).
    Sig = X_pre.T @ X_pre / T0
    Ups = X_pre.T @ y_pre / T0
    g = cp.Variable()
    prob = cp.Problem(cp.Minimize(
        cp.pnorm(Sig @ (np.ones(J) / J) - Ups + g * np.ones(J), "inf")))
    etabar = float(prob.solve(solver=cp.CLARABEL))
    tau = 0.5 * etabar

    # Route scmrelax's hardcoded MOSEK to the open CLARABEL solver.
    _orig = cp.Problem.solve

    def _patched(self, *a, **k):
        if k.get("solver") == cp.MOSEK:
            k["solver"] = cp.CLARABEL
            k.pop("mosek_params", None)
        return _orig(self, *a, **k)

    try:
        cp.Problem.solve = _patched
        ref = L2RelaxationCV(taus=np.array([tau]), cv=2, nonneg=True).fit(X_pre, y_pre)
    finally:
        cp.Problem.solve = _orig

    w = np.asarray(ref.coef_, dtype=float)

    print("== REFERENCE VALUES ==")
    print(f"tau\t{tau:.10f}")
    print(f"weight_sum\t{float(w.sum()):.10f}")
    print(f"weight_min\t{float(w.min()):.10f}")
    for j in range(J):
        print(f"weight\tc{j:03d}\t{float(w[j]):.10f}")
    print("== END ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
