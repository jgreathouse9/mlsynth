"""TDD for robustness of the relaxed-SYNDES weight QP (annealed solver inner step).

``solve_weights_global`` is a simplex-constrained least squares -- always feasible
(uniform weights satisfy the constraints). But its OSQP (first-order ADMM) solve
spuriously reports ``infeasible`` / inaccurate on large-magnitude, ill-scaled
panels, which crashes the annealed solver mid-search (it evaluates many candidate
assignments). The fix: fall back to the robust interior-point CLARABEL solver
before giving up. These tests pin "never crash on a feasible QP; return
simplex-feasible weights matching a direct robust solve."
"""
from __future__ import annotations

import numpy as np
import cvxpy as cp
import pytest

from mlsynth.utils.syndes_helpers.relaxed_formulation import solve_weights_global


def _big_panel(T=40, N=16, seed=0):
    # Large-magnitude, seasonal, partly-collinear panel (mimics real geo sales),
    # the regime where OSQP mis-reports the simplex LSQ as infeasible.
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    season = 500.0 * np.sin(2 * np.pi * t / 12)[:, None]
    levels = 3000.0 + rng.standard_normal(N) * 400.0
    return levels[None, :] + season + rng.standard_normal((T, N)) * 50.0


def _direct(Y, D):
    tr = np.where(D == 1)[0]; co = np.where(D == 0)[0]
    wT = cp.Variable(len(tr)); wC = cp.Variable(len(co))
    p = cp.Problem(cp.Minimize(cp.sum_squares(Y[:, tr] @ wT - Y[:, co] @ wC)),
                   [wT >= 0, wC >= 0, cp.sum(wT) == 1, cp.sum(wC) == 1])
    p.solve(solver=cp.CLARABEL)
    return float(p.value)


class TestRelaxedWeightRobustness:
    def test_never_crashes_on_feasible_qp(self):
        # Across many random assignments on a large-magnitude panel, the weight
        # QP must always return simplex-feasible weights (no MlsynthEstimationError).
        Y = _big_panel(seed=1); N = Y.shape[1]; rng = np.random.default_rng(7)
        for _ in range(40):
            D = np.zeros(N, dtype=int); D[rng.choice(N, 3, replace=False)] = 1
            w = solve_weights_global(Y, D, lam=0.0)
            tr = np.where(D == 1)[0]; co = np.where(D == 0)[0]
            assert np.all(w >= -1e-5)            # feasible to solver tolerance
            assert w[tr].sum() == pytest.approx(1.0, abs=1e-4)
            assert w[co].sum() == pytest.approx(1.0, abs=1e-4)

    def test_matches_direct_robust_solve(self):
        Y = _big_panel(seed=2); N = Y.shape[1]; rng = np.random.default_rng(3)
        for _ in range(10):
            D = np.zeros(N, dtype=int); D[rng.choice(N, 3, replace=False)] = 1
            w = solve_weights_global(Y, D, lam=0.0)
            tr = np.where(D == 1)[0]; co = np.where(D == 0)[0]
            obj = float(np.sum((Y[:, tr] @ w[tr] - Y[:, co] @ w[co]) ** 2))
            assert obj <= _direct(Y, D) * (1 + 1e-4) + 1e-6   # at the optimum
