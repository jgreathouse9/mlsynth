"""Hardening TDD for the native L2 SCM-relaxation OSQP solver.

The bespoke solver already exists and is wired into the RESCM cross-validation
path (``laxscm_helpers.fast_solve.solve_relaxed_l2_osqp``, used by
``crossval.RelaxationCV._solve_relax_problem`` with a cvxpy fall-through). It
solves the L2 relaxation

    min ||w||^2  s.t.  sum(w) = 1,  w >= 0,
                       || (X'y - X'X w) / T0 + gamma ||_inf <= tau   (gamma free)

-- a strictly convex QP over the simplex and the relaxed-balance polytope, hence
a unique optimum. ``test_cv.py`` already pins a single cvxpy-parity instance;
this module hardens the contract: feasibility invariants, the slack/binding tau
regimes, the high-dimensional ``J > T0`` regime the relaxation is built for,
multi-seed parity, infeasibility detection (the balance floor), and the
defensive None-return / CV fall-through paths.

A regression this surfaced and fixed: when ``tau`` is below the balance floor the
program is primal-infeasible; OSQP returns a garbage iterate with status
``"primal infeasible"``, which the solver must reject (return ``None``) rather
than hand a meaningless weight vector to the CV scorer.
"""

from __future__ import annotations

import numpy as np
import osqp
import pytest

from mlsynth.utils.laxscm_helpers.fast_solve import solve_relaxed_l2_osqp


# ---------------------------------------------------------------------------
# cvxpy oracle (the exact program solve_relaxed_l2_osqp mirrors)
# ---------------------------------------------------------------------------

def _l2_relax_cvxpy(X, y, tau):
    import cvxpy as cp

    T0, J = X.shape
    G, h = X.T @ X, X.T @ y
    w, gam = cp.Variable(J), cp.Variable()
    prob = cp.Problem(
        cp.Minimize(cp.sum_squares(w)),
        [cp.sum(w) == 1, w >= 0, cp.norm((h - G @ w) / T0 + gam, "inf") <= tau],
    )
    prob.solve(solver=cp.CLARABEL)
    if w.value is None:
        return None
    return np.asarray(w.value, dtype=float).ravel()


def _data(seed, T0=24, J=8):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((T0, J)) * 2 + 5
    beta = np.zeros(J); beta[: min(3, J)] = [0.5, 0.3, 0.2][: min(3, J)]
    y = X @ beta + rng.standard_normal(T0) * 0.3
    return X, y


def _half_range(X, y, w):
    """Smallest achievable ||resid + gamma||_inf over gamma (the balance slack)."""
    resid = (X.T @ y - X.T @ X @ w) / X.shape[0]
    return 0.5 * (resid.max() - resid.min())


# ---------------------------------------------------------------------------
# Feasibility invariants
# ---------------------------------------------------------------------------

class TestFeasibility:
    def test_on_simplex(self):
        X, y = _data(0)
        w = solve_relaxed_l2_osqp(X, y, 0.5)
        assert w is not None
        assert w.shape == (X.shape[1],)
        assert w.sum() == pytest.approx(1.0, abs=1e-6)
        assert (w >= -1e-7).all()

    @pytest.mark.parametrize("tau", [0.3, 1.0, 5.0])
    def test_balance_polytope_satisfied(self, tau):
        X, y = _data(1)
        w = solve_relaxed_l2_osqp(X, y, tau)
        assert w is not None                      # these tau are above the floor
        assert _half_range(X, y, w) <= tau + 1e-4


# ---------------------------------------------------------------------------
# Regimes
# ---------------------------------------------------------------------------

class TestRegimes:
    def test_slack_tau_is_uniform(self):
        # When the balance constraint is slack, min ||w||^2 over the simplex is
        # the uniform vector 1/J.
        X, y = _data(2, T0=24, J=8)
        big_tau = 100.0 * float(np.abs(X.T @ y).max())
        w = solve_relaxed_l2_osqp(X, y, big_tau)
        np.testing.assert_allclose(w, np.full(8, 1.0 / 8), atol=1e-4)

    def test_binding_tau_departs_from_uniform(self):
        # A tau just below the uniform point's balance half-range makes the
        # constraint bind, so the optimum departs from 1/J (and matches cvxpy).
        X, y = _data(3, T0=24, J=8)
        hr_u = _half_range(X, y, np.full(8, 1.0 / 8))
        tau = 0.8 * hr_u
        w = solve_relaxed_l2_osqp(X, y, tau)
        assert w is not None
        assert np.abs(w - 1.0 / 8).max() > 1e-3
        np.testing.assert_allclose(w, _l2_relax_cvxpy(X, y, tau), atol=1e-3)

    def test_high_dimensional_J_gt_T0(self):
        # The relaxation's reason for being: more donors than pre-periods.
        X, y = _data(4, T0=10, J=25)
        w = solve_relaxed_l2_osqp(X, y, 0.3)
        assert w is not None
        assert w.sum() == pytest.approx(1.0, abs=1e-6)
        np.testing.assert_allclose(w, _l2_relax_cvxpy(X, y, 0.3), atol=1e-3)


# ---------------------------------------------------------------------------
# Parity with cvxpy (unique optimum) across seeds / taus
# ---------------------------------------------------------------------------

class TestParity:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    @pytest.mark.parametrize("tau", [0.3, 1.0])
    def test_matches_cvxpy(self, seed, tau):
        X, y = _data(seed)
        w = solve_relaxed_l2_osqp(X, y, tau)
        ref = _l2_relax_cvxpy(X, y, tau)
        assert w is not None and ref is not None
        np.testing.assert_allclose(w, ref, atol=1e-3)


# ---------------------------------------------------------------------------
# Infeasibility (tau below the balance floor) + defensive None paths
# ---------------------------------------------------------------------------

class TestInfeasibilityAndFailures:
    def test_infeasible_tau_returns_none(self):
        # A tau far below the balance floor is primal-infeasible; the solver must
        # return None (not a garbage iterate), matching cvxpy's infeasibility.
        X, y = _data(0)
        tau = 1e-6
        assert solve_relaxed_l2_osqp(X, y, tau) is None
        assert _l2_relax_cvxpy(X, y, tau) is None

    def test_none_when_solver_returns_nan(self, monkeypatch):
        class _Res:
            x = np.array([np.nan] * 9)
            info = type("I", (), {"status": "solved"})()

        class _Boom:
            def setup(self, *a, **k):
                pass

            def solve(self):
                return _Res()

        monkeypatch.setattr(osqp, "OSQP", lambda: _Boom())
        X, y = _data(0)
        assert solve_relaxed_l2_osqp(X, y, 0.3) is None

    def test_none_when_solution_all_zero(self, monkeypatch):
        class _Res:
            x = np.zeros(9)            # J=8 weights + gamma -> all zero
            info = type("I", (), {"status": "solved"})()

        class _Zero:
            def setup(self, *a, **k):
                pass

            def solve(self):
                return _Res()

        monkeypatch.setattr(osqp, "OSQP", lambda: _Zero())
        X, y = _data(0)
        assert solve_relaxed_l2_osqp(X, y, 0.3) is None

    def test_cv_falls_through_to_cvxpy_when_osqp_none(self, monkeypatch):
        # If the native solver returns None, RelaxationCV must still produce a
        # valid simplex fit via the SCopt/cvxpy reference.
        from mlsynth.utils.laxscm_helpers.crossval import RelaxationCV

        monkeypatch.setattr(
            "mlsynth.utils.laxscm_helpers.fast_solve.solve_relaxed_l2_osqp",
            lambda *a, **k: None,
        )
        X, y = _data(0, T0=30, J=6)
        model = RelaxationCV(tau=0.3, relaxation_type="l2").fit(X, y)
        assert model.coef_ is not None
        assert model.coef_.sum() == pytest.approx(1.0, abs=1e-6)
