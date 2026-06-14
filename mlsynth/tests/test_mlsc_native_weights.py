"""Native (active-set) MLSC weight solvers: parity + independence tests.

The MLSC program is a penalized simplex least squares,

    min_omega ||Y - X omega||^2 + lambda * sigma_y^2 * omega^T Q omega
    s.t. sum(omega) = 1, omega >= 0,

with the block penalty ``Q = R^T R`` (``R_s = I - v_s 1^T`` per block). It is
strictly convex for any lambda > 0 whenever the aggregate control matrix has full
column rank (the penalty fills exactly the part of X's null space the aggregate
directions do not), so the optimum is unique. That lets us solve it with the
library's active-set simplex QP after folding the penalty into the design as a
``sqrt(lambda sigma_y^2) R`` augmentation -- no cvxpy on the default path.

These tests pin, TDD-first:

1. ``R^T R == Q`` (the square-root factor),
2. parity with a cvxpy/CLARABEL oracle on strictly-convex instances
   (lambda > 0 and the lambda = 0 ridge branch),
3. simplex feasibility,
4. cross-validation selecting the same grid penalty as a cvxpy oracle,
5. independence from cvxpy on the default (solver=None) path,
6. the explicit-solver escape hatch still routing through cvxpy.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlsynth.utils.mlsc_helpers.crossval import _DEFAULT_GRID, select_lambda_cv
from mlsynth.utils.mlsc_helpers.optimization import solve_mlsc
from mlsynth.utils.mlsc_helpers.penalty import (
    build_penalty_matrix,
    build_sqrt_factor,
)
from mlsynth.utils.mlsc_helpers.setup import prepare_mlsc_inputs
from mlsynth.utils.mlsc_helpers.simulation import simulate_mlsc_sample
from mlsynth.utils.mlsc_helpers.variance import estimate_variance_components


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _inputs(seed: int = 0):
    s = simulate_mlsc_sample(rng=np.random.default_rng(seed))
    return prepare_mlsc_inputs(
        s.df_agg, s.df_disagg, outcome="y", time="time", treat="treated",
        unitid_agg="state", unitid_disagg="county", agg_id="state",
        weight_col=None,
    )


def _problem(seed: int = 0):
    inputs = _inputs(seed)
    Q = build_penalty_matrix(inputs.v_population, inputs.disagg_to_agg)
    _, sigma_y2 = estimate_variance_components(inputs)
    return inputs, Q, float(sigma_y2)


def _solve_cvxpy_oracle(inputs, Q, lambda_val, sigma_y2):
    """CLARABEL oracle for the penalized simplex program (lambda=0 -> 1e-8 ridge)."""
    import cvxpy as cp

    T0 = inputs.T0
    Y = inputs.Y_agg_treated[:T0]
    X = inputs.X_disagg[:T0, :]
    M = inputs.M
    w = cp.Variable(M)
    ps = float(lambda_val) * float(sigma_y2)
    terms = [cp.sum_squares(Y - X @ w)]
    if ps > 0:
        terms.append(ps * cp.quad_form(w, cp.psd_wrap(Q)))
    else:
        terms.append(1e-8 * cp.sum_squares(w))
    cp.Problem(cp.Minimize(sum(terms)), [cp.sum(w) == 1, w >= 0]).solve(solver=cp.CLARABEL)
    wv = np.clip(np.asarray(w.value, dtype=float), 0.0, None)
    return wv / wv.sum()


# ---------------------------------------------------------------------------
# Square-root factor
# ---------------------------------------------------------------------------

class TestSqrtFactor:
    def test_reconstructs_Q(self):
        inputs = _inputs(0)
        Q = build_penalty_matrix(inputs.v_population, inputs.disagg_to_agg)
        R = build_sqrt_factor(inputs.v_population, inputs.disagg_to_agg)
        np.testing.assert_allclose(R.T @ R, Q, atol=1e-10)

    def test_shape(self):
        inputs = _inputs(0)
        R = build_sqrt_factor(inputs.v_population, inputs.disagg_to_agg)
        assert R.shape == (inputs.M, inputs.M)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            build_sqrt_factor(np.array([0.5, 0.5]), np.array([0]))

    def test_penalty_matrix_length_mismatch_raises(self):
        from mlsynth.utils.mlsc_helpers.penalty import build_penalty_matrix

        with pytest.raises(ValueError):
            build_penalty_matrix(np.array([0.5, 0.5]), np.array([0]))


# ---------------------------------------------------------------------------
# Parity with cvxpy (strictly convex -> unique)
# ---------------------------------------------------------------------------

class TestSolveParity:
    def test_lambda_positive_matches_oracle(self):
        inputs, Q, sigma_y2 = _problem(0)
        lam = 5.0
        omega, agg, status = solve_mlsc(inputs, Q, lam, sigma_y2)
        ref = _solve_cvxpy_oracle(inputs, Q, lam, sigma_y2)
        np.testing.assert_allclose(omega, ref, atol=1e-5)

    def test_lambda_positive_aggregate_weights(self):
        inputs, Q, sigma_y2 = _problem(1)
        omega, agg, _ = solve_mlsc(inputs, Q, 2.0, sigma_y2)
        S = int(inputs.disagg_to_agg.max() + 1)
        expect = np.array([omega[inputs.disagg_to_agg == s].sum() for s in range(S)])
        np.testing.assert_allclose(agg, expect, atol=1e-10)
        assert agg.sum() == pytest.approx(1.0, abs=1e-6)

    def test_lambda_zero_ridge_matches_oracle(self):
        inputs, Q, sigma_y2 = _problem(2)
        omega, _, _ = solve_mlsc(inputs, Q, 0.0, sigma_y2)
        ref = _solve_cvxpy_oracle(inputs, Q, 0.0, sigma_y2)
        np.testing.assert_allclose(omega, ref, atol=1e-4)


class TestFeasibility:
    @pytest.mark.parametrize("lam", [0.0, 1e-3, 1.0, 100.0])
    def test_simplex(self, lam):
        inputs, Q, sigma_y2 = _problem(3)
        omega, agg, _ = solve_mlsc(inputs, Q, lam, sigma_y2)
        assert omega.shape == (inputs.M,)
        assert omega.sum() == pytest.approx(1.0, abs=1e-6)
        assert (omega >= -1e-8).all()


# ---------------------------------------------------------------------------
# Cross-validation parity
# ---------------------------------------------------------------------------

class TestCrossValidation:
    def _cv_cvxpy_oracle(self, inputs, Q, sigma_y2, grid):
        import cvxpy as cp

        T0 = inputs.T0
        t_cv = T0 - 1
        Y = inputs.Y_agg_treated[:T0]
        X = inputs.X_disagg[:T0, :]
        Xtr, Ytr = X[:t_cv], Y[:t_cv]
        Xte, Yte = X[t_cv:T0], Y[t_cv:T0]
        M = Xtr.shape[1]
        err = np.full(len(grid), np.inf)
        for i, v in enumerate(grid):
            w = cp.Variable(M)
            if v == 0.0:
                obj = cp.sum_squares(Ytr - Xtr @ w) + 1e-8 * cp.sum_squares(w)
            else:
                obj = (cp.sum_squares(Ytr - Xtr @ w)
                       + v * sigma_y2 * cp.quad_form(w, cp.psd_wrap(Q)))
            cp.Problem(cp.Minimize(obj), [cp.sum(w) == 1, w >= 0]).solve(solver=cp.CLARABEL)
            if w.value is not None:
                err[i] = float(np.mean((Yte - Xte @ np.asarray(w.value)) ** 2))
        return float(grid[int(np.argmin(err))])

    def test_selects_same_grid_penalty_as_oracle(self):
        inputs, Q, sigma_y2 = _problem(0)
        grid = np.asarray(_DEFAULT_GRID, dtype=float)
        lam = select_lambda_cv(inputs, Q, sigma_y2)
        lam_ref = self._cv_cvxpy_oracle(inputs, Q, sigma_y2, grid)
        # Same grid step, or within one neighbouring step (CV curve can be flat).
        idx = int(np.argmin(np.abs(grid - lam)))
        idx_ref = int(np.argmin(np.abs(grid - lam_ref)))
        assert abs(idx - idx_ref) <= 1

    def test_returns_grid_member(self):
        inputs, Q, sigma_y2 = _problem(1)
        lam = select_lambda_cv(inputs, Q, sigma_y2)
        assert np.isclose(np.asarray(_DEFAULT_GRID, dtype=float), lam).any()

    def test_too_few_periods_raises(self):
        inputs, Q, sigma_y2 = _problem(0)
        with pytest.raises(ValueError):
            select_lambda_cv(inputs, Q, sigma_y2, cv_holdout_periods=inputs.T0)


# ---------------------------------------------------------------------------
# Independence from cvxpy (default path)
# ---------------------------------------------------------------------------

class TestNoCvxpyDefaultPath:
    def test_solve_without_cvxpy(self, monkeypatch):
        import cvxpy

        def boom(*a, **k):  # pragma: no cover - must never be called
            raise AssertionError("native solve_mlsc must not call cvxpy")

        monkeypatch.setattr(cvxpy, "Problem", boom)
        inputs, Q, sigma_y2 = _problem(0)
        omega, _, _ = solve_mlsc(inputs, Q, 3.0, sigma_y2)
        assert omega.sum() == pytest.approx(1.0, abs=1e-6)

    def test_cv_without_cvxpy(self, monkeypatch):
        import cvxpy

        def boom(*a, **k):  # pragma: no cover - must never be called
            raise AssertionError("native select_lambda_cv must not call cvxpy")

        monkeypatch.setattr(cvxpy, "Problem", boom)
        inputs, Q, sigma_y2 = _problem(0)
        lam = select_lambda_cv(inputs, Q, sigma_y2)
        assert np.isfinite(lam)

    def test_modules_do_not_import_cvxpy(self):
        import mlsynth.utils.mlsc_helpers.crossval as cv
        import mlsynth.utils.mlsc_helpers.optimization as op

        assert not hasattr(op, "cp")
        assert not hasattr(cv, "cp")


# ---------------------------------------------------------------------------
# Explicit-solver escape hatch
# ---------------------------------------------------------------------------

class TestExplicitSolverEscapeHatch:
    def test_explicit_solver_matches_native(self):
        import cvxpy as cp

        inputs, Q, sigma_y2 = _problem(0)
        omega_native, _, _ = solve_mlsc(inputs, Q, 5.0, sigma_y2)
        omega_cvxpy, _, _ = solve_mlsc(inputs, Q, 5.0, sigma_y2, solver=cp.CLARABEL)
        np.testing.assert_allclose(omega_native, omega_cvxpy, atol=1e-5)

    def test_explicit_solver_cv_matches_native(self):
        # Routes select_lambda_cv through the cvxpy grid sweep (incl. the
        # lambda=0 county-SC branch); should pick the same grid penalty.
        import cvxpy as cp

        inputs, Q, sigma_y2 = _problem(0)
        grid = np.asarray(_DEFAULT_GRID, dtype=float)
        lam_native = select_lambda_cv(inputs, Q, sigma_y2)
        lam_cvxpy = select_lambda_cv(inputs, Q, sigma_y2, solver=cp.CLARABEL)
        idx = int(np.argmin(np.abs(grid - lam_native)))
        idx_cvxpy = int(np.argmin(np.abs(grid - lam_cvxpy)))
        assert abs(idx - idx_cvxpy) <= 1
