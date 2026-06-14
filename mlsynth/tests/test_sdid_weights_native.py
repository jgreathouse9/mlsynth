"""Native (cvxpy-free) SDID weight solvers: parity + independence tests.

The SDID unit- and time-weight programs are simplex-constrained least squares
with a free intercept (and, for unit weights, an L2 ridge). Both reduce to the
library's active-set simplex QP (``bilevel.active_set.solve_simplex_qp``) once
the intercept is profiled out by centering over the observation axis and the
ridge is folded in by augmenting the design with a ``sqrt(lambda) * I`` block.

These tests pin two things, TDD-first:

1. **Parity** -- on strictly-convex (unique-optimum) instances the native
   solvers reproduce the cvxpy/CLARABEL oracle (weights *and* intercept) to
   high tolerance. Uniqueness is guaranteed by construction: the unit-weight
   program carries a positive ridge; the time-weight instance is overdetermined
   (more donors than pre-periods).
2. **Independence** -- the native solvers never touch cvxpy. We monkeypatch
   ``cvxpy.Problem`` to explode and assert the solvers still return correct
   results.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.sdid_helpers.weights import (
    compute_regularization,
    fit_time_weights,
    unit_weights,
)


# ---------------------------------------------------------------------------
# cvxpy oracles (mirror the pre-refactor implementation exactly)
# ---------------------------------------------------------------------------

def _unit_weights_cvxpy(Y0_pre, y_treated_pre, zeta):
    import cvxpy as cp

    T0, N = Y0_pre.shape
    a = cp.Variable()
    w = cp.Variable(N, nonneg=True)
    pred = a + Y0_pre @ w
    penalty = T0 * (float(zeta) ** 2) * cp.sum_squares(w)
    prob = cp.Problem(
        cp.Minimize(cp.sum_squares(pred - y_treated_pre) + penalty),
        [cp.sum(w) == 1],
    )
    prob.solve(solver=cp.CLARABEL)
    return float(a.value), np.asarray(w.value).ravel()


def _time_weights_cvxpy(Y0_pre, mean_post):
    import cvxpy as cp

    T0, N = Y0_pre.shape
    a = cp.Variable()
    lam = cp.Variable(T0, nonneg=True)
    pred = a + (lam @ Y0_pre)
    prob = cp.Problem(
        cp.Minimize(cp.sum_squares(pred - mean_post)),
        [cp.sum(lam) == 1],
    )
    prob.solve(solver=cp.CLARABEL)
    return float(a.value), np.asarray(lam.value).ravel()


# ---------------------------------------------------------------------------
# Parity (strictly-convex / unique optimum)
# ---------------------------------------------------------------------------

class TestParityWithCvxpy:
    def test_unit_weights_match_oracle(self):
        rng = np.random.default_rng(11)
        T0, N = 18, 6
        Y0_pre = rng.standard_normal((T0, N)) * 3 + 10
        y_pre = rng.standard_normal(T0) * 3 + 10
        zeta = compute_regularization(Y0_pre, 12)
        a_ref, w_ref = _unit_weights_cvxpy(Y0_pre, y_pre, zeta)

        a, w = unit_weights(Y0_pre, y_pre, zeta)
        assert w is not None
        np.testing.assert_allclose(w, w_ref, atol=1e-6)
        assert a == pytest.approx(a_ref, abs=1e-6)

    def test_unit_weights_match_oracle_underdetermined_with_ridge(self):
        # J > T0: the *unregularized* fit is non-unique, but the ridge makes
        # the program strictly convex, so the optimum is unique and the native
        # solver must still reproduce CLARABEL.
        rng = np.random.default_rng(12)
        T0, N = 8, 20
        Y0_pre = rng.standard_normal((T0, N)) * 2 + 5
        y_pre = rng.standard_normal(T0) * 2 + 5
        zeta = compute_regularization(Y0_pre, 10)
        a_ref, w_ref = _unit_weights_cvxpy(Y0_pre, y_pre, zeta)

        a, w = unit_weights(Y0_pre, y_pre, zeta)
        np.testing.assert_allclose(w, w_ref, atol=1e-6)
        assert a == pytest.approx(a_ref, abs=1e-6)

    def test_time_weights_match_oracle_overdetermined(self):
        # More donors than pre-periods -> overdetermined -> unique optimum.
        rng = np.random.default_rng(13)
        T0, N = 10, 40
        Y0_pre = rng.standard_normal((T0, N)) * 4 + 8
        mean_post = rng.standard_normal(N) * 4 + 8
        a_ref, lam_ref = _time_weights_cvxpy(Y0_pre, mean_post)

        a, lam = fit_time_weights(Y0_pre, mean_post)
        assert lam is not None
        np.testing.assert_allclose(lam, lam_ref, atol=1e-6)
        assert a == pytest.approx(a_ref, abs=1e-6)


# ---------------------------------------------------------------------------
# Simplex feasibility + intercept optimality
# ---------------------------------------------------------------------------

class TestFeasibilityAndIntercept:
    def test_unit_weights_simplex(self):
        rng = np.random.default_rng(20)
        Y0_pre = rng.standard_normal((12, 5))
        y_pre = rng.standard_normal(12)
        a, w = unit_weights(Y0_pre, y_pre, 0.1)
        assert w.shape == (5,)
        assert w.sum() == pytest.approx(1.0, abs=1e-9)
        assert (w >= -1e-9).all()
        assert np.isfinite(a)

    def test_time_weights_simplex(self):
        rng = np.random.default_rng(21)
        Y0_pre = rng.standard_normal((10, 25))
        mean_post = rng.standard_normal(25)
        a, lam = fit_time_weights(Y0_pre, mean_post)
        assert lam.shape == (10,)
        assert lam.sum() == pytest.approx(1.0, abs=1e-9)
        assert (lam >= -1e-9).all()
        assert np.isfinite(a)

    def test_unit_weights_intercept_is_profiled_optimum(self):
        # For the returned weights, the optimal free intercept is the mean
        # residual a* = mean(y - Y0 @ w); the solver must return exactly that.
        rng = np.random.default_rng(22)
        T0, N = 15, 6
        Y0_pre = rng.standard_normal((T0, N)) * 2
        y_pre = rng.standard_normal(T0) * 2
        a, w = unit_weights(Y0_pre, y_pre, 0.2)
        a_star = float(np.mean(y_pre - Y0_pre @ w))
        assert a == pytest.approx(a_star, abs=1e-9)


# ---------------------------------------------------------------------------
# Independence from cvxpy
# ---------------------------------------------------------------------------

class TestNoCvxpyDependence:
    def test_unit_weights_without_cvxpy(self, monkeypatch):
        import cvxpy

        def boom(*a, **k):  # pragma: no cover - must never be called
            raise AssertionError("native unit_weights must not call cvxpy")

        monkeypatch.setattr(cvxpy, "Problem", boom)
        rng = np.random.default_rng(30)
        Y0_pre = rng.standard_normal((12, 5))
        y_pre = rng.standard_normal(12)
        a, w = unit_weights(Y0_pre, y_pre, 0.1)
        assert w.sum() == pytest.approx(1.0, abs=1e-9)

    def test_time_weights_without_cvxpy(self, monkeypatch):
        import cvxpy

        def boom(*a, **k):  # pragma: no cover - must never be called
            raise AssertionError("native fit_time_weights must not call cvxpy")

        monkeypatch.setattr(cvxpy, "Problem", boom)
        rng = np.random.default_rng(31)
        Y0_pre = rng.standard_normal((10, 20))
        mean_post = rng.standard_normal(20)
        a, lam = fit_time_weights(Y0_pre, mean_post)
        assert lam.sum() == pytest.approx(1.0, abs=1e-9)

    def test_module_does_not_import_cvxpy(self):
        import mlsynth.utils.sdid_helpers.weights as wmod

        assert not hasattr(wmod, "cp"), "weights.py should no longer import cvxpy as cp"


# ---------------------------------------------------------------------------
# Validation contract preserved
# ---------------------------------------------------------------------------

class TestValidationPreserved:
    def test_unit_weights_rejects_non_array(self):
        with pytest.raises(MlsynthDataError):
            unit_weights([[1.0]], np.zeros(1), 0.1)

    def test_unit_weights_rejects_negative_zeta(self):
        with pytest.raises(MlsynthConfigError):
            unit_weights(np.zeros((2, 2)), np.zeros(2), -1.0)

    def test_unit_weights_rejects_shape_mismatch(self):
        with pytest.raises(MlsynthDataError):
            unit_weights(np.zeros((3, 2)), np.zeros(4), 0.1)

    def test_time_weights_rejects_donor_mismatch(self):
        with pytest.raises(MlsynthDataError):
            fit_time_weights(np.zeros((3, 2)), np.zeros(5))

    def test_time_weights_rejects_1d_donor_matrix(self):
        with pytest.raises(MlsynthDataError):
            fit_time_weights(np.zeros(3), np.zeros(3))

    def test_time_weights_rejects_non_array_donor(self):
        with pytest.raises(MlsynthDataError):
            fit_time_weights([[1.0, 2.0]], np.zeros(2))

    def test_time_weights_rejects_non_array_post_mean(self):
        with pytest.raises(MlsynthDataError):
            fit_time_weights(np.zeros((3, 2)), [0.0, 0.0])

    def test_time_weights_rejects_2d_post_mean(self):
        with pytest.raises(MlsynthDataError):
            fit_time_weights(np.zeros((3, 2)), np.zeros((2, 1)))

    def test_time_weights_rejects_zero_pre_periods(self):
        with pytest.raises(MlsynthDataError):
            fit_time_weights(np.zeros((0, 2)), np.zeros(2))

    def test_time_weights_rejects_zero_donors(self):
        with pytest.raises(MlsynthDataError):
            fit_time_weights(np.zeros((3, 0)), np.zeros(0))

    def test_unit_weights_rejects_non_array_target(self):
        with pytest.raises(MlsynthDataError):
            unit_weights(np.zeros((3, 2)), [0.0, 0.0, 0.0], 0.1)

    def test_unit_weights_rejects_2d_target(self):
        with pytest.raises(MlsynthDataError):
            unit_weights(np.zeros((3, 2)), np.zeros((3, 1)), 0.1)

    def test_unit_weights_rejects_zero_pre_periods(self):
        with pytest.raises(MlsynthDataError):
            unit_weights(np.zeros((0, 2)), np.zeros(0), 0.1)

    def test_unit_weights_rejects_zero_donors(self):
        with pytest.raises(MlsynthDataError):
            unit_weights(np.zeros((3, 0)), np.zeros(3), 0.1)

    def test_unit_weights_rejects_1d_donor_matrix(self):
        with pytest.raises(MlsynthDataError):
            unit_weights(np.zeros(3), np.zeros(3), 0.1)


class TestRegularizationBranches:
    def test_rejects_non_array(self):
        with pytest.raises(MlsynthDataError):
            compute_regularization([[1.0]], 5)

    def test_rejects_1d(self):
        with pytest.raises(MlsynthDataError):
            compute_regularization(np.zeros(5), 5)

    def test_rejects_negative_post_periods(self):
        with pytest.raises(MlsynthConfigError):
            compute_regularization(np.zeros((5, 3)), -1)

    def test_single_pre_period_fallback(self):
        # < 2 pre-periods -> std fallback of 1.0 -> zeta = post^0.25.
        zeta = compute_regularization(np.zeros((1, 3)), 16)
        assert zeta == pytest.approx(16 ** 0.25, abs=1e-9)

    def test_no_donors_fallback(self):
        zeta = compute_regularization(np.zeros((5, 0)), 16)
        assert zeta == pytest.approx(16 ** 0.25, abs=1e-9)

    def test_nan_std_fallback(self):
        # All-NaN diffs -> NaN std -> fallback of 1.0 -> zeta = post^0.25.
        zeta = compute_regularization(np.full((5, 3), np.nan), 16)
        assert zeta == pytest.approx(16 ** 0.25, abs=1e-9)
