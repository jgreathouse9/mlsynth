"""The augsynth path against agents_solvers.md (#960).

Four items, each pinned here:

- ``solve_ridge`` forms ``B B' + lambda I``, which squares the condition number
  of the matching matrix (rule 6). Against an exact rational solve at the
  bottom of the CV grid (condition 1e8) the Gram form erred by 1.2e-6 to
  1.5e-6, the short-panel eigendecomposition of ``B B'`` by 3.5e-6, and the
  thin-SVD form by 7.2e-10 to 8.5e-10.
- The cvxpy fallback of ``simplex_qp`` read no status, retried SCS unchecked,
  and clipped and renormalised without a report (rules 1 and 8).
- ``_scaled_l2`` tested a computed norm against exact zero (rule 4).
- ``BilevelSCM(augment="ridge")`` solved a base simplex QP and threw it away,
  and the caller's warm start never reached the solve that is kept.
"""

import warnings
from fractions import Fraction

import cvxpy as cp
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from mlsynth.exceptions import MlsynthEstimationError
from mlsynth.utils.bilevel import engine as bilevel_engine
from mlsynth.utils.bilevel.engine import BilevelSCM
from mlsynth.utils.geox_helpers.engines import augsynth as augsynth_engine
from mlsynth.utils.solvers import active_set, ridge_augment
from mlsynth.utils.solvers.ridge_augment import (
    SimplexFallbackWarning,
    generate_lambdas,
    simplex_qp,
    solve_ridge,
    solve_ridge_path,
)


# --------------------------------------------------------------------------
# solve_ridge against an exact reference (rule 6, rule 15)
# --------------------------------------------------------------------------

def _exact_correction(A, B, W, lam):
    """``(A - B W)' (B B' + lam I)^{-1} B`` in exact rational arithmetic."""
    m, J = B.shape
    F = [[Fraction(float(B[i, k])) for k in range(J)] for i in range(m)]
    G = [[sum(F[i][k] * F[j][k] for k in range(J)) + (Fraction(lam) if i == j else 0)
          for j in range(m)] for i in range(m)]
    aug = [G[i] + F[i] for i in range(m)]
    for c in range(m):
        p = next(r for r in range(c, m) if aug[r][c] != 0)
        aug[c], aug[p] = aug[p], aug[c]
        piv = aug[c][c]
        aug[c] = [x / piv for x in aug[c]]
        for r in range(m):
            if r != c and aug[r][c] != 0:
                f = aug[r][c]
                aug[r] = [a - f * b for a, b in zip(aug[r], aug[c])]
    X = np.array([[float(x) for x in row[m:]] for row in aug])
    return (A - B @ W) @ X


def _near_rank_deficient(m, J, seed=5):
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(m, 2))
    B = F @ rng.normal(size=(2, J)) + 1e-4 * rng.normal(size=(m, J))
    B -= B.mean(0)
    return F @ rng.normal(size=2), B, np.full(J, 1.0 / J)


@pytest.mark.parametrize("m,J", [(10, 6), (6, 10)])
def test_ridge_correction_is_accurate_at_the_bottom_of_the_grid(m, J):
    """Condition 1e8: the Gram form erred by 1.2e-6 to 1.5e-6 here and the SVD
    form by at most 8.5e-10 (measured on these designs); the bound leaves a
    factor of 12 above the second and lies 120 below the first."""
    A, B, W = _near_rank_deficient(m, J)
    lam = float(generate_lambdas(B)[-1])
    exact = _exact_correction(A, B, W, lam)
    got = solve_ridge(A, B, W, lam)
    assert np.linalg.norm(got - exact) / np.linalg.norm(exact) <= 1e-8


@pytest.mark.parametrize("m,J", [(10, 6), (6, 10)])
def test_the_grid_path_matches_the_single_solve(m, J):
    """One factorization for the grid and one per penalty agree to rounding:
    BLAS sums a one-row product in another order than an L-row one."""
    A, B, W = _near_rank_deficient(m, J)
    lams = generate_lambdas(B)
    path = solve_ridge_path(A, B, W, lams)
    for row, lam in zip(path, lams):
        np.testing.assert_allclose(row, solve_ridge(A, B, W, float(lam)), rtol=1e-13, atol=1e-16)


@settings(max_examples=30, deadline=None)
@given(seed=st.integers(0, 10_000), k=st.integers(-30, 30),
       m=st.integers(3, 15), J=st.integers(2, 15))
def test_ridge_correction_scales_with_the_data(seed, k, m, J):
    """The correction is unitless: (s A, s B) with s^2 lambda gives the same one."""
    rng = np.random.default_rng(seed)
    B, A = rng.normal(size=(m, J)), rng.normal(size=m)
    W = np.full(J, 1.0 / J)
    lam = float(rng.uniform(0.01, 10))
    s = 2.0 ** k
    np.testing.assert_allclose(solve_ridge(s * A, s * B, W, s * s * lam),
                               solve_ridge(A, B, W, lam), rtol=1e-12, atol=1e-14)


def test_ridge_correction_of_an_exact_fit_is_zero():
    """Edge: the base weights already reproduce the target."""
    rng = np.random.default_rng(1)
    B = rng.normal(size=(8, 5))
    W = np.full(5, 0.2)
    np.testing.assert_allclose(solve_ridge(B @ W, B, W, 0.5), 0.0, atol=1e-14)


# --------------------------------------------------------------------------
# The cvxpy fallback (rules 1 and 8)
# --------------------------------------------------------------------------

def _problem(seed=2):
    rng = np.random.default_rng(seed)
    B = rng.normal(size=(30, 12))
    return B - B.mean(0), rng.normal(size=30)


def _refuse(monkeypatch):
    """Make the active set report that it did not converge."""
    real = active_set.solve_simplex_qp

    def refused(B, A, **kw):
        w, info = real(B, A, **kw)
        return w, dict(info, converged=False, certified=False)
    monkeypatch.setattr(active_set, "solve_simplex_qp", refused)


def test_clean_solve_takes_no_fallback_and_is_silent():
    B, A = _problem()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        w = simplex_qp(B, A)
    assert active_set.simplex_kkt_error(B, A, w) <= active_set.CERTIFY_TOL


def test_fallback_is_reported_and_certified(monkeypatch):
    B, A = _problem()
    _refuse(monkeypatch)
    with pytest.warns(SimplexFallbackWarning, match="cvxpy"):
        w = simplex_qp(B, A)
    assert w.min() >= 0 and w.sum() == pytest.approx(1.0)
    assert active_set.simplex_kkt_error(B, A, w) <= active_set.CERTIFY_TOL


def test_fallback_without_an_optimum_raises(monkeypatch):
    """No solver returns an optimum: a typed failure, not a point."""
    B, A = _problem()
    _refuse(monkeypatch)
    monkeypatch.setattr(cp.Problem, "solve", lambda self, *a, **k: None)
    with pytest.warns(SimplexFallbackWarning):
        with pytest.raises(MlsynthEstimationError, match="status"):
            simplex_qp(B, A)


def test_fallback_repair_beyond_rounding_is_reported(monkeypatch):
    """A clip larger than the solver's own tolerance is a repair, and named."""
    B, A = _problem()
    _refuse(monkeypatch)

    def fake_solve(self, *a, **k):
        var = self.variables()[0]
        value = np.full(var.shape, 1.0 / var.shape[0])
        value[0], value[1] = -1e-3, value[1] + 1e-3
        var.save_value(value)                 # bypasses the nonneg projection
        self._status = cp.OPTIMAL
    monkeypatch.setattr(cp.Problem, "solve", fake_solve)
    monkeypatch.setattr(cp.Problem, "status", property(lambda self: getattr(self, "_status", None)))
    with pytest.warns(SimplexFallbackWarning) as record:
        w = simplex_qp(B, A)
    assert any("clipped" in str(r.message) for r in record)
    assert w.min() >= 0 and w.sum() == pytest.approx(1.0)


# --------------------------------------------------------------------------
# _scaled_l2's zero test (rule 4)
# --------------------------------------------------------------------------

def test_scaled_l2_is_undefined_when_the_uniform_fit_is_exact_to_rounding():
    """The target is the donors' mean, computed by another route: the uniform
    imbalance is rounding noise, and the ratio is undefined, not huge."""
    rng = np.random.default_rng(3)
    B = rng.normal(size=(40, 7)) * 1e3
    A = B.mean(axis=1)
    w = rng.dirichlet(np.ones(7))
    assert np.isnan(augsynth_engine._scaled_l2(A, B, w))


@settings(max_examples=30, deadline=None)
@given(seed=st.integers(0, 10_000), k=st.integers(-30, 30))
def test_scaled_l2_is_unitless(seed, k):
    rng = np.random.default_rng(seed)
    B, A = rng.normal(size=(20, 6)), rng.normal(size=20)
    w = rng.dirichlet(np.ones(6))
    s = 2.0 ** k
    assert augsynth_engine._scaled_l2(s * A, s * B, w) == pytest.approx(
        augsynth_engine._scaled_l2(A, B, w), rel=1e-12)


def test_scaled_l2_perfect_fit_is_zero():
    rng = np.random.default_rng(4)
    B = rng.normal(size=(20, 6))
    w = rng.dirichlet(np.ones(6))
    assert augsynth_engine._scaled_l2(B @ w, B, w) == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------
# BilevelSCM(augment="ridge"): no discarded solve, and the warm start is used
# --------------------------------------------------------------------------

def test_ridge_fit_does_not_solve_a_base_it_discards(monkeypatch):
    calls = []
    real = bilevel_engine.simplex_qp
    monkeypatch.setattr(bilevel_engine, "simplex_qp",
                        lambda *a, **k: calls.append(1) or real(*a, **k))
    rng = np.random.default_rng(6)
    Y0 = rng.normal(size=(25, 8)).cumsum(axis=0)
    y = Y0[:, :3].mean(axis=1) + 0.1 * rng.normal(size=25)
    res = BilevelSCM(augment="ridge").fit(y, Y0)
    assert calls == []
    direct = ridge_augment.ridge_augment_weights(y, Y0)
    np.testing.assert_array_equal(res.W, direct.W)


def test_ridge_fit_forwards_the_warm_start(monkeypatch):
    seen = {}
    real = ridge_augment.ridge_augment_weights

    def spy(*a, **k):
        seen.update(k)
        return real(*a, **k)
    monkeypatch.setattr(ridge_augment, "ridge_augment_weights", spy)
    rng = np.random.default_rng(7)
    Y0 = rng.normal(size=(25, 8)).cumsum(axis=0)
    y = Y0[:, :3].mean(axis=1)
    start = np.full(8, 1.0 / 8)
    BilevelSCM(augment="ridge").fit(y, Y0, warm_start=start)
    np.testing.assert_array_equal(seen["warm_start"], start)
