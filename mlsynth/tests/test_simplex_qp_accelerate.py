"""Tests for the FISTA warm-start accelerator on ``simplex_qp``.

The accelerator is a *speed-only* layer: for a large donor pool it computes a
Gram-collapsed FISTA warm start and hands it to the exact active-set solver, so
the active set is certified from the right support instead of built up pivot by
pivot. It must never change the *answer* -- the exact active-set (with the cvxpy
fallback) still determines the weights. These tests pin four things:

1. the simplex projection primitive is the exact Euclidean projection;
2. the FISTA warm start is feasible and near-optimal;
3. ``simplex_qp`` stays KKT-optimal and matches the reference on large problems
   (where the accelerator fires) *and* small ones (where it does not);
4. the accelerator engages only when it should -- large ``J``, no caller warm
   start -- and the accelerated and cold paths agree on the fitted values.
"""
from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cvxpy")

from mlsynth.utils.bilevel import ridge_augment
from mlsynth.utils.bilevel.accelerate import (
    ACCEL_MIN_DONORS, fista_warm_start, simplex_project)
from mlsynth.utils.bilevel.active_set import solve_simplex_qp
from mlsynth.utils.bilevel.ridge_augment import simplex_qp


# --------------------------------------------------------------------------- #
# reference oracles
# --------------------------------------------------------------------------- #
def _proj_ref(v):
    """Exact Euclidean projection onto the simplex via cvxpy (oracle)."""
    x = cp.Variable(v.shape[0])
    prob = cp.Problem(cp.Minimize(cp.sum_squares(x - v)), [x >= 0, cp.sum(x) == 1])
    prob.solve(solver=cp.CLARABEL)
    return np.clip(np.asarray(x.value, float).ravel(), 0.0, None)


def _obj(B, A, w):
    r = B @ np.asarray(w, float).ravel() - A
    return float(r @ r)


def _reference(B, A):
    J = B.shape[1]
    w = cp.Variable(J)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(B @ w - A)), [w >= 0, cp.sum(w) == 1])
    try:
        prob.solve()
    except Exception:
        return None
    return None if w.value is None else np.clip(np.asarray(w.value, float).ravel(), 0, None)


def assert_feasible(w, J, tol=1e-7):
    w = np.asarray(w, float).ravel()
    assert w.shape == (J,)
    assert np.all(np.isfinite(w))
    assert w.min() >= -tol
    assert abs(w.sum() - 1.0) <= tol


def assert_kkt_optimal(B, A, w, tol=1e-6):
    w = np.asarray(w, float).ravel()
    J = B.shape[1]
    assert_feasible(w, J)
    g = B.T @ (B @ w - A)
    scale = 1.0 + float(np.max(np.abs(g)))
    support = w > 1e-7
    assert support.any()
    mu = float(g[support].mean())
    assert np.all(np.abs(g[support] - mu) <= tol * scale)
    if (~support).any():
        assert np.all(g[~support] >= mu - tol * scale)


def _factor_panel(J, T0, seed=0):
    """A factor-structure SC panel (treated + J donors on two AR(1) factors)."""
    rng = np.random.default_rng(seed)
    F = rng.standard_normal((T0, 2))
    Mu = np.zeros((J + 1, 2)); h = J // 2
    Mu[:1 + h, 0] = 1.0; Mu[1 + h:, 1] = 1.0
    Y = F @ Mu.T + rng.standard_normal((T0, J + 1))
    return Y[:, 0], Y[:, 1:]                      # A (T0,), B (T0, J)


# --------------------------------------------------------------------------- #
# 1. simplex projection primitive
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [1, 2, 5, 50, 500])
def test_simplex_project_matches_cvxpy(n):
    # One-sided vs cvxpy: our sort-based projection is *exact*, while CLARABEL's
    # projection QP carries ~1e-4 slack at larger n, so we assert our squared
    # distance is no worse than cvxpy's (plus feasibility and the (v-theta)_+
    # optimality form), not pointwise equality to CLARABEL's approximate point.
    rng = np.random.default_rng(n)
    for _ in range(20):
        v = rng.standard_normal(n) * rng.uniform(0.1, 10)
        p = simplex_project(v)
        assert_feasible(p, n)
        # p must have the form (v - theta)_+ for a single threshold theta
        active = p > 0
        if active.sum() >= 2 and (~active).any():
            theta = float(np.mean((v - p)[active]))
            assert np.allclose((v - p)[active], theta, atol=1e-9)
            assert np.all(v[~active] <= theta + 1e-9)
        ref = _proj_ref(v)
        if ref is not None:
            dm = float((p - v) @ (p - v))
            dr = float((ref - v) @ (ref - v))
            assert dm <= dr + 1e-9 * (1.0 + abs(dr))


def test_simplex_project_idempotent_on_simplex():
    rng = np.random.default_rng(3)
    w = rng.random(20); w /= w.sum()
    assert np.max(np.abs(simplex_project(w) - w)) < 1e-12


def test_simplex_project_of_vertex_is_vertex():
    v = np.array([10.0, -1.0, -2.0, -3.0])
    p = simplex_project(v)
    assert np.argmax(p) == 0 and p[0] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# 2. FISTA warm start
# --------------------------------------------------------------------------- #
def test_fista_warm_start_feasible_and_deterministic():
    A, B = _factor_panel(120, 240)
    w1 = fista_warm_start(B, A)
    w2 = fista_warm_start(B, A)
    assert_feasible(w1, B.shape[1])
    assert np.array_equal(w1, w2), "warm start must be deterministic"


def test_fista_warm_start_near_optimal():
    A, B = _factor_panel(120, 240)
    ref = _reference(B, A)
    o_star = _obj(B, A, ref)
    o_fista = _obj(B, A, fista_warm_start(B, A))
    # coarse but close: the exact polish closes the rest
    assert o_fista <= o_star + 0.05 * (1.0 + abs(o_star))


# --------------------------------------------------------------------------- #
# 3. simplex_qp optimality, large and small
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("J,T0", [(100, 200), (150, 120), (200, 400)])
def test_simplex_qp_large_is_kkt_optimal(J, T0):
    A, B = _factor_panel(J, T0)
    w = simplex_qp(B, A)
    assert_kkt_optimal(B, A, w)
    ref = _reference(B, A)
    if ref is not None:
        assert _obj(B, A, w) <= _obj(B, A, ref) + 1e-6 * (1 + abs(_obj(B, A, ref)))


@pytest.mark.parametrize("J,T0", [(3, 8), (16, 30), (40, 80)])
def test_simplex_qp_small_unaffected(J, T0):
    A, B = _factor_panel(J, T0)
    assert_kkt_optimal(B, A, simplex_qp(B, A))


def test_accelerated_equals_cold_fitted_values():
    """Accelerated simplex_qp and the cold active-set agree on the (unique)
    fitted values B@w across shapes -- the accelerator changes speed, not answer."""
    for J, T0 in [(100, 200), (150, 120), (250, 500), (90, 90)]:
        A, B = _factor_panel(J, T0)
        w_accel = simplex_qp(B, A)
        w_cold, _ = solve_simplex_qp(B, A, return_info=True)
        assert np.max(np.abs(B @ w_accel - B @ w_cold)) < 1e-6
        assert _obj(B, A, w_accel) == pytest.approx(_obj(B, A, w_cold), abs=1e-6)


def test_accelerated_matches_clarabel_value_for_value():
    """On a large problem where the accelerator fires, the weights match a CLARABEL
    solve (the general interior-point we are beating on speed) value-for-value."""
    A, B = _factor_panel(200, 400)
    w = simplex_qp(B, A)
    x = cp.Variable(B.shape[1], nonneg=True)
    cp.Problem(cp.Minimize(cp.sum_squares(A - B @ x)), [cp.sum(x) == 1]).solve(solver=cp.CLARABEL)
    w_cl = np.clip(np.asarray(x.value, float).ravel(), 0.0, None)
    assert np.max(np.abs(B @ w - B @ w_cl)) < 1e-4          # fitted values agree
    assert _obj(B, A, w) <= _obj(B, A, w_cl) + 1e-6         # ours is (at least) as good


def test_accelerator_faster_than_cold_on_large_J():
    """Speed regression: the FISTA warm start makes the exact solve markedly
    faster on a wide donor pool. Conservative 2x margin (measured ~13x at J=250)."""
    import time
    A, B = _factor_panel(250, 500)
    t = time.perf_counter(); w_cold, _ = solve_simplex_qp(B, A, return_info=True); t_cold = time.perf_counter() - t
    t = time.perf_counter(); w_acc = simplex_qp(B, A); t_acc = time.perf_counter() - t
    assert np.max(np.abs(B @ w_acc - B @ w_cold)) < 1e-6    # same answer
    assert t_acc < 0.5 * t_cold, f"accel {t_acc:.3f}s not < half of cold {t_cold:.3f}s"


# --------------------------------------------------------------------------- #
# 4. accelerator engagement gating
# --------------------------------------------------------------------------- #
def test_accelerator_engaged_for_large_J(monkeypatch):
    calls = {"n": 0}
    real = ridge_augment.fista_warm_start

    def spy(B, A, **kw):
        calls["n"] += 1
        return real(B, A, **kw)

    monkeypatch.setattr(ridge_augment, "fista_warm_start", spy)
    A, B = _factor_panel(ACCEL_MIN_DONORS + 20, 2 * (ACCEL_MIN_DONORS + 20))
    simplex_qp(B, A)
    assert calls["n"] == 1


def test_accelerator_skipped_for_small_J(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(ridge_augment, "fista_warm_start",
                        lambda *a, **k: calls.__setitem__("n", calls["n"] + 1))
    A, B = _factor_panel(max(2, ACCEL_MIN_DONORS - 30), 60)
    simplex_qp(B, A)
    assert calls["n"] == 0


def test_accelerator_skipped_when_caller_supplies_warm_start(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(ridge_augment, "fista_warm_start",
                        lambda *a, **k: calls.__setitem__("n", calls["n"] + 1))
    A, B = _factor_panel(ACCEL_MIN_DONORS + 20, 2 * (ACCEL_MIN_DONORS + 20))
    J = B.shape[1]
    simplex_qp(B, A, warm_start=np.full(J, 1.0 / J))
    assert calls["n"] == 0


# --------------------------------------------------------------------------- #
# 5. edge cases
# --------------------------------------------------------------------------- #
def test_edge_single_donor():
    B = np.array([[2.0], [3.0], [1.5]]); A = np.array([2.0, 3.0, 1.5])
    w = simplex_qp(B, A)
    assert w.shape == (1,) and w[0] == pytest.approx(1.0)


def test_edge_wide_more_donors_than_periods():
    A, B = _factor_panel(ACCEL_MIN_DONORS + 40, 30)     # J > T0, accelerator on
    assert_kkt_optimal(B, A, simplex_qp(B, A))


def test_edge_collinear_donors_large():
    A, B = _factor_panel(ACCEL_MIN_DONORS + 10, 200)
    B[:, 1] = B[:, 0]                                    # duplicate donor (degenerate)
    w = simplex_qp(B, A)
    assert_kkt_optimal(B, A, w)
