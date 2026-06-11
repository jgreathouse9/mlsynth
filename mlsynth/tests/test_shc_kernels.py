"""Tests for mlsynth.utils.shc_helpers.kernels.

Covers the local-linear smoother, LOOCV bandwidth selection, the SHC/ASHC
quadratic programs (both branches, the low-variance penalty branch, and the
ASHC argument-validation raise), and the ASHC lambda tuner (default and
explicit grids).

Assertions are structural (shapes, simplex/sum-to-one constraints,
determinism, monotonicity of trivial fits) rather than magic numbers.
"""

import numpy as np
import pytest

from mlsynth.utils.shc_helpers.kernels import (
    loocv_bandwidth,
    smooth,
    solve_shc_qp,
    tune_lambda_ashc,
    _solve_SHC_QP,
)


# --------------------------------------------------------------------------
# smooth
# --------------------------------------------------------------------------
def test_smooth_shape_and_dtype():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(20)
    out = smooth(y, bw=3.0)
    assert out.shape == (20,)
    assert out.dtype == np.float64
    assert np.all(np.isfinite(out))


def test_smooth_recovers_linear_trend():
    # A local-LINEAR smoother should reproduce an exactly linear series
    # (up to numerical error) regardless of bandwidth, since the local
    # design includes an intercept and slope.
    T = 15
    y = 2.0 + 0.5 * np.arange(T)
    out = smooth(y, bw=2.0)
    assert np.allclose(out, y, atol=1e-6)


def test_smooth_determinism():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(12)
    a = smooth(y, bw=2.5)
    b = smooth(y, bw=2.5)
    assert np.array_equal(a, b)


# --------------------------------------------------------------------------
# loocv_bandwidth
# --------------------------------------------------------------------------
def test_loocv_bandwidth_basic():
    rng = np.random.default_rng(0)
    y = np.sin(np.linspace(0, 3, 25)) + 0.05 * rng.standard_normal(25)
    grid = np.array([1.0, 2.0, 4.0, 8.0])
    best_h, cv_errors = loocv_bandwidth(y, grid)
    # best_h is one of the grid points.
    assert best_h in grid
    # one CV error per grid point, all finite and non-negative.
    assert len(cv_errors) == len(grid)
    assert np.all(np.isfinite(cv_errors))
    assert np.all(np.asarray(cv_errors) >= 0)
    # best_h corresponds to the minimum CV error.
    assert best_h == grid[int(np.argmin(cv_errors))]


def test_loocv_bandwidth_determinism():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(15)
    grid = [1.0, 3.0]
    b1, e1 = loocv_bandwidth(y, grid)
    b2, e2 = loocv_bandwidth(y, grid)
    assert b1 == b2
    assert np.allclose(e1, e2)


# --------------------------------------------------------------------------
# solve_shc_qp -- SHC (non-augmented, simplex)
# --------------------------------------------------------------------------
def test_solve_shc_qp_simplex_constraints():
    rng = np.random.default_rng(0)
    L = rng.standard_normal((8, 4))
    ell = rng.standard_normal(8)
    w, obj = solve_shc_qp(L, ell)
    assert w is not None
    assert w.shape == (4,)
    # Simplex: weights sum to 1 and are (numerically) non-negative.
    assert np.isclose(w.sum(), 1.0, atol=1e-6)
    assert np.all(w >= -1e-7)
    assert np.isfinite(obj)


def test_solve_shc_qp_low_variance_penalty_branch():
    # Rank-deficient donor matrix -> G has eigenvalue(s) below tol ->
    # the C.size > 0 low-variance penalty branch is taken.
    rng = np.random.default_rng(0)
    base = rng.standard_normal((6, 2))
    L = np.hstack([base, base[:, [0]]])  # column 2 duplicates column 0
    ell = rng.standard_normal(6)
    w, obj = solve_shc_qp(L, ell)
    assert w is not None
    assert np.isclose(w.sum(), 1.0, atol=1e-6)


def test_solve_shc_qp_full_rank_no_penalty_branch():
    # Full column rank -> no eigenvalues below tol -> penalty == 0 branch.
    rng = np.random.default_rng(1)
    L = rng.standard_normal((10, 3))
    ell = rng.standard_normal(10)
    w, obj = solve_shc_qp(L, ell)
    assert w is not None
    assert np.isclose(w.sum(), 1.0, atol=1e-6)


def test_solve_shc_qp_alias_is_same_function():
    assert _solve_SHC_QP is solve_shc_qp


def test_solve_shc_qp_osqp_matches_cvxpy():
    """The OSQP fast path reproduces the cvxpy reference to high precision."""
    from scipy.linalg import eigh

    from mlsynth.utils.shc_helpers.kernels import _solve_shc_qp_cvxpy

    rng = np.random.default_rng(3)
    for _ in range(8):
        m, N = int(rng.integers(8, 24)), int(rng.integers(3, 12))
        L = rng.standard_normal((m, N))
        ell = rng.standard_normal(m)
        ev, evec = eigh(L.T @ L)
        C = evec[:, ev < 1e-8]
        # SHC (simplex)
        w_fast, _ = solve_shc_qp(L, ell)
        w_ref, _ = _solve_shc_qp_cvxpy(L, ell, C, False, None, None, 1e-6)
        assert np.max(np.abs(w_fast - w_ref)) < 1e-3
        # ASHC (augmented)
        w_shc = np.full(N, 1.0 / N)
        wa, _ = solve_shc_qp(L, ell, use_augmented=True, w_shc=w_shc, lam=0.5)
        wb, _ = _solve_shc_qp_cvxpy(L, ell, C, True, w_shc, 0.5, 1e-6)
        assert np.max(np.abs(wa - wb)) < 1e-3


def test_solve_shc_qp_falls_back_when_osqp_unavailable(monkeypatch):
    """If the OSQP path returns None, the cvxpy fallback still solves."""
    from mlsynth.utils.shc_helpers import kernels

    monkeypatch.setattr(kernels, "_solve_shc_qp_osqp", lambda *a, **k: None)
    rng = np.random.default_rng(0)
    L = rng.standard_normal((8, 4))
    ell = rng.standard_normal(8)
    w, obj = solve_shc_qp(L, ell)
    assert w is not None and np.isclose(w.sum(), 1.0, atol=1e-6)
    assert np.isfinite(obj)


# --------------------------------------------------------------------------
# solve_shc_qp -- ASHC (augmented, ridge toward w_shc)
# --------------------------------------------------------------------------
def test_solve_shc_qp_augmented_runs():
    rng = np.random.default_rng(0)
    L = rng.standard_normal((8, 4))
    ell = rng.standard_normal(8)
    w_shc = np.full(4, 0.25)
    w, obj = solve_shc_qp(L, ell, use_augmented=True, w_shc=w_shc, lam=1.0)
    assert w is not None
    assert w.shape == (4,)
    # ASHC keeps the sum-to-one constraint but drops non-negativity.
    assert np.isclose(w.sum(), 1.0, atol=1e-6)
    assert np.isfinite(obj)


def test_solve_shc_qp_augmented_large_lambda_pulls_toward_w_shc():
    # As lambda -> infinity the deviation penalty vanishes; as lambda -> 0
    # the solution is pulled hard toward w_shc. Use a small lambda to make
    # the ASHC solution close to w_shc.
    rng = np.random.default_rng(2)
    L = rng.standard_normal((8, 4))
    ell = rng.standard_normal(8)
    w_shc = np.array([0.1, 0.2, 0.3, 0.4])
    w_small, _ = solve_shc_qp(L, ell, use_augmented=True, w_shc=w_shc, lam=1e-4)
    w_large, _ = solve_shc_qp(L, ell, use_augmented=True, w_shc=w_shc, lam=1e4)
    # Tiny lambda -> heavy ridge -> closer to w_shc than the weak-ridge fit.
    assert np.linalg.norm(w_small - w_shc) <= np.linalg.norm(w_large - w_shc) + 1e-6


@pytest.mark.parametrize(
    "kwargs",
    [
        {"use_augmented": True, "w_shc": None, "lam": 1.0},
        {"use_augmented": True, "w_shc": np.full(3, 1 / 3), "lam": None},
        {"use_augmented": True, "w_shc": None, "lam": None},
    ],
)
def test_solve_shc_qp_augmented_missing_args_raises(kwargs):
    rng = np.random.default_rng(0)
    L = rng.standard_normal((6, 3))
    ell = rng.standard_normal(6)
    with pytest.raises(ValueError, match="lam and w_shc"):
        solve_shc_qp(L, ell, **kwargs)


# --------------------------------------------------------------------------
# tune_lambda_ashc
# --------------------------------------------------------------------------
def test_tune_lambda_ashc_explicit_grid():
    rng = np.random.default_rng(0)
    L = rng.standard_normal((12, 4))
    ell = rng.standard_normal(12)
    w_shc = np.full(4, 0.25)
    grid = np.array([1e-3, 1e-1, 1.0, 10.0])
    best_lam, errs = tune_lambda_ashc(L, ell, w_shc, lambda_grid=grid)
    assert best_lam in grid
    assert isinstance(errs, dict)
    assert len(errs) == len(grid)
    # best_lam minimises the holdout MSE.
    assert errs[best_lam] == min(errs.values())
    assert all(np.isfinite(v) for v in errs.values())


def test_tune_lambda_ashc_default_grid():
    rng = np.random.default_rng(1)
    L = rng.standard_normal((12, 3))
    ell = rng.standard_normal(12)
    w_shc = np.full(3, 1 / 3)
    best_lam, errs = tune_lambda_ashc(L, ell, w_shc)  # lambda_grid=None branch
    # Default grid is np.logspace(-6, 2, 50).
    assert len(errs) == 50
    assert best_lam in errs
    assert errs[best_lam] == min(errs.values())


def test_tune_lambda_ashc_split_ratio():
    rng = np.random.default_rng(3)
    L = rng.standard_normal((10, 3))
    ell = rng.standard_normal(10)
    w_shc = np.full(3, 1 / 3)
    best_lam, errs = tune_lambda_ashc(
        L, ell, w_shc, lambda_grid=np.array([0.1, 1.0]), split_ratio=0.7
    )
    assert best_lam in (0.1, 1.0)
