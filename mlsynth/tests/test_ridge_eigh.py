"""TDD for the eigendecomposition-based ridge path used in GEOLIFT's ridge CV.

Background
----------
``ridge_augment.solve_ridge`` computes the additive ridge correction

    W_ridge(lambda) = M @ inv(B B^T + lambda I) @ B,   M = A - B W,

inverting an ``m x m`` matrix for *every* lambda. In ``cross_validate`` this is
called once per (fold, lambda), so the inversion dominates the GEOLIFT design
(profiling: ``numpy.linalg.inv`` is the top self-time line, one call per
``solve_ridge``).

Because ``B B^T``, ``M``, and the eigenbasis are fixed across the lambda grid
within a fold, the whole grid can be evaluated from a single eigendecomposition
``B B^T = V diag(d) V^T``:

    inv(B B^T + lambda I) = V diag(1/(d+lambda)) V^T
    W_ridge(lambda)       = (p / (d + lambda)) @ Q,   p = M V,  Q = V^T B.

This is the *same quantity*, algebraically refactored, so it must match the
inversion path to numerical tolerance. These tests pin that parity (the
DSCAR rule: only swap numerics where the result is provably identical) for the
new ``solve_ridge_path`` and for the refactored ``cross_validate``.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlsynth.utils.bilevel.ridge_augment import (
    _HoldoutSplitter,
    cross_validate,
    solve_ridge,
    solve_ridge_path,
)


def _rng(seed=0):
    return np.random.default_rng(seed)


def _simplex_base(B, A):
    """A cheap, deterministic base-weight solver for CV parity tests (the
    parity holds for ANY base_weights_fn since both paths share it)."""
    w = np.linalg.lstsq(B, A, rcond=None)[0]
    w = np.clip(w, 0.0, None)
    s = w.sum()
    return w / s if s > 1e-12 else w


def _naive_cross_validate(base_weights_fn, X0, X1, lambdas, holdout_len=1):
    """Reference implementation: the original per-lambda inv() loop."""
    X0 = np.asarray(X0, float); X1 = np.asarray(X1, float).ravel()
    lambdas = np.asarray(lambdas, float)
    res = []
    for B_t, B_v, A_t, A_v in _HoldoutSplitter(X0, X1, holdout_len=holdout_len):
        W = base_weights_fn(B_t, A_t)
        fold = [float(np.sum((A_v - B_v @ (W + solve_ridge(A_t, B_t, W, lam))) ** 2))
                for lam in lambdas]
        res.append(fold)
    arr = np.asarray(res, float)
    return (lambdas, arr.mean(axis=0), arr.std(axis=0) / np.sqrt(arr.shape[0]))


# ---------------------------------------------------------------------------
# solve_ridge_path parity with the per-lambda inv solve_ridge
# ---------------------------------------------------------------------------

class TestRidgePathParity:
    @pytest.mark.parametrize("m,J", [(20, 5), (8, 12), (30, 30), (5, 1)])
    def test_path_matches_solve_ridge(self, m, J):
        rng = _rng(m * 100 + J)
        B = rng.standard_normal((m, J))
        A = rng.standard_normal(m)
        W = np.clip(rng.standard_normal(J), 0, None)
        lambdas = np.array([1e-6, 1e-3, 0.1, 1.0, 10.0, 100.0])
        path = solve_ridge_path(A, B, W, lambdas)            # (n_lambda, J)
        assert path.shape == (len(lambdas), J)
        for i, lam in enumerate(lambdas):
            ref = solve_ridge(A=A, B=B, W=W, lambda_=lam)
            np.testing.assert_allclose(path[i], ref, rtol=1e-8, atol=1e-9)

    def test_single_lambda(self):
        rng = _rng(7)
        B = rng.standard_normal((15, 6)); A = rng.standard_normal(15)
        W = np.clip(rng.standard_normal(6), 0, None)
        path = solve_ridge_path(A, B, W, [2.5])
        np.testing.assert_allclose(path[0], solve_ridge(A=A, B=B, W=W, lambda_=2.5),
                                   rtol=1e-8, atol=1e-9)

    def test_rank_deficient_b(self):
        # B with linearly dependent columns -> B B^T rank-deficient; +lambda I
        # keeps the solve well-posed and the eigh path must still match.
        rng = _rng(3)
        base = rng.standard_normal((18, 3))
        B = np.hstack([base, base[:, :1] * 2.0])             # rank 3, 4 columns
        A = rng.standard_normal(18); W = np.clip(rng.standard_normal(4), 0, None)
        for lam in (1e-4, 1.0, 50.0):
            np.testing.assert_allclose(
                solve_ridge_path(A, B, W, [lam])[0],
                solve_ridge(A=A, B=B, W=W, lambda_=lam), rtol=1e-7, atol=1e-8)


# ---------------------------------------------------------------------------
# cross_validate parity with the naive inv reference
# ---------------------------------------------------------------------------

class TestCrossValidateParity:
    @pytest.mark.parametrize("m,J,holdout", [(24, 6, 1), (24, 6, 3), (12, 20, 1)])
    def test_matches_naive(self, m, J, holdout):
        rng = _rng(m + J + holdout)
        X0 = rng.standard_normal((m, J))
        X1 = rng.standard_normal(m)
        lambdas = np.array([1e-5, 1e-2, 0.5, 5.0, 50.0])
        lo, eo, so = _naive_cross_validate(_simplex_base, X0, X1, lambdas, holdout)
        ln, en, sn = cross_validate(_simplex_base, X0, X1, lambdas, holdout)
        np.testing.assert_allclose(ln, lo)
        np.testing.assert_allclose(en, eo, rtol=1e-8, atol=1e-9)
        np.testing.assert_allclose(sn, so, rtol=1e-8, atol=1e-9)

    def test_selected_lambda_unchanged(self):
        # The argmin (and hence the selected penalty) must be identical.
        rng = _rng(99)
        X0 = rng.standard_normal((40, 8)); X1 = rng.standard_normal(40)
        lambdas = np.geomspace(1e-6, 1e3, 21)
        _, eo, _ = _naive_cross_validate(_simplex_base, X0, X1, lambdas)
        _, en, _ = cross_validate(_simplex_base, X0, X1, lambdas)
        assert int(np.argmin(en)) == int(np.argmin(eo))
