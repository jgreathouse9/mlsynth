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
    simplex_qp,
    solve_ridge,
    solve_ridge_path,
)
from mlsynth.utils.bilevel.ridge_inference import _reference_stats, _stat


def _naive_reference_stats(resids, post_slice, q, ns, seed):
    """Frozen copy of the original per-permutation conformal reference loop."""
    rng = np.random.default_rng(seed)
    return np.fromiter(
        (_stat(rng.permutation(resids)[post_slice], q) for _ in range(ns)),
        dtype=float, count=ns,
    )


def _inv_solve_ridge(A, B, W, lam):
    """Reference: the original inv()-based single-lambda ridge correction."""
    A = np.asarray(A, float).ravel(); B = np.asarray(B, float)
    W = np.asarray(W, float).ravel()
    M = A - B @ W
    N = np.linalg.inv(B @ B.T + lam * np.identity(B.shape[0]))
    return M @ N @ B


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


# ---------------------------------------------------------------------------
# solve_ridge: np.linalg.solve replaces np.linalg.inv (single-lambda callers)
# ---------------------------------------------------------------------------

class TestSolveRidgeUsesSolve:
    @pytest.mark.parametrize("m,J", [(20, 5), (8, 12), (15, 15), (30, 4)])
    def test_matches_inv_reference(self, m, J):
        rng = _rng(m * 7 + J)
        B = rng.standard_normal((m, J)); A = rng.standard_normal(m)
        W = np.clip(rng.standard_normal(J), 0, None)
        for lam in (1e-5, 0.3, 12.0, 200.0):
            np.testing.assert_allclose(
                solve_ridge(A=A, B=B, W=W, lambda_=lam),
                _inv_solve_ridge(A, B, W, lam), rtol=1e-8, atol=1e-9)


# ---------------------------------------------------------------------------
# cross_validate: warm-starting the per-fold base QP must not change results
# (the base simplex objective is strictly convex under full column rank, so the
# optimum -- and thus the CV curve -- is identical with or without a warm start)
# ---------------------------------------------------------------------------

class TestWarmStartBase:
    def test_warm_vs_cold_parity(self):
        rng = _rng(5)
        X0 = rng.standard_normal((40, 8)); X1 = rng.standard_normal(40)
        lambdas = np.geomspace(1e-6, 1e3, 21)
        _, ew, sw = cross_validate(simplex_qp, X0, X1, lambdas, warm_start_base=True)
        _, ec, sc = cross_validate(simplex_qp, X0, X1, lambdas, warm_start_base=False)
        np.testing.assert_allclose(ew, ec, rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(sw, sc, rtol=1e-7, atol=1e-9)

    def test_warm_vs_cold_collinear(self):
        # Adversarial: collinear donors -> the base objective is only weakly
        # convex; warm-start must still land the same CV curve.
        rng = _rng(8)
        base = rng.standard_normal((30, 4))
        X0 = np.hstack([base, base[:, :2]]); X1 = rng.standard_normal(30)
        lambdas = np.geomspace(1e-5, 1e2, 15)
        _, ew, _ = cross_validate(simplex_qp, X0, X1, lambdas, warm_start_base=True)
        _, ec, _ = cross_validate(simplex_qp, X0, X1, lambdas, warm_start_base=False)
        np.testing.assert_allclose(ew, ec, rtol=1e-6, atol=1e-8)

    def test_cold_matches_naive_with_simplex(self):
        rng = _rng(2)
        X0 = rng.standard_normal((24, 6)); X1 = rng.standard_normal(24)
        lambdas = np.geomspace(1e-5, 1e2, 12)
        _, ec, sc = cross_validate(simplex_qp, X0, X1, lambdas, warm_start_base=False)
        _, en, sn = _naive_cross_validate(simplex_qp, X0, X1, lambdas)
        np.testing.assert_allclose(ec, en, rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(sc, sn, rtol=1e-7, atol=1e-9)


# ---------------------------------------------------------------------------
# Conformal permutation reference: vectorized statistic must be BIT-IDENTICAL
# to the per-permutation loop (the exact hybrid -- same permutations, same
# draws, only the statistic is broadcast). No re-pinning of the conformal /
# augsynth replications.
# ---------------------------------------------------------------------------

class TestReferenceStatsVectorized:
    @pytest.mark.parametrize("T,post,ns,seed", [
        (45, slice(39, 45), 1000, 0),
        (90, slice(80, 90), 500, 3),
        (30, slice(20, 30), 200, 7),
    ])
    def test_iid_bit_identical_q1(self, T, post, ns, seed):
        # q = 1 is the augsynth default (the only value used in practice): the
        # vectorized statistic is BIT-identical to the per-permutation loop.
        rng = _rng(T + ns + seed)
        resids = rng.standard_normal(T)
        ref = _naive_reference_stats(resids, post, 1.0, ns, seed)
        got = _reference_stats(resids, post, 1.0, conformal_type="iid", ns=ns, seed=seed)
        np.testing.assert_array_equal(got, ref)          # exact, not just close

    def test_iid_general_q_matches_to_tol(self):
        # For q != 1 the 2D axis-reduction sums in a different order than the 1D
        # per-row sum, so the match is to floating-point tolerance (~1e-15), not
        # bitwise -- the permutations and draws are still identical.
        rng = _rng(7)
        resids = rng.standard_normal(30)
        post = slice(20, 30)
        ref = _naive_reference_stats(resids, post, 2.0, 200, 7)
        got = _reference_stats(resids, post, 2.0, conformal_type="iid", ns=200, seed=7)
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12)

    def test_iid_array_post_indices(self):
        rng = _rng(11)
        resids = rng.standard_normal(40)
        post = np.array([35, 36, 37, 38, 39])
        ref = _naive_reference_stats(resids, post, 1.0, 300, 2)
        got = _reference_stats(resids, post, 1.0, conformal_type="iid", ns=300, seed=2)
        np.testing.assert_array_equal(got, ref)

    def test_block_unchanged(self):
        # Block conformal is deterministic cyclic shifts; must equal manual roll.
        rng = _rng(2)
        resids = rng.standard_normal(24)
        got = _reference_stats(resids, slice(18, 24), 1.0,
                               conformal_type="block", ns=999, seed=0)
        man = np.array([_stat(np.roll(resids, s)[18:24], 1.0) for s in range(24)])
        np.testing.assert_array_equal(got, man)
