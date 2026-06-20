"""Tests for the in-house non-negative least squares solver.

The MSCMT backend's inner solve is a big-M NNLS called tens of thousands of
times inside differential evolution. We replace ``scipy.optimize.nnls`` with a
pure-NumPy Lawson-Hanson active-set solver so the result is independent of the
installed scipy version (scipy 1.13's ``nnls`` *raises* ``RuntimeError`` on
hitting ``maxiter``; newer scipy returns a best-effort iterate) and never
raises in the hot loop. These tests pin it to the scipy reference where one is
available, and to the KKT conditions directly otherwise.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlsynth.utils.bilevel.nnls import nnls


def _kkt_residual(A: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    """Max KKT violation for ``min ||Aw - b||^2 s.t. w >= 0``.

    At the optimum: ``w >= 0``; the gradient ``g = A'(Aw - b)`` satisfies
    ``g_j >= 0`` on the zero set and ``g_j == 0`` on the support. The
    complementary-slackness scalar ``max(min(w_j, g_j))`` (with sign) captures
    both; we return the largest violation.
    """
    g = A.T @ (A @ w - b)
    viol = 0.0
    viol = max(viol, float(np.max(-np.minimum(w, 0.0))))      # w >= 0
    on_support = w > 1e-9
    if on_support.any():
        viol = max(viol, float(np.max(np.abs(g[on_support]))))  # g == 0 on support
    off = ~on_support
    if off.any():
        viol = max(viol, float(np.max(np.maximum(-g[off], 0.0))))  # g >= 0 off support
    return viol


class TestSmoke:

    def test_returns_w_and_rnorm(self):
        rng = np.random.default_rng(0)
        A = rng.standard_normal((8, 5))
        b = rng.standard_normal(8)
        w, rnorm = nnls(A, b)
        assert w.shape == (5,)
        assert isinstance(rnorm, float)
        assert np.all(w >= 0.0)
        np.testing.assert_allclose(rnorm, np.linalg.norm(A @ w - b), atol=1e-8)


class TestCorrectness:

    @pytest.mark.parametrize("seed", range(8))
    def test_matches_scipy_reference(self, seed):
        scipy_opt = pytest.importorskip("scipy.optimize")
        rng = np.random.default_rng(seed)
        m, n = rng.integers(4, 20), rng.integers(2, 12)
        A = rng.standard_normal((m, n))
        b = rng.standard_normal(m)
        w, _ = nnls(A, b)
        try:
            w_ref, _ = scipy_opt.nnls(A, b)
        except Exception:
            pytest.skip("scipy nnls failed to converge on this draw")
        # NNLS has a unique minimiser when A has full column rank; compare the
        # residuals (robust to a non-unique argmin) and the weights.
        obj = np.linalg.norm(A @ w - b)
        obj_ref = np.linalg.norm(A @ w_ref - b)
        assert obj <= obj_ref + 1e-6
        np.testing.assert_allclose(w, w_ref, atol=1e-6)

    def test_recovers_nonnegative_combination(self):
        rng = np.random.default_rng(3)
        A = rng.standard_normal((30, 6))
        w_true = np.array([0.0, 1.5, 0.0, 2.0, 0.0, 0.5])
        b = A @ w_true
        w, rnorm = nnls(A, b)
        np.testing.assert_allclose(w, w_true, atol=1e-6)
        assert rnorm < 1e-6

    @pytest.mark.parametrize("seed", range(5))
    def test_satisfies_kkt(self, seed):
        rng = np.random.default_rng(100 + seed)
        A = rng.standard_normal((12, 7))
        b = rng.standard_normal(12)
        w, _ = nnls(A, b)
        assert _kkt_residual(A, b, w) < 1e-6

    def test_big_m_simplex_structure(self):
        # The exact shape MSCMT feeds in: predictor-matching rows plus a big-M
        # sum-to-one row. The bespoke solver must handle the ill-conditioning.
        rng = np.random.default_rng(7)
        K, J, M = 6, 20, 1e6
        X0 = rng.standard_normal((K, J))
        X1 = X0 @ rng.dirichlet(np.ones(J))      # a true simplex combination
        A = np.vstack([X0, M * np.ones(J)])
        b = np.concatenate([X1, [M]])
        w, _ = nnls(A, b)
        assert np.all(w >= 0.0)
        # After the big-M row, the weights should nearly sum to one.
        assert abs(w.sum() - 1.0) < 1e-3


class TestEdgeCases:

    def test_single_column(self):
        A = np.array([[2.0], [4.0]])
        b = np.array([2.0, 4.0])
        w, rnorm = nnls(A, b)
        np.testing.assert_allclose(w, [1.0], atol=1e-9)
        assert rnorm < 1e-9

    def test_all_negative_correlation_gives_zero(self):
        # If every column points away from b, the NNLS optimum is w = 0.
        A = np.array([[1.0, 1.0], [1.0, 1.0]])
        b = np.array([-1.0, -1.0])
        w, _ = nnls(A, b)
        np.testing.assert_allclose(w, [0.0, 0.0], atol=1e-9)

    def test_zero_b(self):
        rng = np.random.default_rng(1)
        A = rng.standard_normal((6, 4))
        w, rnorm = nnls(A, np.zeros(6))
        np.testing.assert_allclose(w, np.zeros(4), atol=1e-9)
        assert rnorm < 1e-9

    def test_never_raises_on_degenerate_input(self):
        # Rank-deficient / collinear columns must not raise; a best-effort
        # non-negative iterate is returned.
        A = np.ones((5, 4))
        b = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        w, rnorm = nnls(A, b, maxiter=50)
        assert np.all(w >= 0.0)
        assert np.isfinite(rnorm)

    def test_maxiter_returns_best_effort_not_raise(self):
        # A pathological problem with a tiny maxiter must return, not raise.
        rng = np.random.default_rng(5)
        A = rng.standard_normal((40, 30))
        b = rng.standard_normal(40)
        w, rnorm = nnls(A, b, maxiter=1)
        assert np.all(w >= 0.0)
        assert np.isfinite(rnorm)
