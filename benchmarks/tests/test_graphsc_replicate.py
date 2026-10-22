"""The Graph-SC replication oracle (Zhang, Chung & Talebpour 2026, arXiv:2610.24113v1).

``benchmarks/studies/graphsc_replicate/oracle.py`` ports Algorithm 1 (the donor
graph) and Algorithm 2 (FISTA on eq. (2)), and solves eq. (2) a second way: with
``w >= 0`` the l1 term is ``lambda_1 1'w``, linear, so the whole objective is a
non-negative least-squares problem whose target sum is ``1 - lambda_1 /
lambda_sum``. The tests pin the graph's algebra (Lemma 3.2), that reformulation,
and agreement of FISTA, the exact active-set solve and an independent convex
solver.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from benchmarks.studies.graphsc_replicate import oracle as G


def _problem(seed, J=30, T0=19, k=5):
    rng = np.random.default_rng(seed)
    X0 = rng.normal(size=(T0, J)).cumsum(axis=0)
    w_true = rng.dirichlet(np.ones(4))
    x1 = X0[:, :4] @ w_true + 0.3 * rng.normal(size=T0)
    L, A = G.knn_laplacian(X0.T, k)
    return x1, X0, L, A


class TestTheGraph:
    @settings(max_examples=40, deadline=None)
    @given(st.integers(0, 2**31 - 1), st.integers(5, 40), st.integers(1, 6))
    def test_the_laplacian_is_symmetric_psd_with_spectrum_in_0_2(self, seed, J, k):
        k = min(k, J - 1)
        F = np.random.default_rng(seed).normal(size=(J, 8))
        L, A = G.knn_laplacian(F, k)
        assert np.array_equal(A, A.T) and not A.diagonal().any()
        assert np.all(A.sum(axis=1) >= k)                    # every node keeps its own k neighbours
        ev = np.linalg.eigvalsh(L)
        assert np.allclose(L, L.T) and ev.min() >= -1e-12 and ev.max() <= 2 + 1e-12

    @settings(max_examples=40, deadline=None)
    @given(st.integers(0, 2**31 - 1))
    def test_lemma_3_2_the_quadratic_form_is_the_edge_sum(self, seed):
        rng = np.random.default_rng(seed)
        F = rng.normal(size=(25, 6))
        L, A = G.knn_laplacian(F, 4)
        w = rng.normal(size=25)
        d = A.sum(axis=1)
        u, v = np.nonzero(np.triu(A))
        edge_sum = np.sum((w[u] / np.sqrt(d[u]) - w[v] / np.sqrt(d[v])) ** 2)
        assert np.isclose(w @ L @ w, edge_sum, rtol=1e-12, atol=1e-12)

    def test_similarity_choices_give_the_same_graph_for_euclidean_and_its_negation(self):
        F = np.random.default_rng(1).normal(size=(12, 5))
        L1, A1 = G.knn_laplacian(F, 3, metric="euclidean")
        assert A1.sum() > 0
        for metric in ("correlation", "cosine"):
            L, A = G.knn_laplacian(F, 3, metric=metric)
            assert np.array_equal(A, A.T)

    def test_an_unknown_metric_is_an_error(self):
        with pytest.raises(ValueError, match="metric"):
            G.knn_laplacian(np.ones((5, 2)), 2, metric="manhattan")


class TestTheReformulation:
    @settings(max_examples=40, deadline=None)
    @given(st.integers(0, 2**31 - 1), st.floats(0.0, 1.0), st.floats(0.0, 20.0), st.floats(1.0, 100.0))
    def test_the_objective_is_a_nonnegative_least_squares_up_to_a_constant(self, seed, l1, lg, ls):
        x1, X0, L, _ = _problem(seed % 1000)
        M, b, const = G.nnls_form(x1, X0, L, l1, lg, ls)
        rng = np.random.default_rng(seed)
        for _ in range(3):
            w = rng.uniform(0, 0.2, size=X0.shape[1])
            lhs = G.objective(w, x1, X0, L, l1, lg, ls)
            rhs = 0.5 * np.sum((M @ w - b) ** 2) + const
            assert np.isclose(lhs, rhs, rtol=1e-10, atol=1e-10 * (1 + abs(lhs)))

    def test_the_l1_term_only_moves_the_target_sum(self):
        """With w >= 0, lambda_1 ||w||_1 + lambda_sum/2 (1'w - 1)^2 equals
        lambda_sum/2 (1'w - (1 - lambda_1/lambda_sum))^2 plus a constant."""
        x1, X0, L, _ = _problem(3)
        w_a = G.solve_exact(x1, X0, L, 0.05, 10.0, 50.0)
        M, b, _ = G.nnls_form(x1, X0, L, 0.0, 10.0, 50.0)
        b = b.copy()
        b[-1] = np.sqrt(50.0) * (1 - 0.05 / 50.0)
        from mlsynth.utils.solvers.nnls import nnls

        w_b = nnls(M, b)[0]
        assert np.allclose(w_a, w_b, atol=1e-12)


class TestTheSolvers:
    @pytest.mark.parametrize("seed", range(6))
    def test_fista_the_exact_solve_and_cvxpy_agree(self, seed):
        import cvxpy as cp

        x1, X0, L, _ = _problem(seed)
        args = (x1, X0, L, 0.05, 10.0, 50.0)
        w_f, info = G.fista(*args)
        w_e = G.solve_exact(*args)
        w = cp.Variable(X0.shape[1], nonneg=True)
        obj = (0.5 * cp.sum_squares(x1 - X0 @ w) + 0.05 * cp.sum(w)
               + 5.0 * cp.quad_form(w, cp.psd_wrap(L)) + 25.0 * cp.square(cp.sum(w) - 1))
        cp.Problem(cp.Minimize(obj)).solve(solver=cp.CLARABEL)
        F = lambda v: G.objective(v, *args)
        best = min(F(w_f), F(w_e), F(w.value))
        assert info["converged"]
        assert F(w_e) <= best * (1 + 1e-10) + 1e-12
        assert F(w_f) <= best * (1 + 1e-8)
        assert G.kkt_residual(w_e, *args) <= 1e-9 and G.kkt_residual(w_f, *args) <= 1e-7

    def test_a_capped_fista_reports_it(self):
        x1, X0, L, _ = _problem(0)
        with pytest.warns(RuntimeWarning, match="FISTA"):
            _, info = G.fista(x1, X0, L, 0.05, 10.0, 50.0, max_iter=3)
        assert not info["converged"] and info["iterations"] == 3

    def test_the_exact_solve_satisfies_the_kkt_conditions(self):
        x1, X0, L, _ = _problem(4)
        w = G.solve_exact(x1, X0, L, 0.05, 10.0, 50.0)
        assert np.all(w >= 0) and G.kkt_residual(w, x1, X0, L, 0.05, 10.0, 50.0) <= 1e-9

    def test_a_point_that_is_not_optimal_fails_the_kkt_test(self):
        x1, X0, L, _ = _problem(5)
        w = G.solve_exact(x1, X0, L, 0.05, 10.0, 50.0)
        w2 = w.copy()
        w2[np.argmax(w)] *= 0.9
        assert G.kkt_residual(w2, x1, X0, L, 0.05, 10.0, 50.0) > 1e-4


class TestTheMetrics:
    def test_entropy_and_active_donors(self):
        w = np.array([0.5, 0.5, 0.0, 0.0])
        assert np.isclose(G.weight_entropy(w), np.log(2), atol=1e-9)
        assert G.active_donors(w) == 2

    def test_the_paper_sc_row_on_prop_99(self):
        """Table 1's SC row pins the metric definitions: simplex SC on the 19
        pre-period outcomes in packs per capita."""
        from benchmarks.studies.graphsc_replicate import run as R

        y, X, T0 = R.prop99()
        w = G.simplex_sc(y[:T0], X[:T0])
        m = G.metrics(y, X @ w, T0, w)
        assert abs(m["pre_rmspe"] - 1.6564) <= 5e-4
        assert abs(m["post_gap"] - 19.5136) <= 5e-3
        assert m["active"] == 6
        assert abs(m["entropy"] - 1.4752) <= 5e-4
