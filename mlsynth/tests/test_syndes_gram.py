"""Tests for the Gram reduction of the SYNDES two-way global problem.

Contract. Naming the treated set ``S`` removes every binary from the two-way
MIP, leaving a convex QP that depends on ``Y`` only through ``G = Y'Y / T``:

    f(S) = min { u'(G + lam I)u : u_S >= 0, sum_S u = 1;
                                  u_Sc <= 0, sum_Sc u = -1 }

Writing ``sigma`` for the +/-1 indicator of ``S``, the two normalizations are
``1'u = 0`` and ``sigma'u = 2``. Dropping the sign conditions leaves a
two-equality QP with the closed form ``lb(S) = 4 alpha / (sigma' R sigma)``,
where ``H = (G + lam I)^-1``, ``p = H1``, ``alpha = 1'H1``, ``R = alpha H - pp'``.

What the tests pin:

* ``lb(S) <= f(S)`` for every subset -- soundness, since the solver prunes on it;
* ``lb(S) == f(S)`` on rows flagged exact -- the dropped conditions were slack;
* ``R1 = 0`` and ``R`` PSD, which is what makes the Rayleigh bound valid;
* ``global_lower_bound(K)`` never exceeds the true optimum;
* an ill-conditioned ``M`` yields no bound instead of a wrong one.

``f(S)`` is computed by cvxpy here, independently of the module under test.
"""
from __future__ import annotations

from itertools import combinations

import cvxpy as cp
import numpy as np
import pytest

from mlsynth.exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
)
from mlsynth.utils.syndes_helpers.gram import (
    BoundEngine,
    TwoWayProblem,
    maxcut_candidates,
    signs_from_subsets,
)
from mlsynth.utils.syndes_helpers.optimization import estimate_lambda


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _panel(N=9, T=14, rank=3, noise=0.3, seed=0):
    """A factor-structured ``(T, N)`` pre-period matrix."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((T, rank)) @ rng.standard_normal((rank, N))
            + noise * rng.standard_normal((T, N)))


def _reference_f(problem: TwoWayProblem, treated) -> float:
    """``f(S)`` via cvxpy, independent of the module under test."""
    N = problem.N
    mask = np.zeros(N, dtype=bool)
    mask[list(treated)] = True
    a = cp.Variable(N, nonneg=True)
    b = cp.Variable(N, nonneg=True)
    u = a - b
    cp.Problem(
        cp.Minimize(cp.quad_form(u, problem.G, assume_PSD=True)
                    + problem.lam * (cp.sum_squares(a) + cp.sum_squares(b))),
        [cp.sum(a) == 1, cp.sum(b) == 1, a[~mask] == 0, b[mask] == 0],
    ).solve(solver=cp.CLARABEL)
    av, bv = a.value, b.value
    uu = av - bv
    return float(uu @ problem.G @ uu + problem.lam * (av @ av + bv @ bv))


def _all_subsets(N, K):
    return np.array(list(combinations(range(N), K)), dtype=np.int64)


# ----------------------------------------------------------------------
# smoke
# ----------------------------------------------------------------------


class TestSmoke:
    def test_builds_and_returns_finite_bounds(self):
        problem = TwoWayProblem.from_panel(_panel())
        engine = BoundEngine.from_problem(problem)
        subsets = _all_subsets(problem.N, 3)
        out = engine.bound(signs_from_subsets(problem.N, subsets))

        assert out.lower.shape == (subsets.shape[0],)
        assert out.contrast.shape == (subsets.shape[0], problem.N)
        assert out.exact.shape == (subsets.shape[0],)
        assert np.all(np.isfinite(out.lower))
        assert out.exact.dtype == np.bool_

    def test_gram_shapes_and_symmetry(self):
        Y = _panel()
        problem = TwoWayProblem.from_panel(Y)
        assert problem.G.shape == (problem.N, problem.N)
        assert problem.M.shape == (problem.N, problem.N)
        np.testing.assert_allclose(problem.G, problem.G.T, atol=1e-12)
        np.testing.assert_allclose(problem.G, Y.T @ Y / Y.shape[0], atol=1e-10)

    def test_lambda_default_matches_the_mip_default(self):
        """The two paths must minimize the same objective."""
        Y = _panel()
        assert TwoWayProblem.from_panel(Y).lam == pytest.approx(estimate_lambda(Y))

    def test_explicit_lambda_is_used_verbatim(self):
        problem = TwoWayProblem.from_panel(_panel(), lam=0.25)
        assert problem.lam == pytest.approx(0.25)
        np.testing.assert_allclose(
            problem.M, problem.G + 0.25 * np.eye(problem.N), atol=1e-12)

    def test_evaluate_agrees_with_the_reference_objective(self):
        """``evaluate`` is the objective the whole module is about."""
        problem = TwoWayProblem.from_panel(_panel(N=6, T=10))
        a = np.array([0.6, 0.4, 0.0, 0.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 0.25, 0.25, 0.25, 0.25])
        u = a - b
        expected = float(u @ problem.G @ u
                         + problem.lam * (a @ a + b @ b))
        assert problem.evaluate(a, b) == pytest.approx(expected, rel=1e-12)

    def test_evaluate_matches_the_closed_form_on_an_exact_row(self):
        problem = TwoWayProblem.from_panel(_panel(N=6, T=10))
        engine = BoundEngine.from_problem(problem)
        subsets = _all_subsets(6, 3)
        signs = signs_from_subsets(6, subsets)
        out = engine.bound(signs)
        idx = int(np.flatnonzero(out.exact)[0])
        u = out.contrast[idx]
        assert problem.evaluate(np.maximum(u, 0.0), np.maximum(-u, 0.0)) == (
            pytest.approx(out.lower[idx], rel=1e-9))


# ----------------------------------------------------------------------
# structure of R -- what makes the Rayleigh bound valid
# ----------------------------------------------------------------------


class TestCutMatrix:
    def test_R_annihilates_the_ones_vector(self):
        engine = BoundEngine.from_problem(TwoWayProblem.from_panel(_panel()))
        np.testing.assert_allclose(
            engine.R @ np.ones(engine.N), 0.0, atol=1e-8)

    def test_R_is_psd(self):
        engine = BoundEngine.from_problem(TwoWayProblem.from_panel(_panel()))
        assert float(np.linalg.eigvalsh(engine.R)[0]) > -1e-8

    def test_alpha_positive_and_p_consistent(self):
        problem = TwoWayProblem.from_panel(_panel())
        engine = BoundEngine.from_problem(problem)
        assert engine.alpha > 0.0
        np.testing.assert_allclose(problem.M @ engine.p, np.ones(engine.N), atol=1e-8)
        assert engine.alpha == pytest.approx(float(engine.p.sum()), rel=1e-10)

    def test_cut_value_matches_the_quadratic_form(self):
        engine = BoundEngine.from_problem(TwoWayProblem.from_panel(_panel()))
        subsets = _all_subsets(engine.N, 4)[:20]
        signs = signs_from_subsets(engine.N, subsets)
        expected = np.array([s @ engine.R @ s for s in signs])
        np.testing.assert_allclose(engine.cut_value(signs), expected, rtol=1e-10)


# ----------------------------------------------------------------------
# the bound is sound, and exact where it says it is
# ----------------------------------------------------------------------


class TestBound:
    @pytest.mark.parametrize("K", [1, 2, 4, 8])
    @pytest.mark.parametrize("lam", [None, 0.01, 5.0])
    def test_lower_bounds_every_subset(self, K, lam):
        problem = TwoWayProblem.from_panel(_panel(), lam=lam)
        engine = BoundEngine.from_problem(problem)
        subsets = _all_subsets(problem.N, K)
        lower = engine.bound(signs_from_subsets(problem.N, subsets)).lower
        truth = np.array([_reference_f(problem, s) for s in subsets])
        # scale-relative: CLARABEL's own accuracy is the floor here
        assert np.max((lower - truth) / max(truth.max(), 1e-12)) <= 1e-7

    def test_exact_rows_hit_the_optimum(self):
        problem = TwoWayProblem.from_panel(_panel())
        engine = BoundEngine.from_problem(problem)
        subsets = _all_subsets(problem.N, 4)
        out = engine.bound(signs_from_subsets(problem.N, subsets))
        assert out.exact.any(), "no subset was settled in closed form"
        for idx in np.flatnonzero(out.exact):
            assert out.lower[idx] == pytest.approx(
                _reference_f(problem, subsets[idx]), rel=1e-7, abs=1e-12)

    def test_exact_rows_have_sign_feasible_contrasts(self):
        problem = TwoWayProblem.from_panel(_panel())
        engine = BoundEngine.from_problem(problem)
        subsets = _all_subsets(problem.N, 4)
        signs = signs_from_subsets(problem.N, subsets)
        out = engine.bound(signs)
        for idx in np.flatnonzero(out.exact):
            assert np.all(signs[idx] * out.contrast[idx] >= -1e-9)

    def test_contrast_satisfies_both_normalizations(self):
        problem = TwoWayProblem.from_panel(_panel())
        engine = BoundEngine.from_problem(problem)
        subsets = _all_subsets(problem.N, 3)
        signs = signs_from_subsets(problem.N, subsets)
        out = engine.bound(signs)
        np.testing.assert_allclose(out.contrast.sum(axis=1), 0.0, atol=1e-8)
        np.testing.assert_allclose(
            np.einsum("bi,bi->b", signs, out.contrast), 2.0, atol=1e-8)


class TestRayleighBound:
    @pytest.mark.parametrize("K", [1, 3, 5, 8])
    def test_never_exceeds_the_true_optimum(self, K):
        problem = TwoWayProblem.from_panel(_panel())
        engine = BoundEngine.from_problem(problem)
        subsets = _all_subsets(problem.N, K)
        optimum = min(_reference_f(problem, s) for s in subsets)
        bound = engine.global_lower_bound(K)
        assert bound is not None
        assert bound <= optimum * (1 + 1e-7)

    def test_positive_and_finite(self):
        engine = BoundEngine.from_problem(TwoWayProblem.from_panel(_panel()))
        for K in range(1, engine.N):
            value = engine.global_lower_bound(K)
            assert value is not None and np.isfinite(value) and value > 0.0

    def test_rejects_degenerate_K(self):
        engine = BoundEngine.from_problem(TwoWayProblem.from_panel(_panel()))
        with pytest.raises(MlsynthConfigError):
            engine.global_lower_bound(0)
        with pytest.raises(MlsynthConfigError):
            engine.global_lower_bound(engine.N)


# ----------------------------------------------------------------------
# max-cut candidate search
# ----------------------------------------------------------------------


class TestMaxcutCandidates:
    def test_returns_valid_distinct_k_subsets(self):
        engine = BoundEngine.from_problem(TwoWayProblem.from_panel(_panel()))
        cands = maxcut_candidates(engine, 4, n_restarts=6, seed=0)
        assert cands.ndim == 2 and cands.shape[1] == 4
        for row in cands:
            assert len(set(row.tolist())) == 4
            assert row.min() >= 0 and row.max() < engine.N
        assert len({tuple(r) for r in cands}) == cands.shape[0]

    def test_deterministic_for_a_fixed_seed(self):
        engine = BoundEngine.from_problem(TwoWayProblem.from_panel(_panel()))
        first = maxcut_candidates(engine, 3, n_restarts=5, seed=7)
        second = maxcut_candidates(engine, 3, n_restarts=5, seed=7)
        np.testing.assert_array_equal(first, second)

    def test_candidates_are_swap_local_optima(self):
        """No single swap improves the cut value of a returned candidate."""
        engine = BoundEngine.from_problem(TwoWayProblem.from_panel(_panel()))
        K = 4
        for treated in maxcut_candidates(engine, K, n_restarts=4, seed=1):
            sigma = -np.ones(engine.N)
            sigma[treated] = 1.0
            base = float(sigma @ engine.R @ sigma)
            for i in np.flatnonzero(sigma > 0):
                for j in np.flatnonzero(sigma < 0):
                    swapped = sigma.copy()
                    swapped[i], swapped[j] = -1.0, 1.0
                    assert float(swapped @ engine.R @ swapped) <= base + 1e-9

    def test_beats_a_random_design_on_the_bound(self):
        problem = TwoWayProblem.from_panel(_panel())
        engine = BoundEngine.from_problem(problem)
        K = 4
        best = maxcut_candidates(engine, K, n_restarts=8, seed=0)
        subsets = _all_subsets(problem.N, K)
        lower = engine.bound(signs_from_subsets(problem.N, subsets)).lower
        reached = engine.bound(signs_from_subsets(problem.N, best)).lower.min()
        assert reached <= np.median(lower)


# ----------------------------------------------------------------------
# edge cases
# ----------------------------------------------------------------------


class TestEdges:
    def test_two_units_one_treated(self):
        problem = TwoWayProblem.from_panel(_panel(N=2, T=5))
        engine = BoundEngine.from_problem(problem)
        out = engine.bound(signs_from_subsets(2, np.array([[0]], dtype=np.int64)))
        assert np.isfinite(out.lower[0])
        assert out.lower[0] <= _reference_f(problem, [0]) + 1e-9

    @pytest.mark.parametrize("K", [1, 8])
    def test_extreme_cardinalities(self, K):
        problem = TwoWayProblem.from_panel(_panel())
        engine = BoundEngine.from_problem(problem)
        subsets = _all_subsets(problem.N, K)
        lower = engine.bound(signs_from_subsets(problem.N, subsets)).lower
        truth = np.array([_reference_f(problem, s) for s in subsets])
        assert np.all(lower <= truth + 1e-7 * max(truth.max(), 1.0))

    def test_zero_lambda_with_full_rank_gram(self):
        problem = TwoWayProblem.from_panel(_panel(N=6, T=20, rank=6, noise=1.0),
                                           lam=0.0)
        engine = BoundEngine.from_problem(problem)
        assert engine.reliable
        subsets = _all_subsets(6, 3)
        lower = engine.bound(signs_from_subsets(6, subsets)).lower
        truth = np.array([_reference_f(problem, s) for s in subsets])
        assert np.all(lower <= truth + 1e-7 * max(truth.max(), 1.0))

    def test_collinear_columns_still_bound(self):
        Y = _panel(N=6, T=12, rank=2, noise=0.0)
        Y[:, 1] = Y[:, 0]                                  # exact duplicate donor
        problem = TwoWayProblem.from_panel(Y)
        engine = BoundEngine.from_problem(problem)
        subsets = _all_subsets(6, 2)
        lower = engine.bound(signs_from_subsets(6, subsets)).lower
        truth = np.array([_reference_f(problem, s) for s in subsets])
        assert np.all(lower <= truth + 1e-7 * max(truth.max(), 1.0))


# ----------------------------------------------------------------------
# failures are reported, not swallowed
# ----------------------------------------------------------------------


class TestFailures:
    def test_one_dimensional_panel_raises(self):
        with pytest.raises(MlsynthDataError):
            TwoWayProblem.from_panel(np.arange(6.0))

    def test_single_period_raises(self):
        with pytest.raises(MlsynthConfigError):
            TwoWayProblem.from_panel(np.ones((1, 4)))

    def test_negative_lambda_raises(self):
        with pytest.raises(MlsynthConfigError):
            TwoWayProblem.from_panel(_panel(), lam=-1.0)

    def test_single_unit_raises(self):
        with pytest.raises(MlsynthConfigError):
            TwoWayProblem.from_panel(_panel(N=1, T=5))

    def test_ill_conditioned_gram_reports_no_bound(self):
        """A rank-deficient Gram with a vanishing ridge must not fake a bound."""
        Y = _panel(N=10, T=4, rank=2, noise=0.0)           # rank(G) <= 4 < N
        engine = BoundEngine.from_problem(TwoWayProblem.from_panel(Y, lam=1e-14))
        assert not engine.reliable
        assert engine.global_lower_bound(3) is None
        with pytest.raises(MlsynthEstimationError):
            engine.bound(signs_from_subsets(10, _all_subsets(10, 3)))

    def test_conditioning_threshold_is_configurable(self):
        problem = TwoWayProblem.from_panel(_panel())
        assert BoundEngine.from_problem(problem).reliable
        assert not BoundEngine.from_problem(problem, condition_max=1.0).reliable

    def test_sign_matrix_rejects_out_of_range_indices(self):
        with pytest.raises(MlsynthConfigError):
            signs_from_subsets(4, np.array([[0, 9]], dtype=np.int64))

    def test_bound_rejects_a_mismatched_sign_matrix(self):
        engine = BoundEngine.from_problem(TwoWayProblem.from_panel(_panel()))
        with pytest.raises(MlsynthConfigError):
            engine.bound(np.ones((3, engine.N + 2)))
        with pytest.raises(MlsynthConfigError):
            engine.bound(np.ones(engine.N))          # one-dimensional

    def test_maxcut_rejects_degenerate_K(self):
        engine = BoundEngine.from_problem(TwoWayProblem.from_panel(_panel()))
        with pytest.raises(MlsynthConfigError):
            maxcut_candidates(engine, 0)
        with pytest.raises(MlsynthConfigError):
            maxcut_candidates(engine, engine.N)

    def test_maxcut_refuses_an_unreliable_engine(self):
        problem = TwoWayProblem.from_panel(_panel())
        engine = BoundEngine.from_problem(problem, condition_max=1.0)
        with pytest.raises(MlsynthEstimationError):
            maxcut_candidates(engine, 3)

    def test_cut_value_refuses_an_unreliable_engine(self):
        problem = TwoWayProblem.from_panel(_panel())
        engine = BoundEngine.from_problem(problem, condition_max=1.0)
        with pytest.raises(MlsynthEstimationError):
            engine.cut_value(signs_from_subsets(
                problem.N, _all_subsets(problem.N, 3)))
