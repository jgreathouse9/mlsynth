"""Tests for the SYNDES inner solver: the QP that remains once ``S`` is named.

Contract. With the treated set fixed, the two-way global problem is a strictly
convex quadratic program over a product of two simplices,

    f(S) = min { u'(G + lam I)u : u_S >= 0, sum_S u = 1;
                                  u_Sc <= 0, sum_Sc u = -1 },

which :mod:`mlsynth.utils.syndes_helpers.partition` solves for a whole batch of
treated sets at once. Every solve reports a Frank-Wolfe duality gap, so
``value - gap`` is a valid lower bound and the caller can tell a converged answer
from an unconverged one without consulting a reference solver.

What the tests pin:

* the projection is the Euclidean projection onto the simplex;
* the returned weights are feasible for the program (supports and sums);
* the reported value is the objective at those weights;
* the value agrees with cvxpy where cvxpy is accurate;
* the value agrees with the closed form on rows where the closed form is exact,
  which is the seam between this module and ``gram``;
* batching changes nothing -- a design solved alone and in company agree.
"""
from __future__ import annotations

from itertools import combinations

import cvxpy as cp
import numpy as np
import pytest

from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.syndes_helpers.gram import (
    BoundEngine,
    TwoWayProblem,
    signs_from_subsets,
)
from mlsynth.utils.syndes_helpers.partition import (
    project_simplex,
    solve_partition_batch,
    split_indices,
)


def _panel(N=8, T=13, rank=3, noise=0.35, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((T, rank)) @ rng.standard_normal((rank, N))
            + noise * rng.standard_normal((T, N)))


def _reference_f(problem, treated):
    """``f(S)`` via cvxpy, independent of the module under test."""
    N = problem.N
    mask = np.zeros(N, dtype=bool)
    mask[list(treated)] = True
    a, b = cp.Variable(N, nonneg=True), cp.Variable(N, nonneg=True)
    u = a - b
    cp.Problem(
        cp.Minimize(cp.quad_form(u, problem.G, assume_PSD=True)
                    + problem.lam * (cp.sum_squares(a) + cp.sum_squares(b))),
        [cp.sum(a) == 1, cp.sum(b) == 1, a[~mask] == 0, b[mask] == 0],
    ).solve(solver=cp.CLARABEL)
    av, bv = a.value, b.value
    uu = av - bv
    return float(uu @ problem.G @ uu + problem.lam * (av @ av + bv @ bv))


def _subsets(N, K):
    return np.array(list(combinations(range(N), K)), dtype=np.int64)


# ----------------------------------------------------------------------
# simplex projection
# ----------------------------------------------------------------------


class TestProjection:
    def test_lands_on_the_simplex(self):
        rng = np.random.default_rng(0)
        P = project_simplex(rng.normal(size=(200, 7)))
        np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-12)
        assert P.min() >= -1e-15

    def test_matches_cvxpy_objective(self):
        """CLARABEL settles short, so compare objectives: ours must not be worse."""
        rng = np.random.default_rng(1)
        V = rng.normal(size=(40, 6))
        P = project_simplex(V)
        for i in range(V.shape[0]):
            x = cp.Variable(6, nonneg=True)
            cp.Problem(cp.Minimize(cp.sum_squares(x - V[i])),
                       [cp.sum(x) == 1]).solve(solver=cp.CLARABEL)
            assert ((P[i] - V[i]) ** 2).sum() <= ((x.value - V[i]) ** 2).sum() + 1e-9

    def test_is_idempotent_on_the_simplex(self):
        rng = np.random.default_rng(2)
        P = project_simplex(rng.normal(size=(50, 5)))
        np.testing.assert_allclose(project_simplex(P), P, atol=1e-12)

    def test_single_column_is_the_only_vertex(self):
        np.testing.assert_allclose(
            project_simplex(np.array([[-4.0], [7.0]])), np.ones((2, 1)))

    def test_rejects_a_non_two_dimensional_input(self):
        with pytest.raises(MlsynthConfigError):
            project_simplex(np.ones(5))

    def test_rejects_a_zero_width_input(self):
        with pytest.raises(MlsynthConfigError):
            project_simplex(np.zeros((3, 0)))


# ----------------------------------------------------------------------
# index bookkeeping
# ----------------------------------------------------------------------


class TestStepSize:
    def test_step_is_the_reciprocal_of_twice_the_top_eigenvalue(self):
        problem = TwoWayProblem.from_panel(_panel())
        top = float(np.linalg.eigvalsh(problem.M)[-1])
        assert problem.step == pytest.approx(1.0 / (2.0 * top), rel=1e-12)

    def test_step_is_valid_for_every_restricted_subproblem(self):
        """No treated set may see a curvature above the one the step assumes."""
        problem = TwoWayProblem.from_panel(_panel())
        ceiling = 1.0 / (2.0 * problem.step)
        for subset in _subsets(problem.N, 3):
            signs = -np.ones(problem.N)
            signs[subset] = 1.0
            flipped = problem.M * np.outer(signs, signs)
            assert float(np.linalg.eigvalsh(flipped)[-1]) <= ceiling + 1e-9


class TestSplitIndices:
    def test_control_is_the_complement(self):
        treated = np.array([[0, 3], [1, 2]], dtype=np.int64)
        t, c = split_indices(5, treated)
        np.testing.assert_array_equal(t, treated)
        np.testing.assert_array_equal(c, np.array([[1, 2, 4], [0, 3, 4]]))

    def test_rejects_duplicate_units(self):
        with pytest.raises(MlsynthConfigError):
            split_indices(5, np.array([[1, 1]], dtype=np.int64))

    def test_rejects_out_of_range(self):
        with pytest.raises(MlsynthConfigError):
            split_indices(4, np.array([[0, 9]], dtype=np.int64))


# ----------------------------------------------------------------------
# the inner solve
# ----------------------------------------------------------------------


class TestInnerSolve:
    def test_weights_are_feasible(self):
        problem = TwoWayProblem.from_panel(_panel())
        subsets = _subsets(problem.N, 3)
        t, c = split_indices(problem.N, subsets)
        out = solve_partition_batch(problem, t, c)
        np.testing.assert_allclose(out.treated_weights.sum(axis=1), 1.0, atol=1e-9)
        np.testing.assert_allclose(out.control_weights.sum(axis=1), 1.0, atol=1e-9)
        assert out.treated_weights.min() >= -1e-12
        assert out.control_weights.min() >= -1e-12

    def test_value_is_the_objective_at_the_returned_weights(self):
        problem = TwoWayProblem.from_panel(_panel())
        subsets = _subsets(problem.N, 3)
        t, c = split_indices(problem.N, subsets)
        out = solve_partition_batch(problem, t, c)
        for row in range(subsets.shape[0]):
            a = np.zeros(problem.N); a[t[row]] = out.treated_weights[row]
            b = np.zeros(problem.N); b[c[row]] = out.control_weights[row]
            assert problem.evaluate(a, b) == pytest.approx(out.value[row], rel=1e-9)

    @pytest.mark.parametrize("K", [1, 2, 4, 7])
    def test_agrees_with_cvxpy(self, K):
        problem = TwoWayProblem.from_panel(_panel())
        subsets = _subsets(problem.N, K)
        t, c = split_indices(problem.N, subsets)
        out = solve_partition_batch(problem, t, c, max_iter=6000, tol=1e-14)
        reference = np.array([_reference_f(problem, s) for s in subsets])
        scale = max(reference.max(), 1e-12)
        assert np.max(np.abs(out.value - reference)) <= 1e-7 * scale

    def test_gap_is_nonnegative_and_bounds_below(self):
        problem = TwoWayProblem.from_panel(_panel())
        subsets = _subsets(problem.N, 3)
        t, c = split_indices(problem.N, subsets)
        out = solve_partition_batch(problem, t, c, max_iter=40, tol=0.0)
        assert np.all(out.gap >= 0.0)
        tight = solve_partition_batch(problem, t, c, max_iter=8000, tol=1e-14)
        scale = max(tight.value.max(), 1e-12)
        assert np.all(out.value - out.gap <= tight.value + 1e-9 * scale)

    def test_more_iterations_never_worsen_the_value(self):
        problem = TwoWayProblem.from_panel(_panel())
        subsets = _subsets(problem.N, 3)
        t, c = split_indices(problem.N, subsets)
        short = solve_partition_batch(problem, t, c, max_iter=25, tol=0.0)
        long = solve_partition_batch(problem, t, c, max_iter=4000, tol=1e-14)
        scale = max(long.value.max(), 1e-12)
        assert np.all(long.value <= short.value + 1e-9 * scale)

    def test_batching_changes_nothing(self):
        problem = TwoWayProblem.from_panel(_panel())
        subsets = _subsets(problem.N, 3)
        t, c = split_indices(problem.N, subsets)
        together = solve_partition_batch(problem, t, c, max_iter=3000, tol=1e-14)
        for row in (0, 5, len(subsets) - 1):
            alone = solve_partition_batch(
                problem, t[row][None, :], c[row][None, :],
                max_iter=3000, tol=1e-14)
            assert alone.value[0] == pytest.approx(together.value[row], rel=1e-8)


class TestAgreementWithTheClosedForm:
    """The seam between ``partition`` and ``gram``, checked without a reference."""

    def test_matches_the_closed_form_where_it_is_exact(self):
        problem = TwoWayProblem.from_panel(_panel())
        engine = BoundEngine.from_problem(problem)
        subsets = _subsets(problem.N, 3)
        closed = engine.bound(signs_from_subsets(problem.N, subsets))
        t, c = split_indices(problem.N, subsets)
        iterative = solve_partition_batch(problem, t, c, max_iter=8000, tol=1e-14)

        assert closed.exact.any(), "no subset was settled in closed form"
        scale = max(iterative.value.max(), 1e-12)
        exact_rows = np.flatnonzero(closed.exact)
        assert np.max(np.abs(
            iterative.value[exact_rows] - closed.lower[exact_rows])) <= 1e-7 * scale

    def test_never_falls_below_the_closed_form_bound(self):
        """``lb(S) <= f(S)``, with ``f`` computed here and ``lb`` computed there."""
        problem = TwoWayProblem.from_panel(_panel())
        engine = BoundEngine.from_problem(problem)
        subsets = _subsets(problem.N, 4)
        closed = engine.bound(signs_from_subsets(problem.N, subsets))
        t, c = split_indices(problem.N, subsets)
        iterative = solve_partition_batch(problem, t, c, max_iter=8000, tol=1e-14)
        scale = max(iterative.value.max(), 1e-12)
        assert np.all(closed.lower <= iterative.value + 1e-8 * scale)


# ----------------------------------------------------------------------
# edges and failures
# ----------------------------------------------------------------------


class TestEdges:
    def test_single_treated_unit(self):
        problem = TwoWayProblem.from_panel(_panel())
        t, c = split_indices(problem.N, np.array([[2]], dtype=np.int64))
        out = solve_partition_batch(problem, t, c, max_iter=4000, tol=1e-14)
        assert out.treated_weights.shape == (1, 1)
        assert out.treated_weights[0, 0] == pytest.approx(1.0)
        assert out.value[0] == pytest.approx(_reference_f(problem, [2]), rel=1e-6)

    def test_single_control_unit(self):
        problem = TwoWayProblem.from_panel(_panel())
        treated = np.array([[i for i in range(problem.N) if i != 3]], dtype=np.int64)
        t, c = split_indices(problem.N, treated)
        out = solve_partition_batch(problem, t, c, max_iter=4000, tol=1e-14)
        assert out.control_weights[0, 0] == pytest.approx(1.0)

    def test_two_units(self):
        problem = TwoWayProblem.from_panel(_panel(N=2, T=6))
        t, c = split_indices(2, np.array([[0]], dtype=np.int64))
        out = solve_partition_batch(problem, t, c, max_iter=4000, tol=1e-14)
        assert out.value[0] == pytest.approx(_reference_f(problem, [0]), rel=1e-6)

    def test_duplicate_donors_do_not_break_the_solve(self):
        Y = _panel(N=6, T=12, rank=2, noise=0.0)
        Y[:, 1] = Y[:, 0]
        problem = TwoWayProblem.from_panel(Y)
        t, c = split_indices(6, _subsets(6, 2))
        out = solve_partition_batch(problem, t, c, max_iter=4000, tol=1e-13)
        assert np.all(np.isfinite(out.value))
        assert np.all(out.gap >= 0.0)


class TestFailures:
    def test_rejects_mismatched_batch_sizes(self):
        problem = TwoWayProblem.from_panel(_panel())
        t = np.array([[0, 1], [2, 3]], dtype=np.int64)
        c = np.array([[2, 3, 4, 5, 6, 7]], dtype=np.int64)
        with pytest.raises(MlsynthConfigError):
            solve_partition_batch(problem, t, c)

    def test_rejects_an_empty_side(self):
        problem = TwoWayProblem.from_panel(_panel())
        with pytest.raises(MlsynthConfigError):
            solve_partition_batch(problem, np.zeros((1, 0), dtype=np.int64),
                                  np.arange(problem.N)[None, :])

    def test_rejects_indices_out_of_range(self):
        problem = TwoWayProblem.from_panel(_panel())
        with pytest.raises(MlsynthConfigError):
            solve_partition_batch(problem, np.array([[99]], dtype=np.int64),
                                  np.array([[0, 1]], dtype=np.int64))

    def test_rejects_overlapping_sides(self):
        """A unit cannot be treated and control at once."""
        problem = TwoWayProblem.from_panel(_panel())
        with pytest.raises(MlsynthConfigError):
            solve_partition_batch(problem, np.array([[0, 1]], dtype=np.int64),
                                  np.array([[1, 2]], dtype=np.int64))
