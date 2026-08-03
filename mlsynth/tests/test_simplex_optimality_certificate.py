"""A solver-independent optimality certificate for a simplex weight vector.

Test-first (per ``agents/agents_tests.md``).

The Gram reduction squares the condition number: ``G = R'R`` has
``cond(G) = cond(R)^2``, so a design that is merely awkward at 1e7 -- covariates
in different units, say -- makes the reduced problem singular in float64. The
batched solver then converges on ``G`` to a point that is not a solution of the
program ``G`` came from, and no amount of checking ``G`` reveals it: the answer
has to be checked against the design.

That is what this certificate does. With ``g = B'(Bw - A)`` and ``nu = g'w``, the
KKT conditions for ``min ||Bw - A||^2`` over the simplex say every donor's
reduced gradient is at least ``nu``, with equality on the support. It is
computed from ``B`` and ``A`` directly, so it certifies the point regardless of
which solver produced it or what form that solver worked in.

Optimality and uniqueness are separate questions and both are asked. A point can
be optimal on a face (optimal, not unique) or be the unique optimum's neighbour
that a squared condition number produced (unique optimum, point not optimal).
Only a point that is both certified optimal and known to be the sole optimum can
be asserted equal to what another exact solver returns.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlsynth.utils.bilevel.active_set import solve_simplex_qp
from mlsynth.utils.bilevel.minnorm import simplex_point_is_optimal


def _design(rng, m, J):
    return np.cumsum(rng.normal(size=(m, J)), axis=0) + 50.0


# --------------------------------------------------------------------------- #
# 1. It accepts solutions
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("m,J", [(20, 8), (6, 30), (40, 12), (5, 39), (9, 2)])
def test_accepts_the_exact_solver(m, J):
    rng = np.random.default_rng(m * 13 + J)
    B, A = _design(rng, m, J), _design(rng, m, 1).ravel()
    assert simplex_point_is_optimal(B, A, solve_simplex_qp(B, A))


def test_accepts_a_point_on_a_face():
    """A target the donors reproduce exactly: many optima, each of them optimal."""
    rng = np.random.default_rng(1)
    B = _design(rng, 5, 20)
    A = B @ rng.dirichlet(np.ones(20))
    w = solve_simplex_qp(B, A)
    assert simplex_point_is_optimal(B, A, w)


def test_accepts_a_vertex_when_one_donor_is_the_answer():
    B = np.array([[1.0, 5.0], [2.0, 9.0]])
    A = np.array([1.0, 2.0])
    assert simplex_point_is_optimal(B, A, np.array([1.0, 0.0]))


def test_accepts_the_only_point_when_there_is_one_donor():
    B, A = np.array([[1.0], [2.0]]), np.array([3.0, 4.0])
    assert simplex_point_is_optimal(B, A, np.array([1.0]))


# --------------------------------------------------------------------------- #
# 2. It rejects non-solutions
# --------------------------------------------------------------------------- #
def test_rejects_a_feasible_point_that_is_not_optimal():
    rng = np.random.default_rng(2)
    B, A = _design(rng, 20, 6), _design(rng, 20, 1).ravel()
    assert not simplex_point_is_optimal(B, A, np.full(6, 1.0 / 6.0))


def test_rejects_a_point_perturbed_off_the_optimum():
    rng = np.random.default_rng(3)
    B, A = _design(rng, 25, 5), _design(rng, 25, 1).ravel()
    w = np.asarray(solve_simplex_qp(B, A), dtype=float)
    bad = w.copy()
    bad[0], bad[1] = bad[0] + 0.05, bad[1] - 0.05
    if bad.min() >= 0:                      # only meaningful while still feasible
        assert not simplex_point_is_optimal(B, A, bad)


def test_rejects_a_point_optimal_for_a_different_target():
    rng = np.random.default_rng(4)
    B = _design(rng, 18, 7)
    A1, A2 = _design(rng, 18, 1).ravel(), _design(rng, 18, 1).ravel()
    assert not simplex_point_is_optimal(B, A2, solve_simplex_qp(B, A1))


def test_rejects_the_squared_condition_number_failure_mode():
    """The case this exists for: a design whose Gram is singular in float64, where
    solving in Gram form lands somewhere feasible and wrong."""
    from mlsynth.utils.bilevel.minnorm import simplex_gram, solve_simplex_minnorm

    rng = np.random.default_rng(5)
    B = _design(rng, 8, 20)
    B[:, :4] *= 1e6                          # covariate-scale spread: cond ~ 1e7+
    A = _design(rng, 8, 1).ravel() * 1e3
    w_gram = solve_simplex_minnorm(simplex_gram(B, A))
    w_true = np.asarray(solve_simplex_qp(B, A), dtype=float)
    obj = lambda w: float(np.sum((B @ w - A) ** 2))
    if obj(w_gram) > obj(w_true) * (1.0 + 1e-6):
        assert not simplex_point_is_optimal(B, A, w_gram)
    assert simplex_point_is_optimal(B, A, w_true)


# --------------------------------------------------------------------------- #
# 3. Scale and tolerance
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("s", [1e-4, 1.0, 1e5])
def test_the_verdict_does_not_depend_on_the_units(s):
    rng = np.random.default_rng(6)
    B, A = _design(rng, 15, 9), _design(rng, 15, 1).ravel()
    w = solve_simplex_qp(B, A)
    assert simplex_point_is_optimal(B * s, A * s, w)
    assert not simplex_point_is_optimal(B * s, A * s, np.full(9, 1.0 / 9.0))


def test_a_loose_tolerance_accepts_what_a_tight_one_rejects():
    rng = np.random.default_rng(7)
    B, A = _design(rng, 20, 6), _design(rng, 20, 1).ravel()
    w = np.full(6, 1.0 / 6.0)
    assert not simplex_point_is_optimal(B, A, w, tol=1e-8)
    assert simplex_point_is_optimal(B, A, w, tol=1e9)


# --------------------------------------------------------------------------- #
# 4. Failure modes
# --------------------------------------------------------------------------- #
def test_rejects_an_infeasible_point():
    rng = np.random.default_rng(8)
    B, A = _design(rng, 12, 4), _design(rng, 12, 1).ravel()
    assert not simplex_point_is_optimal(B, A, np.array([-0.2, 0.4, 0.4, 0.4]))
    assert not simplex_point_is_optimal(B, A, np.array([0.5, 0.5, 0.5, 0.5]))


def test_rejects_a_non_finite_point():
    rng = np.random.default_rng(9)
    B, A = _design(rng, 12, 3), _design(rng, 12, 1).ravel()
    assert not simplex_point_is_optimal(B, A, np.array([np.nan, 0.5, 0.5]))


def test_rejects_a_mismatched_length():
    rng = np.random.default_rng(10)
    B, A = _design(rng, 12, 4), _design(rng, 12, 1).ravel()
    with pytest.raises(ValueError, match="len"):
        simplex_point_is_optimal(B, A, np.array([0.5, 0.5]))


def test_rejects_a_non_matrix_design():
    with pytest.raises(ValueError, match="2-D"):
        simplex_point_is_optimal(np.ones(5), np.ones(5), np.ones(5))


def test_rejects_a_mismatched_target_length():
    with pytest.raises(ValueError, match="len"):
        simplex_point_is_optimal(np.ones((5, 3)), np.ones(4), np.ones(3) / 3.0)
