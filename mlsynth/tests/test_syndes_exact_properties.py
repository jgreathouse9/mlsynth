"""Generative property tests for the SYNDES inner solver and exact backend.

Reference: Doudchenko, Khosravi, Imbens et al. (2021), the two-way global
program. Once the treated set is named the program is convex, so the claims here
hold for every panel, penalty and treated set and are asserted over the domain
instead of at a fixture.

Why this layer. :mod:`mlsynth.utils.syndes_helpers.partition` is a Layer 1
helper -- pure linear algebra over a fixed Gram matrix -- which
``agents/agents_tests.md`` makes the first-choice target for property testing.
The exact backend sits a layer up, and only its differential claim is generated
here: on any instance small enough to enumerate, its answer must be the
brute-force optimum. That is a property with a known truth, which is what makes
it generable at all.

Two properties carry the weight.

The Frank-Wolfe gap is what lets every other consumer trust an answer without a
reference solver. ``value - gap`` must lower-bound the converged optimum, or the
exact backend's pruning is unsound and can discard the true optimum.

The differential claim -- exact backend equals brute force -- is the one that
would catch a search bug. Pruning, incumbent seeding and the closed-form
shortcut are all optimisations over "evaluate everything", so any of them going
wrong shows up exactly here.

Conditioning, not solver tolerance, is the limit: the closed form inverts
``G + lam I``, so the generators keep ``T >= N`` and ``lam`` away from zero, and
``assume`` discards any draw the module declines as ill-conditioned.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from mlsynth.utils.syndes_helpers.exact import solve_synthetic_design_exact
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

_entry = st.floats(min_value=-5.0, max_value=5.0,
                   allow_nan=False, allow_infinity=False)
_penalty = st.floats(min_value=1e-3, max_value=10.0,
                     allow_nan=False, allow_infinity=False)


@st.composite
def panels(draw, min_units: int = 2, max_units: int = 6) -> np.ndarray:
    """A ``(T, N)`` pre-period matrix with ``T >= N``."""
    N = draw(st.integers(min_value=min_units, max_value=max_units))
    T = draw(st.integers(min_value=N, max_value=N + 4))
    flat = draw(st.lists(_entry, min_size=T * N, max_size=T * N))
    Y = np.asarray(flat, dtype=float).reshape(T, N)
    assume(np.abs(Y).max() > 1e-6)
    return Y


@st.composite
def instances(draw):
    """A panel, a penalty, the reduced problem, and a treated-set size."""
    Y = draw(panels())
    lam = draw(_penalty)
    problem = TwoWayProblem.from_panel(Y, lam=lam)
    assume(BoundEngine.from_problem(problem).reliable)
    K = draw(st.integers(min_value=1, max_value=problem.N - 1))
    return Y, problem, K


def _all_subsets(N, K):
    return np.array(list(combinations(range(N), K)), dtype=np.int64)


# ----------------------------------------------------------------------
# the projection
# ----------------------------------------------------------------------


@settings(max_examples=200)
@given(st.lists(st.lists(_entry, min_size=1, max_size=8), min_size=1, max_size=12))
def test_projection_lands_on_the_simplex(rows):
    width = min(len(r) for r in rows)
    V = np.array([r[:width] for r in rows], dtype=float)
    P = project_simplex(V)
    assert np.abs(P.sum(axis=1) - 1.0).max() <= 1e-9
    assert P.min() >= -1e-15


@settings(max_examples=150)
@given(st.lists(st.lists(_entry, min_size=1, max_size=8), min_size=1, max_size=12))
def test_projection_is_idempotent(rows):
    width = min(len(r) for r in rows)
    P = project_simplex(np.array([r[:width] for r in rows], dtype=float))
    np.testing.assert_allclose(project_simplex(P), P, atol=1e-10)


# ----------------------------------------------------------------------
# the inner solve
# ----------------------------------------------------------------------


@settings(max_examples=120, deadline=None)
@given(instances())
def test_weights_are_feasible(inst):
    _, problem, K = inst
    t, c = split_indices(problem.N, _all_subsets(problem.N, K))
    out = solve_partition_batch(problem, t, c, max_iter=1500, tol=1e-13)
    assert np.abs(out.treated_weights.sum(axis=1) - 1.0).max() <= 1e-8
    assert np.abs(out.control_weights.sum(axis=1) - 1.0).max() <= 1e-8
    assert out.treated_weights.min() >= -1e-12
    assert out.control_weights.min() >= -1e-12


@settings(max_examples=120, deadline=None)
@given(instances())
def test_value_is_the_objective_at_the_returned_weights(inst):
    _, problem, K = inst
    subsets = _all_subsets(problem.N, K)
    t, c = split_indices(problem.N, subsets)
    out = solve_partition_batch(problem, t, c, max_iter=1500, tol=1e-13)
    for row in range(subsets.shape[0]):
        a = np.zeros(problem.N); a[t[row]] = out.treated_weights[row]
        b = np.zeros(problem.N); b[c[row]] = out.control_weights[row]
        assert abs(problem.evaluate(a, b) - out.value[row]) <= 1e-8 * max(
            1.0, abs(out.value[row]))


@settings(max_examples=120, deadline=None)
@given(instances())
def test_gap_lower_bounds_the_converged_optimum(inst):
    """``value - gap`` must bound below, or the backend's pruning is unsound."""
    _, problem, K = inst
    t, c = split_indices(problem.N, _all_subsets(problem.N, K))
    loose = solve_partition_batch(problem, t, c, max_iter=15, tol=0.0)
    tight = solve_partition_batch(problem, t, c, max_iter=4000, tol=1e-14)
    assert np.all(loose.gap >= 0.0)
    scale = max(float(np.abs(tight.value).max()), 1e-300)
    assert np.all(loose.value - loose.gap <= tight.value + 1e-7 * scale)


@settings(max_examples=120, deadline=None)
@given(instances())
def test_agrees_with_the_closed_form_where_it_is_exact(inst):
    """The seam between ``partition`` and ``gram``, with no reference solver."""
    _, problem, K = inst
    engine = BoundEngine.from_problem(problem)
    subsets = _all_subsets(problem.N, K)
    closed = engine.bound(signs_from_subsets(problem.N, subsets))
    assume(closed.exact.any())
    t, c = split_indices(problem.N, subsets)
    iterative = solve_partition_batch(problem, t, c, max_iter=4000, tol=1e-14)
    scale = max(float(np.abs(iterative.value).max()), 1e-300)
    rows = np.flatnonzero(closed.exact)
    assert np.max(np.abs(iterative.value[rows] - closed.lower[rows])) <= 1e-7 * scale


@settings(max_examples=120, deadline=None)
@given(instances())
def test_closed_form_never_exceeds_the_solved_value(inst):
    """``lb(S) <= f(S)``, with the two sides computed by different modules."""
    _, problem, K = inst
    engine = BoundEngine.from_problem(problem)
    subsets = _all_subsets(problem.N, K)
    closed = engine.bound(signs_from_subsets(problem.N, subsets))
    t, c = split_indices(problem.N, subsets)
    iterative = solve_partition_batch(problem, t, c, max_iter=4000, tol=1e-14)
    scale = max(float(np.abs(iterative.value).max()), 1e-300)
    assert np.all(closed.lower <= iterative.value + 1e-7 * scale)


@settings(max_examples=80, deadline=None)
@given(instances(), st.floats(min_value=0.25, max_value=4.0,
                              allow_nan=False, allow_infinity=False))
def test_scaling_the_panel_scales_the_value_quadratically(inst, c):
    """``Y -> cY`` with ``lam -> c^2 lam`` scales the objective by ``c^2``."""
    Y, problem, K = inst
    scaled = TwoWayProblem.from_panel(Y * c, lam=problem.lam * c * c)
    assume(BoundEngine.from_problem(scaled).reliable)
    ti, ci = split_indices(problem.N, _all_subsets(problem.N, K))
    base = solve_partition_batch(problem, ti, ci, max_iter=3000, tol=1e-14).value
    after = solve_partition_batch(scaled, ti, ci, max_iter=3000, tol=1e-14).value
    np.testing.assert_allclose(after, base * c * c, rtol=1e-5, atol=1e-12)


# ----------------------------------------------------------------------
# the exact backend, against brute force
# ----------------------------------------------------------------------


@settings(max_examples=60, deadline=None)
@given(instances())
def test_exact_backend_matches_brute_force(inst):
    """Pruning, seeding and the closed-form shortcut all fold into this claim."""
    Y, problem, K = inst
    subsets = _all_subsets(problem.N, K)
    t, c = split_indices(problem.N, subsets)
    values = solve_partition_batch(problem, t, c, max_iter=4000, tol=1e-14).value
    best = float(values.min())

    got = solve_synthetic_design_exact(Y=Y, K=K, lam=problem.lam)
    scale = max(abs(best), 1e-300)
    assert got.objective_value <= best + 1e-7 * scale


@settings(max_examples=60, deadline=None)
@given(instances())
def test_exact_backend_returns_a_consistent_design(inst):
    Y, problem, K = inst
    got = solve_synthetic_design_exact(Y=Y, K=K, lam=problem.lam)
    treated = got.assignment.astype(bool)
    assert int(treated.sum()) == K
    assert abs(got.treated_weights.sum() - 1.0) <= 1e-7
    assert abs(got.control_weights.sum() - 1.0) <= 1e-7
    assert np.abs(got.treated_weights[~treated]).max() <= 1e-12
    assert np.abs(got.control_weights[treated]).max() <= 1e-12
    assert got.objective_value == np.float64(got.objective_value)
    np.testing.assert_allclose(got.contrast_series, Y @ got.contrast_weights,
                               atol=1e-9)
