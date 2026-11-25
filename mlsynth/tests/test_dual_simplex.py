"""Tests for the bounded dual simplex (``mlsynth.utils.solvers.dual_simplex``).

The solver takes linear programs whose every variable is boxed,

    min c'x   s.t.   rl <= A x <= ru,   l <= x <= u,   all bounds finite,

and runs Koberstein's dual simplex (Algorithm 7 of "The Dual Simplex Method, Techniques for a Fast and Stable
Implementation", PhD thesis, Universitaet Paderborn, 2005) on ``A x - s = 0``. With every variable boxed, any basis
is dual feasible once the nonbasic variables sit at the bound their reduced costs call for, so there is no dual phase
I, and the Lagrangian value of any row multipliers ``y`` is a lower bound on the optimum. HiGHS (through scipy) is the
reference on the cases both solve.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from hypothesis import HealthCheck, given, settings, strategies as st
from scipy.optimize import linprog

from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.solvers import dual_simplex as ds

TOL = 1e-7


@pytest.fixture(autouse=True, scope="module", params=["numpy", "numba"])
def engine(request):
    """Every test in this module runs once per engine: the NumPy reference and the compiled kernel."""
    if request.param == "numba" and not ds._NUMBA:  # pragma: no cover - the suite runs with numba installed
        pytest.skip("numba is not installed")
    old = ds.DEFAULT_ENGINE
    ds.DEFAULT_ENGINE = request.param
    yield request.param
    ds.DEFAULT_ENGINE = old


def _highs(c, A, rl, ru, l, u):
    """The reference optimum by HiGHS, or None when the LP has no solution."""
    A = sp.csr_matrix(A)
    res = linprog(c, A_ub=sp.vstack([A, -A]).tocsr(), b_ub=np.concatenate([ru, -np.asarray(rl)]),
                  bounds=list(zip(l, u)), method="highs")
    return float(res.fun) if res.status == 0 else None


def _random_lp(rng, m, n, density=0.5, fixed=0.15, equal_rows=0.3):
    A = rng.integers(-2, 3, size=(m, n)) * (rng.random((m, n)) < density)
    l = rng.uniform(-3, 0, n)
    u = l + np.where(rng.random(n) < fixed, 0.0, rng.uniform(0.1, 3, n))
    x0 = rng.uniform(l, u)                                  # a point the rows are built around: often feasible
    act = A @ x0
    rl = act - np.where(rng.random(m) < equal_rows, 0.0, rng.uniform(0, 2, m))
    ru = np.where(rl == act, act, act + rng.uniform(0, 2, m))
    if m:
        shift = np.where(rng.random(m) < 0.15, rng.uniform(2, 6, m), 0.0)  # some rows pushed away: infeasible cases
        rl, ru = rl + shift, ru + shift
    c = rng.normal(size=n)
    return c, A.astype(float), rl, ru, l, u


def _set_partitioning(rng, units, groups, size, k, cliques=0):
    """A cover LP as PANGEO builds them: one equality row per unit, a row sum(x) >= k, and clique rows sum <= 1."""
    members = [np.sort(rng.choice(units, size=int(rng.integers(1, size + 1)), replace=False)) for _ in range(groups)]
    A = np.zeros((units, groups))
    for j, g in enumerate(members):
        A[g, j] = 1.0
    rows, rl, ru = [A], list(np.ones(units)), list(np.ones(units))
    rows.append(np.ones((1, groups)))
    rl.append(float(k))
    ru.append(float(max(groups, k)))                       # k above the group count: valid rows, no point
    for _ in range(cliques):
        r = (rng.random((1, groups)) < 0.3).astype(float)
        rows.append(r)
        rl.append(0.0)
        ru.append(1.0)
    c = rng.uniform(0.1, 5.0, groups).round(2)              # rounded: ties, as real pools have near-ties
    return c, np.vstack(rows), np.array(rl), np.array(ru), np.zeros(groups), np.ones(groups)


def _check_optimal(res, c, A, rl, ru, l, u):
    """Primal feasibility, the row identity, complementary slackness and a zero duality gap."""
    x = res.x
    act = A @ x
    assert np.all(x >= l - TOL) and np.all(x <= u + TOL)
    assert np.all(act >= rl - TOL * (1 + np.abs(rl))) and np.all(act <= ru + TOL * (1 + np.abs(ru)))
    d = c - A.T @ res.y
    np.testing.assert_allclose(res.d, d, atol=1e-8)
    free = (x > l + TOL) & (x < u - TOL)
    ranged = u > l                                          # a fixed variable's reduced cost may take either sign
    assert np.all(np.abs(d[free]) <= 1e-6)
    assert np.all(d[ranged & (x <= l + TOL)] >= -1e-6) and np.all(d[ranged & (x >= u - TOL)] <= 1e-6)
    assert res.objective == pytest.approx(float(c @ x), abs=1e-9)
    assert res.bound == pytest.approx(res.objective, abs=1e-6 * (1 + abs(res.objective)))


# --- smoke ------------------------------------------------------------------------------------------------------------

def test_smoke_a_two_variable_lp_reaches_its_known_optimum():
    """min -x1 - x2 s.t. x1 + x2 <= 1.5, 0 <= x <= 1: the optimum is -1.5."""
    c = np.array([-1.0, -1.0])
    A = np.array([[1.0, 1.0]])
    res = ds.solve_boxed_lp(c, A, [-10.0], [1.5], [0.0, 0.0], [1.0, 1.0])
    assert res.status == "optimal"
    assert res.objective == pytest.approx(-1.5, abs=1e-12)
    _check_optimal(res, c, A, np.array([-10.0]), np.array([1.5]), np.zeros(2), np.ones(2))


def test_smoke_a_set_partitioning_lp_matches_highs():
    rng = np.random.default_rng(3)
    c, A, rl, ru, l, u = _set_partitioning(rng, 8, 30, 3, 3, cliques=2)
    res = ds.solve_boxed_lp(c, A, rl, ru, l, u)
    ref = _highs(c, A, rl, ru, l, u)
    assert ref is not None and res.status == "optimal"
    assert res.objective == pytest.approx(ref, abs=1e-7 * (1 + abs(ref)))
    _check_optimal(res, c, A, rl, ru, l, u)


# --- the Lagrangian bound ---------------------------------------------------------------------------------------------

def test_the_lagrangian_bound_is_the_box_minimum_of_the_lagrangian():
    """Unit: with rows A x - s = 0, s in [rl, ru], the Lagrangian of y is (c - A'y)'x + y's, minimised over the boxes:
    sum_j min(l_j d_j, u_j d_j) + sum_i min(rl_i y_i, ru_i y_i)."""
    c = np.array([1.0, -2.0])
    A = np.array([[1.0, 1.0], [1.0, -1.0]])
    y = np.array([0.5, -1.0])
    d = c - A.T @ y                                          # (1.5, -2.5)
    want = min(0 * d[0], 1 * d[0]) + min(0 * d[1], 2 * d[1]) + min(-1 * 0.5, 3 * 0.5) + min(0 * -1.0, 4 * -1.0)
    got = ds.lagrangian_bound(c, A, np.array([-1.0, 0.0]), np.array([3.0, 4.0]), np.array([0.0, 0.0]),
                              np.array([1.0, 2.0]), y)
    assert got == pytest.approx(want, abs=1e-15)


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**6), m=st.integers(1, 6), n=st.integers(1, 8), scale=st.floats(0.01, 100.0))
def test_property_any_multipliers_give_a_bound_at_most_the_optimum(seed, m, n, scale):
    """Property: weak duality with every variable boxed. Whatever y is, its Lagrangian value is at most the optimum."""
    rng = np.random.default_rng(seed)
    c, A, rl, ru, l, u = _random_lp(rng, m, n)
    ref = _highs(c, A, rl, ru, l, u)
    if ref is None:
        return
    y = rng.normal(size=m) * scale
    assert ds.lagrangian_bound(c, A, rl, ru, l, u, y) <= ref + 1e-9 * (1 + abs(ref)) * (1 + scale)


# --- agreement with HiGHS ---------------------------------------------------------------------------------------------

@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**6), m=st.integers(1, 7), n=st.integers(1, 10))
def test_property_random_boxed_lps_agree_with_highs(seed, m, n):
    """Property: the same status as HiGHS (an optimum or none) and the same optimal value; at an optimum the point is
    feasible, complementary and has no duality gap."""
    rng = np.random.default_rng(seed)
    c, A, rl, ru, l, u = _random_lp(rng, m, n)
    ref = _highs(c, A, rl, ru, l, u)
    res = ds.solve_boxed_lp(c, A, rl, ru, l, u)
    if ref is None:
        assert res.status == "infeasible" and res.bound == np.inf
        return
    assert res.status == "optimal"
    assert res.objective == pytest.approx(ref, abs=1e-7 * (1 + abs(ref)))
    _check_optimal(res, c, A, rl, ru, l, u)


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**6), units=st.integers(3, 10), groups=st.integers(3, 40), size=st.integers(1, 4),
       k=st.integers(0, 4), cliques=st.integers(0, 5))
def test_property_set_partitioning_lps_agree_with_highs(seed, units, groups, size, k, cliques):
    """Property: on cover LPs with the k row and clique rows, which are degenerate and have near-tied costs, the same
    status and optimum as HiGHS."""
    rng = np.random.default_rng(seed)
    c, A, rl, ru, l, u = _set_partitioning(rng, units, groups, min(size, units), k, cliques)
    ref = _highs(c, A, rl, ru, l, u)
    res = ds.solve_boxed_lp(c, A, rl, ru, l, u)
    if ref is None:
        assert res.status == "infeasible"
        return
    assert res.status == "optimal"
    assert res.objective == pytest.approx(ref, abs=1e-7 * (1 + abs(ref)))
    _check_optimal(res, c, A, rl, ru, l, u)


# --- iteration limits: a valid bound at every iterate -----------------------------------------------------------------

@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**6), units=st.integers(4, 9), groups=st.integers(6, 30), k=st.integers(0, 3))
def test_property_every_iterate_bounds_the_optimum_and_the_bound_does_not_fall(seed, units, groups, k):
    """Property: stopped after j iterations, the solve returns ``iteration_limit`` (or its final status), its bound is at
    most the optimum, and the bound does not decrease with j (the dual simplex climbs the dual objective)."""
    rng = np.random.default_rng(seed)
    c, A, rl, ru, l, u = _set_partitioning(rng, units, groups, 3, k)
    full = ds.solve_boxed_lp(c, A, rl, ru, l, u)
    if full.status != "optimal":
        return
    prev = -np.inf
    for j in range(full.iterations + 1):
        res = ds.solve_boxed_lp(c, A, rl, ru, l, u, max_iterations=j)
        assert res.iterations <= j
        assert res.status == ("optimal" if j >= full.iterations else "iteration_limit") or res.status == "optimal"
        assert res.bound <= full.objective + 1e-7 * (1 + abs(full.objective))
        assert res.bound >= prev - 1e-7 * (1 + abs(full.objective))
        assert res.bound == pytest.approx(ds.lagrangian_bound(c, A, rl, ru, l, u, res.y), abs=1e-9)
        prev = res.bound


def test_zero_iterations_return_the_slack_basis_bound():
    """Edge: with no iteration the multipliers are those of the slack basis, y = 0, and the bound is
    sum_j min(l_j c_j, u_j c_j); the row asks for 2, which the slack basis's point (0, 1, 0) does not meet."""
    c = np.array([2.0, -1.0, 0.5])
    A = np.array([[1.0, 1.0, 1.0]])
    l, u = np.zeros(3), np.ones(3)
    res = ds.solve_boxed_lp(c, A, [2.0], [2.0], l, u, max_iterations=0)    # the slack basis has sum 1: infeasible
    assert res.status == "iteration_limit" and res.iterations == 0
    np.testing.assert_array_equal(res.y, [0.0])
    assert res.bound == pytest.approx(-1.0)


# --- warm starts ------------------------------------------------------------------------------------------------------

def test_a_solve_warm_started_from_its_own_optimal_basis_takes_no_iteration():
    rng = np.random.default_rng(11)
    c, A, rl, ru, l, u = _set_partitioning(rng, 9, 40, 3, 2, cliques=3)
    cold = ds.solve_boxed_lp(c, A, rl, ru, l, u)
    warm = ds.solve_boxed_lp(c, A, rl, ru, l, u, basis=cold.basis)
    assert cold.status == warm.status == "optimal" and warm.iterations == 0
    assert warm.objective == pytest.approx(cold.objective, abs=1e-12)


def _child(c, A, rl, ru, l, u, j):
    """The LP left when group j is fixed at 1 in a cover LP: its units' rows and every group meeting it removed, the
    k row lowered by one. Returns the child and the maps from child columns and rows to the parent's."""
    units = A.shape[0] - 1 - (A.shape[0] - 1 - int(np.sum(rl[: A.shape[0]] == 1.0)))
    cover = np.flatnonzero((rl == 1.0) & (ru == 1.0))
    hit = cover[A[cover, j] > 0]
    keep_cols = np.flatnonzero(~np.any(A[hit] > 0, axis=0))
    keep_rows = np.setdiff1d(np.arange(A.shape[0]), hit)
    rl2, ru2 = rl[keep_rows].copy(), ru[keep_rows].copy()
    krow = np.flatnonzero(keep_rows == cover.size)          # the k row sits right after the cover rows
    rl2[krow] = max(rl2[krow][0] - 1.0, 0.0)
    del units
    return (c[keep_cols], A[np.ix_(keep_rows, keep_cols)], rl2, ru2, l[keep_cols], u[keep_cols]), keep_cols, keep_rows


@settings(max_examples=80, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**6), units=st.integers(4, 9), groups=st.integers(8, 40), k=st.integers(1, 3))
def test_property_a_child_warm_started_from_its_parent_reaches_the_cold_optimum(seed, units, groups, k):
    """Property: a child LP (one group fixed) warm-started from the parent's basis, restricted to the columns and rows
    it keeps, reaches the cold solve's status and optimum. The restricted basis is dual feasible by construction:
    surviving columns keep their reduced costs because the removed rows' units are disjoint from them."""
    rng = np.random.default_rng(seed)
    c, A, rl, ru, l, u = _set_partitioning(rng, units, groups, 3, k)
    parent = ds.solve_boxed_lp(c, A, rl, ru, l, u)
    if parent.status != "optimal":
        return
    j = int(np.argmax(parent.x))
    child, cols, rows = _child(c, A, rl, ru, l, u, j)
    if child[1].shape[0] == 0 or child[1].shape[1] == 0:
        return
    basis = parent.basis.restrict(cols, rows)
    warm = ds.solve_boxed_lp(*child, basis=basis)
    cold = ds.solve_boxed_lp(*child)
    assert warm.status == cold.status
    if cold.status == "optimal":
        assert warm.objective == pytest.approx(cold.objective, abs=1e-7 * (1 + abs(cold.objective)))


def test_rows_added_after_a_solve_are_picked_up_from_its_basis():
    """Unit: cut rows appended to a solved LP; their slacks join the basis, which stays dual feasible, and the warm
    solve reaches the cold optimum of the extended LP."""
    rng = np.random.default_rng(5)
    c, A, rl, ru, l, u = _set_partitioning(rng, 8, 30, 3, 2)
    first = ds.solve_boxed_lp(c, A, rl, ru, l, u)
    cuts = (rng.random((3, A.shape[1])) < 0.3).astype(float)
    A2, rl2, ru2 = np.vstack([A, cuts]), np.concatenate([rl, np.zeros(3)]), np.concatenate([ru, np.ones(3)])
    warm = ds.solve_boxed_lp(c, A2, rl2, ru2, l, u, basis=first.basis.with_rows(3))
    cold = ds.solve_boxed_lp(c, A2, rl2, ru2, l, u)
    assert warm.status == cold.status == "optimal"
    assert warm.objective == pytest.approx(cold.objective, abs=1e-9)


def test_a_singular_warm_start_basis_is_repaired_with_slacks():
    """Edge: two copies of one column in the basis cannot be factored; the dependent one is replaced by a slack, the
    result reports the repair, and the solve still reaches the optimum."""
    c = np.array([1.0, 1.0, 2.0])
    A = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
    rl, ru, l, u = np.array([1.0, 0.0]), np.array([1.0, 1.0]), np.zeros(3), np.ones(3)
    bad = ds.BoxedBasis(basic=np.array([0, 1]), at_upper=np.zeros(5, dtype=bool))
    res = ds.solve_boxed_lp(c, A, rl, ru, l, u, basis=bad)
    assert res.basis_repaired and res.status == "optimal"
    assert res.objective == pytest.approx(_highs(c, A, rl, ru, l, u), abs=1e-12)


# --- edge cases -------------------------------------------------------------------------------------------------------

def test_no_rows_puts_each_variable_at_the_bound_its_cost_prefers():
    c = np.array([1.0, -1.0, 0.0])
    res = ds.solve_boxed_lp(c, np.zeros((0, 3)), [], [], [0.0, -2.0, 1.0], [3.0, 5.0, 4.0])
    assert res.status == "optimal" and res.iterations == 0
    assert res.objective == pytest.approx(0.0 * 1 + 5.0 * -1 + 0.0)
    assert res.x[0] == 0.0 and res.x[1] == 5.0


def test_an_infeasible_lp_is_reported_with_an_infinite_bound():
    """x1 + x2 >= 3 with 0 <= x <= 1 has no point: the dual is unbounded, so the bound is +inf."""
    res = ds.solve_boxed_lp(np.array([1.0, 1.0]), np.array([[1.0, 1.0]]), [3.0], [5.0], [0.0, 0.0], [1.0, 1.0])
    assert res.status == "infeasible" and res.bound == np.inf


def test_equal_costs_on_a_degenerate_cover_lp_terminate_at_the_optimum():
    """Edge: every group costs the same, so every cover with the fewest groups ties: the most degenerate case."""
    rng = np.random.default_rng(1)
    c, A, rl, ru, l, u = _set_partitioning(rng, 9, 60, 3, 0)
    c = np.ones_like(c)
    res = ds.solve_boxed_lp(c, A, rl, ru, l, u, max_iterations=10_000)
    ref = _highs(c, A, rl, ru, l, u)
    assert res.status == "optimal" and res.objective == pytest.approx(ref, abs=1e-9)


def test_a_fixed_variable_never_enters_and_keeps_its_value():
    c = np.array([-1.0, -1.0, -5.0])
    A = np.array([[1.0, 1.0, 1.0]])
    res = ds.solve_boxed_lp(c, A, [0.0], [2.0], [0.0, 0.0, 0.5], [1.0, 1.0, 0.5])
    assert res.status == "optimal" and res.x[2] == 0.5
    assert res.objective == pytest.approx(-2.5 - 1.5)


def test_sparse_and_dense_constraint_matrices_give_the_same_answer():
    rng = np.random.default_rng(9)
    c, A, rl, ru, l, u = _set_partitioning(rng, 7, 25, 3, 2, cliques=2)
    dense = ds.solve_boxed_lp(c, A, rl, ru, l, u)
    sparse = ds.solve_boxed_lp(c, sp.csc_matrix(A), rl, ru, l, u)
    assert dense.objective == sparse.objective and dense.iterations == sparse.iterations


# --- failures ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("field, value", [("l", -np.inf), ("u", np.inf), ("rl", -np.inf), ("ru", np.inf)])
def test_an_unbounded_variable_or_row_is_refused(field, value):
    """Failure: the solver is for boxed LPs only; a missing bound is a configuration error, named."""
    args = dict(c=np.array([1.0, 1.0]), A=np.array([[1.0, 1.0]]), rl=np.array([0.0]), ru=np.array([1.0]),
                l=np.zeros(2), u=np.ones(2))
    args[field] = args[field].copy()
    args[field][0] = value
    with pytest.raises(MlsynthConfigError, match="finite"):
        ds.solve_boxed_lp(args["c"], args["A"], args["rl"], args["ru"], args["l"], args["u"])


def test_a_lower_bound_above_its_upper_bound_is_a_data_error():
    with pytest.raises(MlsynthDataError, match="lower bound"):
        ds.solve_boxed_lp(np.ones(2), np.ones((1, 2)), [0.0], [1.0], [0.0, 2.0], [1.0, 1.0])
    with pytest.raises(MlsynthDataError, match="lower bound"):
        ds.solve_boxed_lp(np.ones(2), np.ones((1, 2)), [2.0], [1.0], [0.0, 0.0], [1.0, 1.0])


def test_a_non_finite_cost_or_coefficient_is_a_data_error():
    with pytest.raises(MlsynthDataError, match="finite"):
        ds.solve_boxed_lp(np.array([1.0, np.nan]), np.ones((1, 2)), [0.0], [1.0], np.zeros(2), np.ones(2))
    with pytest.raises(MlsynthDataError, match="finite"):
        ds.solve_boxed_lp(np.ones(2), np.array([[1.0, np.inf]]), [0.0], [1.0], np.zeros(2), np.ones(2))


def test_mismatched_shapes_are_refused():
    with pytest.raises(MlsynthConfigError, match="shape"):
        ds.solve_boxed_lp(np.ones(3), np.ones((1, 2)), [0.0], [1.0], np.zeros(2), np.ones(2))
    with pytest.raises(MlsynthConfigError, match="shape"):
        ds.solve_boxed_lp(np.ones(2), np.ones((1, 2)), [0.0, 0.0], [1.0], np.zeros(2), np.ones(2))


def test_a_warm_start_basis_of_the_wrong_size_is_refused():
    bad = ds.BoxedBasis(basic=np.array([0, 1, 2]), at_upper=np.zeros(4, dtype=bool))
    with pytest.raises(MlsynthConfigError, match="basis"):
        ds.solve_boxed_lp(np.ones(2), np.ones((2, 2)), [0.0, 0.0], [1.0, 1.0], np.zeros(2), np.ones(2), basis=bad)


def test_a_negative_iteration_limit_is_refused():
    with pytest.raises(MlsynthConfigError, match="max_iterations"):
        ds.solve_boxed_lp(np.ones(2), np.ones((1, 2)), [0.0], [1.0], np.zeros(2), np.ones(2), max_iterations=-1)


# --- degeneracy -------------------------------------------------------------------------------------------------------

def _stress_cover(seed, equal=True):
    """The cover LP the stress run built for ``seed`` (units 6-40, up to 400 groups, up to 30 clique rows)."""
    rng = np.random.default_rng(seed)
    units = int(rng.integers(6, 41))
    lp = _set_partitioning(rng, units, int(rng.integers(units, 401)), 4, int(rng.integers(0, 6)),
                           int(rng.integers(0, 31)))
    return (np.ones_like(lp[0]),) + lp[1:] if equal else lp


@pytest.mark.parametrize("seed", [9, 14])
def test_equal_cost_cover_lps_that_stalled_reach_the_optimum(seed):
    """Edge, regression: with every cost equal the cover LP is dual degenerate at every basis. An earlier version
    flipped the ratio test's passed breakpoints explicitly, zero-ratio ones included, which moved the primal point
    without moving the dual objective; these two stalled at the 39,000-iteration cap. They reach HiGHS's optimum."""
    lp = _stress_cover(seed)
    res = ds.solve_boxed_lp(*lp)
    ref = _highs(*lp)
    assert res.status == "optimal" and ref is not None
    assert res.objective == pytest.approx(ref, abs=1e-9)


@settings(max_examples=120, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**6), stall=st.sampled_from([1, 2, 5]), cover=st.booleans(), equal=st.booleans())
def test_property_the_degeneracy_stages_never_change_the_answer(seed, stall, cover, equal):
    """Property: with the stall window shrunk so the costs are perturbed, restored and Bland's rule takes over within a
    few iterations, every LP still ends with HiGHS's status and optimum, and the bound is the Lagrangian value of the
    returned multipliers under the true costs."""
    rng = np.random.default_rng(seed)
    if cover:
        units = int(rng.integers(4, 12))
        lp = _set_partitioning(rng, units, int(rng.integers(units, 60)), 3, int(rng.integers(0, 4)),
                               int(rng.integers(0, 6)))
        if equal:
            lp = (np.ones_like(lp[0]),) + lp[1:]
    else:
        lp = _random_lp(rng, int(rng.integers(1, 8)), int(rng.integers(1, 12)))
    old = ds.STALL_AFTER
    ds.STALL_AFTER = stall
    try:
        res = ds.solve_boxed_lp(*lp)
    finally:
        ds.STALL_AFTER = old
    ref = _highs(*lp)
    if ref is None:
        assert res.status == "infeasible"
        return
    assert res.status == "optimal"
    assert res.objective == pytest.approx(ref, abs=1e-7 * (1 + abs(ref)))
    assert res.bound == pytest.approx(ds.lagrangian_bound(*lp, res.y), abs=1e-12)
    _check_optimal(res, *lp)


def _equal_cover(seed):
    rng = np.random.default_rng(seed)
    units = int(rng.integers(4, 12))
    lp = _set_partitioning(rng, units, int(rng.integers(units, 60)), 3, int(rng.integers(0, 4)), int(rng.integers(0, 6)))
    return (np.ones_like(lp[0]),) + lp[1:]


def _with(constants, fn):
    """Run ``fn`` with module constants of the solver set, restoring them after."""
    old = {k: getattr(ds, k) for k in constants}
    for k, v in constants.items():
        setattr(ds, k, v)
    try:
        return fn()
    finally:
        for k, v in old.items():
            setattr(ds, k, v)


def test_the_result_reports_which_degeneracy_stage_the_solve_reached(engine):
    """Unit: solves that never stall report ``"none"``, the equal-cost cover of the regression test included. With the
    stall window at zero the first iteration that does not raise the dual objective perturbs the costs, and once the
    perturbed optimum is found the solve reports ``"restored"``. With a perturbation as large as the costs (psi = 1)
    the perturbed optimum lies far from the true one and the restored phase stalls into Bland's rule; the engines take
    different paths through ties, so the cover that reaches it is pinned per engine (seeds 5, 7 and 26 for the NumPy
    reference, 4, 10 and 17 for the kernel). Each reaches HiGHS's optimum."""
    rng = np.random.default_rng(3)
    assert ds.solve_boxed_lp(*_set_partitioning(rng, 8, 30, 3, 3, cliques=2)).degeneracy_stage == "none"
    eq = _stress_cover(9)
    assert ds.solve_boxed_lp(*eq).degeneracy_stage == "none"
    res = _with({"STALL_AFTER": 0}, lambda: ds.solve_boxed_lp(*eq))
    assert res.status == "optimal" and res.degeneracy_stage == "restored"
    assert res.objective == pytest.approx(_highs(*eq), abs=1e-9)
    lp = _equal_cover({"numpy": 5, "numba": 4}[engine])
    res = _with({"STALL_AFTER": 0, "PERTURB_REL": 1.0}, lambda: ds.solve_boxed_lp(*lp))
    assert res.status == "optimal" and res.degeneracy_stage == "bland"
    assert res.objective == pytest.approx(_highs(*lp), abs=1e-9)


@settings(max_examples=120, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**6), stall=st.sampled_from([0, 1, 300]), rel=st.sampled_from([0.0, 1e-5, 0.1, 1.0]))
def test_property_the_perturbation_size_never_changes_the_answer(seed, stall, rel):
    """Property: on equal-cost cover LPs, whatever the perturbation's size and the stall window, the solve ends with
    HiGHS's status and optimum."""
    lp = _equal_cover(seed)
    res = _with({"STALL_AFTER": stall, "PERTURB_REL": rel}, lambda: ds.solve_boxed_lp(*lp))
    ref = _highs(*lp)
    if ref is None:
        assert res.status == "infeasible"
        return
    assert res.status == "optimal" and res.objective == pytest.approx(ref, abs=1e-7 * (1 + abs(ref)))


@settings(max_examples=80, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**6), every=st.sampled_from([1, 2, 7]), check=st.booleans())
def test_property_refactoring_never_changes_the_answer(seed, every, check):
    """Property: refactoring the basis every 1, 2 or 7 iterations, or after every pivot element test (its tolerance
    set below zero so the test always fails on an updated inverse), gives HiGHS's status and optimum."""
    rng = np.random.default_rng(seed)
    units = int(rng.integers(4, 12))
    lp = _set_partitioning(rng, units, int(rng.integers(units, 50)), 3, int(rng.integers(0, 4)), int(rng.integers(0, 5)))
    old = ds.REFACTOR_EVERY, ds.PIVOT_CHECK_TOL
    ds.REFACTOR_EVERY = every
    if check:
        ds.PIVOT_CHECK_TOL = -1.0
    try:
        res = ds.solve_boxed_lp(*lp)
    finally:
        ds.REFACTOR_EVERY, ds.PIVOT_CHECK_TOL = old
    ref = _highs(*lp)
    if ref is None:
        assert res.status == "infeasible"
        return
    assert res.status == "optimal" and res.objective == pytest.approx(ref, abs=1e-7 * (1 + abs(ref)))


def test_a_warm_start_on_an_lp_without_rows_is_accepted():
    res = ds.solve_boxed_lp(np.array([1.0, -1.0]), np.zeros((0, 2)), [], [], np.zeros(2), np.ones(2),
                            basis=ds.BoxedBasis(np.zeros(0, dtype=np.int64), np.array([True, False])))
    assert res.status == "optimal" and res.objective == pytest.approx(-1.0) and not res.basis_repaired


def test_a_cost_matrix_in_place_of_a_cost_vector_is_refused():
    with pytest.raises(MlsynthConfigError, match="vector"):
        ds.solve_boxed_lp(np.ones((1, 2)), np.ones((1, 2)), [0.0], [1.0], np.zeros(2), np.ones(2))


# --- the second round: stalls on distinct costs, iteration ceilings, certified infeasibility, cutoffs -------------------

def _tied_cover(seed):
    """A small cover LP with integer costs 1-3, so with ties."""
    rng = np.random.default_rng(seed)
    units = int(rng.integers(4, 9))
    groups = int(rng.integers(units, 13))
    lp = _set_partitioning(rng, units, groups, 3, int(rng.integers(0, 3)), int(rng.integers(0, 3)))
    return (rng.integers(1, 4, groups).astype(float),) + lp[1:]


def test_a_stall_on_distinct_enough_costs_perturbs_them():
    """Unit: on tied costs, with the stall window at zero, the first iteration that does not raise the dual objective
    perturbs them, and the solve still ends at the optimum after restoring them."""
    lp = _tied_cover(1)
    res = _with({"STALL_AFTER": 0}, lambda: ds.solve_boxed_lp(*lp))
    assert res.degeneracy_stage == "restored" and res.status == "optimal"
    assert res.objective == pytest.approx(_highs(*lp), abs=1e-9)


def test_equal_cost_covers_take_few_iterations():
    """Unit, a ceiling: the forty equal-cost covers of the stress run take 2,485 iterations in all. With Harris'
    ratio test taking the smallest pivot in its window they take 9,113, stalling into the perturbed stage; with the
    ratio test's explicit flips of zero-ratio breakpoints (an earlier version) and no perturbation, seeds 9 and 14
    alone reached the 39,000-iteration cap. The ceiling is 4,000."""
    assert sum(ds.solve_boxed_lp(*_stress_cover(s)).iterations for s in range(40)) <= 4000


def test_the_steepest_edge_weights_keep_their_cross_term():
    """Unit, a ceiling: the dual steepest-edge weights are updated by Koberstein (2005), equation 3.50, whose
    cross term -2 (alpha_iq / alpha_rq) tau_i keeps each weight equal to the squared norm of its row of the basis
    inverse. The forty equal-cost covers take 2,485 iterations on the numpy path and 2,377 in the kernel. Dropped
    from the update, the cross term lets the weights drift from those norms with each pivot, and the covers take
    3,580 and 3,664, with the same answers: the weights steer the path, not the optimum. Over 200 covers the kernel
    took 12,700 iterations with the term and 19,400 without, more in 156 of them and fewer in 32. The ceiling is
    3,000."""
    assert sum(ds.solve_boxed_lp(*_stress_cover(s)).iterations for s in range(40)) <= 3000


def test_a_solve_stopped_while_perturbed_bounds_with_the_true_costs():
    """Unit: stopped after three iterations, the last on costs a stall has perturbed, the bound is the Lagrangian
    value of the multipliers under the true costs, and it bounds the true optimum."""
    lp = _stress_cover(9)
    res = _with({"STALL_AFTER": 0}, lambda: ds.solve_boxed_lp(*lp, max_iterations=3))
    assert res.status == "iteration_limit" and res.degeneracy_stage == "perturbed"
    assert res.bound == pytest.approx(ds.lagrangian_bound(*lp, res.y), abs=1e-12)
    assert res.bound <= _highs(*lp) + 1e-9


def test_blands_entering_rule_takes_the_smallest_index_among_the_tied_ratios():
    cand = np.array([9, 4, 7, 2, 5])
    ratio = np.array([0.5, 0.2, 0.2 + 1e-14, 0.3, 0.2])
    assert ds._bland_entering(cand, ratio) == 1                     # index 4 among the ties 4, 7, 5
    assert ds._bland_entering(np.array([3]), np.array([1.0])) == 0


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**6), m=st.integers(1, 7), n=st.integers(1, 10))
def test_property_infeasibility_is_reported_only_with_a_verified_ray(seed, m, n):
    """Property: ``infeasible`` comes with a ray whose Lagrangian under zero costs is positive (a Farkas certificate
    for the boxed LP: Bazaraa, Jarvis and Sherali, Linear Programming and Network Flows, Lemma 5.1; Pan, Linear
    Programming Computation, Lemma 8.3.5), HiGHS agrees there is no point, and a solve with a point has no ray."""
    rng = np.random.default_rng(seed)
    lp = _random_lp(rng, m, n)
    res = ds.solve_boxed_lp(*lp)
    if res.status == "infeasible":
        assert res.ray is not None and _highs(*lp) is None
        assert ds.lagrangian_bound(np.zeros(n), *lp[1:], res.ray) > 0
    else:
        assert res.ray is None


def test_an_unverified_ray_is_not_reported_as_infeasibility():
    """Edge: with the margin raised past any ray's Lagrangian, the infeasible LP of the earlier test is not reported
    infeasible; the solve returns its multipliers' bound with ``iteration_limit``, so a caller cannot prune on it."""
    lp = (np.array([1.0, 1.0]), np.array([[1.0, 1.0]]), [3.0], [5.0], [0.0, 0.0], [1.0, 1.0])
    res = _with({"INFEASIBILITY_MARGIN": 1e300}, lambda: ds.solve_boxed_lp(*lp))
    assert res.status == "iteration_limit" and np.isfinite(res.bound) and res.ray is None


@settings(max_examples=120, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**6), units=st.integers(4, 10), groups=st.integers(6, 40), k=st.integers(0, 3),
       below=st.floats(1e-6, 0.5))
def test_property_a_cutoff_stops_the_solve_once_the_bound_reaches_it(seed, units, groups, k, below):
    """Property: a cutoff under the optimum stops the solve at the first iterate whose bound reaches it, with
    ``cutoff`` (or ``optimal`` when that iterate is the optimal basis); a cutoff above the optimum is never reached,
    and the solve ends ``optimal``."""
    rng = np.random.default_rng(seed)
    lp = _set_partitioning(rng, units, groups, 3, k)
    full = ds.solve_boxed_lp(*lp)
    if full.status != "optimal":
        return
    gap = below * (1.0 + abs(full.objective))
    low = ds.solve_boxed_lp(*lp, cutoff=full.objective - gap)
    assert low.status in ("cutoff", "optimal") and low.bound >= full.objective - gap - 1e-12
    first = next((j for j in range(full.iterations)
                  if ds.solve_boxed_lp(*lp, max_iterations=j).bound >= full.objective - gap), None)
    if first is None:                                      # first met at the optimal basis itself
        assert low.status == "optimal" and low.iterations == full.iterations
    else:
        assert low.status == "cutoff" and low.iterations == first
    high = ds.solve_boxed_lp(*lp, cutoff=full.objective + gap)
    assert high.status == "optimal" and high.objective == pytest.approx(full.objective, abs=1e-12)


def test_a_cutoff_that_is_not_a_number_is_refused():
    with pytest.raises(MlsynthConfigError, match="cutoff"):
        ds.solve_boxed_lp(np.ones(2), np.ones((1, 2)), [0.0], [1.0], np.zeros(2), np.ones(2), cutoff=float("nan"))



# --- engines: the compiled kernel against the NumPy reference --------------------------------------------------------

@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**6), cover=st.booleans())
def test_property_the_engines_agree(seed, cover):
    """Property: the compiled kernel and the NumPy reference give the same status and the same optimum, and each
    returns the Lagrangian value of its own multipliers as its bound."""
    rng = np.random.default_rng(seed)
    if cover:
        units = int(rng.integers(4, 16))
        lp = _set_partitioning(rng, units, int(rng.integers(units, 80)), 4, int(rng.integers(0, 4)),
                               int(rng.integers(0, 8)))
    else:
        lp = _random_lp(rng, int(rng.integers(1, 12)), int(rng.integers(1, 30)))
    a = ds.solve_boxed_lp(*lp, engine="numpy")
    b = ds.solve_boxed_lp(*lp, engine="numba")
    assert a.engine == "numpy" and b.engine == "numba"
    assert a.status == b.status
    if a.status == "optimal":
        assert b.objective == pytest.approx(a.objective, abs=1e-9 * (1 + abs(a.objective)))
    for r in (a, b):
        if r.status != "infeasible":
            assert r.bound == pytest.approx(ds.lagrangian_bound(*lp, r.y), abs=1e-12)


def test_auto_takes_the_compiled_kernel_when_numba_is_installed_and_numpy_when_not():
    lp = _set_partitioning(np.random.default_rng(2), 6, 20, 3, 2)
    assert ds.solve_boxed_lp(*lp, engine="auto").engine == ("numba" if ds._NUMBA else "numpy")
    saved = ds._NUMBA
    ds._NUMBA = False
    try:
        assert ds.solve_boxed_lp(*lp, engine="auto").engine == "numpy"
        with pytest.raises(MlsynthConfigError, match="numba"):
            ds.solve_boxed_lp(*lp, engine="numba")
    finally:
        ds._NUMBA = saved


def test_an_unknown_engine_is_refused():
    with pytest.raises(MlsynthConfigError, match="engine"):
        ds.solve_boxed_lp(np.ones(2), np.ones((1, 2)), [0.0], [1.0], np.zeros(2), np.ones(2), engine="glpk")


def _bound_fixed_child(lp, parent):
    """The child of Pan's section 7.3: the parent LP with its largest group fixed at 1 and every group sharing a unit
    with it fixed at 0, bounds changed and rows kept, so the parent's basis is still a basis."""
    c, A, rl, ru, l, u = lp
    j = int(np.argmax(parent.x))
    cover = np.flatnonzero((rl == 1.0) & (ru == 1.0))
    hit = cover[A[cover, j] > 0]
    clash = np.any(A[hit] > 0, axis=0)
    lo, up = l.copy(), np.where(clash, 0.0, u)
    lo[j] = up[j] = 1.0
    return c, A, rl, ru, lo, up


def test_a_bound_fixed_child_reuses_the_parents_inverse_and_reaches_the_cold_optimum():
    """Unit: the parent's basis carries its inverse; a child that only changes bounds keeps the same basis matrix, so
    the solve starts from that inverse without refactoring (``warm_inverse_used``) and reaches the cold optimum."""
    rng = np.random.default_rng(11)
    lp = _set_partitioning(rng, 9, 40, 3, 2, cliques=3)
    parent = ds.solve_boxed_lp(*lp)
    assert parent.basis.inverse is not None
    child = _bound_fixed_child(lp, parent)
    warm = ds.solve_boxed_lp(*child, basis=parent.basis)
    cold = ds.solve_boxed_lp(*child)
    assert warm.warm_inverse_used and not cold.warm_inverse_used
    assert warm.status == cold.status
    if cold.status == "optimal":
        assert warm.objective == pytest.approx(cold.objective, abs=1e-9)


def test_a_stale_inverse_is_detected_and_refactored():
    """Edge: an inverse carried from a different LP fails the residual check B (B^-1 v) = v on one vector; the solve
    refactors (``warm_inverse_used`` is false) and still reaches the optimum."""
    rng = np.random.default_rng(4)
    lp1 = _set_partitioning(rng, 8, 30, 3, 2)
    lp2 = _set_partitioning(rng, 8, 30, 3, 2)
    first = ds.solve_boxed_lp(*lp1)
    res = ds.solve_boxed_lp(*lp2, basis=first.basis)
    assert not res.warm_inverse_used
    ref = _highs(*lp2)
    if ref is not None:
        assert res.status == "optimal" and res.objective == pytest.approx(ref, abs=1e-9)


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**6), units=st.integers(4, 10), groups=st.integers(8, 40), k=st.integers(1, 3))
def test_property_bound_fixed_children_warm_started_from_the_parent_agree_with_highs(seed, units, groups, k):
    """Property: every bound-fixed child solved from the parent's basis and inverse has HiGHS's status and optimum."""
    rng = np.random.default_rng(seed)
    lp = _set_partitioning(rng, units, groups, 3, k, cliques=int(rng.integers(0, 4)))
    parent = ds.solve_boxed_lp(*lp)
    if parent.status != "optimal":
        return
    child = _bound_fixed_child(lp, parent)
    res = ds.solve_boxed_lp(*child, basis=parent.basis)
    ref = _highs(*child)
    if ref is None:
        assert res.status == "infeasible"
    else:
        assert res.status == "optimal" and res.objective == pytest.approx(ref, abs=1e-7 * (1 + abs(ref)))
