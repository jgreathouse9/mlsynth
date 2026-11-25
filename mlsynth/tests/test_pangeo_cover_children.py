"""PANGEO's second proof search: bound-fixed child LPs on the library's dual simplex, and the plain search first.

The re-optimised search (``test_pangeo_cover_reopt.py``) branches once on the unit with the fewest kept groups and
gives each branch an LP of its own. Each branch's LP is the parent LP (the kept groups, every unit, the cardinality
row, the clique rows) with bounds changed: the fixed group at 1 and every group that meets it at 0 (Pan, Linear
Programming Computation, 2nd ed., section 7.3). The basis matrix is the parent's, so each child starts from the
parent's optimal basis and its inverse and stops once its bound reaches the incumbent (Pan, Lemma 8.3.4).

Re-optimisation pays for one LP per branch. When the incumbent is far above the optimum the branches are many and
the plain search proves first, so the second search runs the plain search for a budget of nodes equal to the
children's column count and re-optimises only if that does not prove.

The oracle for the child bounds is the residual LP solved by HiGHS (:func:`columns._residual_lp`), and for the
designs the full MIP over every candidate group, as in ``test_pangeo_cover.py``.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from mlsynth.exceptions import MlsynthDataError, MlsynthEstimationError
from mlsynth.utils.pangeo_helpers import columns, cover, mip
from mlsynth.utils.pangeo_helpers.parallelism import enumerate_candidate_pairs
from mlsynth.utils.solvers import dual_simplex as ds


def _panel(seed, n, T=12, levels=5.0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, T)).cumsum(axis=1) + rng.normal(size=(n, 1)) * levels


def _setup(seed, n, Q, min_pairs=1, T=12):
    Y = _panel(seed, n, T)
    idx = np.arange(n)
    pool = columns.score_columns(idx, Y, Q)
    k = columns.min_groups(n, Q, min_pairs)
    root = columns.root_lp(pool, k)
    return Y, idx, pool, k, root


def _reference(idx, Y, Q, min_pairs=1):
    cands = enumerate_candidate_pairs(idx, Y, Q)
    return mip.solve_partition(cands, idx, min_pairs=min_pairs, return_diagnostics=True)


def _units(pool, j):
    return set(int(p) for p in pool.members[j][pool.members[j] >= 0])


def _optimum_with(pool, idx, Q, k, row):
    """The least cost over designs holding pool row ``row`` (``inf`` when none exists), by the MIP over that row and
    the groups disjoint from it."""
    g = _units(pool, row)
    rows = [j for j in range(len(pool)) if j == row or not (g & _units(pool, j))]
    try:
        _, d = mip.solve_partition(pool.candidates(rows), idx, min_pairs=k, return_diagnostics=True)
    except MlsynthEstimationError as exc:
        if not str(exc).startswith("PANGEO partition MIP did not solve"):
            raise
        return np.inf
    return d["objective_value"]


def _residual_reference(pool, keep, k, cliques, j):
    """The branch bound the HiGHS residual LP gives: its rigorous Lagrangian bound plus the fixed group's cost."""
    n = int(pool.unit_indices.size)
    G = np.zeros(n, dtype=bool)
    G[list(_units(pool, j))] = True
    left = np.array([r for r in keep if not (_units(pool, r) & _units(pool, j))], dtype=np.int64)
    units = np.flatnonzero(~G)
    kk = max(int(k) - 1, 0)
    if units.size == 0:
        return float(pool.cost_lo[j]) if kk == 0 else np.inf
    found = columns._residual_lp(pool, left, units, kk, cliques)
    if found is None:
        return np.inf
    return columns._down(found[1] + Fraction(float(pool.cost_lo[j])))


def _spy(monkeypatch):
    calls, real = [], ds.solve_boxed_lp

    def rec(*a, **kw):
        out = real(*a, **kw)
        calls.append((kw, out))
        return out

    monkeypatch.setattr(columns, "solve_boxed_lp", rec)
    return calls


# --- smoke --------------------------------------------------------------------------------------------------------

def test_smoke_the_gated_search_proves_the_full_mips_optimum():
    Y, idx, pool, k, root = _setup(1, 14, 2)
    _, d_ref = _reference(idx, Y, 2)
    z = d_ref["objective_value"]
    res = columns._search_gated(pool, root, np.arange(len(pool)), k, 2, z * 1.2, 10**7)
    assert res.status == "proved"
    assert float(pool.cost[res.rows].sum()) == pytest.approx(z, rel=1e-9)
    assert res.lower <= z * (1 + 1e-12)


# --- unit: the child LPs ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("seed,n,with_cliques", [(4, 10, False), (5, 12, True), (6, 14, True)])
def test_each_child_bound_is_the_residual_lps(seed, n, with_cliques):
    """Unit: with no incumbent to stop at, each bound-fixed child LP reaches its optimum, and its rigorous bound is
    the HiGHS residual LP's: the same LP over the same feasible set, so the two agree to rounding. A child that left
    the fixed group free, or the groups meeting it open, would solve a larger LP with a lower value."""
    Y, idx, pool, k, root = _setup(seed, n, 2)
    keep = np.arange(len(pool))
    cliques = columns._separate_cliques(pool, keep, k) if with_cliques else ()
    assert not with_cliques or cliques                                  # the fixtures with cliques have some
    for j, bound, _, _ in columns._branches(pool, root, keep, k, cliques):
        ref = _residual_reference(pool, keep, k, cliques, j)
        if np.isinf(ref):
            assert np.isinf(bound)
        else:
            assert bound == pytest.approx(ref, rel=1e-7, abs=1e-9)


def test_each_child_starts_from_the_parents_basis_and_inverse(monkeypatch):
    """Unit: one LP for the parent, from the slack basis; one per child, from the parent's optimal basis and its
    inverse, taken without a factorization."""
    calls = _spy(monkeypatch)
    Y, idx, pool, k, root = _setup(5, 12, 2)
    branches = columns._branches(pool, root, np.arange(len(pool)), k, ())
    parent_kw, parent = calls[0]
    assert "basis" not in parent_kw and parent.status == "optimal"
    children = calls[1:]
    n = int(pool.unit_indices.size)
    assert len(children) == sum(1 for j, _, left, _ in branches if left.size and len(_units(pool, j)) < n) > 0
    for kw, out in children:
        assert kw["basis"] is parent.basis
        assert out.warm_inverse_used


def test_each_child_stops_at_the_incumbent(monkeypatch):
    """Unit: each child is given the incumbent as its cutoff, in the LP's cost scale. From the optimum some children
    stop there, and every one that stops has a bound that prunes it."""
    calls = _spy(monkeypatch)
    Y, idx, pool, k, root = _setup(6, 14, 2)
    _, d_ref = _reference(idx, Y, 2)
    best = d_ref["objective_value"] * (1 + 1e-9)
    branches = columns._branches(pool, root, np.arange(len(pool)), k, (), best_up=best)
    scale = mip._cost_scale(pool.cost)
    children = calls[1:]
    assert children and all(kw["cutoff"] == pytest.approx(best * scale, rel=1e-15) for kw, _ in children)
    stopped = [out for _, out in children if out.status == "cutoff"]
    assert stopped
    bounds = {j: b for j, b, _, _ in branches}
    assert sum(1 for b in bounds.values() if b >= best) >= len(stopped)


def test_without_an_incumbent_no_child_is_cut_off(monkeypatch):
    calls = _spy(monkeypatch)
    Y, idx, pool, k, root = _setup(6, 12, 2)
    columns._branches(pool, root, np.arange(len(pool)), k, ())
    assert all(kw.get("cutoff") is None for kw, _ in calls)


@pytest.mark.parametrize("seed,n", [(4, 10), (7, 12)])
def test_every_child_bound_is_at_most_the_best_design_through_that_child(seed, n):
    """Unit: rigor. With the incumbent at the optimum, children stop early, and the bound at the point a child stops
    is still a lower bound on every design holding its group."""
    Y, idx, pool, k, root = _setup(seed, n, 2)
    _, d_ref = _reference(idx, Y, 2)
    best = d_ref["objective_value"] * (1 + 1e-9)
    keep = np.arange(len(pool))
    cliques = columns._separate_cliques(pool, keep, k)
    for j, bound, _, _ in columns._branches(pool, root, keep, k, cliques, best_up=best):
        assert bound <= _optimum_with(pool, idx, 2, k, j) * (1 + 1e-12) + 1e-12


def test_a_childs_duals_are_zero_on_the_fixed_groups_units_and_carry_its_cliques():
    """Unit: the duals handed to the child's search cover only the units left, and their cliques are the parent's
    restricted to the groups left, each with at least two members."""
    Y, idx, pool, k, root = _setup(5, 12, 2)
    keep = np.arange(len(pool))
    cliques = columns._separate_cliques(pool, keep, k)
    for j, _, left, duals in columns._branches(pool, root, keep, k, cliques):
        if duals is None:
            continue
        assert np.all(duals.pi[list(_units(pool, j))] == 0.0)
        assert all(K.size >= 2 and set(K.tolist()) <= set(left.tolist()) for K in duals.cliques)
        assert len(duals.mu) == len(duals.cliques) and np.all(duals.mu >= 0)


# --- unit: the gate ---------------------------------------------------------------------------------------------------

def test_the_budget_is_the_childrens_column_count():
    """Unit: the plain stage's node budget is the number of columns over all children, the groups each child
    keeps."""
    Y, idx, pool, k, root = _setup(5, 12, 2)
    keep = np.arange(len(pool))
    expected = sum(left.size for _, _, left, _ in columns._branches(pool, root, keep, k, ()))
    assert columns._reopt_budget(pool, keep) == expected


def test_when_the_plain_search_proves_within_the_budget_no_lp_is_solved(monkeypatch):
    """Unit: on an arm the plain search proves in fewer nodes than the budget, the gated search is the plain search
    and solves no child LP."""
    Y, idx, pool, k, root = _setup(8, 12, 2)
    keep = np.arange(len(pool))
    plain = cover.search(pool, root, keep, k, 2, np.inf, 10**7)
    assert plain.status == "proved" and plain.nodes <= columns._reopt_budget(pool, keep)
    called = []
    monkeypatch.setattr(columns, "_search_reopt", lambda *a, **kw: called.append(1))
    res = columns._search_gated(pool, root, keep, k, 2, np.inf, 10**7)
    assert not called
    assert (res.status, res.nodes, res.upper, res.lower) == (plain.status, plain.nodes, plain.upper, plain.lower)
    assert res.rows.tolist() == plain.rows.tolist()


def test_when_the_plain_search_does_not_prove_the_rest_is_reoptimised(monkeypatch):
    """Unit: a plain stage stopped by its budget hands the re-optimised search the rest of the node limit and the
    better of the two incumbents, and the gated search reports the nodes of both."""
    Y, idx, pool, k, root = _setup(9, 14, 2)
    _, d_ref = _reference(idx, Y, 2)
    z = d_ref["objective_value"]
    monkeypatch.setattr(columns, "_reopt_budget", lambda pool_, keep_: 3)
    seen, real = [], columns._search_reopt

    def rec(pool_, root_, keep_, k_, Q_, best_, limit_):
        out = real(pool_, root_, keep_, k_, Q_, best_, limit_)
        seen.append((best_, limit_, out))
        return out

    monkeypatch.setattr(columns, "_search_reopt", rec)
    keep = np.arange(len(pool))
    first = cover.search(pool, root, keep, k, 2, z * 1.5, 3)
    assert first.status == "node_limit"
    res = columns._search_gated(pool, root, keep, k, 2, z * 1.5, 10**7)
    (best_, limit_, out), = seen
    assert limit_ == 10**7 - first.nodes
    assert best_ == min(z * 1.5, first.upper)
    assert res.nodes == first.nodes + out.nodes
    assert res.status == "proved" and float(pool.cost[res.rows].sum()) == pytest.approx(z, rel=1e-9)


def test_a_design_the_plain_stage_found_is_kept_when_the_reoptimised_stage_finds_none(monkeypatch):
    """Unit: an incumbent the plain stage improved on is the design returned when the re-optimised stage proves
    nothing cheaper exists."""
    Y, idx, pool, k, root = _setup(9, 14, 2)
    _, d_ref = _reference(idx, Y, 2)
    z = d_ref["objective_value"]
    keep = np.arange(len(pool))
    budget = next(b for b in range(1, 10**6)
                  if (lambda r: r.status == "node_limit" and r.rows.size
                      and r.upper <= z * (1 + 1e-9))(cover.search(pool, root, keep, k, 2, np.inf, b)))
    monkeypatch.setattr(columns, "_reopt_budget", lambda pool_, keep_: budget)
    res = columns._search_gated(pool, root, keep, k, 2, np.inf, 10**7)
    assert res.status == "proved" and res.rows.size
    assert float(pool.cost[res.rows].sum()) == pytest.approx(z, rel=1e-9)


# --- edge -------------------------------------------------------------------------------------------------------------

def test_no_kept_groups_proves_nothing_better_than_the_incumbent():
    Y, idx, pool, k, root = _setup(10, 8, 2)
    res = columns._search_gated(pool, root, np.array([], dtype=int), k, 2, 3.5, 10)
    assert res.status == "proved" and res.rows.size == 0 and res.lower == res.upper == 3.5
    assert columns._reopt_budget(pool, np.array([], dtype=int)) == 0


def test_a_child_whose_group_covers_every_unit_is_bounded_by_its_cost():
    """Edge: with four units at Q = 2 a 2-against-2 group covers the arm; with k = 1 its child needs no LP and its
    bound is the group's cost lower bound."""
    Y, idx, pool, k, root = _setup(3, 4, 2)
    assert k == 1
    whole = set(np.flatnonzero((pool.members >= 0).sum(axis=1) == 4).tolist())
    branches = columns._branches(pool, root, np.arange(len(pool)), k, ())
    hit = [(j, b) for j, b, _, _ in branches if j in whole]
    assert hit and all(b == pytest.approx(float(pool.cost_lo[j]), rel=1e-12, abs=1e-15) for j, b in hit)


def test_a_child_that_cannot_hold_enough_groups_is_infeasible():
    """Edge: eight units with at least four groups must be four pairs; a child fixing a triple leaves five units
    for three groups, which no LP point covers (the groups' sizes sum to five, so their x sum to at most 2.5)."""
    Y, idx, pool, k, root = _setup(11, 8, 2, min_pairs=4)
    assert k == 4
    for j, bound, _, _ in columns._branches(pool, root, np.arange(len(pool)), k, ()):
        if len(_units(pool, j)) >= 3:
            assert bound == np.inf


def test_an_infeasible_parent_makes_every_child_infeasible(monkeypatch):
    """Edge: eight units in at least four groups, with only the triples and quadruples kept, have no LP point (the
    kept groups' sizes sum to eight, so their x sum to at most 8/3), and no child is solved."""
    calls = _spy(monkeypatch)
    Y, idx, pool, k, root = _setup(11, 8, 2, min_pairs=4)
    keep = np.flatnonzero((pool.members >= 0).sum(axis=1) >= 3)
    out = columns._branches(pool, root, keep, k, ())
    assert out and all(b == np.inf for _, b, _, _ in out)
    assert len(calls) == 1 and calls[0][1].status == "infeasible"


def test_a_unit_no_kept_group_covers_has_no_branch():
    """Edge: the search branches on the unit with the fewest kept groups; with none holding unit 0 there is no
    branch, and no LP."""
    Y, idx, pool, k, root = _setup(11, 8, 2)
    keep = np.flatnonzero(~(pool.members == 0).any(axis=1))
    assert columns._branches(pool, root, keep, k, ()) == []


@pytest.mark.parametrize("n,Q", [(4, 2), (5, 2), (6, 2), (6, 3)])
def test_tiny_arms(n, Q):
    Y, idx, pool, k, root = _setup(12, n, Q)
    _, d_ref = _reference(idx, Y, Q)
    res = columns._search_gated(pool, root, np.arange(len(pool)), k, Q, np.inf, 10**6)
    assert res.status == "proved"
    assert float(pool.cost[res.rows].sum()) == pytest.approx(d_ref["objective_value"], rel=1e-9, abs=1e-12)


# --- failure ----------------------------------------------------------------------------------------------------------

def test_a_node_limit_of_zero_places_no_node():
    Y, idx, pool, k, root = _setup(8, 14, 2)
    res = columns._search_gated(pool, root, np.arange(len(pool)), k, 2, np.inf, 0)
    assert res.status == "node_limit" and res.nodes == 0


def test_a_node_limit_below_the_budget_stops_the_plain_stage_and_reports_it(monkeypatch):
    """Failure: a node limit the plain stage spends leaves nothing for re-optimisation, and the result says the
    search stopped, not that it proved. The search counts the node that crosses its cap, as
    ``test_the_node_cap_is_shared_across_branches`` pins for the re-optimised search."""
    Y, idx, pool, k, root = _setup(8, 14, 2)
    called = []
    monkeypatch.setattr(columns, "_search_reopt", lambda *a, **kw: called.append(1))
    res = columns._search_gated(pool, root, np.arange(len(pool)), k, 2, np.inf, 5)
    assert res.status == "node_limit" and res.nodes <= 5 + 1 and not called


def test_a_non_finite_cost_reaching_the_child_lps_is_refused():
    """Failure: a group with a non-finite cost is routed to the MIP before any search; one reaching the child LPs
    is refused by the solver with a data error, not solved."""
    Y, idx, pool, k, root = _setup(5, 10, 2)
    cost = pool.cost.copy()
    cost[3] = np.nan
    bad = columns.ColumnPool(**{**pool.__dict__, "cost": cost}) if hasattr(pool, "__dict__") else None
    if bad is None:  # pragma: no cover - ColumnPool is a dataclass
        pytest.skip("ColumnPool has no __dict__")
    with pytest.raises(MlsynthDataError):
        columns._branches(bad, root, np.arange(len(pool)), k, ())


# --- integration ------------------------------------------------------------------------------------------------------

def _second_search_setup(monkeypatch):
    monkeypatch.setattr(cover, "START_NODE_LIMIT", 0)
    monkeypatch.setattr(cover, "SEARCH_QUICK_LIMIT", 1)
    monkeypatch.setattr(columns, "FIRST_PER_UNIT", 0)
    monkeypatch.setattr(columns, "FIRST_MIN", 1)


def test_the_proof_after_the_first_mip_is_the_gated_search(monkeypatch):
    """Integration: the second search, from the first MIP's design, is the gated one, and the diagnostics say which
    stage proved."""
    _second_search_setup(monkeypatch)
    calls, real = [], columns._search_gated

    def rec(*a, **kw):
        out = real(*a, **kw)
        calls.append(out)
        return out

    monkeypatch.setattr(columns, "_search_gated", rec)
    Y = _panel(13, 14)
    _, d = columns.solve_exact(np.arange(14), Y, 3)
    assert len(calls) == 1 and d["search_start"] == "first_mip" and calls[0].status == "proved"
    assert d["second_search"] == calls[0].stage
    assert d["objective_value"] == pytest.approx(_plain_optimum(Y, 3), rel=1e-9)


def _plain_optimum(Y, Q):
    """The optimum by the plain search from the root's duals over every group, run to its proof: an oracle that
    shares no code with the gate or the child LPs."""
    n = Y.shape[0]
    pool = columns.score_columns(np.arange(n), Y, Q)
    k = columns.min_groups(n, Q, 1)
    res = cover.search(pool, columns.root_lp(pool, k), np.arange(len(pool)), k, Q, np.inf, 10**9)
    assert res.status == "proved"
    return float(pool.cost[res.rows].sum())


@pytest.mark.parametrize("budget,stage", [(10**9, "plain"), (0, "reoptimised")])
def test_the_diagnostics_name_the_stage_that_proved(monkeypatch, budget, stage):
    """Integration: an unbounded plain budget proves in the plain stage, and none sends the proof to
    re-optimisation; both prove the optimum."""
    _second_search_setup(monkeypatch)
    monkeypatch.setattr(columns, "_reopt_budget", lambda pool_, keep_: budget)
    Y = _panel(13, 12)
    _, d = columns.solve_exact(np.arange(12), Y, 3)
    assert d["search_start"] == "first_mip" and d["second_search"] == stage
    assert d["objective_value"] == pytest.approx(_plain_optimum(Y, 3), rel=1e-9)


# --- property ---------------------------------------------------------------------------------------------------------

@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**6), n=st.integers(2, 11), Q=st.integers(1, 3), min_pairs=st.integers(1, 4),
       incumbent=st.sampled_from(["none", "optimum", "loose"]), budget=st.sampled_from([None, 0, 1, 50]))
def test_property_the_gated_search_proves_the_full_mips_optimum(seed, n, Q, min_pairs, incumbent, budget):
    """Property: over arms, sizes, cardinality rows, incumbents and plain-stage budgets (the default, none, one
    node, fifty), the gated search proves the full MIP's optimum, and its lower bound never exceeds it."""
    if not columns.feasible(n, Q, min_pairs):
        return
    Y, idx, pool, k, root = _setup(seed, n, Q, min_pairs, T=8)
    _, d_ref = _reference(idx, Y, Q, min_pairs)
    z = d_ref["objective_value"]
    best = {"none": np.inf, "optimum": z * (1 + 1e-9) + 1e-12, "loose": z * 1.1 + 1e-9}[incumbent]
    real = columns._reopt_budget
    try:
        if budget is not None:
            columns._reopt_budget = lambda pool_, keep_: budget
        res = columns._search_gated(pool, root, np.arange(len(pool)), k, Q, best, 10**7)
    finally:
        columns._reopt_budget = real
    tol = 1e-9 * max(1.0, abs(z))
    assert res.status == "proved"
    assert res.lower <= z + tol
    if incumbent != "optimum":
        assert res.rows.size and float(pool.cost[res.rows].sum()) == pytest.approx(z, rel=1e-9, abs=1e-12)


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**6), n=st.integers(4, 10), min_pairs=st.integers(1, 3),
       gap=st.sampled_from([0.0, 0.05, 0.5]), with_cliques=st.booleans())
def test_property_every_child_bound_is_rigorous(seed, n, min_pairs, gap, with_cliques):
    """Property: over arms, cardinality rows, incumbents from the optimum to half again above it, and with or
    without clique rows, every child bound (whether the child reached its optimum or stopped at the incumbent) is
    at most the best design holding the child's group."""
    if not columns.feasible(n, 2, min_pairs):
        return
    Y, idx, pool, k, root = _setup(seed, n, 2, min_pairs, T=8)
    _, d_ref = _reference(idx, Y, 2, min_pairs)
    best = d_ref["objective_value"] * (1 + gap) + 1e-12
    keep = np.arange(len(pool))
    cliques = columns._separate_cliques(pool, keep, k) if with_cliques else ()
    for j, bound, _, _ in columns._branches(pool, root, keep, k, cliques, best_up=best):
        assert bound <= _optimum_with(pool, idx, 2, k, j) * (1 + 1e-12) + 1e-12
