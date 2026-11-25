"""PANGEO's proof search with duals re-optimised one level down.

The search branches once on the unit with the fewest kept groups. Each branch fixes one group, and the cover
problem left over (the other units, the groups disjoint from it, k - 1 groups) gets its own LP. That LP's
Lagrangian bound, evaluated rigorously, prunes the branch when it and the fixed group's cost reach the incumbent;
otherwise the branch is searched with the LP's duals. Measured from the optimum at Q = 2: 2.89 million nodes to
0.88 million at 40 geos, 10.35 million to 6.15 million at 50.

The oracle is the full MIP over every candidate group, as in ``test_pangeo_cover.py``.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from mlsynth.exceptions import MlsynthEstimationError
from mlsynth.utils.pangeo_helpers import columns, cover, mip
from mlsynth.utils.pangeo_helpers.parallelism import enumerate_candidate_pairs


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


def _reference(idx, Y, Q, min_pairs=1, **kw):
    cands = enumerate_candidate_pairs(idx, Y, Q, **kw)
    return mip.solve_partition(cands, idx, min_pairs=min_pairs, return_diagnostics=True)


def _optimum_with(pool, idx, Y, Q, k, row):
    """The least cost over designs holding pool row ``row``: the MIP over that row and the groups disjoint from
    it, with the row forced by being the only group holding its first unit. A design that cannot exist is ``inf``;
    any other failure (no mixed-integer solver installed, a solver error) is raised, so it is never read as one."""
    g = set(int(p) for p in pool.members[row][pool.members[row] >= 0])
    rows = [j for j in range(len(pool)) if j == row or not (g & set(int(p) for p in pool.members[j][pool.members[j] >= 0]))]
    try:
        _, d = mip.solve_partition(pool.candidates(rows), idx, min_pairs=k, return_diagnostics=True)
    except MlsynthEstimationError as exc:
        if not str(exc).startswith("PANGEO partition MIP did not solve"):
            raise
        return np.inf
    return d["objective_value"]


def _design_units(pool, rows):
    return sorted(int(p) for j in rows for p in pool.members[j][pool.members[j] >= 0])


# --- smoke --------------------------------------------------------------------------------------------------------

def test_smoke_the_reoptimised_search_proves_the_full_mips_optimum():
    Y, idx, pool, k, root = _setup(1, 14, 2)
    res = columns._search_reopt(pool, root, np.arange(len(pool)), k, 2, np.inf, 10**7)
    _, d_ref = _reference(idx, Y, 2)
    assert res.status == "proved"
    assert _design_units(pool, res.rows) == list(range(14))
    z = float(pool.cost[res.rows].sum())
    assert z == pytest.approx(d_ref["objective_value"], rel=1e-9)
    assert res.lower <= d_ref["objective_value"] * (1 + 1e-12) <= res.upper * (1 + 1e-12) + 1e-12


# --- unit: the fixed-group search -------------------------------------------------------------------------------------

def test_a_search_with_a_fixed_group_covers_only_the_other_units():
    """Unit: cover.search with ``fixed`` starts with that group's units covered and its cost counted; the design it
    returns holds the fixed group and covers every unit once."""
    Y, idx, pool, k, root = _setup(2, 12, 2)
    j = int(np.argmin(pool.cost))
    g = set(int(p) for p in pool.members[j][pool.members[j] >= 0])
    keep = np.array([r for r in range(len(pool))
                     if not (g & set(int(p) for p in pool.members[r][pool.members[r] >= 0]))])
    res = cover.search(pool, root, keep, k, 2, np.inf, 10**7, fixed=(j,))
    assert res.status == "proved" and j in set(res.rows.tolist())
    assert _design_units(pool, res.rows) == list(range(12))
    assert float(pool.cost[res.rows].sum()) == pytest.approx(_optimum_with(pool, idx, Y, 2, k, j), rel=1e-9)



def test_a_fixed_group_search_against_a_tight_incumbent_still_finds_the_design():
    """Unit: against an incumbent just above the best design holding the fixed group, the search still finds that
    design, so its bound counts only the uncovered units' duals; one that also counted the fixed group's units
    would prune it."""
    Y, idx, pool, k, root = _setup(2, 12, 2)
    j = int(np.argmin(pool.cost))
    g = set(int(p) for p in pool.members[j][pool.members[j] >= 0])
    keep = np.array([r for r in range(len(pool))
                     if not (g & set(int(p) for p in pool.members[r][pool.members[r] >= 0]))])
    z = _optimum_with(pool, idx, Y, 2, k, j)
    assert root.pi[list(g)].sum() > 1e-6 * z                         # the fixed group's duals matter here
    res = cover.search(pool, root, keep, k, 2, z * (1 + 1e-9) + 1e-12, 10**7, fixed=(j,))
    assert res.rows.size and float(pool.cost[res.rows].sum()) == pytest.approx(z, rel=1e-9)

@pytest.mark.parametrize("seed,n", [(2, 12), (5, 14)])
def test_a_fixed_group_whose_cost_reaches_the_incumbent_prunes_at_the_first_level(seed, n):
    """Unit: below a fixed group whose cost alone reaches the incumbent, every group placed at the first level is
    pruned there, so the search places at most as many nodes as there are kept groups holding the unit it branches
    on first, and returns no design. A bound that leaves the fixed group's cost out is still valid, but it searched
    the whole subtree: 42,735 nodes against 129 at 12 geos, 1.2 million against 231 at 14."""
    Y, idx, pool, k, root = _setup(seed, n, 2)
    j = int(np.argmax(pool.cost))
    g = set(int(p) for p in pool.members[j][pool.members[j] >= 0])
    keep = np.array([r for r in range(len(pool))
                     if not (g & set(int(p) for p in pool.members[r][pool.members[r] >= 0]))])
    holders = [sum(1 for r in keep if u in set(int(p) for p in pool.members[r][pool.members[r] >= 0]))
               for u in range(n) if u not in g]
    res = cover.search(pool, root, keep, k, 2, float(pool.cost[j]), 10**7, fixed=(j,))
    assert res.status == "proved" and res.rows.size == 0
    assert res.nodes <= min(holders)


def test_a_fixed_group_that_covers_everything_is_the_design():
    """Edge: with Q large enough that one group holds every unit, fixing it leaves nothing to search."""
    Y, idx, pool, k, root = _setup(3, 4, 2)
    j = int(np.flatnonzero((pool.members >= 0).sum(axis=1) == 4)[0])
    res = cover.search(pool, root, np.array([], dtype=int), 1, 2, np.inf, 10, fixed=(j,))
    assert res.status == "proved" and res.rows.tolist() == [j] and res.nodes == 0



@pytest.mark.parametrize("k,best", [(2, np.inf), (1, 0.0)])
def test_a_fixed_group_that_covers_everything_but_cannot_be_the_design(k, best):
    """Edge: a fixed group holding every unit is no design when the cardinality row asks for more groups, or when
    it does not beat the incumbent."""
    Y, idx, pool, _, root = _setup(3, 4, 2)
    j = int(np.flatnonzero((pool.members >= 0).sum(axis=1) == 4)[0])
    res = cover.search(pool, root, np.array([], dtype=int), k, 2, best, 10, fixed=(j,))
    assert res.status == "proved" and res.rows.size == 0 and res.upper == best

# --- unit: the branch bounds ------------------------------------------------------------------------------------------

def test_every_branch_bound_is_at_most_the_best_design_through_that_branch():
    """Unit: the rigorous residual bound plus the fixed group's cost is a lower bound on every design holding the
    group."""
    Y, idx, pool, k, root = _setup(4, 10, 2)
    keep = np.arange(len(pool))
    for row, bound in columns._branch_bounds(pool, root, keep, k, 2):
        assert bound <= _optimum_with(pool, idx, Y, 2, k, row) * (1 + 1e-12) + 1e-12


def test_the_branch_bounds_are_at_least_the_root_bound():
    """Unit: each branch's LP is a restriction of the root LP with one variable fixed at 1, so its bound is not
    below the root's Lagrangian bound (up to rounding)."""
    Y, idx, pool, k, root = _setup(5, 12, 2)
    for _, bound in columns._branch_bounds(pool, root, np.arange(len(pool)), k, 2):
        assert bound >= root.bound - 1e-9 * max(1.0, abs(root.bound))


def test_a_branch_whose_bound_reaches_the_incumbent_is_not_searched(monkeypatch):
    """Unit: from the optimum as incumbent, a branch whose bound reaches it places no node."""
    Y, idx, pool, k, root = _setup(6, 14, 2)
    _, d_ref = _reference(idx, Y, 2)
    best = d_ref["objective_value"] * (1 + 1e-9)
    bounds = dict(columns._branch_bounds(pool, root, np.arange(len(pool)), k, 2))
    pruned = {r for r, b in bounds.items() if b >= best}
    assert pruned                                                     # the fixture has some
    searched, real = [], cover.search

    def rec(pool_, root_, keep_, k_, Q_, best_, limit_, fixed=()):
        searched.append(tuple(fixed))
        return real(pool_, root_, keep_, k_, Q_, best_, limit_, fixed=fixed)

    monkeypatch.setattr(cover, "search", rec)
    columns._search_reopt(pool, root, np.arange(len(pool)), k, 2, best, 10**7)
    assert not ({f[0] for f in searched if f} & pruned)



def test_each_branch_is_searched_with_its_own_residual_duals(monkeypatch):
    """Unit: a branch's subtree is searched with its residual LP's duals, which are zero on the fixed group's units,
    not with the root's."""
    Y, idx, pool, k, root = _setup(4, 12, 2)
    seen, real = [], cover.search

    def rec(pool_, duals_, keep_, k_, Q_, best_, limit_, fixed=()):
        seen.append((duals_, tuple(fixed)))
        return real(pool_, duals_, keep_, k_, Q_, best_, limit_, fixed=fixed)

    monkeypatch.setattr(cover, "search", rec)
    columns._search_reopt(pool, root, np.arange(len(pool)), k, 2, np.inf, 10**7)
    assert seen
    for duals, fixed in seen:
        assert duals is not root
        g = pool.members[fixed[0]][pool.members[fixed[0]] >= 0]
        assert np.all(duals.pi[g] == 0.0)

def test_the_reoptimised_search_places_fewer_nodes():
    """Unit: from the optimum on a 16-geo arm at Q = 2, re-optimised duals place fewer nodes than the root's."""
    Y, idx, pool, k, root = _setup(7, 16, 2, T=20)
    _, d_ref = _reference(idx, Y, 2)
    best = d_ref["objective_value"] * (1 + 1e-9)
    keep = np.flatnonzero(root.reduced_costs <= columns._up(Fraction(best) - Fraction(root.bound)))
    plain = cover.search(pool, root, keep, k, 2, best, 10**8)
    reopt = columns._search_reopt(pool, root, keep, k, 2, best, 10**8)
    assert plain.status == reopt.status == "proved"
    assert reopt.nodes < plain.nodes


def test_the_node_cap_is_shared_across_branches():
    Y, idx, pool, k, root = _setup(8, 14, 2)
    res = columns._search_reopt(pool, root, np.arange(len(pool)), k, 2, np.inf, 5)
    assert res.status == "node_limit" and res.nodes <= 5 + 1



def test_a_node_limit_of_zero_places_no_node():
    Y, idx, pool, k, root = _setup(8, 14, 2)
    res = columns._search_reopt(pool, root, np.arange(len(pool)), k, 2, np.inf, 0)
    assert res.status == "node_limit" and res.nodes == 0

def test_the_incumbent_carries_from_branch_to_branch(monkeypatch):
    """Unit: a design found in one branch is the incumbent the next branch is searched against."""
    Y, idx, pool, k, root = _setup(9, 12, 2)
    seen, real = [], cover.search

    def rec(pool_, root_, keep_, k_, Q_, best_, limit_, fixed=()):
        out = real(pool_, root_, keep_, k_, Q_, best_, limit_, fixed=fixed)
        seen.append((best_, out.upper))
        return out

    monkeypatch.setattr(cover, "search", rec)
    columns._search_reopt(pool, root, np.arange(len(pool)), k, 2, np.inf, 10**7)
    for (b0, u0), (b1, _) in zip(seen, seen[1:]):
        assert b1 <= min(b0, u0)


# --- edge -------------------------------------------------------------------------------------------------------------

def test_no_kept_groups_proves_nothing_better_than_the_incumbent():
    Y, idx, pool, k, root = _setup(10, 8, 2)
    res = columns._search_reopt(pool, root, np.array([], dtype=int), k, 2, 3.5, 10)
    assert res.status == "proved" and res.rows.size == 0 and res.lower == res.upper == 3.5


def test_a_unit_no_kept_group_covers_has_no_design():
    Y, idx, pool, k, root = _setup(11, 8, 2)
    keep = np.flatnonzero(~(pool.members == 0).any(axis=1))
    res = columns._search_reopt(pool, root, keep, k, 2, np.inf, 10**6)
    assert res.status == "proved" and res.rows.size == 0


@pytest.mark.parametrize("n,Q", [(2, 1), (4, 2), (5, 2), (6, 3)])
def test_tiny_arms(n, Q):
    Y, idx, pool, k, root = _setup(12, n, Q)
    res = columns._search_reopt(pool, root, np.arange(len(pool)), k, Q, np.inf, 10**6)
    _, d_ref = _reference(idx, Y, Q)
    assert res.status == "proved"
    assert float(pool.cost[res.rows].sum()) == pytest.approx(d_ref["objective_value"], rel=1e-9, abs=1e-12)


# --- integration ------------------------------------------------------------------------------------------------------

def test_the_proof_after_the_first_mip_reoptimises_the_duals(monkeypatch):
    """Integration: with no plain-stage budget, the second search, from the first MIP's design, is the re-optimised
    one (``test_pangeo_cover_children.py`` pins the plain stage). The oracle is the plain search run to its proof."""
    monkeypatch.setattr(cover, "START_NODE_LIMIT", 0)
    monkeypatch.setattr(cover, "SEARCH_QUICK_LIMIT", 1)
    monkeypatch.setattr(columns, "FIRST_PER_UNIT", 0)
    monkeypatch.setattr(columns, "FIRST_MIN", 1)
    monkeypatch.setattr(columns, "_reopt_budget", lambda pool_, keep_: 0)
    calls, real = [], columns._search_reopt

    def rec(*a, **kw):
        calls.append(1)
        return real(*a, **kw)

    monkeypatch.setattr(columns, "_search_reopt", rec)
    Y = _panel(13, 12)
    _, d = columns.solve_exact(np.arange(12), Y, 3)
    assert calls == [1] and d["search_start"] == "first_mip" and d["second_search"] == "reoptimised"
    assert d["objective_value"] == pytest.approx(_plain_optimum(Y, 3), rel=1e-9)


def _plain_optimum(Y, Q):
    """The optimum by the plain search from the root's duals over every group, run to its proof."""
    n = Y.shape[0]
    pool = columns.score_columns(np.arange(n), Y, Q)
    k = columns.min_groups(n, Q, 1)
    res = cover.search(pool, columns.root_lp(pool, k), np.arange(len(pool)), k, Q, np.inf, 10**9)
    assert res.status == "proved"
    return float(pool.cost[res.rows].sum())


# --- property ---------------------------------------------------------------------------------------------------------

@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**6), n=st.integers(2, 11), Q=st.integers(1, 3), min_pairs=st.integers(1, 4),
       incumbent=st.sampled_from(["none", "optimum", "loose"]))
def test_property_the_reoptimised_search_proves_the_full_mips_optimum(seed, n, Q, min_pairs, incumbent):
    """Property: over arms, sizes, cardinality rows and incumbents (none, the optimum, 10% above it), the
    re-optimised search proves the full MIP's optimum: its design is the optimum when it beats the incumbent, and
    its lower bound never exceeds the optimum."""
    if not columns.feasible(n, Q, min_pairs):
        return
    Y, idx, pool, k, root = _setup(seed, n, Q, min_pairs, T=8)
    _, d_ref = _reference(idx, Y, Q, min_pairs)
    z = d_ref["objective_value"]
    best = {"none": np.inf, "optimum": z * (1 + 1e-9) + 1e-12, "loose": z * 1.1 + 1e-9}[incumbent]
    res = columns._search_reopt(pool, root, np.arange(len(pool)), k, Q, best, 10**7)
    tol = 1e-9 * max(1.0, abs(z))
    assert res.status == "proved"
    assert res.lower <= z + tol
    if incumbent != "optimum":
        assert res.rows.size and float(pool.cost[res.rows].sum()) == pytest.approx(z, rel=1e-9, abs=1e-12)
        assert len(res.rows) >= k and _design_units(pool, res.rows) == list(range(n))
