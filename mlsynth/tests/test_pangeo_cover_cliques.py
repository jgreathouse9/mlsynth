"""Clique cuts in PANGEO's proof search, with the clique slack credited at every node.

A clique is a family of groups that pairwise share a unit, so a design holds at most one of them: the row
``sum_{G in K} x_G <= 1`` is valid. Cliques are separated on the support of the LP over the kept groups (those
violated by the LP point, of three or more groups, with no unit common to all), then lifted greedily: a kept group
joins only when it meets every member so far. A group that meets only the support's members can miss another
lifted group, and the row would cut off designs.

With duals ``mu >= 0`` on those rows, every design ``x`` costs ``c x = c~ x - sum mu + sum_K mu_K slack_K`` for the
shifted costs ``c~_G = c_G + sum_{K ∋ G} mu_K``. The search runs on the shifted costs, prunes at the incumbent plus
``sum mu``, and at each node credits ``mu_K`` for every clique with no chosen member and no member that still fits,
whose slack is 1 in every completion. Without the credit the shift is charged to every design and the cuts double
the nodes; measured from the optimum at Q = 2 with valid cliques, depth-1 re-optimisation placed 0.74 million nodes
against 0.88 million at 40 geos and 4.78 million against 6.15 million at 50.

The oracle is the full MIP over every candidate group, as in ``test_pangeo_cover.py``.
"""
from __future__ import annotations

import itertools
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

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


def _units(pool, j):
    return set(int(p) for p in pool.members[j][pool.members[j] >= 0])


def _design_units(pool, rows):
    return sorted(int(p) for j in rows for p in pool.members[j][pool.members[j] >= 0])


def _hand_pool(n, groups, cost):
    """A pool over the given groups with exact costs: cost_lo = cost = cost_up."""
    w = max(len(g) for g in groups)
    members = -np.ones((len(groups), w), dtype=np.int64)
    for j, g in enumerate(groups):
        members[j, :len(g)] = sorted(g)
    cost = np.asarray(cost, dtype=float)
    return columns.ColumnPool(np.arange(n), members, cost, np.zeros(len(groups), dtype=np.int64), True,
                              cost.copy(), cost.copy())


def _brute(pool, n, k):
    """The least cost over exact covers with at least k groups, by enumeration (small pools only)."""
    best = np.inf
    groups = [_units(pool, j) for j in range(len(pool))]

    def rec(unc, chosen, cost):
        nonlocal best
        if not unc:
            if len(chosen) >= k:
                best = min(best, cost)
            return
        i = min(unc)
        for j, g in enumerate(groups):
            if i in g and g <= unc:
                rec(unc - g, chosen + [j], cost + pool.cost[j])

    rec(frozenset(range(n)), [], 0.0)
    return best


def _triangles(seed, n=6):
    """Pairs on n units: the pairs inside {0,1,2} and inside {3,4,5} cost about 1, every other pair about 5. The LP
    without cuts takes each triangle's pairs at 1/2; the clique cut on a triangle (pairwise meeting, no unit common
    to all three) removes that point."""
    rng = np.random.default_rng(seed)
    groups = [tuple(p) for p in itertools.combinations(range(n), 2)]
    tri = [set(range(0, 3)), set(range(3, 6))]
    cost = [(1.0 if any(set(g) <= t for t in tri) else 5.0) + rng.uniform(0, 0.5) for g in groups]
    return _hand_pool(n, groups, cost), groups


def _clique_search(pool, keep, k, Q, best, limit=10**7, rounds=None):
    """The search over ``keep`` on the duals of the LP over ``keep`` with its separated cliques."""
    keep = np.asarray(keep, dtype=np.int64)
    n = int(pool.unit_indices.size)
    cliques = columns._separate_cliques(pool, keep, k, rounds=rounds)
    duals, _, _ = columns._residual_lp(pool, keep, np.arange(n), k, cliques)
    return cliques, duals, cover.search(pool, duals, keep, k, Q, best, limit)


# --- smoke --------------------------------------------------------------------------------------------------------

def test_smoke_the_search_on_clique_duals_proves_the_full_mips_optimum():
    Y, idx, pool, k, root = _setup(7, 16, 2, T=20)
    _, d_ref = _reference(idx, Y, 2)
    z = d_ref["objective_value"]
    cliques, duals, res = _clique_search(pool, np.arange(len(pool)), k, 2, np.inf)
    assert len(cliques) > 0 and duals.mu.size == len(cliques) and np.all(duals.mu >= 0)
    assert res.status == "proved"
    assert _design_units(pool, res.rows) == list(range(16))
    assert float(pool.cost[res.rows].sum()) == pytest.approx(z, rel=1e-9)
    assert res.lower <= z * (1 + 1e-12) <= res.upper * (1 + 1e-12)


# --- unit: separation -----------------------------------------------------------------------------------------------

def test_clique_rows_keeps_violated_cliques_of_three_or_more_with_no_common_unit():
    """Unit: on the support of x, the maximal cliques of the conflict graph that x violates, with three or more
    groups and no unit common to all (a cover row already holds a clique with a common unit)."""
    members = np.array([[0, 1], [1, 2], [0, 2], [0, 3], [3, 4], [4, 5]])
    x = np.array([0.5, 0.5, 0.5, 0.0, 0.5, 0.5])
    assert [tuple(sorted(K)) for K in columns._clique_rows(x, members)] == [(0, 1, 2)]
    x = np.array([0.3, 0.3, 0.3, 0.0, 0.5, 0.5])            # 0.9: not violated
    assert columns._clique_rows(x, members) == []
    members = np.array([[0, 1], [0, 2], [0, 3]])             # a star: unit 0 is common to all
    assert columns._clique_rows(np.array([0.5, 0.5, 0.5]), members) == []


def test_clique_rows_orders_by_violation_and_caps_the_count():
    members = np.array([[0, 1], [1, 2], [0, 2], [3, 4], [4, 5], [3, 5]])
    x = np.array([0.5, 0.5, 0.5, 0.6, 0.6, 0.6])
    assert [tuple(sorted(K)) for K in columns._clique_rows(x, members)] == [(3, 4, 5), (0, 1, 2)]
    assert [tuple(sorted(K)) for K in columns._clique_rows(x, members, cap=1)] == [(3, 4, 5)]


def _moon_moser(triples: int):
    """Groups whose conflict graph is the complete multipartite graph K(3, ..., 3): ``3 * triples`` groups, two of
    them adjacent exactly when they come from different triples. It has 3^triples maximal cliques (Moon and Moser,
    Israel J. Math. 3 (1965) 23-28), the most any graph on that many vertices has. Each pair of groups from different
    triples shares a unit of its own, so no unit is common to three groups."""
    verts = [(t, a) for t in range(triples) for a in range(3)]
    units = {}
    for p, q in itertools.combinations(range(len(verts)), 2):
        if verts[p][0] != verts[q][0]:
            units[(p, q)] = len(units)
    width = 3 * (triples - 1)
    members = -np.ones((len(verts), width), dtype=np.int64)
    fill = np.zeros(len(verts), dtype=np.int64)
    for (p, q), u in units.items():
        for v in (p, q):
            members[v, fill[v]] = u
            fill[v] += 1
    return members


class _Deadline(Exception):
    pass


def test_clique_rows_does_bounded_work_on_a_graph_with_exponentially_many_maximal_cliques():
    """Edge: separation is a maximum-weight clique problem, strongly NP-hard, and the methods in use solve it one
    root vertex at a time with a greedy or size-limited search, never by listing every maximal clique (Marzi, Rossi
    and Smriglio, Soft Comput. 23 (2019) 3013-3027, section 2; Atamturk, Nemhauser and Savelsbergh, EJOR 121 (2000)
    40-55, p. 50). On K(3, ..., 3) with 15 triples, 3^15 = 14.3 million maximal cliques: listing them does not finish
    in 10 seconds; the separation must, and it must still return violated cliques."""
    import signal

    members = _moon_moser(15)
    x = np.full(members.shape[0], 0.34)                     # one group per triple: 15 * 0.34 = 5.1 > 1

    def _stop(signum, frame):
        raise _Deadline

    old = signal.signal(signal.SIGALRM, _stop)
    signal.alarm(10)
    try:
        found = columns._clique_rows(x, members)
    except _Deadline:
        pytest.fail("clique separation did not finish in 10 seconds on a graph with 3^15 maximal cliques")
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    assert found
    sets = [set(int(u) for u in row if u >= 0) for row in members]
    for K in found:
        assert len(K) >= 3 and all(sets[a] & sets[b] for a, b in itertools.combinations(K, 2))
        assert float(x[list(K)].sum()) > 1.0 + columns._VIOLATION


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**6), groups=st.integers(3, 24), units=st.integers(3, 10), size=st.integers(1, 4),
       cap=st.integers(1, 8))
def test_property_clique_rows_returns_distinct_violated_cliques_by_violation_within_the_cap(seed, groups, units,
                                                                                         size, cap):
    """Property: whatever the point and the groups, every returned set is a clique of the support's conflict graph
    (pairwise sharing a unit), of three or more groups with no unit common to all, violated by more than the
    tolerance, the list sorted by violation, without repeats, and at most ``cap`` long."""
    rng = np.random.default_rng(seed)
    members = -np.ones((groups, size), dtype=np.int64)
    for j in range(groups):
        g = rng.choice(units, size=int(rng.integers(1, min(size, units) + 1)), replace=False)
        members[j, :g.size] = np.sort(g)
    x = np.where(rng.random(groups) < 0.2, 0.0, rng.random(groups))
    found = columns._clique_rows(x, members, cap=cap)
    sets = [set(int(u) for u in row if u >= 0) for row in members]
    assert len(found) <= cap and len(set(map(tuple, found))) == len(found)
    viol = []
    for K in found:
        assert len(K) >= 3 and all(x[a] > columns._EPS for a in K)
        assert all(sets[a] & sets[b] for a, b in itertools.combinations(K, 2))
        assert not set.intersection(*(sets[a] for a in K))
        viol.append(float(np.sum(x[list(K)])) - 1.0)
        assert viol[-1] > columns._VIOLATION
    assert viol == sorted(viol, reverse=True)


def test_each_root_grows_its_clique_by_the_heaviest_remaining_neighbour():
    """Unit: Algorithm 2 of Marzi, Rossi and Smriglio (2019) with f = x. Root r (lowest degree, first) meets h1, h2
    and l; h1 and h2 meet each other, l meets neither. Heaviest first, r takes h1 then h2: {r, h1, h2} weighs 1.2.
    Lightest first, r would take l and stop at two groups. The fillers f1..f3 meet h1, h2 and l only, raising
    those degrees above r's, and weigh too little to make a violated clique of their own."""
    groups = [(0, 1, 2), (0, 3, 10, 11, 12), (1, 3, 13, 14, 15), (2, 16, 17, 18), (10, 13, 16), (11, 14, 17),
              (12, 15, 18)]
    members = -np.ones((len(groups), 5), dtype=np.int64)
    for j, g in enumerate(groups):
        members[j, :len(g)] = g
    x = np.array([0.3, 0.45, 0.45, 0.1, 0.01, 0.01, 0.01])
    assert columns._clique_rows(x, members) == [(0, 1, 2)]


def test_lifting_adds_a_group_only_when_it_meets_every_member_so_far():
    """Unit: the support clique {1,2,3,4}, {1,5}, {2,5} has no common unit. {1,2,9,10} and {3,5,11,12} each meet all
    three, but not each other, so both cannot join: lifting takes the first in order and refuses the second."""
    members = -np.ones((5, 4), dtype=np.int64)
    for j, g in enumerate([(1, 2, 3, 4), (1, 5), (2, 5), (1, 2, 9, 10), (3, 5, 11, 12)]):
        members[j, :len(g)] = g
    K = columns._lift([0, 1, 2], members, np.array([3, 4]))
    assert sorted(K.tolist()) == [0, 1, 2, 3]
    K = columns._lift([0, 1, 2], members, np.array([4, 3]))
    assert sorted(K.tolist()) == [0, 1, 2, 4]


def test_separated_cliques_are_cliques_of_kept_groups_with_no_common_unit():
    Y, idx, pool, k, root = _setup(7, 16, 2, T=20)
    keep = np.flatnonzero(root.reduced_costs <= np.quantile(root.reduced_costs, 0.5))
    cliques = columns._separate_cliques(pool, keep, k)
    assert len(cliques) > 0
    kept = set(keep.tolist())
    for K in cliques:
        assert len(K) >= 3 and set(K.tolist()) <= kept and len(set(K.tolist())) == len(K)
        sets = [_units(pool, j) for j in K]
        assert all(a & b for a, b in itertools.combinations(sets, 2))
        assert not set.intersection(*sets)


def test_each_separated_clique_is_maximal_over_the_kept_groups():
    Y, idx, pool, k, root = _setup(7, 16, 2, T=20)
    keep = np.arange(len(pool))
    for K in columns._separate_cliques(pool, keep, k):
        sets = [_units(pool, j) for j in K]
        inK = set(K.tolist())
        assert not any(j not in inK and all(_units(pool, j) & s for s in sets) for j in keep)


def test_the_cliques_raise_the_lp_value():
    Y, idx, pool, k, root = _setup(7, 16, 2, T=20)
    keep = np.arange(len(pool))
    cliques = columns._separate_cliques(pool, keep, k)
    _, _, plain = columns._residual_lp(pool, keep, np.arange(16), k, ())
    _, _, cut = columns._residual_lp(pool, keep, np.arange(16), k, cliques)
    assert cut > plain * (1 + 1e-6)


def test_zero_rounds_separate_no_clique():
    Y, idx, pool, k, root = _setup(7, 16, 2, T=20)
    assert columns._separate_cliques(pool, np.arange(len(pool)), k, rounds=0) == ()


# --- unit: the residual LP's duals with cliques ---------------------------------------------------------------------

def test_the_shifted_costs_and_reduced_costs_are_rigorous_lower_bounds():
    """Unit: the shifted cost lower bound of every group is at most cost_lo + sum of the mu of its cliques, and its
    reduced cost at most the shifted cost less its duals and sigma, both in exact arithmetic."""
    Y, idx, pool, k, root = _setup(7, 16, 2, T=20)
    keep = np.arange(len(pool))
    cliques = columns._separate_cliques(pool, keep, k)
    duals, _, _ = columns._residual_lp(pool, keep, np.arange(16), k, cliques)
    shift = [Fraction(0)] * len(pool)
    for K, m in zip(cliques, duals.mu):
        for j in K:
            shift[j] += Fraction(float(m))
    for j in range(0, len(pool), 7):
        lo2 = Fraction(float(pool.cost_lo[j])) + shift[j]
        assert Fraction(float(duals.cost_lo[j])) <= lo2
        rc = lo2 - sum(Fraction(float(duals.pi[p])) for p in _units(pool, j)) - Fraction(float(duals.sigma))
        assert Fraction(float(duals.reduced_costs[j])) <= rc


def test_the_residual_bound_with_cliques_is_at_most_the_optimum():
    Y, idx, pool, k, root = _setup(7, 16, 2, T=20)
    _, d_ref = _reference(idx, Y, 2)
    keep = np.arange(len(pool))
    cliques = columns._separate_cliques(pool, keep, k)
    _, L, value = columns._residual_lp(pool, keep, np.arange(16), k, cliques)
    assert L <= Fraction(d_ref["objective_value"])
    assert float(L) <= value * (1 + 1e-9) + 1e-12


# --- unit: the credit ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(8))
def test_on_triangles_the_credited_search_finds_the_optimum(seed):
    """Unit: on pairs over six units with two cheap triangles, the clique cuts bind and their duals are large, so
    credit given for a clique a chosen pair belongs to, or one a pair can still fill, or a threshold without the sum
    of the duals, prunes the optimum. The search must find it, and its lower bound (from the designs' true costs,
    not the shifted ones) must not pass it."""
    pool, groups = _triangles(seed)
    z = _brute(pool, 6, 3)
    cliques, duals, res = _clique_search(pool, np.arange(len(pool)), 3, 1, np.inf)
    assert len(cliques) == 2 and np.all(duals.mu > 0.5)
    assert res.status == "proved" and res.rows.size
    assert float(pool.cost[res.rows].sum()) == pytest.approx(z, rel=1e-12)
    assert z * (1 - 1e-12) <= res.lower <= z


@pytest.mark.parametrize("seed", range(4))
def test_on_triangles_the_optimum_avoiding_the_cliques_is_found(seed):
    """Unit: with the cross pairs made cheap the optimum holds no triangle pair, so both cliques have slack 1 in it,
    and the credit is what lets the bound reach it."""
    pool, groups = _triangles(seed)
    cost = pool.cost.copy()
    for j, g in enumerate(groups):
        if g in [(0, 3), (1, 4), (2, 5)]:
            cost[j] = 0.2
    pool = _hand_pool(6, groups, cost)
    z = _brute(pool, 6, 3)
    _, duals, res = _clique_search(pool, np.arange(len(pool)), 3, 1, np.inf)
    assert res.status == "proved"
    assert sorted(tuple(sorted(_units(pool, j))) for j in res.rows) == [(0, 3), (1, 4), (2, 5)]
    assert float(pool.cost[res.rows].sum()) == pytest.approx(z, rel=1e-12)


@pytest.mark.parametrize("seed", range(4))
def test_a_fixed_group_in_a_clique_counts_as_hitting_it_against_a_tight_incumbent(seed):
    """Unit: against an incumbent just above the best design holding the fixed group, the search still finds that
    design. The fixed group's cliques are hit in every completion and their duals are already in its shifted cost,
    so crediting one again raises the node bound past that design and the search returns none (it did, on every
    seed, with the exclusion of a fixed group's cliques removed)."""
    pool, groups = _triangles(seed)
    keep_all = np.arange(len(pool))
    cliques = columns._separate_cliques(pool, keep_all, 3)
    duals, _, _ = columns._residual_lp(pool, keep_all, np.arange(6), 3, cliques)
    j = groups.index((0, 1))
    g = _units(pool, j)
    assert any(j in K for K in duals.cliques) and np.all(duals.mu > 0.5)   # the fixed group sits in a binding clique
    left = np.array([r for r in keep_all if not (_units(pool, r) & g)])
    best = min(pool.cost[j] + pool.cost[a] + pool.cost[b]
               for a, b in itertools.combinations(left, 2) if not (_units(pool, a) & _units(pool, b)))
    res = cover.search(pool, duals, left, 3, 1, best * (1 + 1e-9), 10**6, fixed=(j,))
    assert res.status == "proved" and res.rows.size
    assert float(pool.cost[res.rows].sum()) == pytest.approx(best, rel=1e-12)


@pytest.mark.parametrize("seed,n", [(7, 14), (2, 16)])
def test_the_credited_search_places_no_more_nodes_than_the_clique_free_search(seed, n):
    """Unit: from the optimum, the search on the clique LP's duals places no more nodes than the search on the
    clique-free LP's duals over the same groups. Measured: 2,133 against 5,320 nodes (14 geos) and 3,172 against
    11,950 (16 geos). A node bound over the unshifted costs against the threshold raised by the clique duals is
    valid but weaker, and placed 24,045 and 40,531."""
    Y, idx, pool, k, root = _setup(seed, n, 2)
    _, d = _reference(idx, Y, 2)
    best = d["objective_value"] * (1 + 1e-9)
    keep = np.arange(len(pool))
    cliques = columns._separate_cliques(pool, keep, k)
    duals, _, _ = columns._residual_lp(pool, keep, np.arange(n), k, cliques)
    plain, _, _ = columns._residual_lp(pool, keep, np.arange(n), k, ())
    cut = cover.search(pool, duals, keep, k, 2, best, 10**8)
    free = cover.search(pool, plain, keep, k, 2, best, 10**8)
    assert len(cliques) > 0 and cut.status == free.status == "proved"
    assert cut.nodes <= free.nodes


def test_the_slack_credit_holds_the_node_count_on_a_pinned_arm():
    """Unit: a node-count regression pin. On the 16-geo arm of seed 2 at Q = 2, from the optimum, the credited
    search places 3,172 nodes; with the clique slack never credited (the shifted costs and the raised threshold
    alone) it placed 5,139. The bound stays valid either way, so only the count shows the credit. The ceiling sits
    between the two."""
    Y, idx, pool, k, root = _setup(2, 16, 2)
    _, d = _reference(idx, Y, 2)
    keep = np.arange(len(pool))
    cliques = columns._separate_cliques(pool, keep, k)
    duals, _, _ = columns._residual_lp(pool, keep, np.arange(16), k, cliques)
    res = cover.search(pool, duals, keep, k, 2, d["objective_value"] * (1 + 1e-9), 10**8)
    assert res.status == "proved" and res.nodes <= 4000


def test_a_fixed_group_in_a_clique_counts_as_hitting_it():
    """Unit: a fixed group's cliques have slack 0 in every completion: none of them is credited."""
    pool, groups = _triangles(3)
    keep_all = np.arange(len(pool))
    cliques = columns._separate_cliques(pool, keep_all, 3)
    duals, _, _ = columns._residual_lp(pool, keep_all, np.arange(6), 3, cliques)
    j = groups.index((0, 1))
    g = _units(pool, j)
    left = np.array([r for r in keep_all if not (_units(pool, r) & g)])
    sub = _hand_pool(6, groups, pool.cost)
    best = min(pool.cost[j] + pool.cost[a] + pool.cost[b]
               for a, b in itertools.combinations(left, 2) if not (_units(pool, a) & _units(pool, b)))
    res = cover.search(sub, duals, left, 3, 1, np.inf, 10**6, fixed=(j,))
    assert res.status == "proved" and res.rows.size
    assert float(pool.cost[res.rows].sum()) == pytest.approx(best, rel=1e-12)


def test_no_cliques_leave_the_search_unchanged():
    """Unit: duals with no clique search exactly as the root duals they equal."""
    Y, idx, pool, k, root = _setup(4, 12, 2)
    keep = np.arange(len(pool))
    duals, _, _ = columns._residual_lp(pool, keep, np.arange(12), k, ())
    assert duals.cliques == () and duals.mu.size == 0
    a = cover.search(pool, duals, keep, k, 2, np.inf, 10**7)
    plain = columns._Duals(duals.pi, duals.sigma, duals.reduced_costs)
    b = cover.search(pool, plain, keep, k, 2, np.inf, 10**7)
    assert (a.status, a.nodes, a.upper, a.lower) == (b.status, b.nodes, b.upper, b.lower)
    assert np.array_equal(a.rows, b.rows)


@pytest.mark.parametrize("seed,n,Q,fixed", [(4, 12, 2, False), (2, 12, 2, True), (5, 14, 3, False)])
def test_unshifted_costs_give_the_leaf_bound_of_the_running_sum(seed, n, Q, fixed):
    """Unit: without clique shifts the node sums and the leaves' true-cost sums are the same numbers added in the
    same order, so the search may read the running sum at a leaf instead of summing the path again. Forcing the
    leaf sum (duals that carry shifted costs equal to the true ones) and taking the running sum (duals with none)
    must give the same search, bit for bit: status, nodes, bounds and design, with and without a fixed group."""
    Y, idx, pool, k, root = _setup(seed, n, Q)
    keep = np.arange(len(pool))
    fx = ()
    if fixed:
        j = int(np.argmin(pool.cost))
        g = _units(pool, j)
        keep = np.array([r for r in keep if not (_units(pool, r) & g)])
        fx = (j,)
    plain = columns._Duals(np.asarray(root.pi, dtype=float), float(root.sigma), np.asarray(root.reduced_costs))
    forced = columns._Duals(plain.pi, plain.sigma, plain.reduced_costs, cost_lo=np.array(pool.cost_lo, copy=True))
    a = cover.search(pool, plain, keep, k, Q, np.inf, 10**7, fixed=fx)
    b = cover.search(pool, forced, keep, k, Q, np.inf, 10**7, fixed=fx)
    assert a.status == b.status == "proved" and a.nodes == b.nodes
    assert (a.upper, a.lower) == (b.upper, b.lower)
    assert np.array_equal(a.rows, b.rows)


def test_credited_cliques_place_fewer_nodes_in_the_reoptimised_search(monkeypatch):
    """Unit: from the optimum on a 16-geo arm at Q = 2, the depth-1 search with credited cliques places fewer nodes
    than without cliques."""
    Y, idx, pool, k, root = _setup(7, 16, 2, T=20)
    _, d_ref = _reference(idx, Y, 2)
    best = d_ref["objective_value"] * (1 + 1e-9)
    keep = np.flatnonzero(root.reduced_costs <= columns._up(Fraction(best) - Fraction(root.bound)))
    monkeypatch.setattr(columns, "CLIQUE_ROUNDS", 0)
    plain = columns._search_reopt(pool, root, keep, k, 2, best, 10**8)
    monkeypatch.setattr(columns, "CLIQUE_ROUNDS", 10)
    cut = columns._search_reopt(pool, root, keep, k, 2, best, 10**8)
    assert plain.status == cut.status == "proved"
    assert cut.nodes < plain.nodes


def test_the_branches_carry_the_cliques_the_fixed_group_is_not_in(monkeypatch):
    """Unit: each branch's residual LP has the cliques without the fixed group, restricted to the groups left; a
    clique holding the fixed group has no member left."""
    Y, idx, pool, k, root = _setup(7, 16, 2, T=20)
    keep = np.arange(len(pool))
    cliques = columns._separate_cliques(pool, keep, k)
    branches = columns._branches(pool, root, keep, k, cliques)
    assert any(d is not None and len(d.cliques) > 0 for _, _, _, d in branches)
    for j, _, left, duals in branches:
        if duals is None:
            continue
        lset = set(left.tolist())
        for K in duals.cliques:
            assert j not in set(K.tolist()) and set(K.tolist()) <= lset and len(K) >= 2


def test_the_reoptimised_search_separates_cliques_over_the_kept_groups(monkeypatch):
    Y, idx, pool, k, root = _setup(4, 12, 2)
    seen, real = [], columns._separate_cliques

    def rec(pool_, keep_, k_, rounds=None):
        seen.append(np.asarray(keep_).copy())
        return real(pool_, keep_, k_, rounds=rounds)

    monkeypatch.setattr(columns, "_separate_cliques", rec)
    keep = np.arange(0, len(pool))
    columns._search_reopt(pool, root, keep, k, 2, np.inf, 10**7)
    assert len(seen) == 1 and np.array_equal(seen[0], keep)


# --- edge -----------------------------------------------------------------------------------------------------------

def test_separation_over_no_kept_groups_is_empty():
    Y, idx, pool, k, root = _setup(4, 12, 2)
    assert columns._separate_cliques(pool, np.zeros(0, np.int64), k) == ()


def test_separation_over_an_infeasible_keep_is_empty():
    """Edge: groups that cannot cover the arm have no LP point to separate."""
    Y, idx, pool, k, root = _setup(4, 12, 2)
    keep = np.flatnonzero([0 not in _units(pool, j) for j in range(len(pool))])
    assert columns._separate_cliques(pool, keep, k) == ()


def test_separation_over_groups_that_meet_every_unit_but_cover_none_exactly_is_empty():
    """Edge: kept groups {0, 1} and {1, 2} each meet a unit the other misses, so every unit is in some group and the
    first check passes, but unit 1 lies in both and units 0 and 2 in one each: no point covers each unit once, the
    LP has no solution, and separation stops before its first round."""
    pool = _hand_pool(3, [(0, 1), (1, 2)], [1.0, 1.0])
    assert columns._separate_cliques(pool, np.arange(2), 1) == ()


def test_an_integral_lp_separates_no_clique():
    """Edge: pairs on four units (three designs): the LP is integral, nothing is violated."""
    groups = [tuple(p) for p in itertools.combinations(range(4), 2)]
    pool = _hand_pool(4, groups, [1.0, 2.0, 3.0, 3.5, 2.5, 1.5])
    assert columns._separate_cliques(pool, np.arange(6), 2) == ()


def test_clique_members_outside_keep_are_dropped_by_the_search():
    """Edge: a clique handed to the search with members outside keep counts only the kept ones."""
    pool, groups = _triangles(1)
    keep = np.arange(len(pool))
    cliques = columns._separate_cliques(pool, keep, 3)
    duals, _, _ = columns._residual_lp(pool, keep, np.arange(6), 3, cliques)
    z = _brute(pool, 6, 3)
    drop = int(cliques[0][0])
    sub = np.array([j for j in keep if j != drop])
    zs = _brute(_hand_pool(6, [groups[j] for j in sub], pool.cost[sub]), 6, 3)
    res = cover.search(pool, duals, sub, 3, 1, np.inf, 10**6)
    assert res.status == "proved" and res.rows.size
    assert float(pool.cost[res.rows].sum()) == pytest.approx(zs, rel=1e-12) and zs >= z


# --- property -------------------------------------------------------------------------------------------------------

@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**6), n=st.integers(4, 9), Q=st.integers(1, 2), spread=st.floats(0.0, 4.0),
       incumbent=st.sampled_from(["none", "optimum", "loose"]))
def test_property_separated_cliques_are_valid_and_the_credited_search_is_exact(seed, n, Q, spread, incumbent):
    """Property: over hand pools of every group of size 2..2Q with random costs (a spread makes cheap blocks, so
    the LP is fractional and cliques appear), every separated clique pairwise meets and is maximal, and the search
    on its duals finds the enumerated optimum when it beats the incumbent, with a lower bound that does not pass
    it."""
    rng = np.random.default_rng(seed)
    groups = [g for s in range(2, 2 * Q + 1) for g in itertools.combinations(range(n), s)]
    block = rng.integers(0, 2, size=n)
    cost = [rng.uniform(0.5, 1.5) + spread * (len(set(block[list(g)])) > 1) for g in groups]
    pool = _hand_pool(n, groups, cost)
    k = columns.min_groups(n, Q, 1)
    z = _brute(pool, n, k)
    if not np.isfinite(z):
        return
    keep = np.arange(len(pool))
    cliques = columns._separate_cliques(pool, keep, k)
    for K in cliques:
        sets = [_units(pool, j) for j in K]
        assert all(a & b for a, b in itertools.combinations(sets, 2))
        inK = set(K.tolist())
        assert not any(j not in inK and all(_units(pool, j) & s for s in sets) for j in keep)
    duals, L, _ = columns._residual_lp(pool, keep, np.arange(n), k, cliques)
    assert L <= Fraction(z) + Fraction(1, 10**12)
    best = {"none": np.inf, "optimum": z * (1 + 1e-9) + 1e-12, "loose": z * 1.1 + 1e-9}[incumbent]
    res = cover.search(pool, duals, keep, k, Q, best, 10**7)
    assert res.status == "proved"
    assert res.lower <= z + 1e-9 * max(1.0, abs(z))
    if incumbent != "optimum":
        assert res.rows.size and float(pool.cost[res.rows].sum()) == pytest.approx(z, rel=1e-9, abs=1e-12)
        assert _design_units(pool, res.rows) == list(range(n))
        assert res.lower >= z - 1e-9 * max(1.0, abs(z))     # the leaves' bounds are the designs' true costs


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**6), n=st.integers(4, 11), Q=st.integers(1, 3), min_pairs=st.integers(1, 4))
def test_property_the_reoptimised_search_with_cliques_proves_the_full_mips_optimum(seed, n, Q, min_pairs):
    if not columns.feasible(n, Q, min_pairs):
        return
    Y, idx, pool, k, root = _setup(seed, n, Q, min_pairs, T=8)
    _, d_ref = _reference(idx, Y, Q, min_pairs)
    z = d_ref["objective_value"]
    res = columns._search_reopt(pool, root, np.arange(len(pool)), k, Q, np.inf, 10**7)
    assert res.status == "proved"
    assert res.lower <= z + 1e-9 * max(1.0, abs(z))
    assert res.rows.size and float(pool.cost[res.rows].sum()) == pytest.approx(z, rel=1e-9, abs=1e-12)
