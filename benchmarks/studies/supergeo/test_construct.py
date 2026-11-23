"""Design construction for large panels (construct.py): annealing over pairings with a closed-form coupling, then
repair of many groups at once by an exact set-partitioning solve.

On a simulated 235-market panel the root bound is the cheapest triple's cost, 1.78035, and a first-moment count of
designs (Mertens, "The Easiest Hard Problem: Number Partitioning", 2003, Sec. V) puts the optimum within about 1e-5
of it. The study's heuristic (lns_dp, exact repair of 18 markets at a time) stopped near 3.6. Annealing over
pairings reached 1.86 to 1.93 and the repair of 40 to 60 markets at a time 1.8046 in 30 minutes.
"""
import itertools

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st, HealthCheck

from . import construct as cn, dive, test_bcp


def _brute_couple(d):
    """The least cost over every way to split the pairs into couples (cost (d_i - d_j)^2) and lone pairs (d_i^2)."""
    d = list(d)
    best = [np.inf]

    def rec(rest, acc):
        if acc >= best[0]:
            return
        if not rest:
            best[0] = acc
            return
        i, others = rest[0], rest[1:]
        rec(others, acc + d[i] ** 2)
        for k, j in enumerate(others):
            rec(others[:k] + others[k + 1:], acc + (d[i] - d[j]) ** 2)

    rec(list(range(len(d))), 0.0)
    return best[0]


def _cost(z, design):
    return sum(dive._sc(np.asarray(z, float), tuple(sorted(g))) for g in design)


def _is_partition(design, n, lo=2, hi=4):
    seen = sorted(v for g in design for v in g)
    return seen == list(range(n)) and all(lo <= len(g) <= hi for g in design)


@settings(max_examples=200, deadline=None)
@given(d=st.lists(st.floats(0, 100, allow_nan=False), min_size=0, max_size=9))
def test_the_coupling_is_the_brute_force_optimum(d):
    """Property: the DP over sorted differences equals the least cost over every coupling (closed form against the
    general search on the cases they share), and its groups are a partition of the pairs with that cost."""
    cost, groups = cn.couple(np.array(d, float))
    assert cost == pytest.approx(_brute_couple(d), rel=1e-9, abs=1e-9)
    used = sorted(i for g in groups for i in g)
    assert used == list(range(len(d)))
    again = sum(d[g[0]] ** 2 if len(g) == 1 else (d[g[0]] - d[g[1]]) ** 2 for g in groups)
    assert again == pytest.approx(cost, rel=1e-9, abs=1e-9)


def test_a_lone_pair_inside_a_couple_is_never_kept():
    """Unit: for a <= b <= c, coupling b with c and leaving a alone costs no more than coupling a with c and leaving
    b alone ((c - b)^2 + a^2 <= (c - a)^2 + b^2): with 0, 0.5, 1 the DP couples 0.5 with 1."""
    cost, groups = cn.couple(np.array([0.0, 0.5, 1.0]))
    assert cost == pytest.approx(0.25)
    assert sorted(tuple(sorted(g)) for g in groups) == [(0,), (1, 2)]


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**9), n=st.integers(3, 30), kind=st.integers(0, 3))
def test_the_cheapest_triple_is_the_brute_force_one(seed, n, kind):
    """Property: the scan returns a triple of least cost over every triple (its best split of one market against
    two)."""
    z = np.asarray(test_bcp.gen(np.random.default_rng(seed), n, kind), float)
    cost, t = cn.cheapest_triple(z)
    best = min(dive._sc(z, c) for c in itertools.combinations(range(n), 3))
    assert cost == pytest.approx(best, rel=1e-12, abs=1e-12)
    assert len(set(t)) == 3 and dive._sc(z, tuple(sorted(t))) == pytest.approx(cost, rel=1e-12, abs=1e-12)


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**9), n=st.integers(2, 40), kind=st.integers(0, 3), steps=st.integers(0, 3000),
       aseed=st.integers(0, 50))
def test_the_pairing_design_is_a_partition_at_its_reported_cost(seed, n, kind, steps, aseed):
    """Property: the annealed design partitions the markets into groups of 2 to 4 (one triple for an odd count),
    its reported cost is its cost, it costs no more than its start, and the same seed gives the same design."""
    z = np.asarray(test_bcp.gen(np.random.default_rng(seed), n, kind), float)
    design, cost, start = cn.pairing_design(z, steps=steps, seed=aseed)
    assert _is_partition(design, n)
    assert sum(len(g) == 3 for g in design) == n % 2
    assert cost == pytest.approx(_cost(z, design), rel=1e-9, abs=1e-9)
    assert cost <= start + 1e-9 * max(1.0, start)
    again, cost2, _ = cn.pairing_design(z, steps=steps, seed=aseed)
    assert again == design and cost2 == cost


def test_annealing_improves_on_sorted_neighbours():
    """Unit: on 60 normal totals the annealed design costs less than half the sorted-neighbours start."""
    z = np.random.default_rng(11).normal(400.0, 60.0, 60)
    design, cost, start = cn.pairing_design(z, steps=200_000, seed=0)
    assert cost < 0.5 * start


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**9), n=st.integers(8, 28), kind=st.integers(0, 3), steps=st.integers(0, 12),
       rseed=st.integers(0, 20))
def test_the_repair_keeps_a_partition_and_never_costs_more(seed, n, kind, steps, rseed):
    """Property: every repair step leaves a partition into groups of 2 to 4, the trace never rises, its last entry
    is the design's cost, and the same seed gives the same design."""
    z = np.asarray(test_bcp.gen(np.random.default_rng(seed), n, kind), float)
    start, _, _ = cn.pairing_design(z, steps=100, seed=0)
    design, trace = cn.repair(z, start, steps=steps, seed=rseed, k0=2, k_max=4)
    assert _is_partition(design, n)
    assert all(b <= a + 1e-9 * max(1.0, a) for a, b in zip(trace, trace[1:]))
    assert trace[0] == pytest.approx(_cost(z, start), rel=1e-9, abs=1e-9)
    assert trace[-1] == pytest.approx(_cost(z, design), rel=1e-9, abs=1e-9)
    again, trace2 = cn.repair(z, start, steps=steps, seed=rseed, k0=2, k_max=4)
    assert again == design and trace2 == trace


def test_the_repair_of_every_group_at_once_is_the_optimum():
    """Unit: freeing every group, one repair step solves the whole design exactly: on 12 markets it reaches the
    brute-force optimum over all partitions into groups of 2 to 4."""
    z = np.asarray(test_bcp.gen(np.random.default_rng(4), 12, 1), float)
    start, _, _ = cn.pairing_design(z, steps=0, seed=0)
    design, trace = cn.repair(z, start, steps=1, seed=0, k0=len(start), k_max=len(start))
    assert trace[-1] == pytest.approx(test_bcp.brute(z), rel=1e-9, abs=1e-9)


def test_the_freed_set_grows_after_failures():
    """Unit: after `patience` steps in a row without a cheaper repair the number of freed groups grows by one, up to
    k_max; the trace of k is recorded."""
    z = np.asarray(test_bcp.gen(np.random.default_rng(2), 24, 1), float)
    start, _, _ = cn.pairing_design(z, steps=50_000, seed=0)
    design, trace, ks = cn.repair(z, start, steps=40, seed=0, k0=2, k_max=5, patience=3, with_k=True)
    assert ks[0] == 2 and max(ks) <= 5
    assert all(b - a in (0, 1) for a, b in zip(ks, ks[1:]))


def test_the_constructor_reaches_the_bound_on_a_panel_with_a_perfect_design():
    """Smoke: totals made of quadruples a, b, c, a + b - c (and one exact triple) have a design of cost 0 plus the
    triple's; construct finds it."""
    rng = np.random.default_rng(0)
    z = []
    for _ in range(10):
        a, b, c = rng.integers(100, 200, 3).astype(float)
        z += [a, b, c, a + b - c]
    z += [150.0, 70.0, 80.0]
    z = np.array(z)
    design, cost = cn.construct(z, anneal_steps=300_000, repair_steps=30, seed=0)
    assert _is_partition(design, len(z))
    assert cost == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("bad", [dict(steps=-1), dict(steps=1.5), dict(seed=-1)])
def test_bad_annealing_arguments_are_refused(bad):
    z = np.arange(1.0, 11.0)
    kw = dict(steps=10, seed=0) | bad
    with pytest.raises(ValueError):
        cn.pairing_design(z, **kw)


@pytest.mark.parametrize("bad", [dict(k0=0), dict(k0=5, k_max=4), dict(patience=0), dict(node_limit=0),
                                 dict(steps=-2)])
def test_bad_repair_arguments_are_refused(bad):
    z = np.arange(1.0, 13.0)
    start, _, _ = cn.pairing_design(z, steps=0, seed=0)
    kw = dict(steps=3, seed=0, k0=2, k_max=4) | bad
    with pytest.raises(ValueError):
        cn.repair(z, start, **kw)


def test_a_design_that_is_not_a_partition_is_refused():
    z = np.arange(1.0, 9.0)
    with pytest.raises(ValueError, match="partition"):
        cn.repair(z, [(0, 1), (2, 3), (4, 5)], steps=1, seed=0)


def test_too_few_markets_for_a_group():
    """Edge: one market has no design; two are one pair; three are one triple."""
    with pytest.raises(ValueError, match="at least 2"):
        cn.pairing_design(np.array([1.0]), steps=0, seed=0)
    assert cn.pairing_design(np.array([1.0, 4.0]), steps=0, seed=0)[0] == [(0, 1)]
    assert cn.pairing_design(np.array([1.0, 4.0, 3.0]), steps=0, seed=0)[0] == [(0, 1, 2)]


def test_equal_totals_cost_nothing():
    z = np.full(10, 7.0)
    design, cost = cn.construct(z, anneal_steps=1000, repair_steps=2, seed=0)
    assert _is_partition(design, 10) and cost == 0.0


def test_an_odd_group_is_never_freed():
    """Unit: the repair frees only groups of even size. An odd count's triple is the root bound's own column (the
    cheapest triple) and dominates the design's cost, so drawing the dearest group by cost freed it at almost every
    step, every group of the freed markets then cost less than the freed total, and the partitioning MIP had about
    100,000 columns: 100 s a step on 121 markets with no improvement. Freed markets can still form a triple."""
    z = np.asarray(test_bcp.gen(np.random.default_rng(6), 41, 1), float)
    start, _, _ = cn.pairing_design(z, steps=20_000, seed=0)
    triple = [g for g in start if len(g) == 3]
    assert len(triple) == 1
    seen = []
    orig = cn._groups_below

    def spy(z_, F, lo, hi, cur):
        seen.append(set(int(v) for v in F))
        return orig(z_, F, lo, hi, cur)

    cn._groups_below = spy
    try:
        cn.repair(z, start, steps=6, seed=0, k0=3, k_max=3)
    finally:
        cn._groups_below = orig
    assert seen and all(not (set(triple[0]) & F) for F in seen)
