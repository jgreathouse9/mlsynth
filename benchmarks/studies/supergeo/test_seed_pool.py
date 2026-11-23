"""The root's seeded pool (certify_exact._cheap_groups, _seed_pool).

The root started column generation from sg.initial_pool, the sorted neighbours' blocks, and spent most of its time
in the first iterations, whose duals are far from the optimum's: on a simulated 235-market panel 17 pricings, the
first eight of them full scans. At the optimum the market duals are near 0 (every market sits in a group of four
that balances almost exactly) and, for an odd count, the parity dual is the cheapest triple's cost. Seeding the pool
with the cheapest groups, every group of cost at most theta for the least power-of-ten theta whose window holds at
least SEED_MULT * N groups, and with the groups of the starting design, put the first LP near that optimum: 5
pricings and 4.0 s against 17 and 26.1 s. Column generation still runs to convergence, so the bound is the same.
"""
import itertools

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st, HealthCheck

from . import certify_exact as ce, dive, sg, test_bcp


def _all_groups(n, lo=2, hi=4):
    return [c for k in range(lo, hi + 1) for c in itertools.combinations(range(n), k)]


def _cost(z, c):
    return dive._sc(np.asarray(z, float), tuple(c))


def _as_set(rows):
    return {tuple(int(v) for v in r if v >= 0) for r in rows}


@settings(max_examples=80, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**9), n=st.integers(4, 11), kind=st.integers(0, 3), mult=st.integers(0, 12),
       lo=st.integers(2, 3), hi=st.integers(3, 4))
def test_the_cheap_groups_are_exactly_those_at_or_below_theta(seed, n, kind, mult, lo, hi):
    """Property: the rows are every group of lo..hi markets whose cost is at most theta and no other, and theta is
    the least of g 10^e, e = -12, -11, ... (g the squared median spacing, ce._spacing2) at which at least mult * N
    groups cost at most theta, or every group does."""
    if lo > hi:
        return
    z = np.asarray(test_bcp.gen(np.random.default_rng(seed), n, kind), float)
    K = mult * n
    rows, theta = ce._cheap_groups(z, lo, hi, K)
    groups = _all_groups(n, lo, hi)
    costs = np.array([_cost(z, c) for c in groups])
    if K == 0:
        assert len(rows) == 0
        return
    assert _as_set(rows) == {c for c, v in zip(groups, costs) if v <= theta}
    assert (costs <= theta).sum() >= min(K, len(groups))
    g = ce._spacing2(z)
    e = int(round(np.log10(theta / g)))
    assert theta == g * 10.0 ** e and e >= -12
    if e > -12:
        assert (costs <= g * 10.0 ** (e - 1)).sum() < min(K, len(groups))


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**9), n=st.integers(4, 13), kind=st.integers(0, 3))
def test_the_seeded_pool_holds_the_neighbours_the_start_and_the_cheap_groups(seed, n, kind):
    """Property: the seeded pool is the initial pool, the starting design's groups and the cheapest groups, sorted
    and without repeats."""
    z = np.asarray(test_bcp.gen(np.random.default_rng(seed), n, kind), float)
    pool = ce._seed_pool(z, 2, 4)
    assert pool == sorted(set(pool))
    have = set(pool)
    assert set(sg.initial_pool(z, 2, 4)) <= have
    assert {tuple(sorted(int(v) for v in g)) for g in ce._start(z, 2, 4)} <= have
    rows, _ = ce._cheap_groups(z, 2, 4, ce.SEED_MULT * n)
    assert _as_set(rows) <= have
    assert have == set(sg.initial_pool(z, 2, 4)) | {tuple(sorted(int(v) for v in g)) for g in ce._start(z, 2, 4)} \
        | _as_set(rows)


def test_the_default_seeds_at_least_seed_mult_groups_a_market():
    """At the default count, at least SEED_MULT * N groups are seeded when that many exist."""
    z = np.random.default_rng(3).normal(400.0, 50.0, 41)
    rows, theta = ce._cheap_groups(z, 2, 4, ce.SEED_MULT * 41)
    assert ce.SEED_MULT >= 1
    assert len(rows) >= ce.SEED_MULT * 41


def test_a_group_costing_exactly_theta_is_seeded():
    """Integer totals with median spacing 1 make theta a power of ten in exact arithmetic, and a group costing
    exactly 1 lies on the boundary at theta = 1."""
    z = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 20.0, 50.0])
    assert ce._spacing2(z) == 1.0
    rows, theta = ce._cheap_groups(z, 2, 2, 5)
    assert theta == 1.0                                  # pairs cost 1, 4, 9, ...: five cost 1, none less
    assert _as_set(rows) == {(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)}


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**9), n=st.integers(5, 13), kind=st.integers(0, 3))
def test_the_root_bound_is_the_same_from_either_pool(seed, n, kind):
    """Property: column generation runs to convergence from either pool, so the root's bound in the original units,
    (sum pi + tau) / scale, is the LP optimum both times."""
    z = np.asarray(test_bcp.gen(np.random.default_rng(seed), n, kind), float)
    zs, s1, pi1, tau1 = ce._root(z, 2, 4)
    b1 = (pi1.sum() + tau1) / s1
    orig = ce._seed_pool
    try:
        ce._seed_pool = lambda z_, lo, hi: sg.initial_pool(z_, lo, hi)
        zs, s2, pi2, tau2 = ce._root(z, 2, 4)
    finally:
        ce._seed_pool = orig
    b2 = (pi2.sum() + tau2) / s2
    assert b1 == pytest.approx(b2, rel=1e-9, abs=1e-9 * max(1.0, float(np.abs(z).max()) ** 2))


def test_the_root_starts_from_the_seeded_pool(monkeypatch):
    """The root builds its first master from _seed_pool."""
    calls = []
    orig = ce._seed_pool

    def spy(z, lo, hi):
        calls.append(len(z))
        return orig(z, lo, hi)

    monkeypatch.setattr(ce, "_seed_pool", spy)
    z = np.random.default_rng(0).normal(100.0, 10.0, 15)
    ce._root(z, 2, 4)
    assert calls == [15]


def test_equal_totals_end():
    """Every spacing 0: the scale falls back to a positive one, the search for theta ends, and the pool holds every
    group of cost 0; the root runs from it."""
    assert ce._spacing2(np.full(9, 5.0)) > 0
    z = np.full(9, 5.0)
    rows, theta = ce._cheap_groups(z, 2, 4, 5 * 9)
    assert {c for c in _all_groups(9) if _cost(z, c) == 0.0} <= _as_set(rows)
    ce._root(z, 2, 4)


def test_more_groups_asked_than_exist_gives_them_all():
    z = np.array([3.0, 9.0, 1.0, 40.0, 7.0])
    rows, theta = ce._cheap_groups(z, 2, 4, 10_000)
    assert _as_set(rows) == set(_all_groups(5))


def test_none_asked_gives_none():
    z = np.array([3.0, 9.0, 1.0, 40.0, 7.0, 2.0])
    rows, theta = ce._cheap_groups(z, 2, 4, 0)
    assert len(rows) == 0
    assert ce._seed_pool(z, 2, 4, mult=0) == sorted(set(sg.initial_pool(z, 2, 4))
                                                    | {tuple(sorted(g)) for g in ce._start(z, 2, 4)})


def test_a_negative_count_is_refused():
    z = np.array([3.0, 9.0, 1.0, 40.0, 7.0, 2.0])
    with pytest.raises(ValueError, match="mult"):
        ce._seed_pool(z, 2, 4, mult=-1)
    with pytest.raises(ValueError, match="at least 0"):
        ce._cheap_groups(z, 2, 4, -3)
