"""Branch, cut and price against brute force over every partition into groups of 2..4."""
import itertools, pathlib
import numpy as np
import pandas as pd
import pytest
from hypothesis import example, given, settings, strategies as st, HealthCheck
from . import bcp, dive, rank1

GEOLIFT = pathlib.Path(__file__).resolve().parents[3] / "basedata" / "geolift_test_data.csv"


def brute(z):
    n = len(z); best = [np.inf]
    def go(U, acc):
        if acc >= best[0]: return
        if not U: best[0] = acc; return
        i = min(U); rest = sorted(U - {i})
        for k in (1, 2, 3):
            for others in itertools.combinations(rest, k):
                g = (i,) + others
                go(U - set(g), acc + dive._sc(z, g))
    go(frozenset(range(n)), 0.0); return best[0]


def check(z, **kw):
    z = np.asarray(z, float)
    if kw.get("families") == "rank1":
        kw = dict(kw, families=rank1.FAMILIES)
    out = bcp.BCP(z, verbose=False, **kw).solve(time_limit=300)
    want = brute(z)
    assert out["status"] == "optimal"
    assert abs(out["opt"] - want) <= 1e-9 * max(1.0, abs(want)), (out["opt"], want)
    assert sorted(g for c in out["design"] for g in c) == list(range(len(z)))
    assert all(2 <= len(c) <= 4 for c in out["design"])
    assert abs(sum(dive._sc(z, c) for c in out["design"]) - want) <= 1e-9 * max(1.0, abs(want))
    return out


def gen(rng, n, kind):
    return [rng.integers(1, 30, n).astype(float), rng.integers(1, 80, n) / 10.0,
            rng.lognormal(0, 1.3, n) * 100, np.r_[rng.integers(10, 21, n - 1), 47.0]][kind]


@pytest.mark.parametrize("kw", [{}, {"cuts": False}, {"group_branching": False}, {"cuts": False, "group_branching": False},
                                {"families": "rank1"}, {"families": "rank1", "group_branching": False},
                                {"families": "rank1", "purge_after": 1, "deep_depth": 0, "max_active": 20}])
def test_fixed_against_brute_force(kw):
    rng = np.random.default_rng(1)
    for t in range(16):
        check(gen(rng, int(rng.integers(4, 12)), t % 4), **kw)


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**9), n=st.integers(4, 11), kind=st.integers(0, 3))
def test_property_against_brute_force(seed, n, kind):
    check(gen(np.random.default_rng(seed), n, kind))


def _subsets(k, reps):
    df = pd.read_csv(GEOLIFT); dates = sorted(df.date.unique())
    pre = df[df.date.isin(dates[-56:-28])].groupby("location").Y.sum().to_numpy(float)
    rng = np.random.default_rng(2026)
    for _ in range(reps):
        p = pre[sorted(rng.choice(len(pre), k, replace=False))]
        yield p / np.median(np.diff(np.sort(p)))


def test_geolift_12_market_subsets():
    """Five 12-market GeoLift subsets. Subset 0 holds one market 2.3 times any other; the
    root LP spreads it over two near-balanced groups whose leftovers cannot balance, and
    plain branch and price needed 419 nodes. Group branching on that market closes it."""
    nodes = []
    for z in _subsets(12, 5):
        nodes.append(check(z)["nodes"])
    assert nodes[0] < 100, nodes


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**9), n=st.integers(4, 11), kind=st.integers(0, 3))
def test_rank1_property_against_brute_force(seed, n, kind):
    check(gen(np.random.default_rng(seed), n, kind), families="rank1")


def test_geolift_12_market_subsets_rank1():
    """With every rank-1 family the outlier subset is exact at the root."""
    nodes = [check(z, families="rank1")["nodes"] for z in _subsets(12, 5)]
    assert nodes[0] <= 1, nodes


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**9), n=st.integers(4, 11), kind=st.integers(0, 3))
@example(seed=0, n=10, kind=0)   # integer sizes, root LP exactly 0: the cost rescale used to fire at a later node
def test_managed_rank1_property_against_brute_force(seed, n, kind):
    check(gen(np.random.default_rng(seed), n, kind), families="rank1", purge_after=1, deep_depth=0, max_active=15)


def test_cut_management_keeps_the_optimum_and_bounds_the_pool():
    """16-market subset 1: purging and depth-limited separation leave the optimum unchanged,
    purge at least one cut, and never hold more than max_active cuts."""
    z = list(_subsets(16, 2))[1]
    plain = bcp.BCP(z, verbose=False, families=rank1.FAMILIES).solve(heuristic_seconds=2.0)
    managed = bcp.BCP(z, verbose=False, families=rank1.FAMILIES, purge_after=2, deep_depth=1, max_active=60)
    out = managed.solve(heuristic_seconds=2.0)
    assert out["status"] == plain["status"] == "optimal"
    assert abs(out["opt"] - plain["opt"]) <= 1e-9 * plain["opt"]
    assert out["purged"] >= 1 and out["max_pool"] <= 60


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**9), n=st.integers(4, 11), kind=st.integers(0, 3))
@example(seed=0, n=10, kind=0)
def test_tuned_setting_against_brute_force(seed, n, kind):
    """bcp.managed(), the setting tune_cuts.py selected, is exact on every instance."""
    check(gen(np.random.default_rng(seed), n, kind), **bcp.managed())


def test_the_cost_scale_is_fixed_at_the_root():
    """Integer sizes whose root LP is 0 exactly: no later node may change the cost scale, or the
    incumbent and the bounds are in different units (the defect reported 2/128 for an optimum of 2)."""
    z = gen(np.random.default_rng(0), 10, 0)
    for kw in ({}, {"families": "rank1", "purge_after": 1, "deep_depth": 0, "max_active": 15}):
        b = bcp.BCP(np.asarray(z, float), verbose=False, **(dict(kw, families=rank1.FAMILIES) if kw else {}))
        out = b.solve(time_limit=300)
        assert abs(out["opt"] - brute(np.asarray(z, float))) <= 1e-9 * max(1.0, out["opt"])


# --- reduced-cost fixing ----------------------------------------------------------------------------------------

def _sorted_pairs(z):
    """A poor start: neighbours in size order, paired (a triple at the top for odd N), with no repair."""
    o = [int(v) for v in np.argsort(z)]; n = len(z)
    return [tuple(o[i:i + 2]) for i in range(0, n - 3, 2)] + [tuple(o[-3:])] if n % 2 else [tuple(o[i:i + 2]) for i in range(0, n, 2)]


def _designs(n):
    """Every partition of range(n) into groups of 2..4."""
    def go(U):
        if not U:
            yield []
            return
        i = min(U); rest = sorted(U - {i})
        for k in (1, 2, 3):
            for others in itertools.combinations(rest, k):
                g = (i,) + others
                for d in go(U - set(g)):
                    yield [g] + d
    yield from go(frozenset(range(n)))


_ARMS = [{}, {"families": "rank1"}, {"families": "rank1", "purge_after": 1, "deep_depth": 0, "max_active": 15}]


def _bcp(z, **kw):
    if kw.get("families") == "rank1":
        kw = dict(kw, families=rank1.FAMILIES)
    return bcp.BCP(np.asarray(z, float), verbose=False, **kw)


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**9), n=st.integers(4, 9), kind=st.integers(0, 3), arm=st.integers(0, 2))
@example(seed=0, n=9, kind=2, arm=0)    # odd N: the parity dual enters every odd group's reduced cost
def test_fixing_keeps_every_group_of_every_design_no_dearer_than_the_incumbent(seed, n, kind, arm):
    """After the root, the universe holds every group of every design costing at most the incumbent, under
    every cut family; the master's pool stays inside it."""
    z = np.asarray(gen(np.random.default_rng(seed), n, kind), float)
    b = _bcp(z, **_ARMS[arm])
    out = b.solve(incumbent=_sorted_pairs(z), node_limit=1, root_mip_seconds=0.0)
    assert b.universe is not None
    ub = out["opt"]
    for d in _designs(n):
        if sum(dive._sc(z, g) for g in d) <= ub * (1 + 1e-9) + 1e-12:
            assert set(d) <= b.universe, (d, ub)
    assert set(b.cols) <= b.universe | set(out["design"])


def test_fixing_stops_pricing_and_shrinks_with_the_incumbent():
    """A 12-market subset from a poor start: once fixed no node prices, and the universe at the optimum is no
    larger than the one fixed at the start."""
    z = list(_subsets(12, 5))[0]
    b = _bcp(z)
    out = b.solve(incumbent=_sorted_pairs(z), root_mip_seconds=0.0, time_limit=300)
    assert out["status"] == "optimal" and abs(out["opt"] - brute(z)) <= 1e-9 * max(1.0, out["opt"])
    assert out["scans_after_fix"] == 0
    assert out["universe"] <= out["universe_at_fix"]
    assert out["opt"] < sum(dive._sc(z, g) for g in _sorted_pairs(z))       # the start was improved, so a refix ran
    assert out["refixes"] >= 1 and out["universe"] < out["universe_at_fix"]


def test_over_the_universe_limit_nothing_is_fixed():
    rng = np.random.default_rng(5)
    for t in range(6):
        z = gen(rng, int(rng.integers(5, 11)), t % 4)
        b = _bcp(z, universe_limit=0)
        out = check(z, universe_limit=0)
        b.solve(time_limit=300)
        assert b.universe is None and out["universe"] == 0


@pytest.mark.parametrize("kw", [{"fix_columns": False}, {"fix_columns": False, "families": "rank1"}])
def test_without_fixing_against_brute_force(kw):
    rng = np.random.default_rng(2)
    for t in range(12):
        check(gen(rng, int(rng.integers(4, 12)), t % 4), **kw)


# --- the root MIP over the fixed universe ------------------------------------------------------------------------

def _valid(z, design, n):
    assert sorted(g for c in design for g in c) == list(range(n))
    assert all(2 <= len(c) <= 4 for c in design)
    return sum(dive._sc(np.asarray(z, float), tuple(c)) for c in design)


def test_root_mip_improves_a_poor_start():
    """12-market subset 0 from sorted pairs: the MIP over the universe reaches the optimum before any branching,
    and the universe is fixed again at the new incumbent."""
    z = list(_subsets(12, 5))[0]
    start = _sorted_pairs(z)
    out = _bcp(z).solve(incumbent=start, node_limit=1, root_mip_seconds=30.0)
    want = brute(z)
    assert abs(out["opt"] - want) <= 1e-9 * want
    assert abs(_valid(z, out["design"], len(z)) - want) <= 1e-9 * want
    assert out["root_mip_improved"] and out["universe"] < out["universe_at_fix"]


def test_root_mip_closes_a_gap_below_the_solver_default():
    """One market 5,000 against sizes below 40: the second-best design is 8.3e-8 above the optimum (relative),
    under HiGHS's default relative gap of 1e-4. The MIP runs with gap 0, so it still moves to the optimum."""
    z = np.r_[np.random.default_rng(0).integers(1, 40, 8).astype(float), 5000.0]
    start = [(0, 1, 2, 8), (3, 4), (5, 6, 7)]
    assert abs(_valid(z, start, 9) - 24216245.0) < 1e-6
    out = _bcp(z).solve(incumbent=start, node_limit=1, root_mip_seconds=30.0)
    assert out["opt"] == 24216243.0 == brute(z)


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**9), n=st.integers(4, 9), kind=st.integers(0, 3), arm=st.integers(0, 2),
       budget=st.sampled_from([1e-9, 30.0]))
def test_root_mip_returns_a_partition_no_dearer_than_the_start(seed, n, kind, arm, budget):
    """With any budget, even one too short to solve, the incumbent after the root is a partition into groups of
    2 to 4, costs what the search reports, and costs no more than the start."""
    z = np.asarray(gen(np.random.default_rng(seed), n, kind), float)
    start = _sorted_pairs(z)
    out = _bcp(z, **_ARMS[arm]).solve(incumbent=start, node_limit=1, root_mip_seconds=budget)
    c = _valid(z, out["design"], n)
    assert abs(c - out["opt"]) <= 1e-9 * max(1.0, c)
    assert c <= _valid(z, start, n) * (1 + 1e-12)
