"""Branch, cut and price against brute force over every partition into groups of 2..4."""
import itertools, pathlib
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st, HealthCheck
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
                                {"families": "rank1"}, {"families": "rank1", "group_branching": False}])
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
