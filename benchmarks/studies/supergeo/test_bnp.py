"""Branch and price against brute force over every partition into groups of 2..4."""
import itertools, os, sys
import numpy as np
from hypothesis import given, settings, strategies as st, HealthCheck
from . import bnp, dive


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


def check(z):
    b = bnp.BnP(z, verbose=False); out = b.solve(time_limit=120)
    want = brute(np.asarray(z, float))
    assert out["status"] == "optimal"
    assert abs(out["opt"] - want) <= 1e-9 * max(1.0, abs(want)), (out["opt"], want)
    assert sorted(g for c in out["design"] for g in c) == list(range(len(z)))


def test_fixed():
    rng = np.random.default_rng(0)
    for t in range(30):
        n = int(rng.integers(4, 11))
        z = [rng.integers(1, 30, n).astype(float), rng.integers(1, 80, n) / 10.0, rng.lognormal(0, 1, n) * 100][t % 3]
        check(z)


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**9), n=st.integers(4, 11), kind=st.integers(0, 2))
def test_property(seed, n, kind):
    rng = np.random.default_rng(seed)
    z = [rng.integers(1, 30, n).astype(float), rng.integers(1, 80, n) / 10.0, rng.lognormal(0, 1.3, n) * 100][kind]
    check(z)
