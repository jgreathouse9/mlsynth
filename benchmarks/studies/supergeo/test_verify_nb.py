"""verify_nb against brute-force exact-cover enumeration in Fraction arithmetic."""
import pathlib
import itertools, os, sys
from fractions import Fraction
import numpy as np
from hypothesis import given, settings, strategies as st, HealthCheck
from . import verify_nb as vn, mitm_exact as mx


def exact_score(z, c):
    return mx.exact_rc(z, np.zeros(len(z)), c)


def brute(n, cand, cost):
    """Exact optimum over exact covers (None if there is none)."""
    by = {}
    for c in cand: by.setdefault(min(c), []).append(c)
    best = [None]
    def go(U, acc):
        if not U:
            if best[0] is None or acc < best[0]: best[0] = acc
            return
        i = min(U)
        for c in by.get(i, []):
            if set(c) <= U: go(U - set(c), acc + cost[c])
    go(frozenset(range(n)), Fraction(0)); return best[0]


def any_cover(n, cand, cost, rng):
    by = {}
    for c in cand: by.setdefault(min(c), []).append(c)
    def go(U):
        if not U: return []
        i = min(U); opts = [c for c in by.get(i, []) if set(c) <= U]; rng.shuffle(opts)
        for c in opts:
            r = go(U - set(c))
            if r is not None: return [c] + r
        return None
    return go(frozenset(range(n)))


def run_case(z, pi, tau, cand, rng):
    n = len(z); lo = 2
    cost = {c: exact_score(z, c) for c in cand}
    T = Fraction(float(tau))
    rcs = [cost[c] - sum(Fraction(float(pi[g])) for g in c) - (T if len(c) % 2 else 0) for c in cand]
    m = min(min(rcs, default=Fraction(0)), Fraction(0))
    truth = brute(n, cand, cost)
    cover = any_cover(n, cand, cost, rng)
    if cover is None:
        return truth is None
    OPT = sum(cost[c] for c in cover)
    status, nodes, val, design = vn.verify(n, lo, cand, cost, pi, m, float(tau), OPT)
    assert status == 0
    if design is not None:
        assert sorted(g for c in design for g in c) == list(range(n))
    return val == truth


def gen(rng, n, kind):
    if kind == 0: return rng.integers(1, 30, n).astype(float)
    if kind == 1: return rng.integers(1, 80, n) / 10.0
    return rng.lognormal(0, 1, n) * 100


def test_against_brute_force_fixed():
    rng = np.random.default_rng(0); checked = 0
    for trial in range(120):
        n = int(rng.integers(6, 12)); z = gen(rng, n, trial % 3)
        pi = np.zeros(n) if trial % 2 else rng.normal(0, 5, n)
        tau = float(rng.uniform(0, 3)) if n % 2 else 0.0
        allc = [c for k in (2, 3, 4) for c in itertools.combinations(range(n), k)]
        cand = allc if trial % 4 == 0 else [c for c in allc if rng.random() < 0.5]
        assert run_case(z, pi, tau, cand, rng), trial
        checked += 1
    assert checked == 120


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**9), n=st.integers(4, 11), kind=st.integers(0, 2), dens=st.floats(0.3, 1.0))
def test_against_brute_force_property(seed, n, kind, dens):
    rng = np.random.default_rng(seed); z = gen(rng, n, kind)
    pi = rng.normal(0, 3, n) if seed % 2 else np.zeros(n)
    tau = float(rng.uniform(0, 2)) if n % 2 else 0.0
    cand = [c for k in (2, 3, 4) for c in itertools.combinations(range(n), k) if rng.random() < dens]
    assert run_case(z, pi, tau, cand, rng)


def test_float_collapsed_near_tie_is_found():
    """Two covers whose exact costs differ by 2 but whose double costs are equal:
    z = (0, p, p-1, 1); {0,1}{2,3} costs p^2 + (p-2)^2 = 2(p-1)^2 + 2, {0,2}{1,3} costs 2(p-1)^2.
    The worse one is the incumbent; the search must still return the better one."""
    for p in (2.0 ** 52, 2.0 ** 52 - 12345, 2.0 ** 51 + 777, 3.0 * 2 ** 50 + 1):
        z = np.array([0.0, p, p - 1, 1.0]); n = 4
        cand = [(0, 1), (2, 3), (0, 2), (1, 3)]
        cost = {c: exact_score(z, c) for c in cand}
        worse = cost[(0, 1)] + cost[(2, 3)]; better = cost[(0, 2)] + cost[(1, 3)]
        assert worse - better == 2
        assert float(cost[(0, 1)]) + float(cost[(2, 3)]) == float(cost[(0, 2)]) + float(cost[(1, 3)]), "construction must collapse in floating point"
        status, nodes, val, design = vn.verify(n, 2, cand, cost, np.zeros(n), Fraction(0), 0.0, worse)
        assert status == 0 and val == better and sorted(design) == [(0, 2), (1, 3)]


def test_parity_dual_large():
    """Odd N with tau comparable to the costs: the bound may add tau only while an odd group is still owed."""
    rng = np.random.default_rng(5)
    for trial in range(80):
        n = int(rng.choice([5, 7, 9])); z = gen(rng, n, trial % 3)
        allc = [c for k in (2, 3, 4) for c in itertools.combinations(range(n), k)]
        scale = float(np.median([float(exact_score(z, c)) for c in allc]))
        tau = float(rng.uniform(0.5, 5.0)) * scale
        pi = np.zeros(n) if trial % 2 else rng.normal(0, scale / 4, n)
        cand = allc if trial % 3 == 0 else [c for c in allc if rng.random() < 0.6]
        assert run_case(z, pi, tau, cand, rng), trial


def test_node_budget_on_geolift():
    """The fewest-fitting-columns branching keeps the GeoLift proof small (55k nodes when written)."""
    import pandas as pd
    from . import certify_exact as ce
    df = pd.read_csv(str(pathlib.Path(__file__).resolve().parents[3] / "basedata" / "geolift_test_data.csv")); dates = sorted(df.date.unique())
    pre = df[df.date.isin(dates[-56:-28])].groupby("location").Y.sum().to_numpy(float)
    out = ce.certify(pre / np.median(np.diff(np.sort(pre))), verbose=False)
    assert out["proof"].startswith("exact end to end") and out["verify_nodes"] < 1_000_000
    z0 = pre / np.median(np.diff(np.sort(pre)))
    # the certified optimum is the exact cost of the returned design on the ORIGINAL data
    assert sum(exact_score(z0, c) for c in out["design"]) == out["opt_exact"]
    assert abs(out["opt"] - 0.0088722102117969) < 1e-12 * 0.0088722102117969


import json as _json
_W = _json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mutant_witnesses.json")))


import pytest as _pytest
@_pytest.mark.parametrize("name", sorted(_W))
def test_mutant_witness(name):
    """Instances found by fuzzing that separate a broken search from brute force:
    the parity dual added after an odd group is chosen, the incumbent tightened
    below its exact cost, near-tie leaves dropped, and no rounding margin."""
    w = _W[name]
    z = np.array([float.fromhex(v) for v in w["z"]]); pi = np.array([float.fromhex(v) for v in w["pi"]]); tau = float.fromhex(w["tau"])
    cand = [tuple(c) for c in w["cand"]]; n = len(z)
    cost = {c: exact_score(z, c) for c in cand}
    T_ = Fraction(tau)
    m = min(min(cost[c] - sum(Fraction(float(pi[g])) for g in c) - (T_ if len(c) % 2 else 0) for c in cand), Fraction(0))
    status, nodes, val, design = vn.verify(n, 2, cand, cost, pi, m, tau, Fraction(w["incumbent"]))
    assert status == 0 and val == Fraction(w["truth"]) == brute(n, cand, cost)


# --- more than 62 markets: the uncovered set across several words ---------------------------------------------------

def _case(rng, n, kind, dens, pi_scale):
    z = gen(rng, n, kind)
    pi = rng.normal(0, pi_scale, n) if pi_scale else np.zeros(n)
    tau = float(rng.uniform(0, 2)) if n % 2 else 0.0
    cand = [c for k in (2, 3, 4) for c in itertools.combinations(range(n), k) if rng.random() < dens]
    cost = {c: exact_score(z, c) for c in cand}
    T = Fraction(tau)
    m = min([cost[c] - sum(Fraction(float(pi[g])) for g in c) - (T if len(c) % 2 else 0) for c in cand] + [Fraction(0)])
    return z, pi, tau, cand, cost, m


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**9), n=st.integers(4, 11), kind=st.integers(0, 2), dens=st.floats(0.3, 1.0),
       bits=st.integers(3, 5))
def test_several_words_search_exactly_as_one(seed, n, kind, dens, bits):
    """The multi-word search with `bits` markets per word walks the same tree as the one-word search: the same
    status, node count and optimum, and the optimum is the brute-force one."""
    rng = np.random.default_rng(seed)
    z, pi, tau, cand, cost, m = _case(rng, n, kind, dens, 3.0 if seed % 2 else 0.0)
    cover = any_cover(n, cand, cost, rng)
    if cover is None:
        return
    OPT = sum(cost[c] for c in cover)
    one = vn.verify(n, 2, cand, cost, pi, m, tau, OPT)
    many = vn.verify(n, 2, cand, cost, pi, m, tau, OPT, bits=bits)
    assert one[:3] == many[:3]
    assert many[2] == brute(n, cand, cost)


def _random_partition(rng, idx):
    """A partition of idx into groups of 2 to 4 (sizes drawn until they add up)."""
    idx = list(rng.permutation(idx)); out = []
    while idx:
        left = len(idx)
        k = left if left <= 4 else int(rng.choice([s for s in (2, 3, 4) if left - s == 0 or left - s >= 2]))
        out.append(tuple(sorted(int(v) for v in idx[:k]))); idx = idx[k:]
    return out


def test_seventy_markets_in_separate_blocks():
    """70 markets whose candidate groups come from three random partitions of each block of 10, so an exact cover
    is a cover of each block, each block has a handful of covers, and the optimum is the sum of the blocks'
    brute-force optima. Markets 63 to 69 sit in the second word."""
    rng = np.random.default_rng(7); n = 70
    z = gen(rng, n, 2); pi = np.zeros(n)
    cand = []
    for b in range(7):
        idx = list(range(10 * b, 10 * b + 10))
        for _ in range(3):
            cand += [g for g in _random_partition(rng, idx) if g not in cand]
    cost = {c: exact_score(z, c) for c in cand}
    want = Fraction(0)
    for b in range(7):
        idx = list(range(10 * b, 10 * b + 10)); sub = [c for c in cand if c[0] in idx]
        sc = {tuple(g - 10 * b for g in c): cost[c] for c in sub}
        want += brute(10, list(sc), sc)
    cover = any_cover(n, cand, cost, rng)
    status, nodes, val, design = vn.verify(n, 2, cand, cost, pi, Fraction(0), 0.0, sum(cost[c] for c in cover))
    assert status == 0 and val == want
    design = design if design is not None else cover          # None: the starting cover was already optimal
    assert sorted(g for c in design for g in c) == list(range(n))


@settings(max_examples=80, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**9), n=st.integers(4, 11), kind=st.integers(0, 2), dens=st.floats(0.3, 1.0))
def test_rows_search_finds_the_same_optimum(seed, n, kind, dens):
    """verify_rows takes the survivors as an index array and float costs (the capped stages' search) and returns
    every cover it records, as row indices; run to completion the exact best of them is verify's optimum."""
    rng = np.random.default_rng(seed)
    z, pi, tau, cand, cost, m = _case(rng, n, kind, dens, 3.0 if seed % 2 else 0.0)
    cover = any_cover(n, cand, cost, rng)
    if cover is None:
        return
    OPT = sum(cost[c] for c in cover)
    want = vn.verify(n, 2, cand, cost, pi, m, tau, OPT)[2]
    rows = np.full((len(cand), 4), -1, np.int64)
    for j, c in enumerate(cand): rows[j, :len(c)] = c
    fc = np.array([float(cost[c]) for c in cand])
    status, nodes, covers = vn.verify_rows(n, 2, rows, fc, pi, float(m), tau, float(OPT))
    assert status == 0
    got = min([sum(cost[cand[j]] for j in r) for r in covers] + [OPT])     # the exact best of the recorded covers
    assert got == want
