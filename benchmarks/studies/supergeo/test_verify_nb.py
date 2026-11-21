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


def test_rows_search_with_the_lp_duals_from_a_poor_start():
    """The root LP's duals (so the minimum reduced cost is near 0 and the node bound tight) and the start a poor
    cover: the bound rests on each column's full dual sum, so a sum short of a market's dual overstates it and
    prunes the optimum (seed 36: optimum 17 against 25 for the start)."""
    rng = np.random.default_rng(36); n = int(rng.integers(8, 11)) // 2 * 2
    z = gen(rng, n, int(rng.integers(0, 3)))
    cand = [c for k in (2, 3, 4) for c in itertools.combinations(range(n), k)]
    from . import sg
    _, pi, _ = sg.master_lp(cand, np.array([float(exact_score(z, c)) for c in cand]), n)
    pi = np.asarray(pi, float)
    cost = {c: exact_score(z, c) for c in cand}
    m = min([cost[c] - sum(Fraction(float(pi[g])) for g in c) for c in cand] + [Fraction(0)])
    order = list(np.argsort(z)); start = [tuple(sorted(int(v) for v in order[i:i + 2])) for i in range(0, n, 2)]
    OPT = sum(cost[c] for c in start)
    want = brute(n, cand, cost)
    assert want < OPT
    rows = np.full((len(cand), 4), -1, np.int64)
    for j, c in enumerate(cand): rows[j, :len(c)] = c
    status, nodes, covers = vn.verify_rows(n, 2, rows, np.array([float(cost[c]) for c in cand]), pi, float(m), 0.0, float(OPT))
    assert status == 0
    assert min([sum(cost[cand[j]] for j in r) for r in covers] + [OPT]) == want


# --- the search on several threads -----------------------------------------------------------------------------------

@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**9), n=st.integers(4, 11), kind=st.integers(0, 2), dens=st.floats(0.3, 1.0),
       workers=st.integers(2, 4), split=st.integers(1, 200), bits=st.sampled_from([None, 3, 5]))
def test_parallel_search_finds_the_serial_optimum(seed, n, kind, dens, workers, split, bits):
    """The tree cut into at least `split` subtrees (or fewer, when it ends first) and searched by `workers` threads
    sharing the incumbent: the same status and exact optimum as the serial search and as brute force, and the
    design returned is an exact cover costing that optimum."""
    rng = np.random.default_rng(seed)
    z, pi, tau, cand, cost, m = _case(rng, n, kind, dens, 3.0 if seed % 2 else 0.0)
    cover = any_cover(n, cand, cost, rng)
    if cover is None:
        return
    OPT = sum(cost[c] for c in cover)
    one = vn.verify(n, 2, cand, cost, pi, m, tau, OPT)
    par = vn.verify(n, 2, cand, cost, pi, m, tau, OPT, bits=bits, workers=workers, split=split)
    assert par[0] == one[0] == 0
    assert par[2] == one[2] == brute(n, cand, cost)
    if par[3] is not None:
        assert sorted(g for c in par[3] for g in c) == list(range(n))
        assert sum(cost[c] for c in par[3]) == par[2] < OPT


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**9), n=st.integers(4, 11), kind=st.integers(0, 2), dens=st.floats(0.3, 1.0),
       workers=st.integers(1, 4), bits=st.sampled_from([None, 4]))
def test_one_subtree_walks_the_serial_tree(seed, n, kind, dens, workers, bits):
    """With split=1 the whole tree is one task: the subtree search walks the serial tree node for node."""
    rng = np.random.default_rng(seed)
    z, pi, tau, cand, cost, m = _case(rng, n, kind, dens, 3.0 if seed % 2 else 0.0)
    cover = any_cover(n, cand, cost, rng)
    if cover is None:
        return
    OPT = sum(cost[c] for c in cover)
    one = vn.verify(n, 2, cand, cost, pi, m, tau, OPT)
    par = vn.verify(n, 2, cand, cost, pi, m, tau, OPT, bits=bits, workers=workers, split=1)
    assert par[:3] == one[:3]


def test_parallel_search_on_seventy_markets_in_separate_blocks():
    rng = np.random.default_rng(7); n = 70
    z = gen(rng, n, 2); pi = np.zeros(n)
    cand = []
    for b in range(7):
        idx = list(range(10 * b, 10 * b + 10))
        for _ in range(3):
            cand += [g for g in _random_partition(rng, idx) if g not in cand]
    cost = {c: exact_score(z, c) for c in cand}
    cover = any_cover(n, cand, cost, rng); OPT = sum(cost[c] for c in cover)
    one = vn.verify(n, 2, cand, cost, pi, Fraction(0), 0.0, OPT)
    for split in (1, 8, 500):
        par = vn.verify(n, 2, cand, cost, pi, Fraction(0), 0.0, OPT, workers=4, split=split)
        assert par[0] == 0 and par[2] == one[2]


def _hard_case():
    rng = np.random.default_rng(3); n = 12
    z, pi, tau, cand, cost, m = _case(rng, n, 2, 1.0, 0.0)
    o = np.argsort(z)[::-1]                                   # pairs of the largest and smallest: a poor start
    start = [tuple(sorted((int(o[i]), int(o[n - 1 - i])))) for i in range(n // 2)]
    return n, pi, tau, cand, cost, m, sum(cost[c] for c in start)


def test_parallel_node_limit_and_candidate_cap_are_reported():
    n, pi, tau, cand, cost, m, OPT = _hard_case()
    full = vn.verify(n, 2, cand, cost, pi, m, tau, OPT, workers=4, split=16)
    assert full[0] == 0 and full[1] > 200
    assert vn.verify(n, 2, cand, cost, pi, m, tau, OPT, workers=4, split=16, node_limit=50)[0] == 1
    assert vn.verify(n, 2, cand, cost, pi, m, tau, OPT, workers=4, split=1, node_limit=50)[0] == 1
    assert vn.verify(n, 2, cand, cost, pi, m, tau, OPT, workers=2, split=1, cand_cap=0)[0] == 2


def test_bad_parallel_arguments_are_rejected():
    n, pi, tau, cand, cost, m, OPT = _hard_case()
    import pytest
    with pytest.raises(ValueError):
        vn.verify(n, 2, cand, cost, pi, m, tau, OPT, workers=0)
    with pytest.raises(ValueError):
        vn.verify(n, 2, cand, cost, pi, m, tau, OPT, workers=2, split=0)


def _root_task(n, cand, cost, pi, m, tau, OPT):
    """The setup of the parallel search for one task holding the whole tree."""
    p = vn._prepare(n, 2, cand, cost, pi, m, tau, OPT, bits=63)
    U0 = np.zeros(p["W"], np.int64)
    for w in range(p["W"]):
        U0[w] = (1 << min(63, n - w * 63)) - 1
    return p, U0


def test_a_subtree_reads_and_lowers_the_shared_incumbent():
    """A task prunes on the incumbent other threads have written, and writes every better one it finds: started
    from the poor cover it lowers the shared value, and started from the optimum's value it walks fewer nodes."""
    n, pi, tau, cand, cost, m, OPT = _hard_case()
    p, U0 = _root_task(n, cand, cost, pi, m, tau, OPT)
    buf = np.empty((1000, n // 2 + 2), np.int64)
    shared = np.array([p["opt_up"]])
    st1, nodes1, nc1 = vn._search_from(*p["args"], 10**9, np.zeros((1, 8), np.int64), 0, U0, 0.0, p["S0"], False, np.zeros(0, np.int64),
                                       shared, buf, 0)
    assert st1 == 0 and nc1 >= 1 and shared[0] < p["opt_up"]
    best = vn.verify(n, 2, cand, cost, pi, m, tau, OPT)[2]
    p2, _ = _root_task(n, cand, cost, pi, m, tau, best)
    shared2 = np.array([p2["opt_up"]])
    st2, nodes2, nc2 = vn._search_from(*p["args"], 10**9, np.zeros((1, 8), np.int64), 0, U0, 0.0, p["S0"], False, np.zeros(0, np.int64),
                                       shared2, buf, 0)
    assert st2 == 0 and nodes2 < nodes1


def _root_dual_case(seed):
    """All groups of 2 to 4 of an odd number of markets, with the certificate's root duals (the market duals and
    the parity dual tau), so the minimum reduced cost is near 0 and the node bound tight, and a poor start."""
    from . import certify_exact as ce
    rng = np.random.default_rng(seed); n = int(rng.choice([7, 9, 11]))
    z, scale, pi, tau = ce._root(gen(rng, n, 2), 2, 4)
    cand = [c for k in (2, 3, 4) for c in itertools.combinations(range(n), k)]
    cost = {c: exact_score(z, c) for c in cand}
    T = Fraction(tau)
    m = min([cost[c] - sum(Fraction(float(pi[g])) for g in c) - (T if len(c) % 2 else 0) for c in cand] + [Fraction(0)])
    o = [int(v) for v in np.argsort(z)[::-1]]
    start = [tuple(sorted((o[i], o[n - 2 - i]))) for i in range((n - 3) // 2)] + [tuple(sorted(o[(n - 3) // 2:(n - 3) // 2 + 3]))]
    return n, pi, tau, cand, cost, m, sum(cost[c] for c in start)


def test_parallel_search_with_the_root_duals():
    """With the root duals the bound below a frontier node must drop the parity dual once an odd column is on the
    path and must drop the column's market duals: either left in prunes the optimum."""
    for seed in range(12):
        n, pi, tau, cand, cost, m, OPT = _root_dual_case(seed)
        want = brute(n, cand, cost)
        for split in (4, 40, 400):
            par = vn.verify(n, 2, cand, cost, pi, m, tau, OPT, workers=2, split=split)
            assert par[0] == 0 and par[2] == want, (seed, split)


def test_parallel_node_limit_counts_the_total():
    """The node limit bounds the total over the subtrees, not each one: at half the nodes the full search needs
    (one thread, so the walk is fixed) the search stops."""
    n, pi, tau, cand, cost, m, OPT = _hard_case()
    full = vn.verify(n, 2, cand, cost, pi, m, tau, OPT, workers=1, split=64)
    assert full[0] == 0
    assert vn.verify(n, 2, cand, cost, pi, m, tau, OPT, workers=1, split=64, node_limit=full[1] // 2)[0] == 1


def _big_tree_case():
    """Twenty-two lognormal markets, every group of 2 to 4, no duals and a poor start: a tree of 716,236 nodes. (With
    integer markets the tree is a few thousand nodes: many groups balance exactly, the easy phase of Mertens 2001.)"""
    rng = np.random.default_rng(5); n = 22
    z = gen(rng, n, 2)
    cand = [c for k in (2, 3, 4) for c in itertools.combinations(range(n), k)]
    cost = {c: exact_score(z, c) for c in cand}
    o = [int(v) for v in np.argsort(z)[::-1]]
    start = [tuple(sorted((o[i], o[n - 1 - i]))) for i in range(n // 2)]
    return n, np.zeros(n), 0.0, cand, cost, Fraction(0), sum(cost[c] for c in start)


def test_the_node_limit_bounds_the_total_over_the_threads():
    """Each thread adds its nodes to its own slot and checks the sum every 1,024 nodes, so the reported total stays
    within 1,024 nodes per thread of the limit, and a search the limit stops reports status 1."""
    n, pi, tau, cand, cost, m, OPT = _big_tree_case()
    lim = 300_000
    assert vn.verify(n, 2, cand, cost, pi, m, tau, OPT, node_limit=lim)[0] == 1     # the tree is bigger than the limit
    for workers, split in ((4, 64), (4, 4), (2, 1000)):
        st, nodes, val, d = vn.verify(n, 2, cand, cost, pi, m, tau, OPT, node_limit=lim, workers=workers, split=split)
        assert st == 1 and lim < nodes <= lim + 1024 * workers, (workers, split, nodes)


# --- the capped stages' search on threads ------------------------------------------------------------------------------

def _rows_of(cand):
    rows = np.full((len(cand), 4), -1, np.int64)
    for j, c in enumerate(cand): rows[j, :len(c)] = c
    return rows


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**9), n=st.integers(4, 11), kind=st.integers(0, 2), dens=st.floats(0.3, 1.0),
       workers=st.integers(1, 4), split=st.integers(1, 200))
def test_rows_search_on_threads_finds_the_same_optimum(seed, n, kind, dens, workers, split):
    """verify_rows on threads, run to completion: the exact best of the covers it records is verify's optimum, and
    every recorded cover is an exact cover given as row indices."""
    rng = np.random.default_rng(seed)
    z, pi, tau, cand, cost, m = _case(rng, n, kind, dens, 3.0 if seed % 2 else 0.0)
    cover = any_cover(n, cand, cost, rng)
    if cover is None:
        return
    OPT = sum(cost[c] for c in cover)
    want = vn.verify(n, 2, cand, cost, pi, m, tau, OPT)[2]
    fc = np.array([float(cost[c]) for c in cand])
    status, nodes, covers = vn.verify_rows(n, 2, _rows_of(cand), fc, pi, float(m), tau, float(OPT), workers=workers,
                                           split=split)
    assert status == 0
    for r in covers:
        assert sorted(g for j in r for g in cand[j]) == list(range(n))
    assert min([sum(cost[cand[j]] for j in r) for r in covers] + [OPT]) == want


def test_rows_search_on_threads_respects_the_node_limit():
    n, pi, tau, cand, cost, m, OPT = _big_tree_case()
    fc = np.array([float(cost[c]) for c in cand])
    st, nodes, covers = vn.verify_rows(n, 2, _rows_of(cand), fc, pi, float(m), tau, float(OPT), node_limit=300_000,
                                       workers=4)
    assert st == 1 and 300_000 < nodes <= 300_000 + 4 * 1024


def test_rows_search_rejects_bad_workers():
    n, pi, tau, cand, cost, m, OPT = _hard_case()
    fc = np.array([float(cost[c]) for c in cand])
    import pytest
    for w in (0, -2):
        with pytest.raises(ValueError):
            vn.verify_rows(n, 2, _rows_of(cand), fc, pi, float(m), tau, float(OPT), workers=w)


def test_rows_search_with_workers_uses_the_threads(monkeypatch):
    calls = []
    real = vn._parallel_core
    monkeypatch.setattr(vn, "_parallel_core", lambda *a, **k: calls.append(1) or real(*a, **k))
    n, pi, tau, cand, cost, m, OPT = _hard_case()
    fc = np.array([float(cost[c]) for c in cand])
    vn.verify_rows(n, 2, _rows_of(cand), fc, pi, float(m), tau, float(OPT))
    assert calls == []
    vn.verify_rows(n, 2, _rows_of(cand), fc, pi, float(m), tau, float(OPT), workers=2)
    assert calls == [1]
