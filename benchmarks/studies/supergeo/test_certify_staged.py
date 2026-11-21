"""The staged certificate: exact elimination redone at each better incumbent, the exact search between."""
import json, os, pathlib
from fractions import Fraction
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st, HealthCheck
from . import certify_exact as ce, dive, test_bcp

HERE = pathlib.Path(__file__).resolve().parent


def _check(z, **kw):
    z = np.asarray(z, float)
    out = ce.certify_staged(z, **kw)
    want = test_bcp.brute(z)
    assert out["proved"], out["proof"]
    assert abs(out["opt"] - want) <= 1e-9 * max(1.0, want)
    assert sorted(g for c in out["design"] for g in c) == list(range(len(z)))
    assert all(2 <= len(c) <= 4 for c in out["design"])
    assert out["lb"] <= out["opt"] * (1 + 1e-12) + 1e-300
    return out


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**9), n=st.integers(4, 11), kind=st.integers(0, 3))
def test_staged_certificate_against_brute_force(seed, n, kind):
    _check(test_bcp.gen(np.random.default_rng(seed), n, kind), heuristic_seconds=0.0)


def test_a_poor_start_and_tiny_caps_still_reach_the_optimum():
    rng = np.random.default_rng(4)
    for t in range(8):
        z = test_bcp.gen(rng, int(rng.integers(6, 12)), t % 4)
        _check(z, incumbent=test_bcp._sorted_pairs(np.asarray(z, float)), heuristic_seconds=0.0, caps=(1, 10))


def test_stage_survivors_never_grow_and_the_last_stage_is_complete():
    z = list(test_bcp._subsets(16, 3))[1]
    out = ce.certify_staged(z, incumbent=test_bcp._sorted_pairs(z), heuristic_seconds=0.0, caps=(200, 2_000))
    sizes = [s["survivors"] for s in out["stages"]]
    assert sizes == sorted(sizes, reverse=True)
    for a, b in zip(out["stages"], out["stages"][1:]):
        if a["improved"]:
            assert b["survivors"] < a["survivors"], (a, b)       # a better incumbent eliminates more
    assert any(s["improved"] for s in out["stages"][:-1])
    assert out["stages"][-1]["status"] == 0 or out["proof"].startswith("exact: optimum equals")


def test_the_same_exact_optimum_as_the_certificate():
    for k, reps in ((12, 3), (16, 2)):
        for z in test_bcp._subsets(k, reps):
            a = ce.certify(z, verbose=False); b = ce.certify_staged(z)
            assert b["proved"] and a["opt_exact"] == b["opt_exact"]


def test_the_50_state_optimum():
    """The 50-state panel of the SYNDES benchmark (months 34 to 37): the staged certificate from a 1 s heuristic
    reaches the optimum the single-threshold certificate proved (fifty_optima.json)."""
    Y = pd.read_csv(HERE.parents[2] / "basedata" / "urate_cps.csv", header=None).to_numpy(float)
    pre = Y[-7:-3].sum(0); z = pre / np.median(np.diff(np.sort(pre)))
    out = ce.certify_staged(z, heuristic_seconds=1.0)
    ref = json.loads((HERE / "fifty_optima.json").read_text())["urate"]["opt"]
    assert out["proved"] and abs(out["opt"] - ref) <= 1e-12 * ref
    capped = [s for s in out["stages"] if not s["exact"]]   # here the halving rule ends the staging (941 then 549)
    for a, b in zip(capped, capped[1:]):
        assert b["survivors"] <= a["survivors"] / 2


def test_bad_input_is_rejected():
    with pytest.raises(ValueError):
        ce.certify_staged(np.arange(1.0, 9.0), caps=(1000, 10))
    with pytest.raises(ValueError):
        ce.certify_staged(np.arange(1.0, 9.0), caps=(0,))


def test_the_proof_comes_from_an_exact_stage():
    """Capped stages search on floating-point costs and only find incumbents; the proof is the last stage, run on
    exact costs to completion."""
    rng = np.random.default_rng(11)
    for t in range(6):
        z = np.asarray(test_bcp.gen(rng, int(rng.integers(6, 12)), t % 4), float)
        out = ce.certify_staged(z, incumbent=test_bcp._sorted_pairs(z), heuristic_seconds=0.0, caps=(1, 10))
        assert out["proved"]
        if not out["proof"].startswith("exact: optimum equals"):
            last = out["stages"][-1]
            assert last["exact"] and last["status"] == 0
            assert all(not s["exact"] for s in out["stages"][:-1])


def test_automatic_stages_stop_when_the_survivors_stop_halving():
    z = list(test_bcp._subsets(16, 3))[1]
    out = ce.certify_staged(z, incumbent=test_bcp._sorted_pairs(z), heuristic_seconds=0.0, caps="auto")
    assert out["proved"]
    capped = [s for s in out["stages"] if not s["exact"]]
    for a, b in zip(capped, capped[1:]):
        assert a["improved"] and b["survivors"] <= a["survivors"] / 2
    assert out["stages"][-1]["exact"]


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**9), n=st.integers(4, 11), kind=st.integers(0, 3))
def test_automatic_defaults_against_brute_force(seed, n, kind):
    _check(test_bcp.gen(np.random.default_rng(seed), n, kind), caps="auto", heuristic_seconds="auto")


def test_the_exact_stage_on_threads_proves_the_same_optimum():
    """The final stage searched on 1 and 4 threads and on this thread: the same proved optimum, and the stage
    records its thread count."""
    z = list(test_bcp._subsets(16, 3))[1]
    outs = {w: ce.certify_staged(z, incumbent=test_bcp._sorted_pairs(z), heuristic_seconds=0.0, caps=(200,),
                                 workers=w) for w in (None, 1, 4)}
    for w, out in outs.items():
        assert out["proved"] and out["opt_exact"] == outs[None]["opt_exact"]
        assert out["stages"][-1]["workers"] == w
    auto = ce.certify_staged(z, heuristic_seconds=0.0)
    assert auto["stages"][-1]["workers"] == min(4, os.cpu_count() or 1)


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**9), n=st.integers(4, 11), kind=st.integers(0, 3))
def test_the_exact_stage_on_four_threads_against_brute_force(seed, n, kind):
    _check(test_bcp.gen(np.random.default_rng(seed), n, kind), heuristic_seconds=0.0, workers=4)


def test_bad_workers_are_rejected():
    for w in (0, -1, "many"):
        with pytest.raises(ValueError):
            ce.certify_staged(np.arange(1.0, 9.0), workers=w)


def _stage_inputs(z):
    z, scale, pi, tau = ce._root(np.asarray(z, float), 2, 4)
    m, S, LB, neg = ce._exact_bound(z, pi, tau, 2, 4)
    return z, pi, tau, S, m


def _keys(rows):
    return {tuple(int(v) for v in r if v >= 0) for r in rows}


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**9), n=st.integers(5, 12), kind=st.integers(0, 3), f=st.floats(0.0, 1.0))
def test_screening_the_last_survivors_equals_a_new_window(seed, n, kind, f):
    """At an incumbent no dearer than the last one, the exact screen of the last stage's survivors keeps exactly the
    groups a new window and screen keep: survivors at a lower threshold are a subset of those at a higher one."""
    z, pi, tau, S, m = _stage_inputs(test_bcp.gen(np.random.default_rng(seed), n, kind))
    LB = S + Fraction(n, 2) * m
    UB1 = Fraction(test_bcp.brute(z)) * 3 + 1                       # above the optimum
    UB2 = LB + (UB1 - LB) * Fraction(f)                              # anywhere from the bound up to UB1
    rows1, _ = ce._survivor_rows(z, pi, tau, 2, 4, S, m, UB1)
    again = ce._screen_rows(z, pi, tau, 2, S, m, UB2, rows1)
    fresh, _ = ce._survivor_rows(z, pi, tau, 2, 4, S, m, UB2)
    assert _keys(again) == _keys(fresh)
    assert len(again) == len(_keys(again))                           # no duplicates


def test_the_window_is_built_once_for_the_stages(monkeypatch):
    """One window for the bound and one for the first stage; later stages screen the last stage's rows, and record
    how many they screened."""
    calls = []
    real = ce.safe_window
    monkeypatch.setattr(ce, "safe_window", lambda *a, **k: calls.append(a[4]) or real(*a, **k))
    z = list(test_bcp._subsets(16, 3))[1]
    out = ce.certify_staged(z, incumbent=test_bcp._sorted_pairs(z), heuristic_seconds=0.0, caps=(200, 2_000))
    assert out["proved"] and len(out["stages"]) >= 3
    assert len(calls) == 2
    for a, b in zip(out["stages"], out["stages"][1:]):
        assert b["window"] == a["survivors"]


# --- a stated optimality gap ------------------------------------------------------------------------------------------

def _exact_brute(z, t=None):
    """The exact optimum (a Fraction) over every partition into groups of 2 to 4, by recursion on the lowest market;
    with t, over partitions whose every group costs at most t (None when there is none)."""
    from itertools import combinations
    from . import mitm_exact as mx
    n = len(z); zeros = np.zeros(n); memo = {}
    def best(U):
        if not U:
            return Fraction(0)
        if U in memo:
            return memo[U]
        i = min(U); rest = sorted(U - {i}); out = None
        for k in (1, 2, 3):
            for others in combinations(rest, k):
                g = (i,) + others
                left = U - set(g)
                if len(left) == 1:
                    continue
                c = mx.exact_rc(z, zeros, g)
                if t is not None and c > Fraction(t):
                    continue
                r = best(frozenset(left))
                if r is None:
                    continue
                v = c + r
                if out is None or v < out:
                    out = v
        memo[U] = out
        return out
    return best(frozenset(range(n)))


def _design_cost(z, design):
    return sum(min((2 * sum(z[g] for t, g in enumerate(c) if mk >> t & 1) - sum(z[g] for g in c)) ** 2
                   for mk in range(1, (1 << len(c)) - 1)) for c in design)


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**9), n=st.integers(4, 11), kind=st.integers(0, 3),
       gap=st.sampled_from([1e-6, 1e-3, 0.05, 0.3, 0.9]))
def test_a_gap_certificate_against_brute_force(seed, n, kind, gap):
    """With gap g the certificate proves that no design costs less than (1 - g) times the design it returns: the
    brute-force optimum lies between the two, the proven gap is at most g, and the design is a partition costing
    what it reports."""
    z = np.asarray(test_bcp.gen(np.random.default_rng(seed), n, kind), float)
    out = ce.certify_staged(z, gap=gap, heuristic_seconds=0.0)
    want = _exact_brute(z)                        # exact: a gap claims an exact inequality
    assert out["proved"], out["proof"]
    assert want <= out["opt_exact"]
    assert out["opt_exact"] * (1 - Fraction(gap)) <= want
    assert 0 <= out["gap_proved"] <= gap
    assert out["lb_proved_exact"] <= want
    assert sorted(g for c in out["design"] for g in c) == list(range(n))
    assert abs(_design_cost(z, out["design"]) - out["opt"]) <= 1e-9 * max(1.0, out["opt"])


def test_an_exact_certificate_proves_a_zero_gap():
    z = list(test_bcp._subsets(16, 3))[1]
    out = ce.certify_staged(z, incumbent=test_bcp._sorted_pairs(z), heuristic_seconds=0.0)
    assert out["proved"] and out["gap"] == 0 and out["gap_proved"] == 0 and out["lb_proved_exact"] == out["opt_exact"]


def test_a_gap_the_bound_proves_needs_no_exact_search():
    """The 50-state panel's bound is 0.135 of its optimum. The 1 s heuristic's design costs more than ten times the
    optimum, so a capped stage runs; once it has improved the design, the bound proves a gap of 0.95 and the exact
    stage never runs."""
    Y = pd.read_csv(HERE.parents[2] / "basedata" / "urate_cps.csv", header=None).to_numpy(float)
    pre = Y[-7:-3].sum(0); z = pre / np.median(np.diff(np.sort(pre)))
    out = ce.certify_staged(z, gap=0.95, heuristic_seconds=1.0)
    assert out["proved"] and out["proof"].startswith("within the gap: the bound")
    assert not any(s["exact"] for s in out["stages"])
    assert out["gap_proved"] <= 0.95 and out["lb_proved_exact"] == out["lb_exact"]


def test_a_gap_eliminates_more_and_searches_less():
    """With a gap below the bound's own gap the stages still run, but the elimination is at (1 - g) times the
    incumbent, so the first stage keeps fewer groups, and the proof searches no more nodes."""
    z = list(test_bcp._subsets(16, 3))[1]
    kw = dict(incumbent=test_bcp._sorted_pairs(z), heuristic_seconds=0.0, caps=(200,))
    exact = ce.certify_staged(z, **kw)
    g = float(0.5 * (exact["opt_exact"] - exact["lb_exact"]) / exact["opt_exact"])
    loose = ce.certify_staged(z, gap=g, **kw)
    assert loose["proved"] and loose["stages"]
    assert loose["stages"][0]["survivors"] < exact["stages"][0]["survivors"]
    assert loose["stages"][-1]["nodes"] <= exact["stages"][-1]["nodes"]
    assert loose["gap_proved"] <= g


def test_bad_gaps_are_rejected():
    for g in (-0.1, 1.0, 1.5, "big", float("nan")):
        with pytest.raises(ValueError):
            ce.certify_staged(np.arange(1.0, 9.0), gap=g)


def test_a_design_the_exact_stage_finds_is_proved_optimal():
    """From a poor start with a one-node capped stage the exact stage itself finds the better design. A design it
    finds below the target is the optimum, so the proven bound is its cost and the proven gap 0, with or without a
    gap asked for."""
    rng = np.random.default_rng(4); seen = 0
    for t in range(12):
        z = np.asarray(test_bcp.gen(rng, int(rng.integers(6, 12)), t % 4), float)
        for gap in (0.0, 1e-3):
            out = ce.certify_staged(z, incumbent=test_bcp._sorted_pairs(z), heuristic_seconds=0.0, caps=(1,), gap=gap)
            last = out["stages"][-1] if out["stages"] else None
            if last is not None and last["exact"] and last["improved"]:
                seen += 1
                assert out["lb_proved_exact"] == out["opt_exact"] == _exact_brute(z)
                assert out["gap_proved"] == 0
    assert seen >= 3


# --- a balance standard: no group's imbalance above a stated value ---------------------------------------------------

def _group_costs(z):
    from itertools import combinations
    from . import mitm_exact as mx
    n = len(z); zeros = np.zeros(n)
    return sorted(mx.exact_rc(z, zeros, g) for k in (2, 3, 4) for g in combinations(range(n), k))


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**9), n=st.integers(4, 10), kind=st.integers(0, 3),
       q=st.sampled_from([0.01, 0.05, 0.2, 0.6]), gap=st.sampled_from([0.0, 1e-3, 0.3]))
def test_a_balance_standard_against_brute_force(seed, n, kind, q, gap):
    """With max_group_cost t only groups costing at most t are allowed. When no partition meets the standard the
    certificate proves that and returns no design; otherwise its design meets the standard and is within the gap of
    the best design that does (exact brute force over the allowed groups)."""
    z = np.asarray(test_bcp.gen(np.random.default_rng(seed), n, kind), float)
    costs = _group_costs(z)
    t = float(costs[int(q * (len(costs) - 1))])               # a threshold some partitions may not meet
    want = _exact_brute(z, t)
    out = ce.certify_staged(z, max_group_cost=t, gap=gap, heuristic_seconds=0.0)
    assert out["proved"], out["proof"]
    if want is None:
        assert out["feasible"] is False and out["design"] is None and out["opt"] is None
        assert out["proof"].startswith("no design meets the standard")
        return
    assert out["feasible"] is True and out["standard"]["met"]
    assert out["standard"]["max_group"] <= t
    assert want <= out["opt_exact"] and out["opt_exact"] * (1 - Fraction(gap)) <= want
    assert sorted(g for c in out["design"] for g in c) == list(range(n))


def test_a_standard_every_design_meets_changes_nothing():
    z = list(test_bcp._subsets(16, 3))[1]
    kw = dict(incumbent=test_bcp._sorted_pairs(z), heuristic_seconds=0.0)
    free = ce.certify_staged(z, **kw)
    loose = ce.certify_staged(z, max_group_cost=float(max(_group_costs(np.asarray(z, float)))) * 2, **kw)
    assert free["standard"] is None and loose["standard"]["met"]
    assert loose["opt_exact"] == free["opt_exact"] and loose["proved"]


def test_a_start_that_breaks_the_standard_is_not_the_incumbent():
    """The sorted pairs put the two largest markets together; with a threshold below that pair's cost the start
    breaks the standard, and the certificate searches for a design that meets it instead of returning the start."""
    rng = np.random.default_rng(9)
    for _ in range(20):
        z = np.asarray(test_bcp.gen(rng, 8, 2), float)
        start = test_bcp._sorted_pairs(z)
        from . import mitm_exact as mx
        worst = max(mx.exact_rc(z, np.zeros(8), tuple(sorted(g))) for g in start)
        t = float(worst) * 0.5
        want = _exact_brute(z, t)
        if want is None:
            continue
        out = ce.certify_staged(z, incumbent=start, heuristic_seconds=0.0, max_group_cost=t)
        assert out["proved"] and out["standard"]["met"] and out["opt_exact"] == want
        return
    raise AssertionError("no instance with a feasible standard the start breaks")


def test_bad_standards_are_rejected():
    for t in (-1.0, float("nan"), float("inf"), "tight", True):
        with pytest.raises(ValueError):
            ce.certify_staged(np.arange(1.0, 9.0), max_group_cost=t)


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**9), n=st.integers(5, 9), q=st.sampled_from([0.05, 0.2]))
def test_the_standard_is_in_the_units_of_the_markets(seed, n, q):
    """Markets times c = 2^-20 (small enough that the certificate rescales them internally) and the threshold times
    c^2: the same feasibility and the optimum times c^2 exactly."""
    z = np.asarray(test_bcp.gen(np.random.default_rng(seed), n, 2), float)
    costs = _group_costs(z); t = float(costs[int(q * (len(costs) - 1))]); c = 2.0 ** -20
    big = ce.certify_staged(z, max_group_cost=t, heuristic_seconds=0.0)
    small = ce.certify_staged(z * c, max_group_cost=t * c * c, heuristic_seconds=0.0)
    assert small["scale"] != 1.0
    assert big["feasible"] == small["feasible"]
    if big["feasible"]:
        assert small["opt_exact"] == big["opt_exact"] * Fraction(c) ** 2
