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
    """The 50-state panel of the SYNDES benchmark (months 34 to 37): the staged certificate from an 11-step
    heuristic on four threads reaches the optimum the single-threshold certificate proved (fifty_optima.json). Step
    budget and threads are fixed: a time budget makes the stages depend on the machine's speed."""
    Y = pd.read_csv(HERE.parents[2] / "basedata" / "urate_cps.csv", header=None).to_numpy(float)
    pre = Y[-7:-3].sum(0); z = pre / np.median(np.diff(np.sort(pre)))
    out = ce.certify_staged(z, heuristic_steps=11, workers=4)
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
    """One window, at a threshold above both the bound's and the first stage's, serves both; later stages screen
    the last stage's rows, and record how many they screened."""
    calls = []
    real = ce.safe_window
    monkeypatch.setattr(ce, "safe_window", lambda *a, **k: calls.append(a[4]) or real(*a, **k))
    z = list(test_bcp._subsets(16, 3))[1]
    out = ce.certify_staged(z, incumbent=test_bcp._sorted_pairs(z), heuristic_seconds=0.0, caps=(200, 2_000))
    assert out["proved"] and len(out["stages"]) >= 3
    assert len(calls) == 1
    for a, b in zip(out["stages"], out["stages"][1:]):
        assert b["window"] == a["survivors"]


# --- a stated optimality gap ------------------------------------------------------------------------------------------

def _exact_brute(z, t=None, region=None):
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
                if region is not None and len(set(region[list(g)])) > 1:
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
    """The 50-state panel's bound is 0.135 of its optimum. The 11-step heuristic's design costs more than ten times
    the optimum, so a capped stage runs; once it has improved the design, the bound proves a gap of 0.95 and the
    exact stage never runs. The path depends on the heuristic's design: from 5 to 8 steps on four threads the second
    capped stage finds nothing better and the exact stage runs (the certificate still proves the gap), so the test
    fixes the steps and the threads; a 1 s time budget took that path on a slower CI runner (8 steps)."""
    Y = pd.read_csv(HERE.parents[2] / "basedata" / "urate_cps.csv", header=None).to_numpy(float)
    pre = Y[-7:-3].sum(0); z = pre / np.median(np.diff(np.sort(pre)))
    out = ce.certify_staged(z, gap=0.95, heuristic_steps=11, workers=4)
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


# --- an absolute gap -------------------------------------------------------------------------------------------------

@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**9), n=st.integers(4, 11), kind=st.integers(0, 3),
       f=st.sampled_from([0.0, 0.01, 0.3, 2.0]), gap=st.sampled_from([0.0, 0.05]))
def test_an_absolute_gap_against_brute_force(seed, n, kind, f, gap):
    """With abs_gap d (and gap g) the design costs at most max(g UB, d) more than the optimum (exact brute force),
    and the proven absolute gap is at most that allowance."""
    z = np.asarray(test_bcp.gen(np.random.default_rng(seed), n, kind), float)
    want = _exact_brute(z)
    d = float(f * float(want)) if f < 1 else float(f * float(want) + 1.0)   # from none to more than the optimum
    out = ce.certify_staged(z, abs_gap=d, gap=gap, heuristic_seconds=0.0)
    assert out["proved"], out["proof"]
    allow = max(Fraction(gap) * out["opt_exact"], Fraction(d))
    assert want <= out["opt_exact"] <= want + allow
    assert 0 <= out["abs_gap_proved"] and Fraction(out["abs_gap_proved"]) >= out["opt_exact"] - out["lb_proved_exact"]
    assert out["opt_exact"] - out["lb_proved_exact"] <= allow


def _similar(n, seed=2026):
    pre = np.random.default_rng(seed).normal(100, 10, n)
    return pre / np.median(np.diff(np.sort(pre)))


def test_an_absolute_gap_on_similar_markets_needs_no_exact_search():
    """Forty similar-sized markets: the bound is a small fraction of the design, so a relative gap of 0.5 needs
    the exact stage; the absolute gap of the design's own cost is proved by the bound (which is at least 0) once
    the capped stages are done, with no exact stage."""
    z = _similar(40)
    rel = ce.certify_staged(z, gap=0.5, workers=4)              # default step budget: the same design on any machine
    assert rel["lb"] < 0.5 * rel["opt"]                          # the relative gap is not proved by the bound
    out = ce.certify_staged(z, abs_gap=rel["opt"] * 1.5 + 1e-9, workers=4)
    assert out["proved"] and out["proof"].startswith("within the gap: the bound")
    assert not any(s["exact"] for s in out["stages"])
    assert out["abs_gap_proved"] <= rel["opt"] * 1.5 + 1e-9


def test_an_absolute_gap_eliminates_more():
    z = list(test_bcp._subsets(16, 3))[1]
    kw = dict(incumbent=test_bcp._sorted_pairs(z), heuristic_seconds=0.0, caps=(200,))
    exact = ce.certify_staged(z, **kw)
    d = float(0.5 * (exact["opt_exact"] - exact["lb_exact"]))
    loose = ce.certify_staged(z, abs_gap=d, **kw)
    assert loose["proved"] and loose["stages"]
    assert loose["stages"][0]["survivors"] < exact["stages"][0]["survivors"]
    assert loose["abs_gap_proved"] <= d * (1 + 1e-12)


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**9), n=st.integers(5, 9), f=st.sampled_from([0.01, 0.5]))
def test_the_absolute_gap_is_in_the_units_of_the_markets(seed, n, f):
    """Markets times c = 2^-20 (the certificate rescales them) and the gap times c^2: the optimum times c^2 and
    the proven gap within the scaled allowance."""
    z = np.asarray(test_bcp.gen(np.random.default_rng(seed), n, 2), float); c = 2.0 ** -20
    d = float(f * float(_exact_brute(z)))
    small = ce.certify_staged(z * c, abs_gap=d * c * c, heuristic_seconds=0.0)
    assert small["scale"] != 1.0 and small["proved"]
    want = _exact_brute(z) * Fraction(c) ** 2
    assert want <= small["opt_exact"] <= want + Fraction(d) * Fraction(c) ** 2


def test_bad_absolute_gaps_are_rejected():
    for d in (-1.0, float("nan"), float("inf"), "small", True):
        with pytest.raises(ValueError):
            ce.certify_staged(np.arange(1.0, 9.0), abs_gap=d)


def test_a_large_gap_does_not_shrink_the_search_for_a_design_that_meets_the_standard():
    """Until a design meets the standard the search starts from (N // lo) t with no gap applied. With g = 0.9 and
    every allowed design costing more than (1 - g) (N // lo) t, a search shrunk by the gap finds none of them and
    would report the standard infeasible."""
    rng = np.random.default_rng(13); g = 0.9; seen = 0
    from . import mitm_exact as mx
    for _ in range(60):
        n = int(rng.integers(6, 11)); z = np.asarray(test_bcp.gen(rng, n, 2), float)
        start = test_bcp._sorted_pairs(z)
        costs = _group_costs(z)
        for q in (0.05, 0.1, 0.2):
            t = float(costs[int(q * (len(costs) - 1))])
            want = _exact_brute(z, t)
            if want is None or want <= (1 - Fraction(g)) * (n // 2) * Fraction(t):
                continue
            if all(mx.exact_rc(z, np.zeros(n), tuple(sorted(c))) <= Fraction(t) for c in start):
                continue                                  # the start meets the standard: not this case
            out = ce.certify_staged(z, incumbent=start, heuristic_seconds=0.0, max_group_cost=t, gap=g)
            assert out["proved"] and out["feasible"], out["proof"]
            assert want <= out["opt_exact"] and out["opt_exact"] * (1 - Fraction(g)) <= want
            seen += 1
    assert seen >= 3


def test_the_capped_stages_run_on_the_threads(monkeypatch):
    """The capped stages' searches go through the threaded search when workers is set, and record the count."""
    from . import verify_nb as vn
    calls = []
    real = vn._parallel_core
    monkeypatch.setattr(vn, "_parallel_core", lambda *a, **k: calls.append(1) or real(*a, **k))
    z = list(test_bcp._subsets(16, 3))[1]
    out = ce.certify_staged(z, incumbent=test_bcp._sorted_pairs(z), heuristic_seconds=0.0, caps=(200, 2_000), workers=2)
    capped = [s for s in out["stages"] if not s["exact"]]
    assert out["proved"] and capped and all(s["workers"] == 2 for s in capped)
    assert len(calls) >= len(capped)
    none = ce.certify_staged(z, incumbent=test_bcp._sorted_pairs(z), heuristic_seconds=0.0, caps=(200, 2_000), workers=None)
    assert none["opt_exact"] == out["opt_exact"] and all(s["workers"] is None for s in none["stages"])


# --- one window for the bound and the first stage ---------------------------------------------------------------------

@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**9), n=st.integers(5, 12), kind=st.integers(0, 3), f=st.floats(0.0, 1.0),
       extra=st.floats(0.0, 2.0))
def test_a_window_at_a_higher_threshold_screens_to_the_same_groups(seed, n, kind, f, extra):
    """A window built at a threshold at or above the one asked for, screened exactly, keeps the same groups as a
    window built for it: for the bound's negative columns and for a stage's survivors."""
    z, pi, tau, S, m = _stage_inputs(test_bcp.gen(np.random.default_rng(seed), n, kind))
    LB = S + Fraction(n, 2) * m
    UB = Fraction(test_bcp.brute(z)) * 2 + 1
    T = LB + (UB - LB) * Fraction(f)
    th = ce.up(T - S - (Fraction(n, 2) - 1) * m)
    thw = th + extra * (abs(th) + 1.0)                       # any threshold at or above the stage's, and above 0
    thw = max(thw, 0.0)
    W = ce.safe_window(z, pi, 2, 4, thw + tau)
    rows_w, nw = ce._survivor_rows(z, pi, tau, 2, 4, S, m, T, window=(W, thw))
    rows_f, nf = ce._survivor_rows(z, pi, tau, 2, 4, S, m, T)
    assert _keys(rows_w) == _keys(rows_f) and nw == len(W)
    m2, S2, LB2, neg2 = ce._exact_bound(z, pi, tau, 2, 4, window=W)
    m1, S1, LB1, neg1 = ce._exact_bound(z, pi, tau, 2, 4)
    assert (m2, S2, LB2) == (m1, S1, LB1) and _keys(neg2) == _keys(neg1)


def test_a_window_below_the_stage_threshold_is_not_used():
    z, pi, tau, S, m = _stage_inputs(list(test_bcp._subsets(16, 3))[1])
    T = S + 50
    th = ce.up(T - S - (Fraction(16, 2) - 1) * m)
    W = ce.safe_window(z, pi, 2, 4, 0.0 + tau)                # built at 0, below th
    rows_w, nw = ce._survivor_rows(z, pi, tau, 2, 4, S, m, T, window=(W, 0.0))
    rows_f, nf = ce._survivor_rows(z, pi, tau, 2, 4, S, m, T)
    assert th > 0 and _keys(rows_w) == _keys(rows_f) and nw == nf


def test_a_guess_below_the_stage_threshold_builds_a_second_window(monkeypatch):
    """With the threshold guess forced to 0 the shared window is below the first stage's threshold: the stage builds
    its own window and the certificate is unchanged."""
    z = list(test_bcp._subsets(16, 3))[1]
    kw = dict(incumbent=test_bcp._sorted_pairs(z), heuristic_seconds=0.0, caps=(200, 2_000))
    ref = ce.certify_staged(z, **kw)
    calls = []
    real = ce.safe_window
    monkeypatch.setattr(ce, "safe_window", lambda *a, **k: calls.append(a[4]) or real(*a, **k))
    monkeypatch.setattr(ce, "_window_threshold", lambda *a, **k: 0.0)
    out = ce.certify_staged(z, **kw)
    assert len(calls) == 2
    assert out["opt_exact"] == ref["opt_exact"] and out["lb_exact"] == ref["lb_exact"]
    assert out["stages"][0]["survivors"] == ref["stages"][0]["survivors"]


def test_a_negative_guess_still_covers_the_bound(monkeypatch):
    """A threshold guess below 0 is raised to 0, where the window holds every column with a negative reduced cost,
    so the exact bound is unchanged. On this instance 4 columns have a negative reduced cost (m = -5.7e-13), which a
    window built below 0 misses."""
    z = list(test_bcp._subsets(12, 6))[2]
    kw = dict(incumbent=test_bcp._sorted_pairs(z), heuristic_seconds=0.0, caps=(200,))
    ref = ce.certify_staged(z, **kw)
    assert ref["negative_columns"] > 0
    monkeypatch.setattr(ce, "_window_threshold", lambda *a, **k: -1e6)
    out = ce.certify_staged(z, **kw)
    assert out["lb_exact"] == ref["lb_exact"] and out["negative_columns"] == ref["negative_columns"]
    assert out["opt_exact"] == ref["opt_exact"]


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**9), n=st.integers(6, 12), kind=st.integers(0, 3), shift=st.floats(0.01, 0.5),
       f=st.floats(0.0, 1.0))
def test_the_window_guess_is_at_or_above_the_stage_threshold(seed, n, kind, shift, f):
    """With every market's dual raised (the bound holds for any duals), the minimum reduced cost m is clearly
    negative; the guessed window threshold is still at or above the first stage's exact threshold
    T - S - (N / lo - 1) m, so one window serves both."""
    z, pi, tau, S, m = _stage_inputs(test_bcp.gen(np.random.default_rng(seed), n, kind))
    pi2 = pi + shift * (np.median(np.abs(pi)) + 1.0)
    m2, S2, LB2, neg2 = ce._exact_bound(z, pi2, tau, 2, 4)
    assert m2 < -1e-6                                         # the cushion alone does not cover it
    T = LB2 + (abs(LB2) + 1) * Fraction(f)
    th = ce.up(T - S2 - (Fraction(n, 2) - 1) * m2)
    assert ce._window_threshold(z, pi2, tau, 2, 4, S2, T) >= th


# --- a reproducible heuristic ------------------------------------------------------------------------------------------

def test_the_certificate_is_reproducible_by_default():
    """The default heuristic runs a fixed number of steps, max(10, N // 2), from a seed: with the search on this
    thread the same inputs give the same design, with the machine idle and under load; the result records the
    steps and the seed."""
    from .test_lns import _similar, _Load
    z = _similar(40)
    a = ce.certify_staged(z, workers=None, abs_gap=0.05)
    with _Load():
        b = ce.certify_staged(z, workers=None, abs_gap=0.05)
    assert sorted(a["design"]) == sorted(b["design"]) and a["opt_exact"] == b["opt_exact"]
    assert a["heuristic_steps"] == 20 and a["seed"] == 0


def test_the_seed_reaches_the_heuristic():
    """With an absolute gap of 0.5 the sorted start (cost 233) is far from the target and the heuristic works to
    reach it: seed 0 gets there at step 4, seed 1 at step 7, each stops there and the bound proves its design, so
    the design is the heuristic's and the two seeds give two designs. (A gap of 1e6, the earlier fixture, puts the
    start itself within the gap, and the heuristic then takes no step.)"""
    from .test_lns import _similar
    z = _similar(40)
    a = ce.certify_staged(z, workers=None, abs_gap=0.5, seed=0)
    b = ce.certify_staged(z, workers=None, abs_gap=0.5, seed=1)
    assert a["stages"] == [] and b["stages"] == []
    assert a["heuristic_stop"] == b["heuristic_stop"] == "bound" and a["heuristic_steps"] != b["heuristic_steps"]
    assert sorted(a["design"]) != sorted(b["design"]) and b["seed"] == 1


def test_bad_heuristic_budgets_are_rejected():
    for kw in (dict(heuristic_steps=-1), dict(heuristic_steps=2.5), dict(heuristic_steps="lots"),
               dict(seed=-1), dict(seed=1.5)):
        with pytest.raises(ValueError):
            ce.certify_staged(np.arange(1.0, 9.0), **kw)


# --- groups within regions ---------------------------------------------------------------------------------------------

@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**9), n=st.integers(4, 11), kind=st.integers(0, 3), k=st.integers(1, 3),
       gap=st.sampled_from([0.0, 0.05]), f=st.sampled_from([0.0, 0.3]))
def test_regions_against_brute_force(seed, n, kind, k, gap, f):
    """With region labels only groups within one region are allowed. The certificate proves each region on its own
    and adds them: its design keeps every group inside a region, and it is within max(g UB, d) of the exact
    brute-force optimum over within-region designs (or proves there is none)."""
    rng = np.random.default_rng(seed)
    z = np.asarray(test_bcp.gen(rng, n, kind), float)
    region = rng.integers(0, k, n)
    want = _exact_brute(z, region=region)
    d = float(f * float(want)) if want is not None else 0.0
    out = ce.certify_staged(z, region=region, gap=gap, abs_gap=d, heuristic_seconds=0.0)
    assert out["proved"], out["proof"]
    if want is None:
        assert out["feasible"] is False and out["design"] is None
        return
    assert all(len(set(region[list(g)])) == 1 for g in out["design"])
    assert sorted(i for g in out["design"] for i in g) == list(range(n))
    allow = max(Fraction(gap) * out["opt_exact"], Fraction(d))
    assert want <= out["opt_exact"] <= want + allow
    assert out["lb_proved_exact"] <= want and out["opt_exact"] - out["lb_proved_exact"] <= allow


def test_one_region_is_the_unconstrained_problem():
    z = list(test_bcp._subsets(16, 3))[1]
    free = ce.certify_staged(z, heuristic_seconds=0.0)
    one = ce.certify_staged(z, heuristic_seconds=0.0, region=np.zeros(16, int))
    assert one["proved"] and one["opt_exact"] == free["opt_exact"] and len(one["regions"]) == 1


def test_each_region_is_certified_on_its_own():
    """Two regions of eight: two certificates, each on its own markets, whose values add up to the result's."""
    z = np.asarray(list(test_bcp._subsets(16, 3))[1], float); region = np.array([0] * 8 + [1] * 8)
    out = ce.certify_staged(z, heuristic_seconds=0.0, region=region)
    assert [r["n"] for r in out["regions"]] == [8, 8]
    assert sum(r["opt_exact"] for r in out["regions"]) == out["opt_exact"]
    assert sum(r["lb_proved_exact"] for r in out["regions"]) == out["lb_proved_exact"]
    for r, lab in zip(out["regions"], (0, 1)):
        alone = ce.certify_staged(z[region == lab], heuristic_seconds=0.0)
        assert r["label"] == lab and r["opt_exact"] == alone["opt_exact"]


def test_a_region_too_small_to_group_is_infeasible():
    z = np.arange(1.0, 10.0); region = np.array([0] * 8 + [1])
    out = ce.certify_staged(z, heuristic_seconds=0.0, region=region)
    assert out["proved"] and out["feasible"] is False and out["design"] is None
    assert "fewer than 2" in out["proof"]


def test_bad_regions_are_rejected():
    z = np.arange(1.0, 9.0)
    for r in (np.zeros(7), np.zeros((8, 1)), [0, 0, 0, 0, 1, 1, 1, None]):
        with pytest.raises(ValueError):
            ce.certify_staged(z, region=r)


def test_the_absolute_gap_is_shared_between_regions():
    """Two identical regions and an absolute gap d between one region's slack s = UB - LB at its start and 2 s. Each
    region gets d / 2 < s, so each must improve on its start; a region given the whole d would stop at its start,
    and the two starts together would sit more than d above the optimum. The instance is one where the bound is
    tight (the optimum within s / 4 of it), so that excess is real."""
    rng = np.random.default_rng(21)
    for _ in range(200):
        x = np.asarray(test_bcp.gen(rng, 8, 2), float)
        one = ce.certify_staged(x, heuristic_seconds=0.0, caps=(1,))
        start = ce.certify_staged(x, heuristic_seconds=0.0, gap=0.99)           # proved by the bound: the start
        S, LB, O = start["opt_exact"], one["lb_exact"], _exact_brute(x)
        s = S - LB
        if s > 0 and O - LB < s / 4 and S > O:
            break
    else:
        raise AssertionError("no instance with a tight bound")
    d = float(1.5 * s)
    z = np.concatenate([x, x]); region = np.array([0] * 8 + [1] * 8)
    out = ce.certify_staged(z, heuristic_seconds=0.0, region=region, abs_gap=d)
    assert out["proved"]
    assert out["opt_exact"] - 2 * O <= Fraction(d)


def test_one_region_keeps_the_whole_gap_and_no_more():
    """The reproduction (falsified by test_regions_against_brute_force at seed 11, 11 markets, one region, f 0.3):
    the region's share was computed as d * 11 / 11 in floating point, which rounds to one unit in the last place
    above d, and the certificate proved its region within that share: 0.006000000000000018 against the allowed
    0.0060000000000000175. Each region's share must be at most its exact part of d, so the shares sum to at most d."""
    rng = np.random.default_rng(11)
    z = np.asarray(test_bcp.gen(rng, 11, 1), float)
    region = rng.integers(0, 1, 11)
    want = _exact_brute(z, region=region)
    d = float(0.3 * float(want))
    out = ce.certify_staged(z, region=region, abs_gap=d, heuristic_seconds=0.0)
    assert out["proved"] and out["opt_exact"] - out["lb_proved_exact"] <= Fraction(d)


@settings(max_examples=300, deadline=None)
@given(d=st.one_of(st.just(0.0), st.floats(0.0, 1e300, allow_nan=False, allow_infinity=False),
                   st.sampled_from([5e-324, 2.2250738585072014e-308, 0.1, 0.3, 1.0, 1e-17])),
       sizes=st.lists(st.integers(1, 400), min_size=1, max_size=12))
def test_the_shares_of_the_gap_never_add_up_to_more_than_it(d, sizes):
    """Property, over the gap (0, subnormal, tiny, huge) and the regions' sizes: every share is a non-negative float
    at most its exact part d * n_r / n, so the shares sum to at most d, and each is within one unit in the last place
    of that exact part (rounding down loses no more than that)."""
    import math
    shares = ce._gap_shares(d, sizes)
    n = sum(sizes)
    assert len(shares) == len(sizes) and all(isinstance(v, float) and v >= 0 for v in shares)
    for v, k in zip(shares, sizes):
        exact = Fraction(d) * k / n
        assert Fraction(v) <= exact < Fraction(math.nextafter(v, math.inf)) or Fraction(v) == exact
    assert sum(Fraction(v) for v in shares) <= Fraction(d)


# --- reproducible threaded stages ----------------------------------------------------------------------------------

def test_deterministic_stages_give_the_same_run_on_the_threads():
    """With deterministic=True the stages search their subtrees in rounds: on four threads the same inputs give
    the same stages (nodes, status, improvement), the same design and cost, idle and under load, and the stages
    record the schedule."""
    from .test_lns import _Load
    z = list(test_bcp._subsets(16, 3))[1]
    kw = dict(incumbent=test_bcp._sorted_pairs(z), heuristic_seconds=0.0, caps=(200, 2_000), workers=4,
              deterministic=True)
    a = ce.certify_staged(z, **kw)
    with _Load():
        b = ce.certify_staged(z, **kw)
    key = lambda out: [(s["nodes"], s["status"], s["improved"], s["survivors"]) for s in out["stages"]]
    assert a["proved"] and any(not s["exact"] for s in a["stages"])
    assert key(a) == key(b) and sorted(a["design"]) == sorted(b["design"]) and a["opt_exact"] == b["opt_exact"]
    assert all(s["deterministic"] for s in a["stages"]) and a["deterministic"] is True


def test_the_schedule_reaches_every_stage_search(monkeypatch):
    """Both searches, the capped stages' and the exact stage's, receive the schedule asked for."""
    from . import verify_nb as vn
    got = []
    for name in ("verify", "verify_rows"):
        real = getattr(vn, name)
        monkeypatch.setattr(vn, name, (lambda real, name: lambda *a, **k: got.append((name, k.get("deterministic")))
                                       or real(*a, **k))(real, name))
    z = list(test_bcp._subsets(16, 3))[1]
    for det in (True, False):
        got.clear()
        out = ce.certify_staged(z, incumbent=test_bcp._sorted_pairs(z), heuristic_seconds=0.0, caps=(200, 2_000),
                                workers=2, deterministic=det)
        assert {n for n, _ in got} == {"verify", "verify_rows"}, got
        assert all(d is det for _, d in got) and out["deterministic"] is det


def test_the_schedule_reaches_the_regions(monkeypatch):
    from . import verify_nb as vn
    got = []
    real = vn.verify
    monkeypatch.setattr(vn, "verify", lambda *a, **k: got.append(k.get("deterministic")) or real(*a, **k))
    z = np.concatenate([list(test_bcp._subsets(16, 3))[1][:8], list(test_bcp._subsets(16, 3))[1][8:]])
    out = ce.certify_staged(z, region=[0] * 8 + [1] * 8, heuristic_seconds=0.0, workers=2, deterministic=True)
    assert got and all(d is True for d in got) and out["deterministic"] is True


def test_a_bad_schedule_is_rejected():
    for det in ("yes", 1, None):
        with pytest.raises(ValueError):
            ce.certify_staged(np.arange(1.0, 9.0), deterministic=det)


def test_the_certificate_runs_its_threads_in_rounds_by_default():
    """The default schedule is the reproducible one: a run on threads gives the same design every time unless
    deterministic=False asks for the faster dynamic schedule."""
    z = list(test_bcp._subsets(16, 3))[1]
    out = ce.certify_staged(z, incumbent=test_bcp._sorted_pairs(z), heuristic_seconds=0.0, caps=(200,), workers=2)
    assert out["deterministic"] is True and out["stages"] and all(s["deterministic"] for s in out["stages"])


# --- group-size limits (RCA: the heuristic dropped lo and hi) -------------------------------------------------------
#
# certify_staged(z, lo, hi) started from a heuristic called without lo and hi, which therefore built groups of 2 to 4
# whatever was asked, and nothing checked the result: with lo=2, hi=2 it returned the 2-to-4 optimum, groups of four,
# labelled proved with a negative proven gap. Every earlier property held lo=2, hi=4, the heuristic's own defaults.

def _brute_sizes(z, lo, hi):
    """The exact optimum (a Fraction) over every partition into groups of lo to hi markets, or None if there is
    none; recursion on the lowest market."""
    from itertools import combinations
    from . import mitm_exact as mx
    n = len(z); zeros = np.zeros(n); memo = {}
    def best(U):
        if not U:
            return Fraction(0)
        if U in memo:
            return memo[U]
        i = min(U); rest = sorted(U - {i}); out = None
        for k in range(lo - 1, hi):
            for others in combinations(rest, k):
                g = (i,) + others
                sub = best(frozenset(U - set(g)))
                if sub is not None:
                    c = mx.exact_rc(z, zeros, g) + sub
                    out = c if out is None or c < out else out
        memo[U] = out
        return out
    return best(frozenset(range(n)))


def _ten_markets():
    z = np.sort(np.random.default_rng(10).lognormal(0, 0.9, 10))
    z[-1] = 8 * np.median(z)                       # one market eight times the median, as in the incident panel
    return z


def _check_sizes(out, n, lo, hi):
    assert sorted(v for g in out["design"] for v in g) == list(range(n))
    assert all(lo <= len(g) <= hi for g in out["design"]), out["design"]


@pytest.mark.parametrize("lo, hi", [(2, 2), (2, 3), (3, 4), (3, 3), (4, 4), (2, 4)])
def test_the_group_size_limits_hold_with_the_heuristic_on(lo, hi):
    """The reproduction: ten markets, one eight times the median, the default heuristic on. The design keeps every
    group within lo to hi, is the brute-force optimum over such partitions, and the bound lies below it; ten
    markets admit no partition into groups of exactly three or exactly four, and the certificate proves that."""
    z = _ten_markets()
    want = _brute_sizes(z, lo, hi)
    out = ce.certify_staged(z, lo=lo, hi=hi)
    if want is None:
        assert out["design"] is None and not out["feasible"] and out["proved"]
        return
    _check_sizes(out, 10, lo, hi)
    assert out["proved"] and out["opt_exact"] == want
    assert out["lb_exact"] <= want and out["gap_proved"] >= 0


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**9), n=st.integers(2, 11), kind=st.integers(0, 3),
       sizes=st.sampled_from([(2, 2), (2, 3), (2, 4), (3, 3), (3, 4), (4, 4)]),
       heuristic=st.sampled_from([dict(heuristic_seconds=0.0), dict(heuristic_steps=1), dict(heuristic_steps=5), {}]),
       gap=st.sampled_from([0.0, 0.05]), workers=st.sampled_from([None, 2]))
def test_every_group_size_limit_against_brute_force(seed, n, kind, sizes, heuristic, gap, workers):
    """Over the group-size limits, the heuristic (off, one step, five, the default), the gap and the threads: when no
    partition of n markets into groups of lo to hi exists the certificate proves there is no design; otherwise its
    design keeps the limits and is within the gap of the brute-force optimum, and its proven bound is below it."""
    lo, hi = sizes
    z = np.asarray(test_bcp.gen(np.random.default_rng(seed), n, kind), float)
    want = _brute_sizes(z, lo, hi)
    out = ce.certify_staged(z, lo=lo, hi=hi, gap=gap, workers=workers, **heuristic)
    if want is None:
        assert out["design"] is None and not out["feasible"] and out["proved"]
        return
    _check_sizes(out, n, lo, hi)
    assert out["proved"]
    assert out["lb_proved_exact"] <= want <= out["opt_exact"]
    assert out["opt_exact"] * (1 - Fraction(gap)) <= want


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**9), calls=st.lists(st.sampled_from([(2, 2), (2, 3), (2, 4), (3, 4)]),
                                                min_size=2, max_size=4))
def test_calls_in_sequence_are_each_correct(seed, calls):
    """Several certificates with different limits in one process, in any order: each is correct on its own terms
    (no state carried from one call to the next changes a design or a bound)."""
    z = np.asarray(test_bcp.gen(np.random.default_rng(seed), 8, seed % 4), float)
    for lo, hi in calls:
        out = ce.certify_staged(z, lo=lo, hi=hi)
        _check_sizes(out, 8, lo, hi)
        assert out["opt_exact"] == _brute_sizes(z, lo, hi) and out["lb_exact"] <= out["opt_exact"]


@pytest.mark.parametrize("n, lo, hi", [(5, 2, 2), (7, 4, 4), (5, 3, 4), (3, 2, 2), (2, 3, 4)])
def test_sizes_that_admit_no_partition_are_proved_infeasible(n, lo, hi):
    out = ce.certify_staged(np.arange(1.0, n + 1.0), lo=lo, hi=hi)
    assert out["design"] is None and not out["feasible"] and out["proved"]
    assert "cannot be split" in out["proof"]


@pytest.mark.parametrize("lo, hi", [(1, 2), (0, 2), (-1, 2), (3, 2), (2, 5), (2, 1), (2.0, 4), (2, 4.0),
                                    (True, 4), (2, None), ("2", 4)])
def test_bad_group_size_limits_are_rejected(lo, hi):
    """The solvers price and search groups of 2 to 4 markets only (the pricing scan's loops, the search's rows of
    four), so lo and hi must be integers with 2 <= lo <= hi <= 4; anything else raises before any work."""
    with pytest.raises(ValueError, match="lo and hi"):
        ce.certify_staged(np.arange(1.0, 9.0), lo=lo, hi=hi)


@pytest.mark.parametrize("budget", [dict(heuristic_steps=3), dict(heuristic_seconds=0.05)])
def test_the_heuristic_receives_the_limits(monkeypatch, budget):
    """Both heuristic budgets forward lo and hi."""
    from . import lns
    seen = []
    real = lns.lns_dp
    monkeypatch.setattr(lns, "lns_dp", lambda *a, **k: seen.append((k.get("lo"), k.get("hi"))) or real(*a, **k))
    ce.certify_staged(_ten_markets(), lo=2, hi=3, **budget)
    assert seen == [(2, 3)]


def test_the_older_certificate_forwards_the_limits(monkeypatch):
    """certify (the MIP certificate) forwards lo and hi to its heuristic too."""
    from . import lns
    seen = []
    def fake(z, design, **k):
        seen.append((k.get("lo"), k.get("hi")))
        raise RuntimeError("stop after the heuristic call")
    monkeypatch.setattr(lns, "lns_dp", fake)
    with pytest.raises(RuntimeError, match="stop after"):
        ce.certify(_ten_markets(), lo=3, hi=4, verbose=False)
    assert seen == [(3, 4)]


def test_a_design_outside_the_limits_is_caught(monkeypatch):
    """The certificate checks its own result: a heuristic that returns a group outside lo to hi (the defect this
    section pins) makes the run raise, not return a certified design."""
    from . import lns
    real = lns.lns_dp
    monkeypatch.setattr(lns, "lns_dp", lambda z, d, **k: real(z, d, **{**k, "lo": 2, "hi": 4}))
    with pytest.raises(RuntimeError, match="group sizes"):
        ce.certify_staged(_ten_markets(), lo=2, hi=2)


def test_a_bound_above_the_design_is_caught(monkeypatch):
    """The certificate checks its own result: a lower bound above the design's cost is a contradiction, and the run
    raises instead of reading the negative gap as proved."""
    real = ce._exact_bound
    def inflated(*a, **k):
        m, S, LB, neg = real(*a, **k)
        return m, S, LB * 1000 + 10**6, neg
    monkeypatch.setattr(ce, "_exact_bound", inflated)
    with pytest.raises(RuntimeError, match="lower bound"):
        ce.certify_staged(_ten_markets(), lo=2, hi=4)


def test_the_start_fits_the_limits_when_pairs_would_be_cheaper():
    """Six markets in three identical pairs: sorted pairs cost 0, and every partition into triples costs more. With
    lo = hi = 3 and the heuristic off, a pairs start would be the cheapest design the search ever holds, so nothing
    would replace it; the start must itself be two triples, and the result is the brute-force optimum over triples."""
    z = np.array([1.0, 1.0, 5.0, 5.0, 20.0, 20.0])
    out = ce.certify_staged(z, lo=3, hi=3, heuristic_seconds=0.0)
    _check_sizes(out, 6, 3, 3)
    assert out["proved"] and out["opt_exact"] == _brute_sizes(z, 3, 3) > 0


# --- the heuristic stops when it stops paying -------------------------------------------------------------------------
# The default heuristic ran a fixed max(10, N // 2) steps. On the 174 Trimmed Match geos in pairs the sorted start is
# already optimal and the 87 steps were 5.4 of the 5.9 s; on integer panels the heuristic reaches a design of cost 0,
# the root bound, and kept stepping. Two rules now end it early, both counted in steps (the design stays a function
# of the inputs and the seed): heuristic_patience steps in a row without a new lowest cost ("stall"), or a design
# within the requested gap of the root LP bound ("bound"). The result records which ended it.

def _pairs_cost(z):
    from . import mitm_exact as mx
    s = np.argsort(z)
    return sum(mx.exact_rc(np.asarray(z, float), np.zeros(len(z)), (int(s[i]), int(s[i + 1])))
               for i in range(0, len(z) - 1, 2))


def test_a_search_that_cannot_improve_stops_on_a_stall():
    """Smoke: 70 similar-sized markets in pairs. The sorted pairs are optimal, so no step improves them; the default
    patience (30) ends the heuristic at step 30 of its budget of 35, and the certificate proves the sorted pairing."""
    from .test_lns import _similar
    z = _similar(70)
    out = ce.certify_staged(z, lo=2, hi=2)
    assert out["heuristic_steps"] == 30 and out["heuristic_stop"] == "stall"
    assert out["proved"] and out["opt_exact"] == _pairs_cost(z)


def test_a_design_at_the_bound_stops_the_heuristic():
    """Unit: 32 markets in eight quadruples a, a+1, a+2, a+3 (:func:`.test_lns._quadruples`), groups of 2 to 4. The
    root bound is 0 and the heuristic reaches a design of cost 0 before its budget of 16 steps; it stops there,
    and the certificate proves 0."""
    from .test_lns import _quadruples
    z = _quadruples()
    out = ce.certify_staged(z)
    assert out["heuristic_stop"] == "bound" and out["heuristic_steps"] < 16
    assert out["proved"] and out["opt_exact"] == 0


def test_a_search_that_keeps_improving_runs_its_budget():
    """Unit: 40 lognormal markets improve inside the budget of 20 steps, so neither rule fires before it."""
    z = np.asarray(test_bcp.gen(np.random.default_rng(3), 40, 2), float)
    out = ce.certify_staged(z, gap=0.5)
    assert out["heuristic_stop"] == "budget" and out["heuristic_steps"] == 20


def test_the_time_budget_stops_on_a_stall_too():
    """The time budget takes the same rules: pairs on 70 similar-sized markets with 60 s stop after 30 steps."""
    from .test_lns import _similar
    z = _similar(70)
    out = ce.certify_staged(z, lo=2, hi=2, heuristic_seconds=60.0)
    assert out["heuristic_steps"] == 30 and out["heuristic_stop"] == "stall" and out["t"] < 40


def test_no_heuristic_no_stop():
    """Edge: with the heuristic off, a starting design given, or no design possible, nothing stopped a heuristic."""
    z = _ten_markets()
    assert ce.certify_staged(z, heuristic_steps=0)["heuristic_stop"] is None
    assert ce.certify_staged(z, heuristic_seconds=0.0)["heuristic_stop"] is None
    assert ce.certify_staged(z, incumbent=[(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)])["heuristic_stop"] is None
    assert ce.certify_staged(np.arange(1.0, 6.0), lo=2, hi=2)["heuristic_stop"] is None


def test_each_region_reports_its_stop():
    out = ce.certify_staged(_ten_markets(), region=np.array([0] * 6 + [1] * 4))
    assert [r["heuristic_stop"] for r in out["regions"]] == [
        ce.certify_staged(_ten_markets()[:6])["heuristic_stop"], ce.certify_staged(_ten_markets()[6:])["heuristic_stop"]]
    assert out["heuristic_stop"] is None


@pytest.mark.parametrize("p", [0, -1, 2.5, True, "lots", "AUTO", np.float64(2.0), float("nan")])
def test_bad_patience_is_rejected(p):
    """Failure: heuristic_patience is "auto", None (no stall rule) or an integer >= 1."""
    with pytest.raises(ValueError, match="heuristic_patience"):
        ce.certify_staged(np.arange(1.0, 9.0), heuristic_patience=p)


def _spy(monkeypatch):
    from . import lns
    seen = []
    real = lns.lns_dp
    monkeypatch.setattr(lns, "lns_dp", lambda *a, **k: seen.append(k) or real(*a, **k))
    return seen


@pytest.mark.parametrize("budget", [dict(heuristic_steps=3), dict(heuristic_seconds=0.05)])
@pytest.mark.parametrize("p, want", [("auto", 30), (7, 7), (None, None)])
def test_the_heuristic_receives_the_patience(monkeypatch, budget, p, want):
    """Both budgets forward the patience, "auto" being 30."""
    seen = _spy(monkeypatch)
    ce.certify_staged(_ten_markets(), heuristic_patience=p, **budget)
    assert [k.get("patience", "missing") for k in seen] == [want]


def test_the_regions_forward_the_patience(monkeypatch):
    seen = _spy(monkeypatch)
    ce.certify_staged(_ten_markets(), region=np.array([0] * 6 + [1] * 4), heuristic_patience=4)
    assert [k.get("patience") for k in seen] == [4, 4]


@pytest.mark.parametrize("budget", [dict(heuristic_steps=3), dict(heuristic_seconds=0.05)])
@pytest.mark.parametrize("gap, d, size", [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.0, 2.0, 1.0), (0.3, 5.0, 1.0),
                                          (0.0, 2e-6, 1e-3), (0.2, 5e-6, 1e-3)])
def test_the_heuristic_receives_the_gap_as_a_target(monkeypatch, budget, gap, d, size):
    """Both budgets forward the target: the cost at which the design is within the requested gap of the root LP
    bound b (clipped at 0), in the rescaled units, max(b / (1 - g), b + d * scale). At a thousandth of the size the
    root rescales the totals (scale 2^20), so the absolute gap must be scaled with them."""
    z = _ten_markets() * size
    zs, scale, pi, tau = ce._root(z, 2, 4)
    assert (scale == 1.0) == (size == 1.0)
    b = max(float(sum(pi)) + tau, 0.0)
    want = max(b / (1 - gap), b + d * scale)
    seen = _spy(monkeypatch)
    ce.certify_staged(z, gap=gap, abs_gap=d, **budget)
    assert len(seen) == 1 and seen[0]["target"] == pytest.approx(want, rel=1e-12, abs=1e-300)


@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**9), n=st.integers(4, 11), kind=st.integers(0, 3),
       sizes=st.sampled_from([(2, 2), (2, 3), (2, 4), (3, 4)]),
       patience=st.sampled_from(["auto", None, 1, 2, 5]), steps=st.sampled_from(["auto", 0, 1, 8, 40]),
       gap=st.sampled_from([0.0, 0.05, 0.5]))
def test_the_stop_rules_change_when_the_heuristic_ends_and_nothing_else(seed, n, kind, sizes, patience, steps, gap):
    """Property, over the data, the limits, the patience, the budget and the gap. The certificate stays correct (its
    design keeps the limits and is within the gap of the brute-force optimum, its bound below it); the steps that
    ran are within the budget, and the stop it reports agrees with them; and a run given exactly those steps with
    no stall rule holds the same incumbent, so it returns the same design and cost."""
    lo, hi = sizes
    z = np.asarray(test_bcp.gen(np.random.default_rng(seed), n, kind), float)
    want = _brute_sizes(z, lo, hi)
    out = ce.certify_staged(z, lo=lo, hi=hi, gap=gap, heuristic_patience=patience, heuristic_steps=steps)
    if want is None:
        assert out["design"] is None and out["heuristic_stop"] is None
        return
    _check_sizes(out, n, lo, hi)
    assert out["proved"] and out["lb_proved_exact"] <= want <= out["opt_exact"]
    assert out["opt_exact"] * (1 - Fraction(gap)) <= want
    budget = max(10, n // 2) if steps == "auto" else steps
    ran, stop = out["heuristic_steps"], out["heuristic_stop"]
    assert 0 <= ran <= budget
    if budget == 0:
        assert stop is None
    else:
        assert stop in ("budget", "stall", "bound")
        if stop == "budget":
            assert ran == budget
        if stop == "stall":
            p = 30 if patience == "auto" else patience
            assert p is not None and ran >= p
    again = ce.certify_staged(z, lo=lo, hi=hi, gap=gap, heuristic_patience=None, heuristic_steps=ran)
    assert sorted(again["design"]) == sorted(out["design"]) and again["opt_exact"] == out["opt_exact"]


def test_a_root_bound_a_rounding_error_below_zero_still_stops_at_a_perfect_design(monkeypatch):
    """Edge: the root LP's value, summed in floating point, can fall a rounding error below 0 on a panel whose
    optimum is 0. The bound is clipped at 0, so the target is 0 and the quadruples' design of cost 0 still meets it
    and stops the heuristic (at step 4). The duals are shifted here to put the bound 1e-12 below 0; the run is
    stopped after the heuristic, since with those duals the proof is a long search unrelated to this rule."""
    from . import lns
    from .test_lns import _quadruples
    real_root, real_lns, seen = ce._root, lns.lns_dp, {}
    def shifted(*a, **k):
        z, scale, pi, tau = real_root(*a, **k)
        pi = np.array(pi, float); pi[0] -= 1e-12 + float(sum(pi)) + tau
        return z, scale, pi, tau
    class Done(Exception):
        pass
    def spy(*a, **k):
        seen["target"] = k["target"]; seen["out"] = real_lns(*a, **k)
        raise Done
    monkeypatch.setattr(ce, "_root", shifted)
    monkeypatch.setattr(lns, "lns_dp", spy)
    with pytest.raises(Done):
        ce.certify_staged(_quadruples())
    design, loss, trace = seen["out"]
    assert seen["target"] == 0.0 and loss == 0.0 and len(trace) - 1 < 16
