"""The staged certificate: exact elimination redone at each better incumbent, the exact search between."""
import json, pathlib
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


def test_bad_input_is_rejected():
    with pytest.raises(ValueError):
        ce.certify_staged(np.arange(1.0, 64.0))
    with pytest.raises(ValueError):
        ce.certify_staged(np.arange(1.0, 9.0), caps=(1000, 10))
    with pytest.raises(ValueError):
        ce.certify_staged(np.arange(1.0, 9.0), caps=(0,))
