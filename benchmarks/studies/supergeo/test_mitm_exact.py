"""Exact-arithmetic tests of meet-in-the-middle pricing.

Run: python -m pytest test_mitm_exact.py -q -p no:cacheprovider
"""
import itertools, json, pickle, os, sys
from fractions import Fraction
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st, HealthCheck

from . import sg, mitm_exact as mx

HERE = os.path.dirname(os.path.abspath(__file__))


def float_rc(z, pi, g):
    """The kernel's own floating-point reduced cost, as sg._scan computes it."""
    idx = np.array(tuple(g) + (0,) * (4 - len(g)), np.int64)
    r = sg._score(z, idx, len(g))
    for i in g:
        r = r - pi[i]
    return r


# ---- the rounding bound the superset rests on -------------------------------------
finite = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
pos = st.floats(min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False)


@settings(max_examples=3000, deadline=None)
@given(z=st.lists(pos, min_size=4, max_size=4), pi=st.lists(finite, min_size=4, max_size=4), k=st.integers(2, 4))
def test_float_rc_is_within_delta_of_exact(z, pi, k):
    """|float rc - exact rc| <= DELTA_RC for every group: the bound the reference scan uses."""
    z = np.array(z); pi = np.array(pi); g = tuple(range(k))
    err = abs(Fraction(float_rc(z, pi, g)) - mx.exact_rc(z, pi, g))
    assert err <= Fraction(mx.delta_rc(z, pi)), (float(err), mx.delta_rc(z, pi))


# ---- small instances against brute force in exact arithmetic ---------------------
def instances():
    rng = np.random.default_rng(0)
    out = []
    for s in range(4):                                   # generic: lognormal sizes, normal duals
        n = 11; out.append((rng.lognormal(0, 1, n) * 10, rng.normal(0, 3, n)))
    z = np.arange(1.0, 12.0)                             # exact ties: integers, dyadic duals
    out.append((z, np.full(11, 0.125)))
    out.append((z, np.zeros(11)))
    z = np.array([0.1, 0.2, 0.3, 0.7, 1.1, 1.3, 2.9, 0.6, 0.4, 0.5, 0.9])   # non-dyadic: 0.1 + 0.2 != 0.3
    out.append((z, np.zeros(11)))
    out.append((z, np.full(11, 1e-17)))
    return out


@pytest.mark.parametrize("case", range(8))
def test_exact_set_at_ties_and_generic_thresholds(case):
    z, pi = instances()[case]
    groups = [g for k in (2, 3, 4) for g in itertools.combinations(range(len(z)), k)]
    rng = np.random.default_rng(case)
    thetas = [0.0, -1e-9, 1e-12] + [float(mx.exact_rc(z, pi, groups[j])) for j in rng.choice(len(groups), 6, replace=False)]
    for th in thetas:
        got, _ = mx.price_exact(z, pi, 2, 4, th)
        assert got == mx.exact_set(z, pi, 2, 4, th), th


# ---- generative: random instances, thresholds placed on exact reduced costs -------
dyadic = st.integers(-64, 64).map(lambda k: k / 8.0)


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(data=st.data(), n=st.integers(4, 9))
def test_price_exact_equals_brute_force(data, n):
    kind = data.draw(st.sampled_from(["float", "int", "decimal"]))
    if kind == "float":
        z = data.draw(st.lists(st.floats(0.01, 1e4, allow_nan=False), min_size=n, max_size=n))
    elif kind == "int":
        z = data.draw(st.lists(st.integers(1, 30), min_size=n, max_size=n))
    else:
        z = data.draw(st.lists(st.integers(1, 40), min_size=n, max_size=n)); z = [v / 10 for v in z]
    pi = data.draw(st.lists(st.one_of(dyadic, st.floats(-50, 50, allow_nan=False)), min_size=n, max_size=n))
    z = np.array(z, float); pi = np.array(pi, float)
    groups = [g for k in (2, 3, 4) for g in itertools.combinations(range(n), k)]
    g = groups[data.draw(st.integers(0, len(groups) - 1))]
    th = data.draw(st.sampled_from([float(mx.exact_rc(z, pi, g)), 0.0, -1e-9, data.draw(st.floats(-20, 200, allow_nan=False))]))
    got, _ = mx.price_exact(z, pi, 2, 4, th)
    assert got == mx.exact_set(z, pi, 2, 4, th)


# ---- N = 210, the steered run's final duals --------------------------------------
@pytest.mark.parametrize("theta", ["zero", "cg", "thousand", "margin"])
def test_n210_against_exact_reference(theta):
    stt = json.load(open(os.path.join(HERE, "n210_duals.json")))
    z = np.array([float.fromhex(v) for v in stt["z"]]); pi = np.array([float.fromhex(v) for v in stt["pi"]])
    th = {"zero": 0.0, "cg": -1e-9 * 160.919, "thousand": 1e3, "margin": 221075.0 - float(np.sum(pi))}[theta]
    got, _ = mx.price_exact(z, pi, 2, 4, th)
    assert got == mx.reference_exact(z, pi, 2, 4, th)


# --- the safe window on threads ---------------------------------------------------------------------------------------

@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**9), n=st.integers(4, 16), th=st.floats(-5.0, 500.0), dual=st.floats(0.0, 3.0),
       sizes=st.sampled_from([(2, 4), (2, 3), (3, 4), (2, 2), (4, 4)]))
def test_the_parallel_window_emits_the_serial_rows(seed, n, th, dual, sizes):
    """The window counted and filled in parallel (offsets from the per-half counts) emits the serial window's
    rows: the same number, and the same rows once sorted, for every range of group sizes."""
    lo, hi = sizes
    from . import mitm
    rng = np.random.default_rng(seed)
    z = rng.lognormal(0, 1, n) * 10; pi = rng.normal(0, dual, n)
    hv, ha, hb, hsa, hsb, hw = mitm._halves(z, pi); order = np.argsort(hv, kind="stable")
    M = float(np.max(np.abs(z))); P = float(np.max(np.abs(pi)))
    wmax = float(max(pi.max(), 0.0) + max(np.sort(pi)[-2:].sum(), pi.max(), 0.0))
    slack_r2 = 8 * mx.U * (abs(th) + 6 * P) + 2 * mx.U * abs(wmax); slack_v = 16 * mx.U * 4 * M
    args = (hv, ha, hb, hsa, hsb, hw, order, lo, hi, float(th), wmax, slack_r2, slack_v)
    _, k = mx._window_safe(*args, 0)
    serial, _ = mx._window_safe(*args, k)
    par = mx._window_safe_par(*args)
    assert par.shape == serial.shape
    key = lambda r: sorted(map(tuple, r.tolist()))
    assert key(par) == key(serial)
