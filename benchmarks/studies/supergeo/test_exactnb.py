"""exactnb.member (numba, error-free transformations) against Fraction arithmetic."""
import itertools, json, math, os, pickle, sys
from fractions import Fraction
import numpy as np
from hypothesis import given, settings, strategies as st
from . import exactnb as ex, mitm_exact as mx, mitm_par

HERE = os.path.dirname(os.path.abspath(__file__))


def near(F):
    """The doubles just below, at (rounded) and just above an exact rational."""
    f = float(F)
    return [math.nextafter(f, -math.inf), f, math.nextafter(f, math.inf)]


@settings(max_examples=4000, deadline=None)
@given(data=st.data())
def test_member_matches_fraction(data):
    k = data.draw(st.integers(2, 4))
    zs = st.one_of(st.floats(1e-6, 1e9, allow_nan=False), st.integers(1, 100).map(lambda v: v / 10), st.integers(1, 1000).map(float))
    z = np.array(data.draw(st.lists(zs, min_size=k, max_size=k)))
    pi = np.array(data.draw(st.lists(st.one_of(st.floats(-1e6, 1e6, allow_nan=False), st.integers(-64, 64).map(lambda v: v / 8)), min_size=k, max_size=k)))
    g = np.arange(k, dtype=np.int64)
    rc = mx.exact_rc(z, pi, tuple(range(k)))
    th = data.draw(st.sampled_from(near(rc) + [0.0, data.draw(st.floats(-1e6, 1e6, allow_nan=False))]))
    assert ex.member(z, pi, g, k, th) == (rc <= Fraction(th))


def test_decimal_near_ties():
    rng = np.random.default_rng(3); checked = 0
    for _ in range(300):
        n = 6; z = rng.integers(1, 80, n) / 10.0; pi = np.zeros(n)
        for g in itertools.chain.from_iterable(itertools.combinations(range(n), k) for k in (2, 3, 4)):
            rc = mx.exact_rc(z, pi, g)
            for th in near(rc):
                assert ex.member(z, pi, np.array(g, np.int64), len(g), th) == (rc <= Fraction(th)); checked += 1
    assert checked > 10000


def test_n210_margin_set_matches_fraction():
    stt = json.load(open(os.path.join(HERE, "n210_duals.json"))); z = np.array([float.fromhex(v) for v in stt["z"]]); pi = np.array([float.fromhex(v) for v in stt["pi"]])
    assert ex.in_range(z, pi)
    th = 221075.0 - float(np.sum(pi))
    rows = mitm_par.price(z, pi, 2, 4, th * 1.01)[0]           # a superset straddling the threshold
    got = ex.members(z, pi, rows, th)
    T = Fraction(th)
    want = np.array([mx.exact_rc(z, pi, tuple(int(v) for v in r if v >= 0)) <= T for r in rows])
    assert (got == want).all() and want.sum() > 0 and (~want).sum() > 0


# --- the floating-point filter in front of the exact test -------------------------------------------------------------

_zs = st.one_of(st.floats(1e-6, 1e9, allow_nan=False), st.integers(1, 100).map(lambda v: v / 10),
                st.integers(1, 1000).map(float))
_pis = st.one_of(st.floats(-1e6, 1e6, allow_nan=False), st.integers(-64, 64).map(lambda v: v / 8))


def _exact_minus(z, pi, g, tau, th):
    return mx.exact_rc(z, pi, tuple(g)) - (Fraction(tau) if len(g) % 2 else 0) - Fraction(th)


@settings(max_examples=3000, deadline=None)
@given(data=st.data())
def test_the_float_value_is_within_its_bound(data):
    """The filter's claim: the floating-point rc(g) - theta is within its bound of the exact value (or the bound is
    infinite and the row goes to the exact test)."""
    k = data.draw(st.integers(2, 4))
    z = np.array(data.draw(st.lists(_zs, min_size=k, max_size=k)))
    pi = np.array(data.draw(st.lists(_pis, min_size=k, max_size=k)))
    tau = data.draw(st.sampled_from([0.0, 0.5, 3.25]))
    g = np.arange(k, dtype=np.int64)
    exact = _exact_minus(z, pi, g, tau, 0.0)
    th = data.draw(st.sampled_from(near(exact) + [0.0, data.draw(st.floats(-1e6, 1e6, allow_nan=False))]))
    v, b = ex._float_rc(z, pi, g, k, th, tau)
    if math.isfinite(b):
        assert abs(Fraction(v) - _exact_minus(z, pi, g, tau, th)) <= Fraction(b)


@settings(max_examples=400, deadline=None)
@given(data=st.data())
def test_members2_matches_fraction_near_ties(data):
    """members2 (float filter, then the exact test for the rows it cannot decide) against Fraction arithmetic, the
    threshold at the doubles around one row's exact reduced cost, strict and not."""
    n = data.draw(st.integers(4, 7))
    z = np.array(data.draw(st.lists(_zs, min_size=n, max_size=n)))
    pi = np.array(data.draw(st.lists(_pis, min_size=n, max_size=n)))
    tau = data.draw(st.sampled_from([0.0, 0.5]))
    groups = [g for k in (2, 3, 4) for g in itertools.combinations(range(n), k)]
    rows = np.full((len(groups), 4), -1, np.int64)
    for j, g in enumerate(groups): rows[j, :len(g)] = g
    pick = groups[data.draw(st.integers(0, len(groups) - 1))]
    th = data.draw(st.sampled_from(near(_exact_minus(z, pi, pick, tau, 0.0)) + [0.0]))
    strict = data.draw(st.booleans())
    got = ex.members2(z, pi, rows, th, tau, strict)
    for j, g in enumerate(groups):
        d = _exact_minus(z, pi, g, tau, th)
        assert got[j] == (d < 0 if strict else d <= 0), (g, th, strict)


def test_values_outside_the_error_model_go_to_the_exact_test():
    """Markets scaled by 2^-560 or 2^420: outside [2^-400, 2^400] the relative error model does not hold (squares
    near the subnormal range, or past 2^800), so the filter's bound is infinite and every row gets member2's
    answer. (member2 itself is exact only inside that range, and its callers rescale into it: the certificate
    rescales the markets by a power of 4.)"""
    rng = np.random.default_rng(3); n = 7
    groups = [g for k in (2, 3, 4) for g in itertools.combinations(range(n), k)]
    rows = np.full((len(groups), 4), -1, np.int64)
    for j, g in enumerate(groups): rows[j, :len(g)] = g
    for scale in (2.0 ** -560, 2.0 ** 420):
        z = rng.integers(1, 50, n).astype(float) * scale; pi = np.zeros(n)
        th = float(mx.exact_rc(z, pi, groups[5]))
        assert not math.isfinite(ex._float_rc(z, pi, rows[5], len(groups[5]), th, 0.0)[1])
        got = ex.members2(z, pi, rows, th, 0.0, False)
        want = [ex.member2(z, pi, rows[j], len(g), th, 0.0, False) for j, g in enumerate(groups)]
        assert [bool(v) for v in got] == [bool(v) for v in want]
