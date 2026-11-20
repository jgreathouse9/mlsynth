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
