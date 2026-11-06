"""The floor's certificate without its per-call costs (#965).

A group's floor is the hull bound at its NNLS point over the box {0 <= x <= cap}:
with lo = 0 and m the number of controls the knapsack's sum constraint is implied
by the box, so the knapsack's multiplier is taken as 0 (weak duality makes any
multiplier valid, and at 0 the value is the box minimum), the four rationals of the
last step become three additions each rounded one step down, and the slope cap is
computed in doubles with a relative margin of 1e-9, the Fractions kept for the
cases outside the range that argument needs.
"""
import math
from fractions import Fraction as Q
from itertools import combinations

import numpy as np
import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st
from scipy.optimize import nnls

from . import treatment
from .brute import geo_panel, panel
from .certificate import (ExactData, _greedy_lambda, add_down, box_bound_float, hull_bound, hull_bound_float,
                          knapsack_dual)
from . import certificate
from .treatment import _split, gram_floor, root_above, rss_floor, slope_cap, square_sum, yy_bracket

SETTINGS = settings(max_examples=120, deadline=None, suppress_health_check=[HealthCheck.too_slow])
FINITE = st.floats(allow_nan=False, allow_infinity=False)


def _old_floor(M, T, lam):
    """The floor as it was computed before: Fraction cap, the general hull bound."""
    y, X, _ = _split(M, T)
    yy = square_sum(y)
    cap = Q(math.sqrt(float(yy / lam)) * (1 + 1e-9))
    while cap * cap * lam < yy:
        cap *= Q(1000001, 1000000)
    w, _ = nnls(X, y)
    fl = hull_bound_float(y, X, np.minimum(w, float(cap)), 0.0, float(cap), X.shape[1])
    return max(Q(0), fl) if fl != -np.inf else Q(0)


def _box_point(seed, shift=0.0):
    y, X = panel(seed)
    w, _ = nnls(X, y)
    hi = 2.0 * float(w.max()) + 0.5
    w = np.clip(w + shift * np.random.default_rng(seed).normal(size=w.shape), 0.0, hi)
    return y, X, w, hi


# ---------------------------------------------------------------- smoke


def test_smoke_a_geo_floor_is_finite_and_below_the_group_rss():
    M = geo_panel(5)
    T = (0, 3)
    fl = rss_floor(M, T, gram_floor(M))
    y, X, _ = _split(M, T)
    w, rnorm = nnls(X, y)
    assert 0 <= fl <= Q(float(rnorm) ** 2) * (1 + Q(1, 10 ** 9))


# ---------------------------------------------------------------- rounding down


@SETTINGS
@given(a=FINITE, b=FINITE)
@example(a=1.0, b=2.0 ** -53 + 2.0 ** -105)          # rounds up to nearest
@example(a=2.0, b=-(2.0 ** -53))                       # just below a power of two
@example(a=0.0, b=0.0)
def test_property_add_down_is_at_most_the_exact_sum(a, b):
    s = add_down(a, b)
    exact = Q(a) + Q(b)
    if not math.isfinite(a + b):
        assert s == -math.inf
    elif s == -math.inf:                                           # one step below the lowest double
        assert a + b == -1.7976931348623157e308
    else:
        assert Q(s) <= exact
        assert exact - Q(s) <= 2 * abs(Q(math.ulp(a + b)))       # one step below nearest at most


def test_add_down_where_nearest_rounds_up():
    a, b = 1.0, 2.0 ** -53 + 2.0 ** -105
    assert a + b > 1.0 and Q(a + b) > Q(a) + Q(b)                   # nearest rounds up here
    assert Q(add_down(a, b)) <= Q(a) + Q(b)


@pytest.mark.parametrize("a,b", [(1.7e308, 1.7e308), (math.inf, 1.0), (-math.inf, 1.0), (math.nan, 1.0)])
def test_add_down_gives_no_bound_off_the_range(a, b):
    assert add_down(a, b) == -math.inf


# ---------------------------------------------------------------- the box bound


@SETTINGS
@given(seed=st.integers(0, 10_000), shift=st.floats(0.0, 1.0), k=st.integers(-30, 30))
@example(seed=0, shift=0.0, k=30)
@example(seed=0, shift=0.0, k=-30)
def test_property_the_box_bound_never_exceeds_the_exact_one(seed, shift, k):
    y, X, w, hi = _box_point(seed, shift)
    y, X, w, hi = y * 2.0 ** k, X * 2.0 ** k, w, hi
    assert Q(box_bound_float(y, X, w, hi)) <= hull_bound(ExactData(y, X), w, 0.0, hi, X.shape[1])


@SETTINGS
@given(seed=st.integers(0, 10_000), shift=st.floats(0.0, 1.0))
def test_property_the_box_bound_is_the_hull_bound_to_rounding(seed, shift):
    """The same value as the general certificate at lo = 0, m = p: the multiplier 0
    gives the box minimum, so only the last roundings differ."""
    y, X, w, hi = _box_point(seed, shift)
    exact = hull_bound(ExactData(y, X), w, 0.0, hi, X.shape[1])
    b = Q(box_bound_float(y, X, w, hi))
    h = Q(hull_bound_float(y, X, w, 0.0, hi, X.shape[1]))
    scale = max(Q(1), abs(exact))
    assert abs(b - h) <= scale * Q(1, 10 ** 12)


@SETTINGS
@given(g=st.lists(st.floats(-1e6, 1e6), min_size=1, max_size=12), hi=st.floats(1e-3, 1e3))
@example(g=[-5e-324], hi=1.5)                     # a subnormal product: its rounding is absolute
def test_property_the_greedy_multiplier_is_zero_on_the_box_once_a_gradient_is_nonnegative(g, hi):
    """Why the multiplier 0 loses nothing: with lo = 0 and m = p the fill stops at
    the first nonnegative cost, where the multiplier is 0; with every cost negative
    the fill takes them all and the dual at any multiplier is the same sum."""
    g = np.array(g)
    if (g >= 0).any():
        assert _greedy_lambda(g, 0.0, hi, g.size) == 0.0
    exact_min = sum(Q(hi) * min(Q(0), Q(float(v))) for v in g)
    got = Q(knapsack_dual(g, 0.0, hi, g.size, lam=0.0))
    # relative to the sum, plus half a subnormal step for each product and the sum
    # in the subnormal range, where rounding is absolute
    assert abs(got - exact_min) <= abs(exact_min) * Q(1, 10 ** 12) + (g.size + 1) * Q(2) ** -1074


def test_every_gradient_negative():
    """The case where the greedy multiplier is not 0: the box bound still holds."""
    y, X = panel(1)
    w = np.zeros(X.shape[1])                  # at the origin every control lowers the RSS
    hi = 5.0
    assert Q(box_bound_float(y, X, w, hi)) <= hull_bound(ExactData(y, X), w, 0.0, hi, X.shape[1])


@pytest.mark.parametrize("k", [-1040, -1060])
def test_subnormal_data_keep_the_box_bound_below_the_exact_one(k):
    y, X, w, hi = _box_point(0)
    y, X = y * 2.0 ** k, X * 2.0 ** k
    assert Q(box_bound_float(y, X, w, hi)) <= hull_bound(ExactData(y, X), w, 0.0, hi, X.shape[1])


def test_an_overflow_gives_no_box_bound():
    y, X, w, hi = _box_point(0)
    assert box_bound_float(y * 2.0 ** 1000, X * 2.0 ** 1000, w, hi) == -np.inf


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_a_non_finite_point_is_refused(bad):
    y, X, w, hi = _box_point(0)
    w = w.copy(); w[1] = bad
    with pytest.raises(ValueError, match="finite"):
        box_bound_float(y, X, w, hi)


def test_a_point_of_the_wrong_shape_is_refused():
    y, X, w, hi = _box_point(0)
    with pytest.raises(ValueError, match="one weight per control"):
        box_bound_float(y, X, w[:-1], hi)


def _patched(monkeypatch, f, f_err, g, d, gw, gw_err, dot=lambda k, m: 0.0):
    monkeypatch.setattr(certificate, "residual_bound", lambda y, X, w: (np.zeros(1), np.zeros(1)))
    monkeypatch.setattr(certificate, "objective_bound", lambda r, e: (f, f_err))
    monkeypatch.setattr(certificate, "gradient_bound", lambda X, r, e: (np.array(g), np.array(d)))
    monkeypatch.setattr(certificate, "linear_bound", lambda g_, d_, w: (gw, gw_err))
    monkeypatch.setattr(certificate, "_dot_err", dot)
    X = np.ones((1, len(g))); return np.zeros(1), X, np.zeros(len(g))


def test_the_last_steps_round_down_where_nearest_rounds_up(monkeypatch):
    """f + kn - g'w with no error terms, at a sum whose nearest double is above it:
    the bound is below the exact sum of its own terms."""
    kn_term = -(2.0 ** -53 + 2.0 ** -105)
    y, X, w = _patched(monkeypatch, 1.0, 0.0, [kn_term], [0.0], 0.0, 0.0)
    got = box_bound_float(y, X, w, 1.0)
    exact = Q(1) + Q(kn_term)
    assert Q(1.0 + kn_term) > exact                                # nearest rounds up here
    assert Q(got) <= exact


def test_the_box_bound_subtracts_every_error_term(monkeypatch):
    """With the pieces fixed, the bound is f + kn - g'w less f's error, the
    knapsack's rounding, its perturbation max(d) p hi and g'w's error, inflated,
    each step rounded down."""
    dot = lambda k, m: 1e-3 * m + k * 1e-6
    g, d, hi = [-2.0, 0.5, -1.0], [0.01, 0.03, 0.02], 1.5
    y, X, w = _patched(monkeypatch, 10.0, 0.25, g, d, 0.75, 0.125, dot)
    terms = hi * np.minimum(0.0, np.array(g))
    kn = float(terms.sum()); kn_err = dot(6, float(np.abs(terms).sum())) + 0.03 * (3 * hi)
    err = certificate.inflate(0.25 + kn_err + 0.125)
    want = add_down(add_down(add_down(10.0, kn), -0.75), -err)
    assert box_bound_float(y, X, w, hi) == want


def test_an_infinite_term_gives_no_box_bound(monkeypatch):
    for f, gw in ((math.inf, 0.0), (0.0, math.nan)):
        y, X, w = _patched(monkeypatch, f, 0.0, [1.0], [0.0], gw, 0.0)
        assert box_bound_float(y, X, w, 1.0) == -np.inf


# ---------------------------------------------------------------- the slope cap


@SETTINGS
@given(a=st.integers(1, 10 ** 40), b=st.integers(1, 10 ** 40))
@example(a=4, b=1)                                                 # a perfect square
@example(a=1, b=9)
def test_property_the_root_is_above_and_tight(a, b):
    q = Q(a, b)
    r = root_above(q)
    assert r * r > q
    assert (r - Q(1, q.denominator * treatment._ROOT_SCALE)) ** 2 <= q




@SETTINGS
@given(yy=st.floats(0.0, 2.0 ** 900), num=st.integers(1, 10 ** 30), den=st.integers(1, 10 ** 30),
       e=st.integers(-800, 800))
@example(yy=1.0, num=1, den=3, e=0)
def test_property_the_float_cap_covers_the_exact_quotient(yy, num, den, e):
    lam = Q(num, den) * Q(2) ** e
    cap = slope_cap(yy, lam)
    if cap == math.inf:                                            # a quotient past the doubles
        assert Q(yy) / lam > Q(2) ** 2046
        return
    assert Q(cap) * Q(cap) * lam >= Q(yy)
    # and not loose: cap^2 lambda at most y'y (1 + 3e-9)^2, or one subnormal step
    assert Q(cap) * Q(cap) * lam <= Q(yy) * (1 + Q(3, 10 ** 9)) ** 2 + \
        (2 * Q(cap) + Q(2) ** -1074) * Q(2) ** -1074 * lam


def test_a_zero_series_has_a_zero_cap():
    assert slope_cap(0.0, Q(1, 7)) == 0.0


@pytest.mark.parametrize("yy,lam", [(1.0, Q(1, 2 ** 1100)), (2.0 ** 1000, Q(1, 2 ** 60)), (2.0 ** -1060, Q(1)),
                                    (2.0 ** -1000, Q(2 ** 900))])
def test_a_quotient_off_the_range_takes_the_exact_cap(yy, lam):
    """A Gram floor below the doubles, a quotient past their top or bottom, or a
    subnormal y'y: the float argument does not hold, and the cap is the exact
    construction's, a double whose square times lambda is at least y'y."""
    cap = slope_cap(yy, lam)
    assert 0 < cap < math.inf and Q(cap) * Q(cap) * lam >= Q(yy)


def test_a_cap_past_the_doubles_is_infinite_and_the_floor_clears_nothing():
    assert slope_cap(1.0, Q(1, 2 ** 2100)) == math.inf
    y, X, w, _ = _box_point(0)
    assert box_bound_float(y, X, w, math.inf) == -np.inf


def test_the_exact_cap_is_tight():
    """The integer root is within a relative 2^-64 of sqrt(y'y / lambda)."""
    yy, lam = Q(1), Q(3)
    cap = Q(treatment._cap_exact(yy, lam))
    assert cap * cap * lam >= yy and cap * cap * lam <= yy * (1 + Q(1, 2 ** 50))


def test_an_exact_yy_takes_the_exact_cap():
    yy = square_sum(np.array([3.0, 4.0]))
    cap = slope_cap(yy, Q(1, 3))
    assert Q(cap) * Q(cap) * Q(1, 3) >= yy


# ---------------------------------------------------------------- the floor


def test_the_floor_is_the_old_floor_to_rounding_on_every_geo_group():
    """Below the old floor by a few rounding steps at most, on every pair of a
    seven-geo panel: the same bound, computed without its per-call costs."""
    M = geo_panel(5)
    lam = gram_floor(M)
    for T in combinations(range(M.shape[1]), 2):
        new, old = rss_floor(M, T, lam), _old_floor(M, T, lam)
        assert Q(new) <= Q(old) + abs(Q(old)) * Q(1, 10 ** 15) + Q(1, 10 ** 300)
        assert Q(new) >= Q(old) - abs(Q(old)) * Q(1, 10 ** 12)


@pytest.mark.parametrize("raw", [-1.0, -1e-300, -math.inf])
def test_a_box_bound_at_or_below_zero_gives_a_zero_floor(monkeypatch, raw):
    """A bound below zero (an exact or near-exact fit, where the rounding error
    exceeds the residual) or -inf (an overflow) is a floor of 0. A panel cannot
    make the first case with a Gram floor: an exact fit makes the Gram singular."""
    M = geo_panel(5)
    lam = gram_floor(M)
    monkeypatch.setattr(treatment, "box_bound_float", lambda y, X, w, hi: raw)
    fl = rss_floor(M, (1, 4), lam)
    assert fl == 0 and isinstance(fl, Q)


def test_an_overflow_floor_is_zero_not_minus_infinity():
    """Past the doubles' range the box bound is -inf; the floor is 0."""
    M = geo_panel(5) * 2.0 ** 1000
    lam = gram_floor(M)
    fl = rss_floor(M, (1, 4), lam)
    assert fl == 0 and isinstance(fl, Q)


def test_the_floor_takes_a_bracket_top_as_yy():
    M = geo_panel(5)
    lam = gram_floor(M)
    y = _split(M, (1, 4))[0]
    lo, hi = yy_bracket(y)
    assert rss_floor(M, (1, 4), lam, yy=hi) == rss_floor(M, (1, 4), lam, yy=hi)
    assert Q(rss_floor(M, (1, 4), lam, yy=hi)) <= Q(_old_floor(M, (1, 4), lam)) * (1 + Q(1, 10 ** 15))


def test_the_floors_certify_the_same_design():
    M = geo_panel(5)
    out = treatment.certify_treatment(M, 2, n_test=8, restarts=3)
    assert out.certified and out.treatment is not None
