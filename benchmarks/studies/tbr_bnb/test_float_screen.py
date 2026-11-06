"""The floors' y'y as a float bracket, and the screen read in doubles (#965).

Every group needs y'y for its floor's slope cap and for the screen against the
incumbent's ceiling, and the exact sum of squares in Python integers took 6.1 s of
the 48 s four-geo GeoLift run; the screen's Fraction comparisons took 2.4 s more.
yy_bracket bounds y'y in doubles, lo <= y'y <= hi, from math.fsum of the squares.
The ceiling is nondecreasing in y'y, so a group competes for every y'y in the
bracket when its floor is below ceiling(lo), and for none when it is at least
ceiling(hi); compete_bracket decides that in doubles across a margin, and the
exact y'y is computed only for a group it cannot decide or one that is searched or
certified."""
from fractions import Fraction as Q
from itertools import combinations

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthDataError

from . import treatment
from .brute import geo_panel
from .treatment import (EDGE, SLACK, _competes, _split, can_compete, ceiling, certify_treatment,
                        compete_bracket, square_sum, yy_bracket)

N_TEST = 8
SETTINGS = settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
SAFE = st.floats(min_value=-2.0 ** 400, max_value=2.0 ** 400, allow_nan=False, allow_infinity=False).filter(
    lambda v: v == 0 or abs(v) >= 2.0 ** -400)
CLASSES = st.sampled_from([None] + [Q(c, 100) for c in (50, 80, 95, 97, 98, 99, 100)])


def _double_ceiling(yy, r_star, rss_star):
    """The ceiling as compete_bracket computes it in doubles."""
    a, b, R = treatment._class_doubles(r_star, rss_star)
    second = min(R, yy * b)
    return second if a is None else max(yy * a, second)


# ---------------------------------------------------------------- smoke


def test_smoke_the_bracket_holds_on_a_geo_group():
    M = geo_panel(5)
    y = _split(M, (0, 3))[0]
    lo, hi = yy_bracket(y)
    assert Q(lo) <= square_sum(y) <= Q(hi) and hi - lo <= 2.0 ** -45 * hi


def test_smoke_the_search_certifies_the_same_design():
    M = geo_panel(5)
    out = certify_treatment(M, 2, n_test=N_TEST, restarts=3)
    assert out.certified and out.treatment is not None


# ---------------------------------------------------------------- the bracket


@SETTINGS
@given(st.lists(SAFE, min_size=0, max_size=60))
def test_property_the_bracket_holds_the_exact_sum_of_squares(v):
    v = np.array(v, dtype=float)
    b = yy_bracket(v)
    assert b is not None
    lo, hi = b
    exact = square_sum(v)
    assert Q(lo) <= exact <= Q(hi)
    assert hi - lo <= 2.0 ** -45 * hi                     # a few ulps, not a loose bound


@SETTINGS
@given(st.lists(st.floats(min_value=1.0, max_value=2.0, allow_nan=False), min_size=2, max_size=40),
       st.integers(-300, 300))
def test_property_the_bracket_holds_at_every_scale(v, e):
    v = np.ldexp(np.array(v), e)
    lo, hi = yy_bracket(v)
    assert Q(lo) <= square_sum(v) <= Q(hi)


def test_the_bracket_holds_where_the_float_sum_rounds_down():
    """Squares that round down and a sum that rounds down: the float sum is below
    the exact one, so a bracket without its margin misses it."""
    v = np.full(50, 1.0 + 2.0 ** -27)                     # (1 + 2^-27)^2 = 1 + 2^-26 + 2^-54
    s = float(np.sum(v * v))
    assert Q(s) < square_sum(v)
    lo, hi = yy_bracket(v)
    assert Q(lo) <= square_sum(v) <= Q(hi)


def test_an_all_zero_series_is_bracketed_at_zero():
    assert yy_bracket(np.zeros(5)) == (0.0, 0.0)
    assert yy_bracket(np.zeros(0)) == (0.0, 0.0)


@pytest.mark.parametrize("v", [[1.0, 2.0 ** -520], [2.0 ** 520, 1.0], [5e-324], [-(2.0 ** 512)]])
def test_a_square_outside_the_normal_range_leaves_no_bracket(v):
    """A square below the normal range loses the relative bound on its rounding,
    and one above it overflows: the exact sum is taken for those groups."""
    assert yy_bracket(np.array(v)) is None


@pytest.mark.parametrize("v", [[1.0, np.nan], [np.inf, 1.0]])
def test_a_non_finite_series_leaves_no_bracket(v):
    assert yy_bracket(np.array(v)) is None


def test_too_many_periods_leave_no_bracket(monkeypatch):
    """The sum of 2^20 squares of up to 2^500 each is below the overflow threshold;
    a longer series could exceed it."""
    monkeypatch.setattr(treatment, "_MAX_PERIODS", 4)
    assert yy_bracket(np.ones(5)) is None and yy_bracket(np.ones(4)) == pytest.approx((4.0, 4.0))


# ---------------------------------------------------------------- the screen in doubles


@SETTINGS
@given(v=st.lists(SAFE.filter(lambda x: abs(x) < 2.0 ** 300), min_size=1, max_size=30),
       r_star=CLASSES, rss_frac=st.fractions(0, 2), floor_frac=st.fractions(0, Q(11, 10)))
def test_property_a_decided_bracket_is_the_exact_screen(v, r_star, rss_frac, floor_frac):
    """Whatever compete_bracket decides is can_compete's answer at the exact y'y."""
    v = np.array(v, dtype=float)
    yy = square_sum(v)
    assume(yy > 0)
    lo, hi = yy_bracket(v)
    rss_star = None if r_star is None else yy * rss_frac
    floor = Q(float(yy * floor_frac))
    got = compete_bracket(floor, lo, hi, r_star, rss_star)
    if got is not None:
        assert got == can_compete(yy, floor, r_star, rss_star)


@SETTINGS
@given(lo=st.floats(1e-200, 1e200), width=st.integers(0, 64), r_star=CLASSES.filter(lambda r: r is not None),
       rss_frac=st.fractions(0, 2), at=st.sampled_from(["lo", "hi", "mid"]),
       floor_frac=st.fractions(0, Q(11, 10)))
def test_property_a_decided_bracket_holds_for_every_yy_in_it(lo, width, r_star, rss_frac, at, floor_frac):
    """The answer holds across the bracket: at both ends and between them."""
    hi = float(np.nextafter(lo, np.inf)) if width else lo
    for _ in range(width - 1):
        hi = float(np.nextafter(hi, np.inf))
    yy = {"lo": Q(lo), "hi": Q(hi), "mid": (Q(lo) + Q(hi)) / 2}[at]
    rss_star = Q(lo) * rss_frac
    floor = Q(float(Q(lo) * floor_frac))
    got = compete_bracket(floor, lo, hi, r_star, rss_star)
    if got is not None:
        assert got == can_compete(yy, floor, r_star, rss_star)


def test_the_bracket_decides_far_from_the_ceiling():
    r, rss = Q(98, 100), Q(10)
    c = ceiling(Q(1000), r, rss)
    assert compete_bracket(c / 2, 1000.0, 1000.0, r, rss) is True
    assert compete_bracket(c * 2, 1000.0, 1000.0, r, rss) is False


def test_a_floor_at_the_ceiling_is_left_to_the_exact_screen():
    """A floor equal to the ceiling cannot be told from one just below it in doubles."""
    r, rss = Q(98, 100), Q(10)
    for yy in (Q(1000), Q(3951050), Q(7, 3)):
        c = ceiling(yy, r, rss)
        assert compete_bracket(c, float(yy), float(yy), r, rss) is None


def test_a_floor_at_the_ceiling_where_the_double_ceiling_rounds_up_does_not_compete():
    """At class 0.50 and y'y = 1000.37 the ceiling in doubles is above the exact one,
    and the floor at the exact ceiling, rounded, is below it: without the margin the
    group would be taken as competing, though floor < ceiling is false."""
    r, rss, lo = Q(50, 100), Q(10 ** 12), 1000.37
    c = ceiling(Q(lo), r, rss)
    assert float(c) < _double_ceiling(lo, r, rss)
    assert compete_bracket(c, lo, lo, r, rss) is not True
    assert not can_compete(Q(lo), c, r, rss)


def test_a_floor_just_below_the_ceiling_where_the_double_ceiling_rounds_down_competes():
    """At class 0.98 and y'y = 1000.37 the ceiling in doubles is below the exact one,
    and a floor just below the exact ceiling rounds to a double above it: without
    the margin the group would be screened out, though floor < ceiling."""
    r, rss, lo = Q(98, 100), Q(10 ** 12), 1000.37
    c = ceiling(Q(lo), r, rss)
    floor = c * (1 - Q(1, 2 ** 70))
    assert float(floor) > _double_ceiling(lo, r, rss)
    assert compete_bracket(floor, lo, lo, r, rss) is not False
    assert can_compete(Q(lo), floor, r, rss)


def test_a_bracket_straddling_the_ceiling_is_undecided():
    """The ceiling at lo is below the floor and at hi above it: the answer depends
    on where y'y is."""
    r = Q(98, 100)
    up = r + EDGE - SLACK
    yy_lo, yy_hi = 1000.0, 1000.0 * (1 + 2.0 ** -40)
    floor = (Q(yy_lo) + Q(yy_hi)) / 2 * (1 - up * up)
    assert compete_bracket(floor, yy_lo, yy_hi, r, Q(0)) is None


def test_with_no_incumbent_every_group_competes():
    assert compete_bracket(Q(10 ** 9), 1.0, 1.0, None, None) is True


def test_a_zero_floor_competes_against_a_positive_ceiling():
    assert compete_bracket(Q(0), 1000.0, 1000.0, Q(98, 100), Q(10)) is True


def test_a_ceiling_near_the_bottom_of_the_range_is_left_to_the_exact_screen():
    """Below the normal range a product loses the relative bound on its rounding."""
    r = Q(98, 100)
    assert compete_bracket(Q(0), 2.0 ** -1060, 2.0 ** -1060, r, Q(0)) is None


def test_a_zero_ceiling_is_left_to_the_exact_screen():
    """y'y = 0 gives a zero ceiling; a zero floor does not compete against it."""
    got = compete_bracket(Q(0), 0.0, 0.0, Q(98, 100), Q(0))
    assert got is None or got == can_compete(Q(0), Q(0), Q(98, 100), Q(0))


# ---------------------------------------------------------------- the fallback


def test_an_undecided_group_reads_the_exact_yy():
    r, rss = Q(98, 100), Q(10)
    yy = Q(1000)
    c = ceiling(yy, r, rss)
    calls = []

    def exact():
        calls.append(1)
        return yy

    assert _competes(c, (1000.0, 1000.0), exact, r, rss) is can_compete(yy, c, r, rss)
    assert calls == [1]


def test_a_decided_group_does_not_read_the_exact_yy():
    r, rss = Q(98, 100), Q(10)
    c = ceiling(Q(1000), r, rss)

    def exact():
        raise AssertionError("the exact y'y was read for a decided group")

    assert _competes(c / 2, (1000.0, 1000.0), exact, r, rss) is True
    assert _competes(c * 2, (1000.0, 1000.0), exact, r, rss) is False


def test_an_exact_yy_is_read_directly():
    """A group whose series leaves no bracket carries its exact y'y."""
    r, rss = Q(98, 100), Q(10)
    yy = Q(1000)
    c = ceiling(yy, r, rss)

    def exact():
        raise AssertionError("the row's exact y'y should be used")

    for floor in (c / 2, c, c * 2):
        assert _competes(floor, yy, exact, r, rss) is can_compete(yy, floor, r, rss)


# ---------------------------------------------------------------- the search


def _count_exact(monkeypatch):
    real, calls = treatment._yy, []
    monkeypatch.setattr(treatment, "_yy", lambda M, T: calls.append(tuple(T)) or real(M, T))
    return calls


def test_the_exact_yy_is_read_only_for_groups_searched_or_certified(monkeypatch):
    M = geo_panel(5)
    calls = _count_exact(monkeypatch)
    out = certify_treatment(M, 2, n_test=N_TEST, restarts=3, pass_restarts=5)
    groups = len(list(combinations(range(M.shape[1]), 2)))
    reached = {tuple(s["treatment"]) for s in out.survivors}
    assert set(calls) <= reached | set(combinations(range(M.shape[1]), 2))
    assert len(set(calls)) <= len(reached) + out.searched and len(calls) < groups
    assert len(calls) == len(set(calls))                  # computed once a group


def test_every_survivor_is_certified_against_the_exact_ceiling(monkeypatch):
    M = geo_panel(5)
    out = certify_treatment(M, 2, n_test=N_TEST, restarts=3, pass_restarts=5)
    for s in out.survivors:
        c = s["ceiling"]
        assert c is None or isinstance(c, Q)


def test_an_undecided_screen_everywhere_certifies_the_same_design(monkeypatch):
    """With every group left to the exact screen the search is the exact one."""
    M = geo_panel(5)
    want = certify_treatment(M, 2, n_test=N_TEST, restarts=3)
    monkeypatch.setattr(treatment, "compete_bracket", lambda *a: None)
    calls = _count_exact(monkeypatch)
    got = certify_treatment(M, 2, n_test=N_TEST, restarts=3)
    assert (got.treatment, got.key, got.rss, got.certified) == (want.treatment, want.key, want.rss, want.certified)
    assert len(got.survivors) == len(want.survivors) and got.screened_out == want.screened_out
    assert calls and len(calls) == len(set(calls))       # every group read exactly, once


def test_a_panel_too_small_for_the_bracket_takes_the_exact_route():
    """Scaled by 2^-520 no group has a bracket, and the certified design is the
    unscaled panel's: correlation and the gates do not see the scale."""
    M = geo_panel(5)
    want = certify_treatment(M, 2, n_test=N_TEST, restarts=3)
    small = np.ldexp(M, -520)
    assert all(yy_bracket(_split(small, T)[0]) is None for T in combinations(range(small.shape[1]), 2))
    got = certify_treatment(small, 2, n_test=N_TEST, restarts=3)
    assert got.certified and got.treatment == want.treatment


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_the_floors_refuse_a_non_finite_series(bad):
    """A non-finite series leaves no bracket, and its exact y'y is refused."""
    M = geo_panel(5).copy()
    M[3, 2] = bad
    with pytest.raises(MlsynthDataError, match="finite"):
        treatment._floor_chunk(M, treatment.gram_floor(geo_panel(5)), [(1, 2)])
    assert treatment._floor_chunk(M, None, [(0, 1)])[0][1] is not None    # a finite group still has its y'y
