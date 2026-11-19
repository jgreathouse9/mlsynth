"""The floors' y'y as a float bracket, and the relative screen read in doubles (#965).

Every group needs y'y for its floor's slope cap, and the exact sum of squares in
Python integers took 6.1 s of the 48 s four-geo GeoLift run. yy_bracket bounds y'y
in doubles, lo <= y'y <= hi, from math.fsum of the squares, and the cap reads the
bracket's top.

The screen against the incumbent reads the relative ceiling bar L_T^2 in doubles
(treatment._drop_side): a sign whose floor is at or above it times 1 + margin is
dropped, and the margin covers the roundings of the level's sum, its square and the
product (treatment._screen_terms). Dropping is the one direction that has to be
exact: a row kept is decided again in rationals when it is read."""
from fractions import Fraction as Q

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthDataError

from mlsynth.utils.tbr_helpers.design.exact import fastfloor, treatment
from .brute import geo_panel
from .brute import raw_panel
from mlsynth.utils.tbr_helpers.design.exact.treatment import _drop_side, _screen_terms, _split, certify_treatment, group_level, level_doubles, relative_bar, relative_ceiling, square_sum, yy_bracket

N_TEST = 8
SETTINGS = settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
SAFE = st.floats(min_value=-2.0 ** 400, max_value=2.0 ** 400, allow_nan=False, allow_infinity=False).filter(
    lambda v: v == 0 or abs(v) >= 2.0 ** -400)


# ---------------------------------------------------------------- smoke


def test_smoke_the_bracket_holds_on_a_geo_group():
    M = geo_panel(5)
    y = _split(M, (0, 3))[0]
    lo, hi = yy_bracket(y)
    assert Q(lo) <= square_sum(y) <= Q(hi) and hi - lo <= 2.0 ** -45 * hi


def test_smoke_the_search_certifies():
    M = raw_panel(5)
    out = certify_treatment(M, 2, n_test=N_TEST, restarts=3, blank_periods=0)
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

LEVELS = st.lists(st.fractions(min_value=Q(1, 10 ** 6), max_value=10 ** 6), min_size=1, max_size=8)


def _exact_ceiling(sums, T, n, bar):
    return relative_ceiling(group_level(sums, T, n), bar)


@SETTINGS
@given(sums=LEVELS, n=st.integers(1, 400), rss=st.fractions(min_value=Q(1, 10 ** 6), max_value=10 ** 6),
       lev=st.fractions(min_value=Q(1, 10 ** 6), max_value=10 ** 6), g=st.sampled_from([Q(0), Q(1, 100), Q(1, 3)]),
       scale=st.fractions(min_value=Q(1, 2), max_value=2), ulps=st.integers(-3, 3))
def test_property_a_dropped_side_cannot_compete(sums, n, rss, lev, g, scale, ulps):
    """A side the doubles drop has its floor at or above the exact ceiling, for
    floors at, just around and well away from the ceiling."""
    bar = relative_bar(rss, lev, g)
    T = tuple(range(len(sums)))
    c = _exact_ceiling(sums, T, n, bar)
    f = float(c * scale)
    for _ in range(abs(ulps)):
        f = np.nextafter(f, np.inf if ulps > 0 else -np.inf)
    ell = level_doubles(sums, n)
    terms = _screen_terms(bar, ell, len(T))
    assume(terms is not None)
    b, ell, margin = terms
    if _drop_side([f], [float(ell[list(T)].sum())], b, margin)[0]:
        assert Q(f) >= c


@SETTINGS
@given(sums=LEVELS, n=st.integers(1, 400), rss=st.fractions(min_value=Q(1, 10 ** 6), max_value=10 ** 6),
       lev=st.fractions(min_value=Q(1, 10 ** 6), max_value=10 ** 6))
def test_property_the_screen_decides_away_from_the_ceiling(sums, n, rss, lev):
    """A floor at twice the ceiling is dropped and one at half of it is kept: the
    margin is a few ulps, not a loose bound."""
    bar = relative_bar(rss, lev, Q(0))
    T = tuple(range(len(sums)))
    c = _exact_ceiling(sums, T, n, bar)
    b, ell, margin = _screen_terms(bar, level_doubles(sums, n), len(T))
    L = [float(ell[list(T)].sum())]
    assert _drop_side([float(2 * c)], L, b, margin)[0]
    assert not _drop_side([float(c / 2)], L, b, margin)[0]
    assert margin <= 2.0 ** -40


def test_a_floor_at_the_ceiling_is_kept():
    """At the ceiling exactly the floor cannot compete, but the doubles cannot show
    it across the margin: the row is kept and decided in rationals."""
    sums, n = [Q(3), Q(7)], 5
    bar = relative_bar(Q(2), Q(1), Q(0))
    c = _exact_ceiling(sums, (0, 1), n, bar)
    b, ell, margin = _screen_terms(bar, level_doubles(sums, n), 2)
    assert not _drop_side([float(c)], [float(ell.sum())], b, margin)[0]


def test_no_incumbent_has_no_screen():
    assert _screen_terms(None, np.array([1.0, 2.0]), 1) is None


@pytest.mark.parametrize("bar,ell", [(Q(1, 2 ** 1100), [1.0, 2.0]), (Q(1), [2.0 ** -1010, 1.0]),
                                     (Q(2 ** 1030), [1.0, 1.0])])
def test_a_bar_or_level_the_doubles_do_not_carry_has_no_screen(bar, ell):
    """A bar or level below 2^-1000 rounds absolutely, not relatively, and a bar past
    the doubles is infinite: no row is dropped in doubles."""
    assert _screen_terms(bar, np.array(ell), 1) is None


def test_a_nan_floor_or_an_overflowing_ceiling_is_never_dropped():
    b, margin = 1.0, 2.0 ** -40
    assert not _drop_side([np.nan], [1.0], b, margin)[0]
    assert not _drop_side([1e308], [1e200], b, margin)[0]          # the ceiling overflows
    assert not _drop_side([1.0], [2.0 ** -520], b, margin)[0]      # the level's square is below the normal range


def test_a_zero_floor_is_kept():
    assert not _drop_side([0.0], [3.0], 1.0, 2.0 ** -40)[0]


def test_rows_with_no_bracket_certify_the_same_design(monkeypatch):
    """With no group bracketed, every floor's cap reads the exact y'y: the search is
    the same."""
    M = raw_panel(5)
    monkeypatch.setattr(fastfloor, "AVAILABLE", False)
    want = certify_treatment(M, 2, n_test=N_TEST, restarts=3, blank_periods=0)
    monkeypatch.setattr(treatment, "yy_bracket", lambda y: None)
    got = certify_treatment(M, 2, n_test=N_TEST, restarts=3, blank_periods=0)
    assert (got.treatment, got.controls, got.key, got.rss, got.certified) == \
        (want.treatment, want.controls, want.key, want.rss, want.certified)
    assert got.screened_out == want.screened_out


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_the_floors_refuse_a_non_finite_series(bad):
    """A non-finite series leaves no bracket, and its exact y'y is refused."""
    M = geo_panel(5).copy()
    M[3, 2] = bad
    with pytest.raises(MlsynthDataError, match="finite"):
        treatment._floor_chunk(M, treatment.gram_floor(geo_panel(5)), [(1, 2)])
    assert treatment._floor_chunk(M, None, [(0, 1)])[0][:2] == (Q(0), Q(0))   # no Gram floor: nothing to bound
