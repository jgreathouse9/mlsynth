"""The floors of many treatment groups in one compiled loop (fastfloor), and the
treatment search's use of it when numba is installed.

The kernel computes, for each group, what rss_floor computes: the group's series as
_split sums it, a bracket on y'y, the slope cap in doubles, an NNLS point (Lawson
and Hanson's active set on the group's Gram, warm-started from the previous group's
support), and box_bound_float's certificate at that point. A group it cannot do in
doubles (an entry outside the bracket's range, a quotient off the cap's range, eight
or more treated geos) comes back as nan and takes rss_floor's path. The tests check
the kernel against _split, square_sum, the exact certificate and rss_floor, and the
search with the kernel against the search without it.
"""
import math
from fractions import Fraction as Q
from itertools import combinations

import numpy as np
import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st
from scipy.optimize import nnls

from mlsynth.exceptions import MlsynthDataError

from . import fastfloor, treatment
from .brute import geo_panel
from .certificate import ExactData, hull_bound
from .treatment import _split, certify_treatment, gram_floor, rss_floor, square_sum

needs_numba = pytest.mark.skipif(not fastfloor.AVAILABLE, reason="numba is not installed")

SETTINGS = settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
N_TEST = 8


def _panel(seed, n, p, scale=1.0):
    rng = np.random.default_rng(seed)
    base = rng.normal(100.0, 10.0, size=(n, 1))
    return np.ascontiguousarray((base + rng.normal(0.0, 5.0, size=(n, p)).cumsum(axis=0) / 4) * scale)


def _parts(M, T):
    """The kernel on one group; a panel with no Gram floor gets 0, so the kernel leaves
    its floor (nan) but still builds the series and the bracket."""
    lam = gram_floor(M)
    return fastfloor.floor_parts(np.ascontiguousarray(M), np.ascontiguousarray(M.T @ M),
                                 np.asarray(T, np.int64), float(lam) if lam is not None else 0.0), lam


# ---------------------------------------------------------------- smoke


@needs_numba
def test_smoke_the_kernel_gives_a_floor_and_a_bracket_for_every_group():
    M = geo_panel(5)
    groups = np.array(list(combinations(range(M.shape[1]), 2)), np.int64)
    fl, lo, hi = fastfloor.floors(np.ascontiguousarray(M), np.ascontiguousarray(M.T @ M), groups,
                                  float(gram_floor(M)))
    assert fl.shape == lo.shape == hi.shape == (len(groups),)
    assert np.all(np.isfinite(fl)) and np.all(lo <= hi)


# ---------------------------------------------------------------- the kernel


@needs_numba
@SETTINGS
@given(seed=st.integers(0, 10_000), n=st.integers(5, 60), p=st.integers(9, 14), k=st.integers(1, 7))
def test_property_the_series_is_splits_bit_for_bit(seed, n, p, k):
    """The kernel's y is _split's, so its floor bounds the RSS the search scores."""
    M = _panel(seed, n, p)
    T = tuple(sorted(np.random.default_rng(seed).choice(p, size=k, replace=False).tolist()))
    (fl, lo, hi, w, y, cap), _ = _parts(M, T)
    assert np.array_equal(y, _split(M, T)[0])


@needs_numba
@SETTINGS
@given(seed=st.integers(0, 10_000), n=st.integers(5, 60), e=st.integers(-400, 400))
def test_property_the_bracket_holds_the_exact_sum_of_squares(seed, n, e):
    M = np.ldexp(_panel(seed, n, 7), e)
    (fl, lo, hi, w, y, cap), _ = _parts(M, (0, 3))
    yy = square_sum(y)
    assert Q(lo) <= yy <= Q(hi)


@needs_numba
@SETTINGS
@given(seed=st.integers(0, 10_000), k=st.integers(1, 3), e=st.integers(-30, 30))
@example(seed=0, k=2, e=30)
@example(seed=0, k=2, e=-30)
def test_property_the_floor_never_exceeds_the_exact_certificate_at_its_point(seed, k, e):
    """At the kernel's own point, its floor is at most the exact hull certificate
    over the box: the bound holds whatever point the solver returns."""
    M = np.ldexp(_panel(seed, 24, 7), e)
    T = tuple(range(k))
    (fl, lo, hi, w, y, cap), lam = _parts(M, T)
    if lam is None or not math.isfinite(fl):
        return
    X = _split(M, T)[1]
    assert np.all(w >= 0) and np.all(w <= cap)
    assert Q(fl) <= hull_bound(ExactData(y, X), w, 0.0, cap, X.shape[1])


@needs_numba
@SETTINGS
@given(seed=st.integers(0, 10_000), k=st.integers(1, 3))
def test_property_the_point_is_as_good_as_scipys(seed, k):
    """The active set ends at the NNLS optimum: its objective is scipy's to rounding."""
    M = _panel(seed, 30, 8)
    T = tuple(range(k))
    (fl, lo, hi, w, y, cap), lam = _parts(M, T)
    X = _split(M, T)[1]
    ours = float(np.sum((X @ w - y) ** 2))
    theirs = float(nnls(X, y)[1] ** 2)
    assert ours <= theirs * (1 + 1e-9) + 1e-12 * float(y @ y)


@needs_numba
def test_the_floors_are_rss_floors_to_rounding_on_every_geo_group():
    M = geo_panel(5)
    lam = gram_floor(M)
    groups = list(combinations(range(M.shape[1]), 2))
    fl, lo, hi = fastfloor.floors(np.ascontiguousarray(M), np.ascontiguousarray(M.T @ M),
                                  np.array(groups, np.int64), float(lam))
    for f, T in zip(fl, groups):
        old = float(rss_floor(M, T, lam))
        assert abs(max(f, 0.0) - old) <= 1e-9 * max(1.0, abs(old))


@needs_numba
def test_the_order_of_the_groups_does_not_change_a_floor():
    """The warm start changes where the solve begins, not where it ends."""
    M = geo_panel(5)
    lam = float(gram_floor(M))
    groups = np.array(list(combinations(range(M.shape[1]), 3)), np.int64)
    args = (np.ascontiguousarray(M), np.ascontiguousarray(M.T @ M))
    a = fastfloor.floors(*args, groups, lam)[0]
    b = fastfloor.floors(*args, groups[::-1].copy(), lam)[0][::-1]
    assert np.allclose(a, b, rtol=1e-9, atol=0)


# ---------------------------------------------------------------- edges


@needs_numba
@pytest.mark.parametrize("bad", [2.0 ** -520, 2.0 ** 520])
def test_a_series_entry_off_the_brackets_range_is_left_to_the_exact_path(bad):
    """The range is the series', as yy_bracket's is: one treated geo carries the entry."""
    M = geo_panel(5).copy()
    M[2, 0] = bad
    (fl, lo, hi, w, y, cap), _ = _parts(M, (0,))
    assert y[2] == bad
    assert math.isnan(fl) and math.isnan(lo) and math.isnan(hi)


@needs_numba
def test_a_tiny_entry_another_column_absorbs_keeps_the_kernel():
    M = geo_panel(5).copy()
    M[2, 0] = 2.0 ** -520
    (fl, lo, hi, w, y, cap), _ = _parts(M, (0, 1))
    assert y[2] == M[2, 1] and math.isfinite(fl)


@needs_numba
def test_eight_treated_geos_are_left_to_the_exact_path():
    """numpy sums eight or more columns pairwise, not left to right."""
    M = _panel(1, 30, 12)
    (fl, lo, hi, w, y, cap), _ = _parts(M, tuple(range(8)))
    assert math.isnan(fl)


@needs_numba
def test_a_cap_off_the_range_is_left_to_the_exact_path():
    M = geo_panel(5)
    (fl, lo, hi, w, y, cap) = fastfloor.floor_parts(np.ascontiguousarray(M), np.ascontiguousarray(M.T @ M),
                                                     np.array([0, 1], np.int64), 2.0 ** -1060)
    assert math.isnan(fl) and math.isfinite(hi)


@needs_numba
def test_a_zero_series_has_a_zero_cap_and_a_floor_at_or_below_zero():
    M = geo_panel(5).copy()
    M[:, 0] = 0.0; M[:, 1] = 0.0
    (fl, lo, hi, w, y, cap), _ = _parts(M, (0, 1))
    assert lo == hi == 0.0 and cap == 0.0 and np.all(w == 0) and fl <= 0


# ---------------------------------------------------------------- the search


def _rows(M, groups, monkeypatch=None, fast=True):
    if monkeypatch is not None:
        monkeypatch.setattr(fastfloor, "AVAILABLE", fast)
    return treatment._floor_chunk(M, gram_floor(M), groups)


@needs_numba
def test_the_rows_with_the_kernel_are_the_rows_without_it(monkeypatch):
    M = geo_panel(5)
    groups = list(combinations(range(M.shape[1]), 2))
    fast = _rows(M, groups, monkeypatch, True)
    slow = _rows(M, groups, monkeypatch, False)
    for (fa, ya, Ta), (fs, ys, Ts) in zip(fast, slow):
        assert Ta == Ts
        assert abs(float(fa) - float(fs)) <= 1e-9 * max(1.0, abs(float(fs)))
        assert isinstance(ya, tuple) and isinstance(ys, tuple)
        assert ya[0] <= float(square_sum(_split(M, Ta)[0])) <= ya[1]


@needs_numba
@pytest.mark.parametrize("k", [2, 3])
def test_the_search_with_the_kernel_certifies_the_same_design(monkeypatch, k):
    M = geo_panel(5)
    monkeypatch.setattr(fastfloor, "AVAILABLE", False)
    want = certify_treatment(M, k, n_test=N_TEST, restarts=3)
    monkeypatch.setattr(fastfloor, "AVAILABLE", True)
    got = certify_treatment(M, k, n_test=N_TEST, restarts=3)
    assert (got.treatment, got.key, got.rss, got.certified) == (want.treatment, want.key, want.rss, want.certified)


@needs_numba
def test_a_group_the_kernel_cannot_do_takes_the_exact_path(monkeypatch):
    """An entry off the bracket's range: that group's row is rss_floor's, with the
    exact y'y, and the other groups keep the kernel's."""
    M = geo_panel(5).copy()
    M[2, 0] = 2.0 ** -520
    calls = []
    real = treatment.rss_floor
    monkeypatch.setattr(treatment, "rss_floor", lambda *a, **k: calls.append(tuple(a[1])) or real(*a, **k))
    groups = [(j,) for j in range(M.shape[1])]
    rows = treatment._floor_chunk(M, gram_floor(M), groups)
    with_zero = [T for T in groups if 0 in T]
    assert sorted(calls) == sorted(with_zero)
    for floor, yy, T in rows:
        if 0 in T:
            assert isinstance(yy, Q) and yy == square_sum(_split(M, T)[0])


def test_without_numba_the_floors_are_rss_floors(monkeypatch):
    """The fallback is the whole of the old path."""
    monkeypatch.setattr(fastfloor, "AVAILABLE", False)
    M = geo_panel(5)
    lam = gram_floor(M)
    rows = treatment._floor_chunk(M, lam, [(0, 1), (2, 3)])
    for floor, yy, T in rows:
        assert floor == rss_floor(M, T, lam, yy=yy[1] if isinstance(yy, tuple) else yy)


@needs_numba
def test_no_gram_floor_takes_the_old_path():
    """A singular panel has no Gram floor: every floor is 0, as before."""
    M = geo_panel(5).copy()
    M[:, 1] = M[:, 0]
    rows = treatment._floor_chunk(M, gram_floor(M), [(0, 2), (3, 4)])
    assert all(floor == 0 for floor, _, _ in rows)


# ---------------------------------------------------------------- failure


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_a_non_finite_series_is_refused_with_or_without_the_kernel(monkeypatch, bad):
    for fast in (False, True) if fastfloor.AVAILABLE else (False,):
        monkeypatch.setattr(fastfloor, "AVAILABLE", fast)
        M = geo_panel(5).copy()
        M[3, 2] = bad
        with pytest.raises(MlsynthDataError, match="finite"):
            treatment._floor_chunk(M, gram_floor(geo_panel(5)), [(1, 2)])
