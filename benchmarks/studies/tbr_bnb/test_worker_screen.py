"""The treatment floors screened where they are computed.

Of the 1,581,580 four-geo groups of an 80-geo panel the floors leave a handful, yet
every floor came back to the parent as a Python row, was sorted and was screened one
at a time. The floors now take the incumbent's class as it stands when they are
computed (the start's design), and a worker drops each row the float screen shows
cannot compete, counting it as screened out. The incumbent only improves, so its
ceiling only falls (ceiling's docstring): a row that cannot compete then cannot
compete later, and the search is the search it was. The tests pin the vectorized
screen to compete_bracket, the screened rows to the unscreened ones less the rows
compete_bracket refuses, and the search with the worker screen to the search
without it.
"""
from fractions import Fraction as Q
from itertools import combinations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthDataError

from . import fastfloor, treatment
from .brute import geo_panel
from .treatment import compete_bracket, gram_floor

SETTINGS = settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
N_TEST = 8


def _class(M, k):
    """The class (r_star, rss_star) of the start's design on M, as the search builds it."""
    T0, Cs = treatment._start(M, k, N_TEST, 5)
    C0, key0, rss0, rest0 = treatment._design(M, T0, Cs, N_TEST)
    assert key0 is not None
    return Q(key0[4]), rss0


def _unscreened(M, k, cls):
    groups = list(combinations(range(M.shape[1]), k))
    rows = treatment._floor_rows(M, groups, gram_floor(M), None, 1)
    keep = [r for r in rows if not (isinstance(r[1], tuple) and compete_bracket(r[0], r[1][0], r[1][1], *cls) is False)]
    return groups, rows, keep


def _same(a, b):
    assert len(a) == len(b)
    for (fa, ya, Ta), (fb, yb, Tb) in zip(a, b):
        assert Ta == Tb and fa == fb and ya == yb


# ---------------------------------------------------------------- smoke


def test_smoke_the_screened_rows_and_the_count_cover_every_group():
    M = geo_panel(5)
    cls = _class(M, 3)
    groups = list(combinations(range(M.shape[1]), 3))
    rows, dropped = treatment._screened_floor_rows(M, groups, gram_floor(M), None, 1, cls)
    assert dropped > 0 and len(rows) + dropped == len(groups)


# ---------------------------------------------------------------- unit


@pytest.mark.parametrize("k", [2, 3])
def test_the_screened_rows_are_the_rows_compete_bracket_keeps(k):
    M = geo_panel(5)
    cls = _class(M, k)
    groups, rows, keep = _unscreened(M, k, cls)
    got, dropped = treatment._screened_floor_rows(M, groups, gram_floor(M), None, 1, cls)
    _same(got, keep)
    assert dropped == len(rows) - len(keep)


def test_two_processes_screen_as_one_does():
    M = geo_panel(5)
    cls = _class(M, 3)
    groups = list(combinations(range(M.shape[1]), 3))
    one = treatment._screened_floor_rows(M, groups, gram_floor(M), None, 1, cls)
    with treatment._pool(2) as pool:
        two = treatment._screened_floor_rows(M, groups, gram_floor(M), pool, 2, cls)
    _same(one[0], two[0])
    assert one[1] == two[1]


@pytest.mark.parametrize("k,gap", [(2, Q(0)), (3, Q(0)), (3, Q(1, 100))])
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_the_search_with_the_worker_screen_is_the_search_without_it(monkeypatch, k, gap, n_jobs):
    M = geo_panel(5)
    with_screen = treatment.certify_treatment(M, k, n_test=N_TEST, restarts=5, pass_restarts=3, gap=gap, n_jobs=n_jobs)
    monkeypatch.setattr(treatment, "_screened_floor_rows",
                        lambda M, groups, lam, pool, n_jobs, cls: (treatment._floor_rows(M, groups, lam, pool, n_jobs), 0))
    without = treatment.certify_treatment(M, k, n_test=N_TEST, restarts=5, pass_restarts=3, gap=gap, n_jobs=n_jobs)
    for attr in ("treatment", "controls", "rss", "key", "passing", "certified", "screened_out", "searched", "replaced"):
        assert getattr(with_screen, attr) == getattr(without, attr), attr
    assert [s["treatment"] for s in with_screen.survivors] == [s["treatment"] for s in without.survivors]
    assert with_screen.screened_out > 0


# ---------------------------------------------------------------- edge


def test_no_incumbent_drops_nothing():
    M = geo_panel(5)
    groups = list(combinations(range(M.shape[1]), 3))
    rows, dropped = treatment._screened_floor_rows(M, groups, gram_floor(M), None, 1, (None, None))
    assert dropped == 0
    _same(rows, treatment._floor_rows(M, groups, gram_floor(M), None, 1))


def test_without_the_kernel_the_same_rows_are_dropped(monkeypatch):
    M = geo_panel(5)
    cls = _class(M, 3)
    groups, rows, keep = _unscreened(M, 3, cls)
    monkeypatch.setattr(fastfloor, "AVAILABLE", False)
    got, dropped = treatment._screened_floor_rows(M, groups, gram_floor(M), None, 1, cls)
    assert [r[2] for r in got] == [r[2] for r in keep] and dropped == len(rows) - len(keep)


def test_a_group_the_kernel_leaves_is_kept_for_the_exact_path(monkeypatch):
    """A nan floor (the kernel could not do the group) is never dropped by the mask:
    the row takes rss_floor's path and the parent screens it."""
    M = geo_panel(5)
    cls = _class(M, 2)
    groups = list(combinations(range(M.shape[1]), 2))
    real = fastfloor.floors

    def some_nan(Mc, Gf, g, lamf):
        fl, lo, hi = real(Mc, Gf, g, lamf)
        fl = fl.copy(); fl[::3] = np.nan
        return fl, lo, hi

    monkeypatch.setattr(fastfloor, "floors", some_nan)
    got, dropped = treatment._screened_floor_rows(M, groups, gram_floor(M), None, 1, cls)
    kept = {r[2] for r in got}
    assert all(groups[i] in kept for i in range(0, len(groups), 3))
    assert len(got) + dropped == len(groups)


def test_no_groups():
    M = geo_panel(5)
    assert treatment._screened_floor_rows(M, [], gram_floor(M), None, 1, _class(M, 2)) == ([], 0)


def test_the_mask_keeps_a_floor_at_or_below_zero():
    """A floor at or below zero is a floor of 0: below every positive ceiling."""
    m = treatment._screen_mask(np.array([-5.0, 0.0]), np.array([10.0, 10.0]), np.array([10.0, 10.0]),
                               Q(9, 10), Q(1))
    assert not m.any()


# ---------------------------------------------------------------- failure


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_a_non_finite_series_is_refused_through_the_screened_floors(monkeypatch, bad):
    """With or without the kernel, a group whose series is not finite raises, it is not
    dropped as unable to compete: the screen reads only finite rows."""
    cls = _class(geo_panel(5), 2)
    for fast in (False, True) if fastfloor.AVAILABLE else (False,):
        monkeypatch.setattr(fastfloor, "AVAILABLE", fast)
        M = geo_panel(5).copy()
        M[3, 2] = bad
        with pytest.raises(MlsynthDataError, match="finite"):
            treatment._floor_chunk_screened(M, gram_floor(geo_panel(5)), [(1, 2)], cls)


# ---------------------------------------------------------------- property


@SETTINGS
@given(fl=st.lists(st.floats(-1e6, 1e6, allow_nan=False), min_size=1, max_size=30),
       lo_e=st.lists(st.floats(-3, 6, allow_nan=False), min_size=30, max_size=30),
       widen=st.floats(0, 1e-6, allow_nan=False),
       r=st.integers(-99, 99), rss_e=st.floats(-3, 6, allow_nan=False))
def test_property_the_mask_is_compete_bracket_refusing(fl, lo_e, widen, r, rss_e):
    """Row by row, the vectorized mask is True exactly where compete_bracket returns
    False for the row as _floor_chunk builds it (floor 0 when the kernel's is not
    positive)."""
    n = len(fl)
    f = np.array(fl)
    lo = 10.0 ** np.array(lo_e[:n])
    hi = lo * (1 + widen)
    r_star, rss_star = Q(r, 100), Q(10.0 ** rss_e)
    m = treatment._screen_mask(f, lo, hi, r_star, rss_star)
    for i in range(n):
        floor = f[i] if f[i] > 0 else Q(0)
        assert bool(m[i]) == (compete_bracket(floor, float(lo[i]), float(hi[i]), r_star, rss_star) is False)
