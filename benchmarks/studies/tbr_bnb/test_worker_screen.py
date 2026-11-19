"""The treatment floors screened where they are computed.

Of the 1,581,580 four-geo groups of an 80-geo panel the floors leave a handful, yet
every floor came back to the parent as a Python row, was sorted and was screened one
at a time. The floors take the incumbent's bar as it stands when they are computed
(the start's design), and a worker drops each row whose two floors, on y and on -y,
the float screen shows are at or above the group's ceiling (treatment._drop_side),
counting it as screened out. The incumbent only improves, so its bar only falls: a
row that cannot compete then cannot compete later, and the search is the search it
was. The tests pin the screened rows to the unscreened ones less the rows the screen
drops, and the search with the worker screen to the search without it.
"""
from fractions import Fraction as Q
from itertools import combinations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthDataError

from mlsynth.utils.tbr_helpers.design.exact import fastfloor, treatment
from .brute import geo_panel, raw_panel
from mlsynth.utils.tbr_helpers.design.exact.treatment import _drop_side, gram_floor

SETTINGS = settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
N_TEST = 8


def _centred(M):
    return M - M.mean(axis=0)


def _scr(M, k):
    """The screen terms of the start's design on the raw panel M, as the search builds them."""
    Mc = _centred(M)
    T0, C0 = treatment._start(M, k, N_TEST, 5)
    d = treatment._design(Mc, M, T0, C0, N_TEST)
    assert d is not None
    sums = treatment.geo_sums(M)
    n = M.shape[0]
    bar = treatment.relative_bar(d.rss, treatment.group_level(sums, T0, n), 0)
    return treatment._screen_terms(bar, treatment.level_doubles(sums, n), k)


def _dropped(row, scr):
    b, ell, margin = scr
    L = [float(ell[list(row[2])].sum())]
    return bool(_drop_side([float(row[0])], L, b, margin)[0] and _drop_side([float(row[1])], L, b, margin)[0])


def _unscreened(M, k, scr):
    Mc = _centred(M)
    groups = list(combinations(range(M.shape[1]), k))
    rows = treatment._floor_rows(Mc, groups, gram_floor(Mc), None, 1)
    return groups, rows, [r for r in rows if not _dropped(r, scr)]


def _same(a, b):
    assert len(a) == len(b)
    for ra, rb in zip(a, b):
        assert ra == rb


# ---------------------------------------------------------------- smoke


def test_smoke_the_screened_rows_and_the_count_cover_every_group():
    M = raw_panel(5, p=7)
    scr = _scr(M, 3)
    groups = list(combinations(range(M.shape[1]), 3))
    rows, dropped = treatment._screened_floor_rows(_centred(M), groups, gram_floor(_centred(M)), None, 1, scr)
    assert dropped > 0 and len(rows) + dropped == len(groups)


# ---------------------------------------------------------------- unit


@pytest.mark.parametrize("k", [2, 3])
def test_the_screened_rows_are_the_rows_the_screen_keeps(k):
    M = raw_panel(5, p=7)
    scr = _scr(M, k)
    groups, rows, keep = _unscreened(M, k, scr)
    got, dropped = treatment._screened_floor_rows(_centred(M), groups, gram_floor(_centred(M)), None, 1, scr)
    _same(got, keep)
    assert dropped == len(rows) - len(keep)


def test_two_processes_screen_as_one_does():
    M = raw_panel(5, p=7)
    Mc = _centred(M)
    scr = _scr(M, 3)
    groups = list(combinations(range(M.shape[1]), 3))
    one = treatment._screened_floor_rows(Mc, groups, gram_floor(Mc), None, 1, scr)
    with treatment._pool(2) as pool:
        two = treatment._screened_floor_rows(Mc, groups, gram_floor(Mc), pool, 2, scr)
    _same(one[0], two[0])
    assert one[1] == two[1]


@pytest.mark.parametrize("k,gap", [(2, Q(0)), (3, Q(0)), (3, Q(1, 100))])
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_the_search_with_the_worker_screen_is_the_search_without_it(monkeypatch, k, gap, n_jobs):
    M = raw_panel(5, p=7)
    with_screen = treatment.certify_treatment(M, k, n_test=N_TEST, restarts=5, pass_restarts=3, gap=gap, n_jobs=n_jobs, blank_periods=0)
    monkeypatch.setattr(treatment, "_streamed_floor_rows", lambda M, p, k, lam, pool, n_jobs, scr, until=None:
                        (treatment._floor_rows(M, list(combinations(range(p), k)), lam, pool, n_jobs), 0))
    without = treatment.certify_treatment(M, k, n_test=N_TEST, restarts=5, pass_restarts=3, gap=gap, n_jobs=n_jobs, blank_periods=0)
    for attr in ("treatment", "controls", "rss", "key", "slope", "certified", "searched", "replaced"):
        assert getattr(with_screen, attr) == getattr(without, attr), attr
    assert [s["treatment"] for s in with_screen.survivors] == [s["treatment"] for s in without.survivors]
    assert with_screen.screened_out == without.screened_out > 0


def test_the_search_hands_the_floors_the_start_bar(monkeypatch):
    """The search passes the start design's screen to the floors, so the workers drop
    rows: a search that passed none would be the same search, only slower."""
    M = raw_panel(5, p=7)
    seen = []
    real = treatment._streamed_floor_rows

    def spy(M, p, k, lam, pool, n_jobs, scr, until=None):
        rows, dropped = real(M, p, k, lam, pool, n_jobs, scr, until=until)
        seen.append((scr, dropped))
        return rows, dropped

    monkeypatch.setattr(treatment, "_streamed_floor_rows", spy)
    treatment.certify_treatment(M, 3, n_test=N_TEST, restarts=5, pass_restarts=3, gap=Q(0), n_jobs=1, blank_periods=0)
    (scr, dropped), = seen
    want = _scr(M, 3)
    assert scr[0] == want[0] and scr[2] == want[2] and np.array_equal(scr[1], want[1]) and dropped > 0


# ---------------------------------------------------------------- edge


def test_no_incumbent_drops_nothing():
    M = raw_panel(5, p=7)
    Mc = _centred(M)
    groups = list(combinations(range(M.shape[1]), 3))
    rows, dropped = treatment._screened_floor_rows(Mc, groups, gram_floor(Mc), None, 1, None)
    assert dropped == 0
    _same(rows, treatment._floor_rows(Mc, groups, gram_floor(Mc), None, 1))


def test_without_the_kernel_the_same_rows_are_dropped(monkeypatch):
    M = raw_panel(5, p=7)
    scr = _scr(M, 3)
    groups, rows, keep = _unscreened(M, 3, scr)
    monkeypatch.setattr(fastfloor, "AVAILABLE", False)
    got, dropped = treatment._screened_floor_rows(_centred(M), groups, gram_floor(_centred(M)), None, 1, scr)
    assert [r[2] for r in got] == [r[2] for r in keep] and dropped == len(rows) - len(keep)


@pytest.mark.parametrize("which", [1, -1])
def test_a_group_the_kernel_leaves_is_kept_for_the_exact_path(monkeypatch, which):
    """A nan floor on either sign (the kernel could not do the group) is never
    dropped: that sign takes rss_floor's path."""
    M = raw_panel(5, p=7)
    Mc = _centred(M)
    scr = _scr(M, 2)
    groups = list(combinations(range(M.shape[1]), 2))
    real = fastfloor.floors

    def some_nan(Mc, Gf, g, lamf, sign=1.0):
        fl, lo, hi = real(Mc, Gf, g, lamf, sign)
        if sign == which:
            fl = fl.copy(); fl[::3] = np.nan
        return fl, lo, hi

    monkeypatch.setattr(fastfloor, "floors", some_nan)
    got, dropped = treatment._screened_floor_rows(Mc, groups, gram_floor(Mc), None, 1, scr)
    kept = {r[2]: r for r in got}
    for i in range(0, len(groups), 3):
        r = kept[groups[i]]
        side = r[0] if which == 1 else r[1]
        assert side == treatment.rss_floor(Mc, groups[i], gram_floor(Mc), sign=which)
    assert len(got) + dropped == len(groups)


def test_a_panel_without_a_gram_floor_keeps_every_row():
    """With no Gram floor (lam None) every floor is 0, below every ceiling."""
    M = raw_panel(5, p=7)
    scr = _scr(M, 2)
    groups = list(combinations(range(M.shape[1]), 2))
    got, dropped = treatment._floor_chunk_screened(_centred(M), None, groups, scr)
    assert dropped == 0 and [r[2] for r in got] == groups


def test_a_row_is_dropped_only_when_both_signs_are():
    """A row one of whose signs can compete is kept: the group still has designs of
    that sign to certify."""
    b, margin = 1.0, 2.0 ** -40
    ell = np.array([1.0, 1.0])
    assert not _dropped((100.0, 1.0, (0, 1)), (b, ell, margin))
    assert not _dropped((1.0, 100.0, (0, 1)), (b, ell, margin))
    assert _dropped((100.0, 100.0, (0, 1)), (b, ell, margin))


def test_no_groups():
    M = raw_panel(5, p=7)
    assert treatment._screened_floor_rows(_centred(M), [], gram_floor(_centred(M)), None, 1, _scr(M, 2)) == ([], 0)


# ---------------------------------------------------------------- failure


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_a_non_finite_series_is_refused_through_the_screened_floors(monkeypatch, bad):
    """With or without the kernel, a group whose series is not finite raises, it is not
    dropped as unable to compete: the screen reads only finite rows."""
    scr = _scr(raw_panel(5, p=7), 2)
    for fast in (False, True) if fastfloor.AVAILABLE else (False,):
        monkeypatch.setattr(fastfloor, "AVAILABLE", fast)
        M = geo_panel(5).copy()
        M[3, 2] = bad
        with pytest.raises(MlsynthDataError, match="finite"):
            treatment._floor_chunk_screened(M, gram_floor(geo_panel(5)), [(1, 2)], scr)


# ---------------------------------------------------------------- property


@SETTINGS
@given(seed=st.integers(0, 500), k=st.integers(1, 3))
def test_property_the_kernel_screen_is_the_python_screen(seed, k):
    """Row for row, the compiled screen keeps what the Python screen keeps."""
    M = raw_panel(seed, p=6)
    Mc = _centred(M)
    scr = _scr(M, k)
    groups = list(combinations(range(M.shape[1]), k))
    got, d = treatment._floor_chunk_screened(Mc, gram_floor(Mc), groups, scr)
    rows = [treatment._floor_row(Mc, gram_floor(Mc), T) for T in groups]
    keep = [r for r in rows if not _dropped(r, scr)]
    assert [r[2] for r in got] == [r[2] for r in keep] and d == len(rows) - len(keep)
