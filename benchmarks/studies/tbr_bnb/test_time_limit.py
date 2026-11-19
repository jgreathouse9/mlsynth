"""A time limit on the certified search (#965, v1).

certify_treatment stops at ``time_limit`` seconds (TIME_LIMIT, two minutes, unless
told otherwise; None for no limit) and reports the best design found with
``certified`` False, ``timed_out`` True and the proven gap: no design's key exceeds
1 + ``proven_gap`` times the reported design's. The gap comes from the floors of the
groups the search did not resolve, the groups it never reached and those whose
certificate did not close: a group with level L and floor f has no design whose key
exceeds sqrt(L^2 RSS* / (level*^2 f)) times the incumbent's. A group whose floors were
never computed bounds nothing, and the gap is None. Every stage that can stop early
is stopped here at a chosen point by a clock the test throws (deadline.clock), and
the gap is checked against enumeration."""
import inspect
from fractions import Fraction as Q

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthDataError

from . import deadline
from . import slopefree as sf
from . import treatment
from .brute import panel, raw_panel, relative_optimum
from .treatment import TIME_LIMIT, _GroupOutcome, certify_treatment, proven_gap

N_TEST = 4
SETTINGS = settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow])


class Switch:
    """A clock at 0 until thrown, then past every deadline."""

    def __init__(self):
        self.on = False

    def __call__(self):
        return 1e18 if self.on else 0.0


class Ticks:
    """A clock that moves one second each time it is read."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 1.0
        return self.t


def _switch(monkeypatch):
    s = Switch()
    monkeypatch.setattr(deadline, "clock", s)
    return s


def _throw_after(monkeypatch, s, name):
    real = getattr(treatment, name)

    def wrapped(*a, **k):
        r = real(*a, **k)
        s.on = True
        return r

    monkeypatch.setattr(treatment, name, wrapped)


def _run(M, **kw):
    kw = dict(n_test=N_TEST, restarts=3, blank_periods=0, time_limit=60.0) | kw
    return certify_treatment(M, 2, **kw)


def _within(out, M):
    """The enumerated optimum's key is at most 1 + proven_gap times the reported key."""
    best, _, _ = relative_optimum(M, 2, N_TEST)
    assert out.proven_gap is not None
    assert best <= (1 + float(out.proven_gap)) * out.key * (1 + 1e-9) + 1e-12
    return best


# ---------------------------------------------------------------- smoke


def test_smoke_a_search_out_of_time_reports_a_design_and_a_gap(monkeypatch):
    M = raw_panel(5, n=30, p=6)
    s = _switch(monkeypatch)
    _throw_after(monkeypatch, s, "_streamed_floor_rows")
    out = _run(M)
    assert out.timed_out and not out.certified
    assert out.treatment is not None and np.isfinite(out.key)
    assert out.unresolved > 0
    _within(out, M)


# ---------------------------------------------------------------- unit


def test_the_default_limit_is_two_minutes():
    assert TIME_LIMIT == 120.0
    assert inspect.signature(certify_treatment).parameters["time_limit"].default == TIME_LIMIT


def test_a_search_that_finishes_reports_its_requested_gap():
    M = raw_panel(5, n=30, p=6)
    for g in (0, Q(1, 100)):
        out = _run(M, gap=g)
        assert out.certified and not out.timed_out
        assert out.proven_gap == Q(g) and out.unresolved == 0


def test_no_limit_and_a_far_limit_are_the_same_search():
    M = raw_panel(6, n=30, p=6)
    a = _run(M, time_limit=None)
    b = _run(M, time_limit=1e9)
    assert (a.treatment, a.controls, a.key, a.certified, a.proven_gap) == \
           (b.treatment, b.controls, b.key, b.certified, b.proven_gap)
    assert not a.timed_out and not b.timed_out


def test_the_gap_is_the_largest_floor_bound():
    """sqrt(L^2 RSS* / (level*^2 f)) - 1 at the smaller floor, rounded up to a
    rational (root_above), the largest over the groups, never below the requested gap."""
    one = proven_gap([(Q(1), Q(4), Q(2))], Q(1), Q(1), Q(0))
    assert Q(1) <= one <= Q(1) + Q(1, 10 ** 9)
    two = proven_gap([(Q(1), Q(4), Q(2)), (Q(9), Q(9), Q(9))], Q(1), Q(1), Q(0))
    assert Q(2) <= two <= Q(2) + Q(1, 10 ** 9)
    assert proven_gap([(Q(100), Q(100), Q(1))], Q(1), Q(1), Q(1, 10)) == Q(1, 10)
    assert proven_gap([], Q(1), Q(1), Q(1, 20)) == Q(1, 20)
    assert proven_gap([(Q(0), Q(4), Q(2)), (Q(1), Q(1), Q(1))], Q(1), Q(1), Q(0)) is None
    scaled = proven_gap([(Q(2), Q(2), Q(2))], Q(3), Q(1), Q(0))      # q = 4 * 3 / 2 = 6
    assert Q(6) < (scaled + 1) ** 2 <= Q(6) + Q(1, 10 ** 6)


def test_the_floors_stop_at_the_limit(monkeypatch):
    """Out of time before any floor is computed, no group is bounded: the start's
    design is reported and the gap is unknown."""
    M = raw_panel(5, n=30, p=6)
    s = _switch(monkeypatch)
    _throw_after(monkeypatch, s, "_start")
    out = _run(M)
    assert out.timed_out and not out.certified
    assert out.proven_gap is None
    assert out.unresolved == 15 and out.survivors == [] and out.searched == 0
    assert out.treatment is not None


def test_the_pass_stops_at_the_limit(monkeypatch):
    M = raw_panel(5, n=30, p=6)
    s = _switch(monkeypatch)
    _throw_after(monkeypatch, s, "_streamed_floor_rows")
    out = _run(M)
    assert out.searched == 0 and out.survivors == []
    assert out.unresolved == out.groups - out.screened_out
    _within(out, M)


def test_the_window_stops_at_the_limit(monkeypatch):
    """Out of time after the first group: it is the only survivor, and every group
    after it that could compete is counted as unresolved."""
    M = raw_panel(5, n=30, p=6)
    whole = _run(M)
    assert len(whole.survivors) > 1
    s = _switch(monkeypatch)
    _throw_after(monkeypatch, s, "_optimum_timed")
    out = _run(M)
    assert len(out.survivors) == 1 and out.timed_out and not out.certified
    assert out.unresolved >= 1
    _within(out, M)


def _first_certified(M, **kw):
    return tuple(_run(M, time_limit=None, **kw).survivors[0]["treatment"])


def _fail(monkeypatch, T):
    """Group T's certification left open on its turn, in every run."""
    real = treatment._optimum_timed

    def open_at_T(*a):
        if tuple(a[2]) == tuple(T):
            return _GroupOutcome(None, False, (1, -1)), 0.0
        return real(*a)

    monkeypatch.setattr(treatment, "_optimum_timed", open_at_T)


def test_the_retries_stop_at_the_limit(monkeypatch):
    """A group left open on its turn is certified again after the window, unless the
    time is up: then it stays open and bounds the gap."""
    M = raw_panel(5, n=30, p=6)
    _fail(monkeypatch, _first_certified(M, pass_restarts=0, start=[0, 5]))
    control = _run(M, pass_restarts=0, start=[0, 5])
    assert control.retried == 1 and control.certified and not control.timed_out
    s = _switch(monkeypatch)
    _throw_after(monkeypatch, s, "_window")
    out = _run(M, pass_restarts=0, start=[0, 5])
    assert out.retried == 0 and not out.certified and out.timed_out
    assert out.unresolved == 1
    _within(out, M)


def test_an_open_group_the_incumbent_rules_out_is_resolved(monkeypatch):
    """A group left open on its turn and not retried for lack of time is resolved
    when the final incumbent's ceiling is at or below both its floors. The group's
    floor is set between its ceiling on its turn and the final one, so that it is
    admitted and then ruled out, and the queue keeps it first, where it was."""
    M = raw_panel(5, n=30, p=6)
    kw = dict(pass_restarts=0, start=[0, 5])
    X = _first_certified(M, **kw)
    base = _run(M, time_limit=None, **kw)
    lev = treatment.group_level(treatment.geo_sums(M), X, M.shape[0])
    c_then = base.survivors[0]["ceiling"]
    c_final = treatment.relative_ceiling(lev, treatment.relative_bar(base.rss, base.level, Q(0)))
    assert c_final < c_then
    f = float((c_then + c_final) / 2)
    assert c_final <= Q(f) < c_then
    real = treatment._streamed_floor_rows

    def raised(*a, **k):
        rows, dropped = real(*a, **k)
        return [(f, f, T) if tuple(T) == X else (fp, fm, T) for fp, fm, T in rows], dropped

    order = treatment._bound_order
    monkeypatch.setattr(treatment, "_streamed_floor_rows", raised)
    monkeypatch.setattr(treatment, "_bound_order", lambda r, ell: -1.0 if tuple(r[2]) == X else order(r, ell))
    _fail(monkeypatch, X)
    s = _switch(monkeypatch)
    _throw_after(monkeypatch, s, "_window")
    out = _run(M, **kw)
    assert [r["treatment"] for r in out.survivors][0] == list(X)
    assert out.retried == 0 and out.unresolved == 0 and out.certified and not out.timed_out
    assert out.proven_gap == 0


def test_every_certification_is_given_the_deadline(monkeypatch):
    """The deadline reaches certify_slope_free from the window and from the retries."""
    M = raw_panel(5, n=30, p=6)
    kw = dict(pass_restarts=0, start=[0, 5])
    _fail(monkeypatch, _first_certified(M, **kw))
    seen, real = [], sf.certify_slope_free

    def spy(*a, **k):
        seen.append(k.get("deadline"))
        return real(*a, **k)

    monkeypatch.setattr(sf, "certify_slope_free", spy)
    _switch(monkeypatch)
    out = _run(M, **kw)
    assert out.retried == 1 and len(seen) >= 2
    assert set(seen) == {60.0}


def test_the_floors_read_nothing_past_the_deadline():
    """In this process and in each worker, a range past its deadline reads no block:
    no row, nothing dropped, so every group is unread."""
    M = raw_panel(5, n=30, p=6)
    Mc = M - M.mean(axis=0)
    lam = treatment.gram_floor(Mc)
    assert treatment._streamed_floor_rows(Mc, 6, 2, lam, None, 1, None, until=-1.0) == ([], 0)
    with treatment._pool(2) as pool:
        assert treatment._streamed_floor_rows(Mc, 6, 2, lam, pool, 2, None, until=-1.0) == ([], 0)
        rows, dropped = treatment._streamed_floor_rows(Mc, 6, 2, lam, pool, 2, None, until=None)
    assert len(rows) + dropped == 15


def test_an_open_group_bounds_the_gap_without_a_limit(monkeypatch):
    """A group whose certificate stays open is unresolved whether or not time ran
    out: the search is not certified, not timed out, and the gap counts every group
    left open, here all of them."""
    M = raw_panel(5, n=30, p=6)

    def never(*a, **k):
        return _GroupOutcome(None, False, (1, -1))

    _fail(monkeypatch, _first_certified(M, pass_restarts=0, start=[0, 5]))
    monkeypatch.setattr(treatment, "_optimum", never)
    out = _run(M, pass_restarts=0, start=[0, 5], time_limit=None)
    assert out.survivors and not out.certified and not out.timed_out
    assert out.unresolved == len(out.survivors)
    _within(out, M)


def test_the_branch_stops_at_the_deadline():
    """A size's branch and bound entered after the deadline leaves its root open, so
    the certificate is not certified; with no deadline or a far one it closes. The
    size of every control is one design, evaluated at its root."""
    y, X = panel(0, p=8)
    free = sf.certify_slope_free(y, X, restarts=3, group_screen=False)
    assert free.certified and free.sizes_after_hull
    late = sf.certify_slope_free(y, X, restarts=3, group_screen=False, deadline=-1.0)
    assert not late.certified
    assert sorted(late.open) == sorted((m, ()) for m in late.sizes_after_hull if m < X.shape[1])
    assert late.open
    far = sf.certify_slope_free(y, X, restarts=3, group_screen=False, deadline=deadline.clock() + 1e9)
    assert far.certified and far.controls == free.controls


def test_a_group_out_of_time_is_open(monkeypatch):
    """The deadline reaches a group's certification: past it, every sign left to
    certify leaves the group open."""
    M = raw_panel(5, n=30, p=6)
    s = _switch(monkeypatch)
    Mc = M - M.mean(axis=0)
    row = next(r for r in treatment._floor_rows(Mc, [(0, 1)], treatment.gram_floor(Mc), None, 1))
    s.on = True
    res = treatment._optimum(Mc, M, (0, 1), N_TEST, 3, None, row[:2], {}, until=0.5)
    assert not res.certified


@given(seed=st.integers(0, 10_000), limit=st.integers(1, 600))
@SETTINGS
def test_property_the_gap_holds_wherever_the_search_stops(seed, limit):
    """On a clock that moves a second a read, a limit of ``limit`` seconds stops the
    search at a different point for each limit; wherever it stops, a finished search
    is the enumerated optimum and a stopped one holds its gap against enumeration."""
    M = raw_panel(seed, n=30, p=6)
    before = deadline.clock
    deadline.clock = Ticks()
    try:
        out = _run(M, time_limit=float(limit))
    finally:
        deadline.clock = before
    assert out.certified != out.timed_out
    assert out.treatment is not None
    if out.certified:
        best, _, _ = relative_optimum(M, 2, N_TEST)
        assert out.key == pytest.approx(best, rel=1e-9) and out.proven_gap == 0
    elif out.proven_gap is not None:
        _within(out, M)
    else:
        assert out.survivors == []


# ---------------------------------------------------------------- edge


def test_a_pool_out_of_time_holds_its_gap(monkeypatch):
    M = raw_panel(7, n=30, p=6)
    s = _switch(monkeypatch)
    _throw_after(monkeypatch, s, "_streamed_floor_rows")
    out = _run(M, n_jobs=2)
    assert out.timed_out and not out.certified and out.survivors == []
    _within(out, M)


def test_one_control_out_of_time():
    """k = p - 1: one group per control, each a single design."""
    M = raw_panel(3, n=30, p=3)
    before = deadline.clock
    deadline.clock = Ticks()
    try:
        out = certify_treatment(M, 2, n_test=N_TEST, restarts=3, blank_periods=0, time_limit=3.0)
    finally:
        deadline.clock = before
    assert out.treatment is not None and (out.certified or out.timed_out)


# ---------------------------------------------------------------- failure


@pytest.mark.parametrize("bad", [0, 0.0, -1.0, float("nan"), float("inf"), True, "60", Q(1)])
def test_a_time_limit_that_is_not_a_positive_number_of_seconds_raises(bad):
    with pytest.raises(MlsynthDataError, match="time_limit"):
        _run(raw_panel(5, n=30, p=6), time_limit=bad)
