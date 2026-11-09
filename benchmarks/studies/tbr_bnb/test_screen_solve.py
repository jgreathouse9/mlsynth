"""The group screen's last tier, by size (#965).

group_certificate ends with the size-free relaxation solved by Clarabel when no
first-order schedule closes, but only below SCREEN_SOLVE_BELOW candidate controls
(screen_solves); from there the screen ends open after its schedules and the sizes
decide. The verdicts are checked against enumeration on small panels."""
from fractions import Fraction as Q
from itertools import combinations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthConfigError

from . import slopefree
from .brute import panel, positive_optimum
from .certificate import ExactData
from .slopefree import (SCREEN_SOLVE_BELOW, certify_slope_free, dyadic_below, group_certificate,
                        screen_solves)

SETTINGS = settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def _data(seed, p):
    y, X = panel(seed, p=p)
    D = ExactData(y, X)
    return y, X, D.c, D.G


def _best_ratio(c, G):
    """max (c'u)^2 / u'Gu over nonzero binary u with c'u > 0, exactly; 0 when none."""
    best = Q(0)
    p = len(c)
    for r in range(1, p + 1):
        for C in combinations(range(p), r):
            u = np.zeros(p, dtype=object); u[:] = Q(0); u[list(C)] = Q(1)
            cu = c @ u
            if cu > 0:
                best = max(best, cu * cu / (u @ G @ u))
    return best


def _stalled(monkeypatch):
    """The first-order tier gives up at every schedule."""
    monkeypatch.setattr(slopefree, "dual_multipliers", lambda *a, **k: (None, {"evals": 0}))


def _solves(monkeypatch):
    """Count the relaxation solves the screen makes."""
    calls = []
    real = slopefree.solve

    def spy(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(slopefree, "solve", spy)
    return calls


# ---------------------------------------------------------------- smoke


def test_smoke_a_screen_without_its_solve_ends_open_after_its_schedules(monkeypatch):
    _, _, c, G = _data(3, 6)
    _stalled(monkeypatch)
    calls = _solves(monkeypatch)
    res = group_certificate(c, G, _best_ratio(c, G) * Q(12, 10), final_solve=False)
    assert not res.certified and res.tier == "open" and res.rung is None
    assert calls == []


# ---------------------------------------------------------------- unit


@pytest.mark.parametrize("p, solves", [(1, True), (36, True), (SCREEN_SOLVE_BELOW - 1, True),
                                       (SCREEN_SOLVE_BELOW, False), (98, False), (500, False)])
def test_the_screen_solves_below_the_threshold_only(p, solves):
    assert screen_solves(p) is solves


def test_the_threshold_is_ninety_candidate_controls():
    """Measured: at 56 candidate controls the solve closed screens in 25 s or less,
    at 96 and 98 it took 129-204 s and lost to the sizes' own certificates."""
    assert SCREEN_SOLVE_BELOW == 90


def test_by_default_the_screen_asks_screen_solves_for_its_number_of_controls(monkeypatch):
    _, _, c, G = _data(3, 6)
    _stalled(monkeypatch)
    seen = []
    monkeypatch.setattr(slopefree, "screen_solves", lambda p: (seen.append(p), False)[1])
    calls = _solves(monkeypatch)
    res = group_certificate(c, G, _best_ratio(c, G) * Q(12, 10))
    assert seen == [6] and calls == [] and res.tier == "open"


def test_by_default_a_small_group_still_reaches_the_solve(monkeypatch):
    _, _, c, G = _data(3, 6)
    _stalled(monkeypatch)
    calls = _solves(monkeypatch)
    res = group_certificate(c, G, _best_ratio(c, G) * Q(12, 10))
    assert calls == [1] and res.certified and res.tier == "full"


def test_an_explicit_solve_overrides_the_size(monkeypatch):
    _, _, c, G = _data(3, 6)
    _stalled(monkeypatch)
    monkeypatch.setattr(slopefree, "screen_solves", lambda p: False)
    calls = _solves(monkeypatch)
    res = group_certificate(c, G, _best_ratio(c, G) * Q(12, 10), final_solve=True)
    assert calls == [1] and res.certified and res.tier == "full"


@pytest.mark.parametrize("seed", range(4))
def test_a_schedule_that_closes_needs_no_solve(seed):
    """Without its solve the screen still certifies whatever its schedules close."""
    _, _, c, G = _data(seed, 6)
    lam = _best_ratio(c, G) * 2
    with_solve = group_certificate(c, G, lam, final_solve=True)
    without = group_certificate(c, G, lam, final_solve=False)
    if with_solve.tier == "dual":
        assert without.certified and without.tier == "dual" and without.rung == with_solve.rung


@SETTINGS
@given(seed=st.integers(0, 10_000), p=st.integers(1, 6),
       f=st.sampled_from([Q(1, 2), Q(99, 100), Q(101, 100), Q(11, 10), Q(2)]))
def test_property_a_screen_without_its_solve_certifies_only_what_holds(seed, p, f):
    _, _, c, G = _data(seed % 80, p)
    best = _best_ratio(c, G)
    lam = best * f if best > 0 else f
    res = group_certificate(c, G, lam, final_solve=False)
    if res.certified:
        assert res.tier == "dual" and res.bound <= 0 and best <= dyadic_below(lam)
    else:
        assert res.tier == "open"


@SETTINGS
@given(seed=st.integers(0, 10_000), f=st.sampled_from([Q(99, 100), Q(101, 100), Q(13, 10)]))
def test_property_the_search_reaches_the_same_verdict_without_the_screens_solve(seed, f):
    """The screen proves only that no size needs searching; without its solve the
    sizes decide, and the search certifies the same design at the same ratio."""
    y, X = panel(seed % 80, p=6)
    rss, _ = positive_optimum(y, X)
    ceiling = Q(rss) * f
    on = certify_slope_free(y, X, restarts=3, ceiling=ceiling)
    orig = slopefree.SCREEN_SOLVE_BELOW
    try:
        slopefree.SCREEN_SOLVE_BELOW = 0
        off = certify_slope_free(y, X, restarts=3, ceiling=ceiling)
    finally:
        slopefree.SCREEN_SOLVE_BELOW = orig
    assert on.certified and off.certified
    assert (on.ratio, on.controls, on.ceiling_held) == (off.ratio, off.controls, off.ceiling_held)


# ---------------------------------------------------------------- edges


def test_one_candidate_control_without_the_solve(monkeypatch):
    _, _, c, G = _data(4, 1)
    best = _best_ratio(c, G)
    res = group_certificate(c, G, best * 2 if best > 0 else Q(1), final_solve=False)
    assert res.tier in ("dual", "open")
    if best > 0:
        assert not group_certificate(c, G, best * Q(1, 2), final_solve=False).certified


# ---------------------------------------------------------------- failure


@pytest.mark.parametrize("bad", ["yes", 1, 0, 1.0])
def test_a_solve_that_is_not_a_boolean_is_refused(bad):
    _, _, c, G = _data(3, 4)
    with pytest.raises(MlsynthConfigError, match="final_solve"):
        group_certificate(c, G, Q(1), final_solve=bad)
