"""The group screen's schedule ladder (#965).

The screen's first-order multipliers come from L-BFGS-B on a smoothed certificate
whose smoothing shrinks tenfold a stage. At 150 evaluations a stage they stop far
from the relaxation's optimum on the 100-geo groups: on (4, 33, 35, 58) the best
certificate after 1,143 evaluations was 3,725 y'y, where Clarabel's was below zero.
Given 1,000 or 3,000 evaluations a stage the same method closed 17 of the 19 groups
that search certified, in 4 to 25 s, against Clarabel's 78 to 140 s, and the two
it left open Clarabel left open too. So the screen tries the schedules in turn, the
size certificates' own first, and hands what none closes to Clarabel. The longer
schedules run only from SCREEN_LADDER_FROM candidate controls: below it Clarabel's
solve costs less than a long schedule that does not close."""
from fractions import Fraction as Q
from itertools import combinations

import numpy as np
import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthConfigError

from . import slopefree
from .brute import geo_panel, panel
from .certificate import ExactData
from .search import incumbent
from .slopefree import (DUAL_BUDGET, SCREEN_LADDER_FROM, SCREEN_SCHEDULES, dyadic_below, group_certificate,
                        screen_schedules)
from .treatment import _split

SETTINGS = settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def _ratio(D, C):
    p = len(D.c)
    u = np.zeros(p, dtype=object); u[:] = Q(0); u[sorted(C)] = Q(1)
    cu = D.c @ u
    return cu * cu / (u @ D.G @ u) if cu > 0 else None


def _best_ratio(c, G):
    """max (c'u)^2 / u'Gu over nonzero binary u with c'u > 0, exactly; 0 when none."""
    best, p = Q(0), len(c)
    for r in range(1, p + 1):
        for C in combinations(range(p), r):
            u = np.zeros(p, dtype=object); u[:] = Q(0); u[list(C)] = Q(1)
            cu = c @ u
            if cu > 0:
                best = max(best, cu * cu / (u @ G @ u))
    return best


def _group(seed, T, f, n=20, p=30):
    """A treatment group of a simulated panel, at the incumbent's ratio times f. When
    every c_j < 0 no design has a positive slope, the incumbent has no ratio, and the
    group is tested at f: every design is below any positive ratio."""
    y, X, _ = _split(geo_panel(seed, n=n, p=p), T)
    D = ExactData(y, X)
    r = _ratio(D, incumbent(y, X, restarts=5, seed=0)[1])
    return D.c, D.G, dyadic_below((Q(1) if r is None else r) * f)


@pytest.fixture(scope="module")
def stalled():
    """A 30-geo group, 26 candidate controls, at 1.001 times its incumbent's ratio:
    the first schedule gives up after its 400 evaluations and the second closes."""
    return _group(0, (0, 4, 19, 23), Q(1001, 1000))


def _spy(monkeypatch):
    real = slopefree.dual_multipliers
    calls = []

    def dual(A, R, m, c, warm=None, **k):
        calls.append((k["maxfun"], k["stages"], k["budget"]))
        return real(A, R, m, c, warm=warm, **k)

    monkeypatch.setattr(slopefree, "dual_multipliers", dual)
    return calls


# ---------------------------------------------------------------- smoke


def test_smoke_a_longer_schedule_closes_what_the_first_gives_up(stalled):
    c, G, lam = stalled
    first = group_certificate(c, G, lam, schedules=SCREEN_SCHEDULES[:1])
    ladder = group_certificate(c, G, lam, schedules=SCREEN_SCHEDULES)
    assert first.certified and first.tier == "full" and first.rung is None
    assert ladder.certified and ladder.tier == "dual" and ladder.rung == 1 and ladder.bound <= 0


# ---------------------------------------------------------------- unit


def test_the_first_schedule_is_the_size_certificates():
    assert SCREEN_SCHEDULES[0] == (150, 6, DUAL_BUDGET)
    assert all(a[0] < b[0] for a, b in zip(SCREEN_SCHEDULES, SCREEN_SCHEDULES[1:]))


def test_the_schedules_are_tried_in_turn_until_one_closes(monkeypatch, stalled):
    calls = _spy(monkeypatch)
    c, G, lam = stalled
    assert group_certificate(c, G, lam, schedules=SCREEN_SCHEDULES).rung == 1
    assert calls == list(SCREEN_SCHEDULES[:2])


def test_a_schedule_whose_exact_bound_does_not_close_hands_on(monkeypatch, stalled):
    """Multipliers the float certificate accepts are checked exactly; when the exact
    bound at them is above zero the next schedule runs. Zero multipliers leave the
    exact bound at (1 + p) lambda_max of the objective, above zero."""
    real = slopefree.dual_multipliers
    calls = []

    def dual(A, R, m, c_, warm=None, **k):
        calls.append(k["maxfun"])
        if len(calls) == 1:
            return slopefree.Multipliers("full", np.zeros(R.n_eq), np.zeros(R.n_in)), {"evals": 1}
        return real(A, R, m, c_, warm=warm, **k)

    monkeypatch.setattr(slopefree, "dual_multipliers", dual)
    c, G, lam = stalled
    res = group_certificate(c, G, lam, schedules=SCREEN_SCHEDULES)
    assert res.certified and res.tier == "dual" and res.rung == 1 and calls == [150, 1000]


def test_clarabel_decides_what_no_schedule_closes(monkeypatch, stalled):
    calls = []
    monkeypatch.setattr(slopefree, "dual_multipliers", lambda *a, **k: (calls.append(1), (None, {"evals": 0}))[1])
    c, G, lam = stalled
    res = group_certificate(c, G, lam, schedules=SCREEN_SCHEDULES)
    assert res.certified and res.tier == "full" and res.rung is None and len(calls) == len(SCREEN_SCHEDULES)


@pytest.mark.parametrize("p, rungs", [(1, 1), (SCREEN_LADDER_FROM - 1, 1),
                                      (SCREEN_LADDER_FROM, len(SCREEN_SCHEDULES)), (500, len(SCREEN_SCHEDULES))])
def test_the_longer_schedules_run_from_the_ladder_size(p, rungs):
    assert screen_schedules(p) == SCREEN_SCHEDULES[:rungs]


def test_the_screen_picks_its_schedules_by_the_number_of_controls(monkeypatch):
    seen = []
    monkeypatch.setattr(slopefree, "screen_schedules", lambda p: (seen.append(p), SCREEN_SCHEDULES[:1])[1])
    y, X = panel(0, p=5)
    D = ExactData(y, X)
    group_certificate(D.c, D.G, Q(10) ** 9)
    assert seen == [5]


def test_the_slope_free_search_screens_with_the_ladder_of_its_size(monkeypatch):
    seen = []
    real = slopefree.group_certificate

    def screen(c, G, lam, **k):
        seen.append(k)
        return real(c, G, lam, **k)

    monkeypatch.setattr(slopefree, "group_certificate", screen)
    from .brute import positive_optimum
    y, X = panel(0, p=6)
    best, _ = positive_optimum(y, X)
    out = slopefree.certify_slope_free(y, X, restarts=10, ceiling=Q(best) * Q(95, 100))
    assert out.certified and out.ceiling_held and seen == [{}]


# ---------------------------------------------------------------- properties


@SETTINGS
@given(seed=st.integers(0, 10_000), p=st.integers(1, 6),
       f=st.sampled_from([Q(1, 2), Q(9, 10), Q(99, 100), Q(101, 100), Q(11, 10), Q(2)]))
def test_property_the_ladder_certifies_only_what_holds_and_all_the_first_rung_does(seed, p, f):
    y, X = panel(seed % 80, p=p)
    D = ExactData(y, X)
    best = _best_ratio(D.c, D.G)
    lam = best * f if best > 0 else f
    ladder = group_certificate(D.c, D.G, lam, schedules=SCREEN_SCHEDULES)
    first = group_certificate(D.c, D.G, lam, schedules=SCREEN_SCHEDULES[:1])
    if ladder.certified:
        assert best <= dyadic_below(lam) and ladder.bound <= 0
    if first.certified:
        assert ladder.certified


@SETTINGS
@given(seed=st.integers(0, 10_000), f=st.sampled_from([Q(999, 1000), Q(1001, 1000), Q(101, 100), Q(11, 10)]))
@example(seed=6393, f=Q(999, 1000))   # panel 33, group (2, 6, 9, 11): every c_j < 0
def test_property_on_geo_groups_no_design_is_above_a_certified_ratio(seed, f):
    """On simulated geo panels, eight candidate controls, every design enumerated."""
    rng = np.random.default_rng(seed)
    T = tuple(sorted(rng.choice(12, 4, replace=False).tolist()))
    c, G, lam = _group(seed % 40, T, f, n=10, p=12)
    res = group_certificate(c, G, lam, schedules=SCREEN_SCHEDULES)
    if res.certified:
        assert _best_ratio(c, G) <= lam


# ---------------------------------------------------------------- edges


def test_no_schedule_is_clarabel_alone(monkeypatch, stalled):
    calls = _spy(monkeypatch)
    c, G, lam = stalled
    res = group_certificate(c, G, lam, schedules=())
    assert res.certified and res.tier == "full" and res.rung is None and calls == []


def test_one_candidate_control_with_the_ladder():
    y, X = panel(4, p=1)
    D = ExactData(y, X)
    best = _best_ratio(D.c, D.G)
    assert group_certificate(D.c, D.G, best * 2 if best > 0 else Q(1), schedules=SCREEN_SCHEDULES).certified
    if best > 0:
        assert not group_certificate(D.c, D.G, best * Q(1, 2), schedules=SCREEN_SCHEDULES).certified


# ---------------------------------------------------------------- failures


@pytest.mark.parametrize("schedules", [((0, 6, 400),), ((150, 0, 400),), ((150, 6, 0),), ((150, 6),),
                                       ((150, 6, 400, 1),), ((150.0, 6, 400),), [(150, 6, 400)]])
def test_a_malformed_schedule_is_refused(schedules):
    y, X = panel(0, p=4)
    D = ExactData(y, X)
    with pytest.raises(MlsynthConfigError, match="schedule"):
        group_certificate(D.c, D.G, Q(1), schedules=schedules)
