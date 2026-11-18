"""A size's relaxation rounds stop early when the bound cannot close (#965, D2).

On the hotel_med certified search 237 of the 300 size_certificate calls that reached
the relaxation rounds ended open and branched anyway, and they took 500.7 s of the 555 s
those rounds cost; 108 ran all seven rounds. Every call that closed had fallen to at most
0.458 of its round-0 bound by round 1, while the open calls' median was 0.55. So after
round HOPELESS_FROM a round whose bound is still above HOPELESS_RATIO times round 0's
ends the size open (tier "hopeless") and the node branches on that round's point.

Branching is sound whatever the point, so the rule can never certify anything wrongly;
what it can do is give up on a node that would have closed, which costs nodes, never
validity. The checks: the stop happens exactly at the first qualifying round; a sequence
that reaches zero first closes as before; the settings are read at the call and refused
when malformed; a search finds the same optimum with the rule on and off."""
import itertools
from fractions import Fraction as Q

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthConfigError

from . import progress, slopefree
from .brute import panel, positive_optimum
from .certificate import ExactData
from .slopefree import _ratio, certify_slope_free, size_certificate

SETTINGS = settings(max_examples=80, deadline=None, suppress_health_check=[HealthCheck.too_slow,
                                                                          HealthCheck.function_scoped_fixture])


def _open_size():
    """A size held open at every round: its own best ratio less a little (test_pentagons)."""
    y, X = panel(3, p=8)
    D = ExactData(y, X); c, G = D.c, D.G
    m = 4
    best = max(_ratio(c, G, set(C))[1] for C in itertools.combinations(range(8), m))
    return c, G, best * Q(999, 1000), m


@pytest.fixture
def scripted(monkeypatch):
    """solve() returns the next scripted bound each round (a point, zero multipliers), and
    every separation finds one new triangle, so only the bounds decide when rounds end."""
    state = {"bounds": iter(()), "solves": 0}
    tri = iter(itertools.combinations(range(8), 3))

    def solve(c, G, lam, R, solver=None):
        state["solves"] += 1
        n = R.n; u = np.full(n - 1, 0.5)
        Z = np.empty((n, n)); Z[0, 0] = 1.0; Z[0, 1:] = Z[1:, 0] = u; Z[1:, 1:] = 0.25; np.fill_diagonal(Z[1:, 1:], 0.5)
        b = next(state["bounds"])
        return slopefree.Solve(0.0, 1.0, np.zeros(R.n_eq), np.zeros(R.n_in), Z, "optimal",
                               bound=None if b is None else Q(b))

    monkeypatch.setattr(slopefree, "solve", solve)
    monkeypatch.setattr(slopefree, "separate_triangles", lambda u, U, k=400, tol=1e-6: [(0.1, 1, next(tri))])

    def run(bounds, rounds=6, **k):
        state["bounds"] = iter(bounds); state["solves"] = 0
        c, G, lam, m = _open_size()
        res = size_certificate(c, G, lam, m, schedules=(), rounds=rounds, solver="clarabel", pentagons=False, **k)
        return res, state["solves"]

    return run


# ---------------------------------------------------------------- the rule


def test_a_round_still_above_the_ratio_ends_the_size_open(scripted, monkeypatch):
    monkeypatch.setattr(slopefree, "HOPELESS_RATIO", 0.6); monkeypatch.setattr(slopefree, "HOPELESS_FROM", 1)
    res, solves = scripted([10, 7, 5, 3, 1, -1, -2])
    assert not res.certified and res.tier == "hopeless" and res.rounds == 1 and solves == 2
    assert res.bound == 7 and res.u is not None


def test_a_bound_that_falls_below_the_ratio_keeps_its_rounds(scripted, monkeypatch):
    monkeypatch.setattr(slopefree, "HOPELESS_RATIO", 0.6); monkeypatch.setattr(slopefree, "HOPELESS_FROM", 1)
    res, solves = scripted([10, 5, 4, 2, -1])
    assert res.certified and res.rounds == 4 and solves == 5 and res.tier == "full"


def test_the_ratio_is_against_round_zero_not_the_last_round(scripted, monkeypatch):
    monkeypatch.setattr(slopefree, "HOPELESS_RATIO", 0.6); monkeypatch.setattr(slopefree, "HOPELESS_FROM", 1)
    res, _ = scripted([10, 5, 4.9, 4.8, 4.7, 4.6, 4.5])     # each round 98% of the last, all below 6
    assert not res.certified and res.tier == "full" and res.rounds == 6


def test_the_rule_starts_at_hopeless_from(scripted, monkeypatch):
    monkeypatch.setattr(slopefree, "HOPELESS_RATIO", 0.6); monkeypatch.setattr(slopefree, "HOPELESS_FROM", 2)
    res, solves = scripted([10, 9, 8, 7])
    assert res.tier == "hopeless" and res.rounds == 2 and solves == 3


def test_none_turns_the_rule_off(scripted, monkeypatch):
    monkeypatch.setattr(slopefree, "HOPELESS_RATIO", None)
    res, solves = scripted([10, 9.9, 9.8, 9.7, 9.6, 9.5, 9.4])
    assert res.tier == "full" and res.rounds == 6 and solves == 7


def test_the_last_round_ends_as_before(scripted, monkeypatch):
    monkeypatch.setattr(slopefree, "HOPELESS_RATIO", 0.6); monkeypatch.setattr(slopefree, "HOPELESS_FROM", 1)
    res, solves = scripted([10, 9], rounds=1)
    assert res.tier == "full" and res.rounds == 1 and solves == 2


def test_a_round_without_a_bound_is_not_judged(scripted, monkeypatch):
    monkeypatch.setattr(slopefree, "HOPELESS_RATIO", 0.6); monkeypatch.setattr(slopefree, "HOPELESS_FROM", 1)
    res, _ = scripted([10, None, 4, -1])
    assert res.certified and res.rounds == 3


def test_no_round_zero_bound_turns_the_rule_off_for_the_size(scripted, monkeypatch):
    monkeypatch.setattr(slopefree, "HOPELESS_RATIO", 0.6); monkeypatch.setattr(slopefree, "HOPELESS_FROM", 1)
    res, solves = scripted([None, 9, 9, 9], rounds=3)
    assert res.tier == "full" and solves == 4


def test_the_settings_are_read_at_the_call(scripted, monkeypatch):
    monkeypatch.setattr(slopefree, "HOPELESS_FROM", 1)
    monkeypatch.setattr(slopefree, "HOPELESS_RATIO", 0.95)
    assert scripted([10, 9, 8, 7, 6, 5, 4])[0].tier == "full"
    monkeypatch.setattr(slopefree, "HOPELESS_RATIO", 0.85)
    assert scripted([10, 9, 8, 7, 6, 5, 4])[0].tier == "hopeless"


def test_a_stop_is_reported(scripted, monkeypatch):
    monkeypatch.setattr(slopefree, "HOPELESS_RATIO", 0.6); monkeypatch.setattr(slopefree, "HOPELESS_FROM", 1)
    seen = []
    with progress.reporting(progress=seen.append):
        scripted([10, 8, 1])
    ev = [e for e in seen if e.kind == "hopeless"]
    assert len(ev) == 1 and ev[0].fields["round"] == 1 and ev[0].fields["ratio"] == pytest.approx(0.8)


# ---------------------------------------------------------------- failure


@pytest.mark.parametrize("bad", [0, 0.0, 1.5, -0.2, "0.6", True, float("nan")])
def test_a_bad_ratio_is_refused(scripted, monkeypatch, bad):
    monkeypatch.setattr(slopefree, "HOPELESS_RATIO", bad)
    with pytest.raises(MlsynthConfigError):
        scripted([10, 9, 8])


@pytest.mark.parametrize("bad", [0, -1, 1.0, "1", None, True])
def test_a_bad_start_round_is_refused(scripted, monkeypatch, bad):
    monkeypatch.setattr(slopefree, "HOPELESS_RATIO", 0.6); monkeypatch.setattr(slopefree, "HOPELESS_FROM", bad)
    with pytest.raises(MlsynthConfigError):
        scripted([10, 9, 8])


# ---------------------------------------------------------------- the real relaxation


def test_smoke_a_real_open_size_stays_open_and_a_closable_size_closes(monkeypatch):
    monkeypatch.setattr(slopefree, "HOPELESS_RATIO", 0.6); monkeypatch.setattr(slopefree, "HOPELESS_FROM", 1)
    c, G, lam, m = _open_size()
    res = size_certificate(c, G, lam, m, schedules=(), rounds=4, solver="clarabel")
    assert not res.certified and res.bound is not None and res.bound > 0
    res = size_certificate(c, G, lam * 2, m, schedules=(), rounds=4, solver="clarabel")
    assert res.certified and res.bound <= 0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_search_finds_the_same_optimum_with_the_rule_on_and_off(monkeypatch, seed):
    y, X = panel(seed, p=8)
    monkeypatch.setattr(slopefree, "PRICE_LIMIT", 0)            # bound every node: none is priced
    monkeypatch.setattr(slopefree, "HOPELESS_RATIO", None)
    off = certify_slope_free(y, X, restarts=1, deep_restarts=0, hull=False)
    monkeypatch.setattr(slopefree, "HOPELESS_RATIO", 0.3); monkeypatch.setattr(slopefree, "HOPELESS_FROM", 1)
    on = certify_slope_free(y, X, restarts=1, deep_restarts=0, hull=False)
    assert on.certified and off.certified and sorted(on.controls) == sorted(off.controls)
    assert sorted(on.controls) == sorted(positive_optimum(y, X)[1])


# ---------------------------------------------------------------- property


@SETTINGS
@given(st.lists(st.one_of(st.none(), st.fractions(-5, 20, max_denominator=8)), min_size=2, max_size=7),
       st.sampled_from([0.3, 0.5, 0.6, 0.9, 1.0]), st.integers(1, 4))
def test_property_the_stop_is_the_first_qualifying_round_and_never_hides_a_closure(scripted, monkeypatch,
                                                                                   bounds, ratio, start):
    monkeypatch.setattr(slopefree, "HOPELESS_RATIO", ratio); monkeypatch.setattr(slopefree, "HOPELESS_FROM", start)
    rounds = len(bounds) - 1
    res, solves = scripted(bounds, rounds=rounds)
    b0 = bounds[0]
    expect_close = expect_stop = None
    for r, b in enumerate(bounds):
        if b is not None and b <= 0:
            expect_close = r
            break
        if (r >= start and r < rounds and b is not None and b0 is not None and b0 > 0 and b > Q(ratio) * b0):
            expect_stop = r
            break
    if expect_close is not None:
        assert res.certified and res.rounds == expect_close
    elif expect_stop is not None:
        assert not res.certified and res.tier == "hopeless" and res.rounds == expect_stop and solves == expect_stop + 1
    else:
        assert not res.certified and res.tier == "full" and solves == rounds + 1
