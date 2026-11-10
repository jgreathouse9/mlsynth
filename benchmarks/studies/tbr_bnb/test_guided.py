"""The floor-guided search over treatment groups, uncertified (guided.py).

The certified search (treatment.py) without its certificates: the library's start,
the exact floors of every treatment group, and a search of the controls of every
group the floors cannot rule out, the start group included, best floor first. Its
answer is a design, not a proof, and it reports how far from the optimum the floors
allow that design to be. Every claim is checked against enumeration of all treatment
and control groups with the library's own key."""
from fractions import Fraction as Q
from itertools import combinations
from math import comb

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.tbr_helpers.design.objective import score_split

from . import guided, treatment
from .brute import geo_panel, treatment_optimum
from .guided import group_search, guided_treatment
from .treatment import can_compete, design_moments, gram_floor

SETTINGS = settings(max_examples=12, deadline=None, suppress_health_check=[HealthCheck.too_slow])
N_TEST = 8


def _key(M, T, C):
    y = M[:, sorted(T)].sum(axis=1)
    return score_split(y, M[:, sorted(C)].sum(axis=1), objective="reference", n_test=N_TEST).key


def _rss(M, T, C):
    y, X, rest = treatment._split(M, T)
    cu, uGu, yy = design_moments(y, X, sorted(rest.index(j) for j in C))
    return yy - cu * cu / uGu


def _floors(M, k):
    """Each group's floor, by the floor function the search uses, and its exact y'y."""
    rows = treatment._floor_rows(M, list(combinations(range(M.shape[1]), k)), gram_floor(M), None, 1)
    return {tuple(T): (f, treatment._yy(M, T)) for f, _, T in rows}


def _competing(M, k, out):
    """The groups whose floor can still beat the reported design."""
    r, rss = Q(out.key[4]), out.rss
    return {T for T, (f, yy) in _floors(M, k).items() if can_compete(yy, f, r, rss)}


def _check_bound(M, k, out):
    """The reported design is a real design scored by the library's key, no better
    than the optimum, and the optimum is no better than the bound allows."""
    key, T, C = treatment_optimum(M, k, N_TEST)
    if not out.passing:
        assert out.treatment is None and out.key is None
        return key
    assert key is not None
    assert tuple(out.key) == tuple(_key(M, out.treatment, out.controls))
    assert out.rss == _rss(M, out.treatment, out.controls)
    assert tuple(out.key) <= tuple(key)
    if not out.above_class:
        assert Q(key[4]) == Q(out.key[4])
        assert _rss(M, T, C) >= out.rss_bound
        assert 1 / key[5] >= out.impact_bound * (1 - 1e-9)
    if out.proven:
        assert tuple(out.key) == tuple(key)
    return key


# ---------------------------------------------------------------- smoke


def test_smoke_a_passing_design_and_a_bound_that_holds():
    M = geo_panel(0)                       # no open group can round above the design's class
    out = guided_treatment(M, 2, n_test=N_TEST, restarts=5)
    assert out.passing and out.groups == comb(7, 2) and not out.above_class
    assert out.rss_bound <= out.rss and out.impact_bound <= 1 / out.key[5]
    assert out.seconds > 0
    _check_bound(M, 2, out)


# ---------------------------------------------------------------- unit


@pytest.mark.parametrize("seed,k", [(0, 1), (0, 2), (1, 3), (4, 2), (5, 1), (5, 3)])
def test_the_bound_holds_on_small_panels(seed, k):
    M = geo_panel(seed)
    _check_bound(M, k, guided_treatment(M, k, n_test=N_TEST, restarts=5))


def test_every_group_is_screened_or_searched():
    out = guided_treatment(geo_panel(5), 2, n_test=N_TEST, restarts=5)
    assert out.screened_out + out.searched == out.groups and out.screened_out > 0


def test_every_group_that_can_beat_the_design_was_searched_and_is_open():
    M = geo_panel(5)
    out = guided_treatment(M, 2, n_test=N_TEST, restarts=5)
    assert set(out.open) == _competing(M, 2, out)
    assert set(out.open) <= set(out.searched_groups)


def test_the_start_group_is_searched_from_its_start_design(monkeypatch):
    calls = []
    real = guided.group_search

    def spy(M, T, n_test, restarts, start=None):
        calls.append((tuple(T), start))
        return real(M, T, n_test, restarts, start=start)

    monkeypatch.setattr(guided, "group_search", spy)
    M = geo_panel(5)
    T0, C0 = treatment._start(M, 2, N_TEST, 5)
    guided_treatment(M, 2, n_test=N_TEST, restarts=5)
    assert (tuple(T0), sorted(C0)) in [(T, sorted(s) if s else s) for T, s in calls]
    assert all(s is None for T, s in calls if T != tuple(T0))
    assert calls[0][0] == tuple(T0)                     # searched first


def test_the_start_group_stays_open_while_its_floor_can_beat_the_design():
    """The start group is a group like any other once searched: its floor enters the
    bound whenever it can still beat the reported design."""
    M = geo_panel(5)
    out = guided_treatment(M, 2, n_test=N_TEST, restarts=5)
    T0, _ = treatment._start(M, 2, N_TEST, 5)
    f, yy = _floors(M, 2)[tuple(T0)]
    assert (tuple(T0) in out.open) == can_compete(yy, f, Q(out.key[4]), out.rss)
    assert tuple(T0) in out.open


def test_a_start_given_is_the_first_incumbent():
    M = geo_panel(5)
    out = guided_treatment(M, 2, n_test=N_TEST, restarts=5, start=[0, 1])
    assert (0, 1) in out.searched_groups
    _check_bound(M, 2, out)


def test_the_bound_is_the_least_floor_of_the_open_groups():
    M = geo_panel(0)
    out = guided_treatment(M, 2, n_test=N_TEST, restarts=5)
    assert not out.above_class
    floors = _floors(M, 2)
    assert out.rss_bound == min([floors[T][0] for T in out.open] + [out.rss])
    assert out.impact_bound == pytest.approx((1 / out.key[5]) * float(out.rss_bound / out.rss) ** 0.5, rel=1e-12)


def test_a_group_that_can_round_above_the_class_leaves_no_impact_bound(monkeypatch):
    """An open group whose correlation bound rounds above the design's class could hold a
    design of a higher class at any impact: the bound then says so and gives no impact."""
    real = guided._floor_rows
    monkeypatch.setattr(guided, "_floor_rows", lambda M, groups, lam: [(Q(0), yy, T) for _, yy, T in real(M, groups, lam)])
    M = geo_panel(5)
    out = guided_treatment(M, 2, n_test=N_TEST, restarts=5)
    if out.key[4] < 1:
        assert out.above_class and out.impact_bound is None and not out.proven
    _check_bound(M, 2, out)


def test_proven_only_when_no_group_is_open(monkeypatch):
    """With every floor at the design's own RSS, nothing can beat it and the design is
    the optimum."""
    M = geo_panel(5)
    out = guided_treatment(M, 2, n_test=N_TEST, restarts=5)
    assert out.proven == (not out.open)
    big = out.rss
    real = guided._floor_rows
    monkeypatch.setattr(guided, "_floor_rows", lambda M, groups, lam: [(big * 10 ** 6, yy, T) for _, yy, T in real(M, groups, lam)])
    out2 = guided_treatment(M, 2, n_test=N_TEST, restarts=5)
    assert out2.open == [] and out2.proven and out2.rss_bound == out2.rss
    assert out2.searched == 1 and out2.screened_out == out2.groups - 1


# ---------------------------------------------------------------- the group search


def test_smoke_group_search_returns_a_passing_design_scored_by_the_key():
    M = geo_panel(5)
    g = group_search(M, (0, 1), N_TEST, 5)
    if g.key is not None:
        assert tuple(g.key) == tuple(_key(M, (0, 1), g.controls))
        assert g.rss == _rss(M, (0, 1), g.controls)


def test_a_failing_rss_design_does_not_end_the_group_search(monkeypatch):
    """The RSS descent's design fails a gate; the gate-aware descent from it still
    finds the group's passing design."""
    M = geo_panel(5)
    T = (0, 1)
    want = group_search(M, T, N_TEST, 5)
    assert want.key is not None
    real = treatment._local

    def failing(M_, T_, n_test, restarts):
        C, key, rss, rest = real(M_, T_, n_test, restarts)
        return C, None, rss, rest

    monkeypatch.setattr(guided, "_local", failing)
    got = group_search(M, T, N_TEST, 5)
    assert got.key is not None and tuple(got.key) == tuple(_key(M, T, got.controls))


def test_the_group_search_is_never_worse_than_its_start():
    M = geo_panel(5)
    T0, C0 = treatment._start(M, 2, N_TEST, 5)
    g = group_search(M, T0, N_TEST, 1, start=C0)
    assert tuple(g.key) >= tuple(_key(M, T0, C0))


def test_the_start_design_is_descended_from(monkeypatch):
    """A start given to the group search is a start of the key's descent, whatever the
    RSS descent finds; the descent never ends below its start, so neither does the
    group search."""
    calls = []
    real = guided._key_descent

    def spy(M, T, C, n_test):
        calls.append(sorted(C))
        return real(M, T, C, n_test)

    monkeypatch.setattr(guided, "_key_descent", spy)
    M = geo_panel(5)
    T0, C0 = treatment._start(M, 2, N_TEST, 5)
    group_search(M, T0, N_TEST, 3, start=C0)
    assert sorted(C0) in calls


def test_the_group_search_reaches_the_group_optimum_the_rss_descent_misses():
    """The case the RCA found: the RSS-best design fails a gate, and the best passing
    design is a neighbour of it under the library's moves."""
    for seed in range(40):
        M = geo_panel(seed)
        for T in combinations(range(7), 2):
            C, key, _, rest = treatment._local(M, T, N_TEST, 5)
            if key is None:
                g = group_search(M, T, N_TEST, 5)
                if g.key is not None:
                    return
    pytest.fail("no group whose RSS-best design fails a gate while a passing one exists")


# ---------------------------------------------------------------- properties


@given(seed=st.integers(0, 10_000), k=st.integers(1, 3))
@SETTINGS
def test_property_the_bound_holds(seed, k):
    M = geo_panel(seed)
    _check_bound(M, k, guided_treatment(M, k, n_test=N_TEST, restarts=3))


@given(seed=st.integers(0, 10_000), k=st.integers(1, 2))
@SETTINGS
def test_property_every_group_that_can_beat_the_design_was_searched(seed, k):
    M = geo_panel(seed)
    out = guided_treatment(M, k, n_test=N_TEST, restarts=3)
    if out.passing:
        assert _competing(M, k, out) == set(out.open) and set(out.open) <= set(out.searched_groups)
    else:
        assert set(out.searched_groups) == set(combinations(range(7), k))


@given(seed=st.integers(0, 10_000), T=st.sampled_from(list(combinations(range(7), 2))))
@SETTINGS
def test_property_the_group_search_never_loses_a_passing_design(seed, T):
    """Whatever the RSS descent reports, the group search returns a design at least as
    good as each of its starts that passes."""
    M = geo_panel(seed)
    C, key, _, rest = treatment._local(M, T, N_TEST, 3)
    g = group_search(M, T, N_TEST, 3)
    if key is not None:
        assert g.key is not None and tuple(g.key) >= tuple(key)


# ---------------------------------------------------------------- edge


def test_no_design_passes():
    M = geo_panel(2)
    key, _, _ = treatment_optimum(M, 1, N_TEST)
    out = guided_treatment(M, 1, n_test=N_TEST, restarts=3)
    assert key is None and not out.passing and out.treatment is None
    assert out.searched == out.groups and out.screened_out == 0
    assert out.above_class and out.impact_bound is None and not out.proven


def test_one_control_left():
    M = geo_panel(2, p=4)
    out = guided_treatment(M, 3, n_test=N_TEST, restarts=3)
    key = _check_bound(M, 3, out)
    if key is not None and out.passing:
        assert len(out.controls) == 1


def test_a_singular_gram_with_a_negative_entry_gives_zero_floors_and_searches_every_group():
    M = geo_panel(5)
    M[:, 2] = -(M[:, 0] + 0.5 * M[:, 1])   # collinear, a negative Gram entry, no design a perfect fit
    assert gram_floor(M) is None
    out = guided_treatment(M, 1, n_test=N_TEST, restarts=3)
    assert out.screened_out == 0 and out.searched == out.groups
    _check_bound(M, 1, out)


def test_a_singular_gram_with_no_negative_entry_still_screens():
    """The diagonal bound (certificate.gram_lower) gives a floor where the eigenvalue
    bound gives none, so the screen still rules groups out."""
    M = geo_panel(5)
    M[:, 2] = M[:, 0] + 0.5 * M[:, 1]      # collinear, no design a perfect fit
    assert np.linalg.matrix_rank(M) < M.shape[1] and (M.T @ M).min() >= 0
    assert gram_floor(M) is not None
    out = guided_treatment(M, 1, n_test=N_TEST, restarts=3)
    assert out.screened_out > 0
    _check_bound(M, 1, out)


def test_one_restart_still_finds_a_passing_design():
    M = geo_panel(5)
    _check_bound(M, 2, guided_treatment(M, 2, n_test=N_TEST, restarts=1))


# ---------------------------------------------------------------- failure


@pytest.mark.parametrize("k", [0, 7, -1])
def test_a_size_without_a_control_raises(k):
    with pytest.raises(MlsynthDataError, match="treatment"):
        guided_treatment(geo_panel(0), k, n_test=N_TEST)


def test_a_start_of_the_wrong_size_raises():
    with pytest.raises(MlsynthDataError, match="start"):
        guided_treatment(geo_panel(0), 2, n_test=N_TEST, start=[0])


@pytest.mark.parametrize("r", [0, -1])
def test_fewer_than_one_restart_raises(r):
    with pytest.raises(MlsynthDataError, match="restarts"):
        guided_treatment(geo_panel(0), 2, n_test=N_TEST, restarts=r)
