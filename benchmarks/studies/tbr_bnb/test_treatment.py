"""The search over treatment groups (#965, step 9).

Every treatment group of the size gets an exact floor on the RSS of its designs;
a group whose floor shows that none of its designs can reach the incumbent's
rounded correlation, or can reach it only with a larger RSS, is ruled out, and
every other group has its own passing optimum certified (certify_gated). Every
claim is checked against enumeration of all treatment and control groups with the
library's own key."""
from fractions import Fraction as Q
from itertools import combinations
from math import comb

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthDataError

from .brute import geo_panel, treatment_optimum
from .certificate import ExactData
from .gated import certify_gated
from .treatment import (EDGE, SLACK, _split, can_compete, certify_treatment, gram_floor,
                        rss_floor)

SETTINGS = settings(max_examples=12, deadline=None, suppress_health_check=[HealthCheck.too_slow])
N_TEST = 8


def _designs(M, T):
    """Every positive-slope control group of T and its exact RSS."""
    y = M[:, sorted(T)].sum(axis=1)
    rest = [j for j in range(M.shape[1]) if j not in T]
    X = M[:, rest]
    D = ExactData(y, X)
    for r in range(1, len(rest) + 1):
        for C in combinations(range(len(rest)), r):
            u = np.zeros(len(rest), dtype=object); u[list(C)] = 1
            cu = D.c @ u
            if cu > 0:
                yield D.yy - cu * cu / (u @ D.G @ u)


# ---------------------------------------------------------------- smoke


def test_smoke_the_certified_design_is_the_key_optimum():
    M = geo_panel(5)
    out = certify_treatment(M, 2, n_test=N_TEST, restarts=5)
    key, T, C = treatment_optimum(M, 2, N_TEST)
    assert out.certified and out.passing
    assert tuple(out.key) == pytest.approx(tuple(key), rel=1e-12)
    assert set(out.treatment) == T and set(out.controls) == C


# ---------------------------------------------------------------- unit


@pytest.mark.parametrize("seed,k", [(0, 1), (0, 2), (1, 3), (4, 2), (5, 1), (5, 3)])
def test_the_key_optimum_on_small_panels(seed, k):
    M = geo_panel(seed)
    out = certify_treatment(M, k, n_test=N_TEST, restarts=5)
    key, T, C = treatment_optimum(M, k, N_TEST)
    assert out.certified
    assert out.passing == (key is not None)
    if key is not None:
        assert tuple(out.key) == pytest.approx(tuple(key), rel=1e-12)


def test_every_group_is_screened_or_certified():
    M = geo_panel(5)
    out = certify_treatment(M, 2, n_test=N_TEST, restarts=5)
    assert out.groups == comb(M.shape[1], 2)
    assert out.screened_out + len(out.survivors) == out.groups
    assert all(s["certified"] for s in out.survivors)


def test_the_screen_rules_out_groups():
    """With the incumbent in hand most treatment groups are ruled out by their floor
    alone, without a certification."""
    M = geo_panel(5)
    out = certify_treatment(M, 2, n_test=N_TEST, restarts=5)
    assert out.screened_out > 0


def test_a_better_group_replaces_a_poor_start():
    """Started from the passing group whose own optimum has the smallest key, the
    search still ends at the key's optimum: a survivor that beats the incumbent
    replaces it. The start passes, so the incumbent is set before any survivor is
    certified and has to be replaced, not merely filled."""
    M = geo_panel(5)
    key, T, _ = treatment_optimum(M, 2, N_TEST)
    own = {}
    for t in combinations(range(M.shape[1]), 2):
        res = certify_gated(*_split(M, t)[:2], n_test=N_TEST, restarts=3)
        if res.passing:
            own[t] = res.key
    worst = min(own, key=own.get)
    assert set(worst) != T and own[worst] < tuple(key)
    out = certify_treatment(M, 2, n_test=N_TEST, start=list(worst), restarts=5)
    assert out.certified and set(out.treatment) == T
    assert tuple(out.key) == pytest.approx(tuple(key), rel=1e-12)
    assert out.replaced >= 1


@given(seed=st.integers(0, 10_000), k=st.integers(1, 3), t=st.integers(0, 20))
@SETTINGS
def test_property_the_floor_is_below_every_design(seed, k, t):
    M = geo_panel(seed % 300, p=6)
    groups = list(combinations(range(6), k))
    T = groups[t % len(groups)]
    floor = rss_floor(M, T, gram_floor(M))
    assert floor >= 0
    assert all(floor <= rss for rss in _designs(M, T))


@given(seed=st.integers(0, 10_000))
@SETTINGS
def test_property_the_gram_floor_holds_for_every_treatment_group(seed):
    """The full Gram matrix's eigenvalue floor bounds every principal submatrix's
    smallest eigenvalue (Cauchy interlacing), so one floor serves every group."""
    M = geo_panel(seed % 300, p=6)
    lam = gram_floor(M)
    assert lam is not None and lam > 0
    for T in combinations(range(6), 2):
        X = np.delete(M, list(T), axis=1)
        assert float(lam) <= np.linalg.eigvalsh(X.T @ X)[0] * (1 + 1e-9)


@given(seed=st.integers(0, 10_000), k=st.integers(1, 2))
@SETTINGS
def test_property_a_group_ruled_out_has_no_better_design(seed, k):
    """Against the certified optimum, every group the screen rules out has, by
    enumeration, no passing design with a key above it."""
    M = geo_panel(seed % 200, p=6)
    out = certify_treatment(M, k, n_test=N_TEST, restarts=3)
    key, _, _ = treatment_optimum(M, k, N_TEST)
    assert out.certified and out.passing == (key is not None)
    if key is not None:
        assert tuple(out.key) == pytest.approx(tuple(key), rel=1e-12)


# ---------------------------------------------------------------- can_compete


def test_a_floor_above_the_incumbent_in_its_class_rules_out():
    # corr bound sqrt(1 - 10/1000) = 0.99499 rounds to 0.99, the incumbent's class
    assert not can_compete(Q(1000), Q(10), Q(99, 100), Q(5))
    assert can_compete(Q(1000), Q(10), Q(99, 100), Q(11))


def test_a_correlation_below_the_class_rules_out_whatever_the_rss():
    # corr bound sqrt(1 - 100/1000) = 0.9487 cannot round to 0.99
    assert not can_compete(Q(1000), Q(100), Q(99, 100), Q(10**6))


def test_a_correlation_that_can_round_above_the_class_competes_whatever_the_rss():
    # corr bound sqrt(1 - 1/1000) = 0.9995 can round to 1.00, above 0.99
    assert can_compete(Q(1000), Q(1), Q(99, 100), Q(0))


def test_no_incumbent_competes_with_everything():
    assert can_compete(Q(1000), Q(999), None, None)


def test_the_class_edges_are_half_a_rounding_step_with_a_slack():
    assert EDGE == Q(5, 1000) and 0 < SLACK < Q(1, 10**6)
    # just below the upper edge of 0.99's class: cannot round above it
    c = Q(99, 100) + EDGE - 2 * SLACK
    yy = Q(1); floor = 1 - c * c
    assert not can_compete(yy, floor, Q(99, 100), floor)
    # within the slack of it: can, since the key's correlation is a double
    c = Q(99, 100) + EDGE - SLACK / 2
    assert can_compete(yy, 1 - c * c, Q(99, 100), Q(0))
    # the same at the lower edge: within the slack, the class is still in reach
    c = Q(99, 100) - EDGE - SLACK / 2
    assert can_compete(yy, 1 - c * c, Q(99, 100), Q(1))


# ---------------------------------------------------------------- edge


def test_no_design_passes():
    """On this panel every design fails a gate, at every size."""
    M = geo_panel(2)
    key, _, _ = treatment_optimum(M, 1, N_TEST)
    out = certify_treatment(M, 1, n_test=N_TEST, restarts=3)
    assert key is None and out.certified and not out.passing
    assert out.treatment is None and out.key is None


def test_one_control_left():
    M = geo_panel(2, p=4)
    out = certify_treatment(M, 3, n_test=N_TEST, restarts=3)
    key, T, C = treatment_optimum(M, 3, N_TEST)
    assert out.certified and out.passing == (key is not None)
    if key is not None:
        assert set(out.treatment) == T and len(out.controls) == 1


def test_a_singular_gram_gives_no_floor_and_rules_nothing_out():
    M = geo_panel(1, p=5)
    M = np.column_stack([M, M[:, 0]])          # a duplicated geo
    assert gram_floor(M) is None
    assert rss_floor(M, (1,), None) == 0


def test_each_survivor_stops_once_its_rounds_cannot_compete(monkeypatch):
    """Every group certified after the incumbent is set carries a stop rule: a round
    whose RSS rules the group out ends that group's search."""
    from . import treatment as tr

    real = tr.certify_gated
    rules = []

    def spy(*a, stop=None, **kw):
        rules.append(stop)
        return real(*a, stop=stop, **kw)

    monkeypatch.setattr(tr, "certify_gated", spy)
    out = certify_treatment(geo_panel(5), 2, n_test=N_TEST, restarts=3)
    assert out.passing and len(rules) == len(out.survivors) >= 2
    assert rules[0] is None                       # the start, before any incumbent
    for stop in rules[1:]:
        assert stop(Q(10) ** 30) is True           # an RSS no design of the class can have
        assert stop(Q(0)) is False                 # a perfect fit always competes
    assert all("stopped" in s for s in out.survivors)


def test_a_refused_certificate_is_reported(monkeypatch):
    """A group whose own certification is cut short leaves the search uncertified."""
    from . import treatment as tr

    real = tr.certify_gated
    calls = []

    def short(*a, **kw):
        res = real(*a, **kw)
        calls.append(1)
        if len(calls) == 2:
            res.certified = False
        return res

    monkeypatch.setattr(tr, "certify_gated", short)
    out = certify_treatment(geo_panel(5), 2, n_test=N_TEST, restarts=3)
    assert len(calls) >= 2 and not out.certified


# ---------------------------------------------------------------- failure


@pytest.mark.parametrize("k", [0, 7, -1])
def test_a_size_without_a_control_raises(k):
    with pytest.raises(MlsynthDataError, match="treatment"):
        certify_treatment(geo_panel(0), k, n_test=N_TEST)


def test_a_start_of_the_wrong_size_raises():
    with pytest.raises(MlsynthDataError, match="start"):
        certify_treatment(geo_panel(0), 2, n_test=N_TEST, start=[0])
