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
from . import treatment
from .treatment import (EDGE, SLACK, _split, can_compete, ceiling, certify_treatment,
                        gram_floor, rss_floor, square_sum)

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
    out = certify_treatment(M, 2, n_test=N_TEST, start=list(worst), restarts=5, pass_restarts=0)
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
    smallest eigenvalue, so one floor serves every group."""
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


# ---------------------------------------------------------------- y'y in integers


def _fraction_sum(v):
    return sum(Q(x) ** 2 for x in v)


def test_smoke_square_sum_is_exact_on_mixed_exponents():
    v = [0.1, -3.5, 1e300, 5e-324, 2.0 ** -1074, -1e-300, 7.0]
    s = square_sum(np.array(v))
    assert isinstance(s, Q) and s == _fraction_sum(v)


def test_square_sum_of_a_panel_column():
    M = geo_panel(3, p=6)
    y, _, _ = _split(M, (0, 2))
    assert square_sum(y) == _fraction_sum(y.tolist())


@given(st.lists(st.floats(allow_nan=False, allow_infinity=False, allow_subnormal=True), max_size=40))
@settings(max_examples=300, deadline=None)
def test_property_square_sum_is_the_exact_sum_of_squares(v):
    assert square_sum(np.array(v, dtype=float)) == _fraction_sum(v)


def test_square_sum_of_nothing_is_zero():
    assert square_sum(np.array([], dtype=float)) == 0
    assert square_sum(np.zeros(5)) == 0


def test_square_sum_of_integers():
    assert square_sum(np.array([3.0, -4.0])) == 25


@pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan])
def test_square_sum_of_a_non_finite_value_raises(bad):
    with pytest.raises(MlsynthDataError, match="finite"):
        square_sum(np.array([1.0, bad]))


def test_the_floor_reads_a_given_yy():
    M = geo_panel(5, p=6); lam = gram_floor(M); T = (1, 4)
    y, _, _ = _split(M, T)
    assert rss_floor(M, T, lam, yy=square_sum(y)) == rss_floor(M, T, lam)


def test_the_screen_computes_yy_once_per_group(monkeypatch):
    calls = []
    real = treatment.square_sum
    monkeypatch.setattr(treatment, "square_sum", lambda v: calls.append(1) or real(v))
    M = geo_panel(2, p=6)
    out = certify_treatment(M, 2, n_test=N_TEST, restarts=5, pass_restarts=0)
    assert out.certified
    # once for each group the screen reads, and once for the start group if it is left open
    assert len(calls) <= out.groups


def test_a_singular_gram_gives_no_floor_and_rules_nothing_out():
    M = geo_panel(1, p=5)
    M = np.column_stack([M, M[:, 0]])          # a duplicated geo
    assert gram_floor(M) is None
    assert rss_floor(M, (1,), None) == 0


def test_each_survivor_is_certified_against_the_incumbents_ceiling(monkeypatch):
    """Every group certified after the incumbent is set is asked only whether it has
    a design below its ceiling, the RSS at which it would stop competing."""
    from . import treatment as tr

    real = tr.certify_gated
    seen = []

    def spy(y, X, *a, ceiling=None, **kw):
        seen.append((sum(Q(v) ** 2 for v in y.tolist()), ceiling))
        return real(y, X, *a, ceiling=ceiling, **kw)

    monkeypatch.setattr(tr, "certify_gated", spy)
    out = certify_treatment(geo_panel(5), 2, n_test=N_TEST, restarts=3)
    assert out.passing and len(seen) == len(out.survivors) >= 2
    assert seen[0][1] is None                      # the start, before any incumbent
    assert all(c is not None and 0 < c < yy for yy, c in seen[1:])
    assert all("stopped" in s for s in out.survivors)


def _compete_reference(yy, floor, r_star, rss_star):
    """can_compete read directly off the rounding, the form ceiling() replaces."""
    corr2 = 1 - floor / yy
    up, low = r_star + EDGE - SLACK, r_star - EDGE - SLACK
    if corr2 >= up * up:
        return True
    if corr2 < low * low:
        return False
    return floor < rss_star


@given(yy=st.integers(10 ** 3, 10 ** 9), f=st.fractions(0, 1), r=st.sampled_from([Q(81, 100), Q(9, 10), Q(99, 100), Q(1)]),
       g=st.fractions(0, 1))
@SETTINGS
def test_property_the_ceiling_agrees_with_the_rounding(yy, f, r, g):
    """A group competes exactly when its floor is below its ceiling (equality, a
    single RSS, aside)."""
    yy = Q(yy); floor = yy * f; rss_star = yy * g
    c = ceiling(yy, r, rss_star)
    assume_away = floor == c
    if not assume_away:
        assert can_compete(yy, floor, r, rss_star) == _compete_reference(yy, floor, r, rss_star)
        assert can_compete(yy, floor, r, rss_star) == (floor < c)


def test_no_incumbent_has_no_ceiling():
    assert ceiling(Q(10), None, None) is None


def _worst_passing(M, k=2):
    """The passing group whose own optimum has the smallest key."""
    own = {}
    for t in combinations(range(M.shape[1]), k):
        res = certify_gated(*_split(M, t)[:2], n_test=N_TEST, restarts=3)
        if res.passing:
            own[t] = res.key
    return list(min(own, key=own.get))


def _flaky(monkeypatch, fail_calls):
    """certify_gated, cut short on the given calls (1-based), as if a group's
    certificate had not closed in time; returns the ceilings each call saw."""
    from . import treatment as tr

    real = tr.certify_gated
    seen = []

    def wrapped(y, X, *a, ceiling=None, **kw):
        res = real(y, X, *a, ceiling=ceiling, **kw)
        seen.append(ceiling)
        if len(seen) in fail_calls:
            res.certified = False
        return res

    monkeypatch.setattr(tr, "certify_gated", wrapped)
    return seen


def test_a_group_left_open_is_retried_against_the_final_ceiling(monkeypatch):
    """A survivor whose certificate does not close against the ceiling of its turn
    is certified again at the end against the final incumbent's, which is lower
    when the incumbent improved since: the search is certified once it closes."""
    M = geo_panel(5)
    start = _worst_passing(M)
    plain = certify_treatment(M, 2, n_test=N_TEST, start=start, restarts=3, pass_restarts=0)
    assert plain.replaced >= 1                           # the incumbent improves during the search
    seen = _flaky(monkeypatch, {2})                      # the first survivor after the start
    out = certify_treatment(M, 2, n_test=N_TEST, start=start, restarts=3, pass_restarts=0)
    assert out.certified and out.retried == 1
    assert seen[-1] is not None and seen[-1] < seen[1]  # retried at a lower ceiling
    assert out.key == plain.key


def test_a_retried_group_can_replace_the_incumbent(monkeypatch):
    """The winning group cut short on its turn, before its passing design is found:
    the groups after it lower the ceiling, so it is retried, and its optimum, found
    on the retry, replaces the incumbent."""
    from . import treatment as tr

    M = geo_panel(5)
    start = _worst_passing(M)
    plain = certify_treatment(M, 2, n_test=N_TEST, start=start, restarts=3, pass_restarts=0)
    turn = 1 + [s["treatment"] for s in plain.survivors].index(sorted(plain.treatment))
    assert turn > 1                                     # the winner is not the start
    real = tr.certify_gated
    calls = []

    def wrapped(y, X, *a, **kw):
        res = real(y, X, *a, **kw)
        calls.append(1)
        if len(calls) == turn:
            res.certified, res.passing = False, False
        return res

    monkeypatch.setattr(tr, "certify_gated", wrapped)
    out = certify_treatment(M, 2, n_test=N_TEST, start=start, restarts=3, pass_restarts=0)
    assert out.certified and out.retried == 1
    assert out.key == plain.key and sorted(out.treatment) == sorted(plain.treatment)


def test_a_group_still_open_after_its_retry_leaves_the_search_uncertified(monkeypatch):
    M = geo_panel(5)
    start = _worst_passing(M)
    calls = certify_treatment(M, 2, n_test=N_TEST, start=start, restarts=3, pass_restarts=0)
    n = len(calls.survivors)
    seen = _flaky(monkeypatch, {2, n + 1})               # open on its turn and on its retry
    out = certify_treatment(M, 2, n_test=N_TEST, start=start, restarts=3, pass_restarts=0)
    assert not out.certified and out.retried == 1


# ---------------------------------------------------------------- the pass and the queue


def test_the_pass_needs_no_more_certifications_than_without_it():
    """From a poor start, a local search in every group the start keeps finds a better
    incumbent before anything is certified, so no more groups need certifying, and the
    answer is the key's optimum either way."""
    M = geo_panel(5)
    start = _worst_passing(M)
    key, T, _ = treatment_optimum(M, 2, N_TEST)
    off = certify_treatment(M, 2, n_test=N_TEST, start=start, restarts=3, pass_restarts=0)
    on = certify_treatment(M, 2, n_test=N_TEST, start=start, restarts=3, pass_restarts=10)
    for out in (off, on):
        assert out.certified and set(out.treatment) == T
        assert tuple(out.key) == pytest.approx(tuple(key), rel=1e-12)
    assert off.searched == 0 and on.searched >= 1
    assert len(on.survivors) <= len(off.survivors)
    assert on.replaced <= off.replaced


def test_the_pass_adopts_its_best_design_as_the_incumbent():
    """From the worst passing start, the pass's best local design is the key's optimum
    on this panel, so it is the incumbent before any group is certified and no
    certification replaces it."""
    M = geo_panel(5)
    key, T, _ = treatment_optimum(M, 2, N_TEST)
    out = certify_treatment(M, 2, n_test=N_TEST, start=_worst_passing(M), restarts=3, pass_restarts=10)
    assert out.pass_key is not None
    assert tuple(out.pass_key) == pytest.approx(tuple(key), rel=1e-12)
    assert out.replaced == 0


def test_groups_are_certified_best_local_design_first():
    M = geo_panel(5)
    out = certify_treatment(M, 2, n_test=N_TEST, start=_worst_passing(M), restarts=3, pass_restarts=10)
    keys = [s["pass_key"] for s in out.survivors if not s.get("start")]
    passing = [k for k in keys if k is not None]
    assert passing == sorted(passing, reverse=True)
    assert all(k is None for k in keys[len(passing):])     # groups with no passing local design last


def test_each_certification_starts_from_its_groups_local_design(monkeypatch):
    from . import treatment as tr

    real = tr.certify_gated
    seen = []

    def spy(y, X, *a, start=None, **kw):
        seen.append(start)
        return real(y, X, *a, start=start, **kw)

    monkeypatch.setattr(tr, "certify_gated", spy)
    M = geo_panel(5)
    out = certify_treatment(M, 2, n_test=N_TEST, start=_worst_passing(M), restarts=3, pass_restarts=10)
    assert len(seen) == len(out.survivors) >= 2
    assert seen[0] is None                                # the start group, before the pass
    assert all(s is not None and len(s) >= 1 for s in seen[1:])


def test_a_negative_pass_is_refused():
    with pytest.raises(MlsynthDataError, match="pass_restarts"):
        certify_treatment(geo_panel(0), 2, n_test=N_TEST, pass_restarts=-1)


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
