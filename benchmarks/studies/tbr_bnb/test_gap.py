"""A certificate within a proven gap (#965).

certify_treatment with ``gap`` g screens and certifies every treatment group against
the ceiling bar L_T^2 with the bar RSS* / (level*^2 (1 + g)^2), in place of the
incumbent's own: a certified search proves that no design's relative key exceeds
1 + g times the reported design's. Every claim is checked against enumeration of all
treatment and control groups with the library's own key."""
from fractions import Fraction as Q
from itertools import combinations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthDataError
from mlsynth.utils.tbr_helpers.design.objective import score_split

from .brute import raw_panel, relative_optimum
from mlsynth.utils.tbr_helpers.design.exact.treatment import certify_treatment, geo_sums, group_level, relative_bar, relative_ceiling

N_TEST = 8
SETTINGS = settings(max_examples=12, deadline=None, suppress_health_check=[HealthCheck.too_slow])
REL = 1e-9


def _within(M, k, out, g):
    """The gap's claim against enumeration: the optimum's key is at most (1 + g)
    times the reported design's."""
    best, _, _ = relative_optimum(M, k, N_TEST)
    return out.key is not None and best <= (1 + float(g)) * out.key * (1 + REL)


def _own_optima(M, k):
    """Each treatment group's own best key, by enumeration of its control groups."""
    p = M.shape[1]
    own = {}
    for T in combinations(range(p), k):
        y = M[:, list(T)].sum(axis=1)
        rest = [j for j in range(p) if j not in T]
        own[T] = max(score_split(y, M[:, list(C)].sum(axis=1), objective="relative", n_test=N_TEST).key[0]
                     for r in range(1, len(rest) + 1) for C in combinations(rest, r))
    return own


def _runner_up(M, k):
    """The best group, the group whose own optimum is the next best, and the gap that
    just covers it."""
    own = _own_optima(M, k)
    ranked = sorted(own, key=lambda t: -own[t])
    best, second = ranked[0], ranked[1]
    ratio = own[best] / own[second]                  # > 1
    g = Q(ratio) * Q(1001, 1000) - 1                 # 1 + g just above the ratio
    return best, second, g, Q(ratio) - 1


# ---------------------------------------------------------------- smoke


def test_smoke_a_gap_certificate_is_within_its_gap():
    M = raw_panel(5)
    g = Q(1, 20)
    out = certify_treatment(M, 2, n_test=N_TEST, restarts=5, gap=g, blank_periods=0)
    assert out.certified and out.gap == g
    assert _within(M, 2, out, g)


# ---------------------------------------------------------------- unit


def test_no_gap_is_the_exact_search():
    """gap = 0 is the search without one: the same design, survivors and ceilings."""
    M = raw_panel(5)
    plain = certify_treatment(M, 2, n_test=N_TEST, restarts=5, blank_periods=0)
    zero = certify_treatment(M, 2, n_test=N_TEST, restarts=5, gap=0, blank_periods=0)
    assert plain.gap == zero.gap == 0
    assert (zero.treatment, zero.controls, zero.rss) == (plain.treatment, plain.controls, plain.rss)
    assert [(s["treatment"], s["ceiling"]) for s in zero.survivors] == \
           [(s["treatment"], s["ceiling"]) for s in plain.survivors]
    assert zero.screened_out == plain.screened_out


def test_each_survivor_is_certified_at_the_gaps_ceiling():
    """Every certification is asked about RSS* (L_T / level*)^2 / (1 + g)^2, squared
    because the key is the level over a constant times sqrt(RSS)."""
    M = raw_panel(5)
    g = Q(1, 50)
    out = certify_treatment(M, 2, n_test=N_TEST, restarts=5, gap=g, blank_periods=0)
    assert out.replaced == 0 and out.survivors     # the incumbent is the pass's throughout
    sums, n = geo_sums(M), M.shape[0]
    for s in out.survivors:
        L = group_level(sums, s["treatment"], n)
        assert s["ceiling"] == out.rss * (L / out.level) ** 2 / (1 + g) ** 2
        assert s["ceiling"] == relative_ceiling(L, relative_bar(out.rss, out.level, g))


def test_a_start_within_the_gap_is_kept():
    """Started from the runner-up group with a gap that covers it, the search keeps
    the start's design and certifies it; with no gap the best group replaces it."""
    M = raw_panel(5)
    best, second, g, _ = _runner_up(M, 2)
    kept = certify_treatment(M, 2, n_test=N_TEST, start=list(second), restarts=5, pass_restarts=0, gap=g, blank_periods=0)
    assert kept.certified and tuple(kept.treatment) == second and kept.replaced == 0
    assert _within(M, 2, kept, g)
    exact = certify_treatment(M, 2, n_test=N_TEST, start=list(second), restarts=5, pass_restarts=0, blank_periods=0)
    assert exact.certified and tuple(exact.treatment) == best and exact.replaced >= 1


def test_a_start_just_outside_the_gap_is_replaced():
    """A gap that falls short of the runner-up's ratio leaves the best group able to
    beat it, so the best group replaces the start."""
    M = raw_panel(5)
    best, second, _, just = _runner_up(M, 2)
    short = (1 + just) * Q(999, 1000) - 1
    out = certify_treatment(M, 2, n_test=N_TEST, start=list(second), restarts=5, pass_restarts=0, gap=short, blank_periods=0)
    assert out.certified and tuple(out.treatment) == best


def test_a_gap_screens_out_at_least_as_many_groups():
    M = raw_panel(5, p=7)
    plain = certify_treatment(M, 2, n_test=N_TEST, restarts=5, pass_restarts=0, blank_periods=0)
    wide = certify_treatment(M, 2, n_test=N_TEST, restarts=5, pass_restarts=0, gap=1, blank_periods=0)
    assert wide.screened_out >= plain.screened_out
    assert len(wide.survivors) <= len(plain.survivors)


@given(seed=st.integers(0, 10_000), k=st.integers(1, 2),
       g=st.fractions(0, 1, max_denominator=1000))
@SETTINGS
def test_property_a_certified_design_is_within_its_gap(seed, k, g):
    M = raw_panel(seed, n=30, p=6)
    out = certify_treatment(M, k, n_test=N_TEST, restarts=3, gap=g, blank_periods=0)
    assert out.certified and out.gap == g
    assert _within(M, k, out, g)


# ---------------------------------------------------------------- edge


def test_a_huge_gap_keeps_the_start_and_still_certifies():
    """With a gap of a million every ceiling is far below any design: the start's
    design is reported, certified within the gap."""
    M = raw_panel(5)
    out = certify_treatment(M, 2, n_test=N_TEST, restarts=5, gap=10 ** 6, blank_periods=0)
    assert out.certified and out.replaced == 0
    assert _within(M, 2, out, Q(10 ** 6))


def test_a_float_gap_is_read_exactly():
    M = raw_panel(5)
    out = certify_treatment(M, 2, n_test=N_TEST, restarts=5, gap=0.05, blank_periods=0)
    assert out.gap == Q(0.05) and isinstance(out.gap, Q)


# ---------------------------------------------------------------- failure


@pytest.mark.parametrize("bad", [-Q(1, 100), -1, float("nan"), float("inf"), True, "0.01", None])
def test_a_gap_that_is_not_a_nonnegative_number_raises(bad):
    with pytest.raises(MlsynthDataError, match="gap"):
        certify_treatment(raw_panel(5), 2, n_test=N_TEST, restarts=3, gap=bad, blank_periods=0)
