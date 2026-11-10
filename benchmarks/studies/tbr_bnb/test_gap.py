"""A certificate within a proven gap (#965).

certify_treatment with ``gap`` g screens and certifies every treatment group against
the ceiling of a target RSS, the incumbent's RSS over (1 + g)^2, in place of the
incumbent's own RSS. Every claim is checked against enumeration of all treatment and
control groups with the library's own key."""
import math
from fractions import Fraction as Q
from itertools import combinations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthDataError

from .brute import geo_panel, treatment_optimum
from .gated import certify_gated
from .treatment import _split, ceiling, certify_treatment, design_moments

N_TEST = 8
SETTINGS = settings(max_examples=12, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def _rss(M, T, C):
    """The exact RSS of the design (T, C), C in geo indices."""
    y, X, rest = _split(M, sorted(T))
    cu, uGu, yy = design_moments(y, X, sorted(rest.index(j) for j in C))
    return yy - cu * cu / uGu


def _within(M, k, out, g):
    """The gap's claim against enumeration: the reported design is the optimum, or the
    optimum is in its correlation class with an RSS at least its RSS over (1 + g)^2,
    so that its required impact is at most (1 + g) times the optimum's."""
    key, T, C = treatment_optimum(M, k, N_TEST)
    if key is None:
        return not out.passing
    if not out.passing:
        return False
    if tuple(out.key) == pytest.approx(tuple(key), rel=1e-12):
        return True
    return out.key[4] == key[4] and out.rss <= (1 + g) ** 2 * _rss(M, T, C)


def _own_optima(M, k):
    """Each passing treatment group's own optimum: group -> (key, exact RSS)."""
    own = {}
    for t in combinations(range(M.shape[1]), k):
        y, X, rest = _split(M, t)
        res = certify_gated(y, X, n_test=N_TEST, restarts=3)
        if res.passing:
            own[t] = (res.key, res.rss)
    return own


def _runner_up(M, k):
    """The best group, and the passing group whose own optimum is the best of the
    others in the same correlation class, with the gap that just covers it."""
    own = _own_optima(M, k)
    best = max(own, key=lambda t: own[t][0])
    same = [t for t in own if t != best and own[t][0][4] == own[best][0][4]]
    second = max(same, key=lambda t: own[t][0])
    ratio = own[second][1] / own[best][1]            # > 1: the runner-up's RSS is larger
    g = Q(math.sqrt(float(ratio))) * Q(1001, 1000) - 1   # (1 + g)^2 just above the ratio
    assert (1 + g) ** 2 > ratio
    return best, second, g


# ---------------------------------------------------------------- smoke


def test_smoke_a_gap_certificate_is_within_its_gap():
    M = geo_panel(5)
    g = Q(1, 20)
    out = certify_treatment(M, 2, n_test=N_TEST, restarts=5, gap=g)
    assert out.certified and out.passing and out.gap == g
    assert _within(M, 2, out, g)


# ---------------------------------------------------------------- unit


def test_no_gap_is_the_exact_search():
    """gap = 0 is the search without one: the same design, survivors and ceilings."""
    M = geo_panel(5)
    plain = certify_treatment(M, 2, n_test=N_TEST, restarts=5)
    zero = certify_treatment(M, 2, n_test=N_TEST, restarts=5, gap=0)
    assert plain.gap == zero.gap == 0
    assert (zero.treatment, zero.controls, zero.rss) == (plain.treatment, plain.controls, plain.rss)
    assert [(s["treatment"], s["ceiling"]) for s in zero.survivors] == \
           [(s["treatment"], s["ceiling"]) for s in plain.survivors]
    assert zero.screened_out == plain.screened_out


def test_each_survivor_is_certified_at_the_gaps_ceiling():
    """Every certification is asked about the ceiling of the incumbent's RSS over
    (1 + g)^2, squared because the required impact is sqrt(RSS) times a constant."""
    M = geo_panel(5)
    g = Q(1, 50)
    out = certify_treatment(M, 2, n_test=N_TEST, restarts=5, gap=g)
    assert out.replaced == 0 and out.survivors   # the incumbent is the pass's throughout
    r_star = Q(out.key[4])
    for s in out.survivors:
        y, _, _ = _split(M, s["treatment"])
        yy = sum(Q(v) ** 2 for v in y.tolist())
        assert s["ceiling"] == ceiling(yy, r_star, out.rss / (1 + g) ** 2)


def test_a_start_within_the_gap_is_kept():
    """Started from the runner-up group with a gap that covers it, the search keeps
    the start's design and certifies it; with no gap the best group replaces it."""
    M = geo_panel(5)
    best, second, g = _runner_up(M, 2)
    kept = certify_treatment(M, 2, n_test=N_TEST, start=list(second), restarts=5, pass_restarts=0, gap=g)
    assert kept.certified and tuple(kept.treatment) == second and kept.replaced == 0
    assert _within(M, 2, kept, g)
    exact = certify_treatment(M, 2, n_test=N_TEST, start=list(second), restarts=5, pass_restarts=0)
    assert exact.certified and tuple(exact.treatment) == best and exact.replaced >= 1


def test_a_start_just_outside_the_gap_is_replaced():
    """A gap that falls short of the runner-up's ratio leaves the best group able to
    beat it, so the best group replaces the start."""
    M = geo_panel(5)
    best, second, g = _runner_up(M, 2)
    short = (1 + g) * Q(999, 1000) ** 2 - 1        # (1 + short)^2 below the ratio
    out = certify_treatment(M, 2, n_test=N_TEST, start=list(second), restarts=5, pass_restarts=0, gap=short)
    assert out.certified and tuple(out.treatment) == best


def test_a_gap_screens_out_at_least_as_many_groups():
    M = geo_panel(5)
    plain = certify_treatment(M, 2, n_test=N_TEST, restarts=5, pass_restarts=0)
    wide = certify_treatment(M, 2, n_test=N_TEST, restarts=5, pass_restarts=0, gap=1)
    assert wide.screened_out >= plain.screened_out
    assert len(wide.survivors) <= len(plain.survivors)
    assert wide.screened_out > plain.screened_out


@given(seed=st.integers(0, 10_000), k=st.integers(1, 2),
       g=st.fractions(0, 1, max_denominator=1000))
@SETTINGS
def test_property_a_certified_design_is_within_its_gap(seed, k, g):
    M = geo_panel(seed, n=30, p=6)
    out = certify_treatment(M, k, n_test=N_TEST, restarts=3, gap=g)
    assert out.certified and out.gap == g
    assert _within(M, k, out, g)


# ---------------------------------------------------------------- edge


def test_a_huge_gap_keeps_the_start_and_still_certifies():
    """With a gap of a million every group's ceiling in the class is far below any
    design: only a group that could round above the class survives, and the start's
    design is the one reported unless such a group beats it."""
    M = geo_panel(5)
    out = certify_treatment(M, 2, n_test=N_TEST, restarts=5, gap=10 ** 6)
    assert out.certified and out.passing
    assert _within(M, 2, out, Q(10 ** 6))


def test_a_gap_with_no_passing_design():
    """No incumbent, no ceiling: the gap has nothing to scale."""
    M = geo_panel(2)
    out = certify_treatment(M, 1, n_test=N_TEST, restarts=3, gap=Q(1, 10))
    assert out.certified and not out.passing and out.treatment is None


def test_a_float_gap_is_read_exactly():
    M = geo_panel(5)
    out = certify_treatment(M, 2, n_test=N_TEST, restarts=5, gap=0.05)
    assert out.gap == Q(0.05) and isinstance(out.gap, Q)


# ---------------------------------------------------------------- failure


@pytest.mark.parametrize("bad", [-Q(1, 100), -1, float("nan"), float("inf"), True, "0.01", None])
def test_a_gap_that_is_not_a_nonnegative_number_raises(bad):
    with pytest.raises(MlsynthDataError, match="gap"):
        certify_treatment(geo_panel(5), 2, n_test=N_TEST, restarts=3, gap=bad)
