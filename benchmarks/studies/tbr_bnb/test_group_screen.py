"""The group screen (#965): one certificate for every size of a treatment group.

Asked whether any design of a group has an RSS below a ceiling, the slope-free
search first tries one relaxation without a size: the rows of every size, with the
size equality and its products replaced by sum u >= 1 and its products with
1 - u_i, and the trace bound 1 + p. When its certificate is at most zero at the
ratio y'y - ceiling, no design of any size is below the ceiling, and no size is
searched."""
from fractions import Fraction as Q
from itertools import combinations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthDataError

from . import slopefree
from .brute import panel, positive_optimum
from .certificate import ExactData
from .slopefree import Rows, certify_slope_free, dyadic_below, group_bound, group_certificate, objective

SETTINGS = settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def _designs(p):
    for r in range(1, p + 1):
        for C in combinations(range(p), r):
            u = np.zeros(p, dtype=int); u[list(C)] = 1
            yield set(C), u


def _lifted(u):
    z = np.concatenate([[1], u]).astype(object)
    return np.array([[Q(int(a * b)) for b in z] for a in z], dtype=object)


def _best_ratio(c, G):
    """max (c'u)^2 / u'Gu over nonzero binary u with c'u > 0, exactly; 0 when none."""
    best = Q(0)
    for _, u in _designs(len(c)):
        uq = u.astype(object)
        cu = c @ uq
        if cu > 0:
            best = max(best, cu * cu / (uq @ G @ uq))
    return best


def _data(seed, p):
    y, X = panel(seed, p=p)
    D = ExactData(y, X)
    return y, X, D.c, D.G


# ---------------------------------------------------------------- smoke


def test_smoke_a_held_ceiling_is_proved_without_searching_a_size():
    y, X = panel(0, p=6)
    best, _ = positive_optimum(y, X)
    out = certify_slope_free(y, X, restarts=10, ceiling=Q(best) * Q(95, 100))
    assert out.certified and out.ceiling_held
    assert out.group_tier in ("dual", "full") and out.sizes == {}


# ---------------------------------------------------------------- the rows


@pytest.mark.parametrize("seed", range(3))
def test_every_design_of_every_size_meets_the_size_free_rows(seed):
    _, _, c, _ = _data(seed, 5)
    R = Rows(c, None)
    for C, u in _designs(5):
        if c @ u.astype(object) >= 0:
            assert R.holds(_lifted(u)), C


def test_the_size_free_rows_exclude_the_empty_design():
    _, _, c, _ = _data(1, 4)
    assert not Rows(c, None).holds(_lifted(np.zeros(4, dtype=int)))


def test_the_size_free_trace_bound_is_one_plus_the_controls():
    """tr Z = 1 + sum u <= 1 + p on the size-free rows, so a matrix with one positive
    eigenvalue is charged 1 + p times it."""
    _, _, c, _ = _data(2, 4)
    R = Rows(c, None)
    A = np.full((4, 4), Q(0), dtype=object); A[0, 0] = Q(1)
    b = group_bound(A, R, np.zeros(R.n_eq), np.zeros(R.n_in))
    assert Q(5) <= b <= Q(5) * (1 + Q(1, 10 ** 9))


@SETTINGS
@given(seed=st.integers(0, 10_000), p=st.integers(2, 6), cuts=st.integers(0, 2))
def test_property_the_size_free_rows_hold_at_every_design(seed, p, cuts):
    _, _, c, _ = _data(seed % 80, p)
    rng = np.random.default_rng(seed)
    cut = [set(rng.choice(p, size=int(rng.integers(1, p + 1)), replace=False).tolist()) for _ in range(cuts)]
    R = Rows(c, None, cuts=cut)
    for C, u in _designs(p):
        if c @ u.astype(object) >= 0:
            assert R.holds(_lifted(u)) == (C not in cut), (C, cut)


# ---------------------------------------------------------------- the certificate


@pytest.mark.parametrize("seed", range(5))
def test_a_ratio_above_every_design_is_certified(seed):
    _, _, c, G = _data(seed, 6)
    res = group_certificate(c, G, _best_ratio(c, G) * Q(12, 10))
    assert res.certified and res.bound <= 0 and res.tier in ("dual", "full")


@pytest.mark.parametrize("seed", range(5))
def test_a_ratio_below_the_best_design_is_not_certified(seed):
    _, _, c, G = _data(seed, 6)
    res = group_certificate(c, G, _best_ratio(c, G) * Q(99, 100))
    assert not res.certified


@SETTINGS
@given(seed=st.integers(0, 10_000), p=st.integers(1, 6),
       f=st.sampled_from([Q(1, 2), Q(9, 10), Q(99, 100), Q(101, 100), Q(11, 10), Q(2)]))
def test_property_a_certificate_holds_for_every_design(seed, p, f):
    """Certified at lam only when no design of any size has a ratio above lam."""
    _, _, c, G = _data(seed % 80, p)
    best = _best_ratio(c, G)
    lam = best * f if best > 0 else f
    res = group_certificate(c, G, lam)
    if res.certified:
        assert best <= dyadic_below(lam)
        assert res.bound <= 0


def test_the_first_order_multipliers_are_checked_exactly(monkeypatch):
    """Multipliers the first-order method returns certify only when the exact bound
    at them is at most zero: at zero multipliers the bound is (1 + p) lambda_max of
    the objective, above zero, so the relaxation's own multipliers decide, and a
    ratio below the best design's is not certified."""
    _, _, c, G = _data(3, 6)

    def zeros(A, R, m, c_, warm=None, **k):
        return slopefree.Multipliers("full", np.zeros(R.n_eq), np.zeros(R.n_in)), {"evals": 1}

    monkeypatch.setattr(slopefree, "dual_multipliers", zeros)
    assert not group_certificate(c, G, _best_ratio(c, G) * Q(99, 100)).certified
    res = group_certificate(c, G, _best_ratio(c, G) * Q(12, 10))
    assert res.certified and res.tier == "full"


def test_the_full_relaxation_decides_when_the_dual_gives_up(monkeypatch):
    _, _, c, G = _data(3, 6)
    monkeypatch.setattr(slopefree, "dual_multipliers", lambda *a, **k: (None, {"evals": 0}))
    res = group_certificate(c, G, _best_ratio(c, G) * Q(12, 10))
    assert res.certified and res.tier == "full"


# ---------------------------------------------------------------- edges and failures


def test_one_candidate_control():
    _, _, c, G = _data(4, 1)
    best = _best_ratio(c, G)
    assert group_certificate(c, G, best * 2 if best > 0 else Q(1)).certified
    if best > 0:
        assert not group_certificate(c, G, best * Q(1, 2)).certified


@pytest.mark.parametrize("lam", [Q(0), Q(-5)])
def test_a_ratio_at_or_below_zero_is_not_certified(lam):
    """Every design with a positive slope has a positive ratio."""
    _, _, c, G = _data(0, 5)
    assert not group_certificate(c, G, lam).certified


def test_no_candidate_control_is_refused():
    with pytest.raises(MlsynthDataError, match="no candidate control"):
        group_certificate(np.array([], dtype=object), np.zeros((0, 0), dtype=object), Q(1))


# ---------------------------------------------------------------- in the slope-free search


def test_the_screen_is_not_tried_without_a_ceiling():
    y, X = panel(0, p=6)
    out = certify_slope_free(y, X, restarts=10)
    assert out.certified and out.group_tier is None


def test_the_screen_is_not_tried_when_the_incumbent_is_below_the_ceiling():
    y, X = panel(0, p=6)
    best, _ = positive_optimum(y, X)
    out = certify_slope_free(y, X, restarts=10, ceiling=Q(best) * Q(105, 100))
    assert out.certified and not out.ceiling_held and out.group_tier is None


def test_a_screen_that_does_not_close_leaves_the_sizes_to_decide(monkeypatch):
    y, X = panel(0, p=6)
    best, _ = positive_optimum(y, X)
    monkeypatch.setattr(slopefree, "group_certificate",
                        lambda c, G, lam: slopefree.GroupResult(False, None, "open", 0.0))
    out = certify_slope_free(y, X, restarts=10, ceiling=Q(best) * Q(95, 100))
    assert out.certified and out.ceiling_held and out.group_tier == "open" and out.sizes


@SETTINGS
@given(seed=st.integers(0, 10_000), f=st.sampled_from([Q(9, 10), Q(98, 100), Q(102, 100), Q(11, 10)]))
def test_property_the_screen_changes_no_verdict(seed, f):
    y, X = panel(seed % 60, p=5)
    best, _ = positive_optimum(y, X)
    if best == np.inf:
        return
    on = certify_slope_free(y, X, restarts=5, ceiling=Q(best) * f)
    off = certify_slope_free(y, X, restarts=5, ceiling=Q(best) * f, group_screen=False)
    assert (on.certified, on.ceiling_held, on.incumbent) == (off.certified, off.ceiling_held, off.incumbent)
    assert off.group_tier is None


def test_the_treatment_search_certifies_the_same_design_with_the_screen(monkeypatch):
    from . import gated, treatment
    from .brute import geo_panel

    M = geo_panel(5, p=6)
    on = treatment.certify_treatment(M, 2, n_test=8, restarts=5)
    real = slopefree.certify_slope_free
    monkeypatch.setattr(gated, "certify_slope_free", lambda *a, **k: real(*a, group_screen=False, **k))
    plain = treatment.certify_treatment(M, 2, n_test=8, restarts=5)
    assert (on.treatment, on.controls, on.key, on.certified) == \
        (plain.treatment, plain.controls, plain.key, plain.certified)
