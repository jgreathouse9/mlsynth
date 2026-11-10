"""A no-good cut for every incumbent the search adopts (#965, step 6).

The search cut the start design from every size's relaxation, and no design adopted
after it: one found by rounding a node's point, at a leaf or by pricing. The node
whose subtree holds such a design bounds the design itself, at the ratio it set, so
its certificate cannot fall below zero and the subtree runs to the depth limit. On
the beds 10 of the 14 nodes left open there held the final incumbent, and 10 of the
11 uncertified groups had one. Each bounded node now cuts the incumbent it is bounded
against: every group of the beds certifies, with 45% fewer nodes and 59% fewer full
relaxations (results/incumbent_cuts.log)."""
from fractions import Fraction as Q

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from . import slopefree
from .beds import BEDS
from .brute import panel, positive_optimum
from .slopefree import _ratio, certify_slope_free

SETTINGS = settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow])


@pytest.fixture(autouse=True)
def _shallow_incumbent(monkeypatch):
    """These tests are of the search from a few restarts' incumbent, so the second
    descent is off (test_deep_descent.py checks it)."""
    monkeypatch.setattr(slopefree, "DEEP_RESTARTS", 0)


@pytest.fixture(autouse=True)
def _bound_every_node(monkeypatch):
    monkeypatch.setattr(slopefree, "PRICE_LIMIT", 0)


def _stuck():
    """Group 8 of s20_k3 (seed 27, treated 0, 4, 13): on main rounding adopted a
    better design at a size-8 node, and the subtree holding it ran to the depth limit,
    one node open at depth 12."""
    return BEDS["s20_k3"].panel(8)


def _spy_bounded(monkeypatch):
    """Each bounded node's (c, G, ratio it is bounded at, cuts)."""
    seen = []
    real = slopefree.size_certificate

    def spy(c, G, lam, m, cuts=(), **k):
        seen.append((c, G, lam, [set(C) for C in cuts]))
        return real(c, G, lam, m, cuts=cuts, **k)

    monkeypatch.setattr(slopefree, "size_certificate", spy)
    return seen


def _cuts_the_incumbent(c, G, lam, cuts) -> bool:
    return any(_ratio(c, G, C)[1] == lam for C in cuts)


# ---------------------------------------------------------------- smoke


def test_smoke_a_group_left_open_by_its_own_incumbent_certifies():
    y, X = _stuck()
    out = certify_slope_free(y, X, restarts=5)
    best, C = positive_optimum(y, X)
    assert out.certified and not out.open
    assert set(out.controls) == C and float(out.incumbent) == pytest.approx(best, rel=1e-9)
    assert out.rounded > 0                  # the design that closed the gap was adopted mid-search


# ---------------------------------------------------------------- unit


def test_every_bounded_node_cuts_the_incumbent_it_is_bounded_against(monkeypatch):
    seen = _spy_bounded(monkeypatch)
    y, X = _stuck()
    out = certify_slope_free(y, X, restarts=5)
    ratios = {lam for _, _, lam, _ in seen}
    assert len(ratios) > 1                  # the incumbent changed during the search
    assert all(_cuts_the_incumbent(c, G, lam, cuts) for c, G, lam, cuts in seen)
    assert out.certified


def test_the_excluded_designs_stay_cut(monkeypatch):
    y, X = _stuck()
    first = certify_slope_free(y, X, restarts=5)
    seen = _spy_bounded(monkeypatch)
    second = certify_slope_free(y, X, restarts=5, exclude=[set(first.controls)])
    assert seen and all(set(first.controls) in cuts for _, _, _, cuts in seen)
    assert set(second.controls) != set(first.controls)


# ---------------------------------------------------------------- properties


@SETTINGS
@given(seed=st.integers(0, 10_000), rounding=st.booleans(), every=st.integers(1, 2))
def test_property_every_node_cuts_its_incumbent_and_the_optimum_is_certified(seed, rounding, every):
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(slopefree, "PRICE_LIMIT", 0)
        seen = _spy_bounded(mp)
        y, X = panel(seed % 60, p=7)
        best, C = positive_optimum(y, X)
        if C is None:
            return
        out = certify_slope_free(y, X, restarts=3, rounding=rounding, clarabel_every=every)
        assert all(_cuts_the_incumbent(c, G, lam, cuts) for c, G, lam, cuts in seen)
        if out.certified:
            assert set(out.controls) == C
            assert float(out.incumbent) == pytest.approx(best, rel=1e-9)
        else:
            assert out.open
    finally:
        mp.undo()


# ---------------------------------------------------------------- edges


def test_a_ceiling_search_bounds_at_the_ceiling_and_still_cuts_the_incumbent(monkeypatch):
    """Against a ceiling the nodes are bounded at y'y - ceiling, above the incumbent's
    ratio, so the incumbent's cut excludes a design the claim does not need; without
    the group screen the sizes are searched, and the ceiling is held."""
    y, X = _stuck()
    best, _ = positive_optimum(y, X)
    seen = _spy_bounded(monkeypatch)
    out = certify_slope_free(y, X, restarts=5, ceiling=Q(best) * Q(99, 100), group_screen=False)
    assert out.certified and out.ceiling_held
    assert seen and all(lam > out.ratio for _, _, lam, _ in seen)
    assert all(set(out.controls) in cuts for _, _, _, cuts in seen)
