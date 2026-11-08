"""Rounding a node's relaxed point to a design (#965).

At a node that solves the relaxations and stays open, the relaxed point ``u`` is
rounded to a design of the node's size: the controls fixed in, and the free controls
with the largest relaxed values. The design is evaluated exactly, and one better than
the incumbent replaces it, so the rest of the tree is searched at the higher ratio.
The rounding never prunes a node by itself: what is proved is unchanged, and only the
threshold the later nodes are bounded against rises. Huang et al. (SoCS 2025) fix
the most decided variables of a relaxation the same way (Relax-Search, Algorithm 2)."""
from fractions import Fraction as Q

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthConfigError

from . import slopefree
from .brute import geo_panel, panel, positive_optimum
from .certificate import ExactData
from .slopefree import certify_slope_free, rounded_design
from .treatment import _split

SETTINGS = settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])


@pytest.fixture(autouse=True)
def _bound_every_node(monkeypatch):
    """Every node is bounded, so the search branches on a 26-control group."""
    monkeypatch.setattr(slopefree, "PRICE_LIMIT", 0)


@pytest.fixture(scope="module")
def bed_group():
    """30 simulated units, 20 periods, treatment (0, 6, 10, 15, 16, 18, 19, 22, 23, 28):
    a group whose certificate branches and whose start design rounding improves."""
    return _split(geo_panel(0, n=20, p=30), (0, 6, 10, 15, 16, 18, 19, 22, 23, 28))[:2]


# ---------------------------------------------------------------- smoke


def test_smoke_rounding_shrinks_the_tree_and_keeps_the_design(bed_group):
    y, X = bed_group
    plain = certify_slope_free(y, X, restarts=5, rounding=False)
    rounded = certify_slope_free(y, X, restarts=5)          # rounding is on by default
    assert plain.certified and rounded.certified
    assert (rounded.incumbent, rounded.controls) == (plain.incumbent, plain.controls)
    assert rounded.rounded > 0
    assert rounded.nodes < plain.nodes


# ---------------------------------------------------------------- unit


def test_the_rounded_design_takes_the_largest_free_values():
    u = np.array([0.9, 0.1, 0.6, 0.4, 0.7])
    assert rounded_design(u, [], 2) == {0, 4}
    assert rounded_design(u, [(0, 0)], 2) == {4, 2}          # 0 is fixed out
    assert rounded_design(u, [(1, 1)], 2) == {1, 0}          # 1 is fixed in, whatever its value


def test_ties_go_to_the_lower_index():
    u = np.array([0.5, 0.5, 0.5, 0.5])
    assert rounded_design(u, [], 2) == {0, 1}
    assert rounded_design(u, [(0, 0)], 2) == {1, 2}


def test_a_better_rounded_design_becomes_the_incumbent(monkeypatch, bed_group):
    """Each improvement is a rounded design, and the last one found is the result's
    design unless a leaf beat it later."""
    y, X = bed_group
    real = slopefree.rounded_design
    taken = []

    def spy(u, fixes, m):
        C = real(u, fixes, m)
        taken.append(frozenset(C))
        return C

    monkeypatch.setattr(slopefree, "rounded_design", spy)
    out = certify_slope_free(y, X, restarts=5)
    assert out.rounded > 0 and len(taken) >= out.rounded
    D = ExactData(y, X)
    for C in taken:
        cu, r = slopefree._ratio(D.c, D.G, C)
        assert cu <= 0 or r <= out.ratio                     # no rounded design beats the result


def _worst_design(D, m):
    """The positive-slope design of size m with the smallest ratio, and its ratio."""
    from itertools import combinations
    p = len(D.c)
    scored = [(r, set(C)) for C in combinations(range(p), m)
              for cu, r in [slopefree._ratio(D.c, D.G, set(C))] if cu > 0]
    return min(scored, key=lambda t: t[0])


@pytest.mark.parametrize("above, taken", [(0, False), (1, True)])
def test_only_a_strictly_better_rounded_design_is_taken(monkeypatch, above, taken):
    """A rounded design at the incumbent's ratio leaves it; one just above replaces it."""
    y, X = panel(5, p=8)
    D = ExactData(y, X)
    r0, C0 = _worst_design(D, 3)
    monkeypatch.setattr(slopefree, "rounded_design", lambda u, fixes, m: set(C0))
    out = slopefree.SlopeFreeCertificate(incumbent=Q(0), controls=[], ratio=Q(0), sizes_after_hull=[])
    lam = r0 - Q(1, 10 ** 12) if above else r0
    inc = {"lam": lam, "C": {"start"}, "every": 1, "round": True}
    slopefree._branch(D.c, D.G, out, 3, [], [], 0, 400, 0, 0, inc, [])
    assert out.rounded == int(taken)
    assert inc["C"] == (C0 if taken else {"start"})
    assert inc["lam"] == r0                                  # taken, or the incumbent's own


def test_a_negative_slope_rounded_design_is_never_taken(monkeypatch):
    """A design whose slope is not positive has no RSS in the slope-free search:
    _ratio scores it 0, which never beats the threshold, at least 0, so it never
    replaces the incumbent."""
    from itertools import combinations
    y, X = panel(5, p=8)
    D = ExactData(y, X)
    neg = next(set(C) for C in combinations(range(8), 3) if slopefree._ratio(D.c, D.G, set(C))[0] <= 0)
    assert slopefree._ratio(D.c, D.G, neg)[1] == 0
    monkeypatch.setattr(slopefree, "rounded_design", lambda u, fixes, m: set(neg))
    out = slopefree.SlopeFreeCertificate(incumbent=Q(0), controls=[], ratio=Q(0), sizes_after_hull=[])
    inc = {"lam": Q(10) ** -9, "C": {"start"}, "every": 1, "round": True}
    slopefree._branch(D.c, D.G, out, 3, [], [], 0, 400, 0, 0, inc, [])
    assert out.rounded == 0 and inc["C"] == {"start"}


def test_an_excluded_rounded_design_is_never_taken(monkeypatch):
    y, X = panel(5, p=8)
    D = ExactData(y, X)
    out = slopefree.SlopeFreeCertificate(incumbent=Q(0), controls=[], ratio=Q(0), sizes_after_hull=[])
    C = {0, 1, 2}
    monkeypatch.setattr(slopefree, "rounded_design", lambda u, fixes, m: set(C))
    inc = {"lam": Q(0), "C": set(), "every": 1, "round": True}
    slopefree._branch(D.c, D.G, out, 3, [], [], 0, 400, 0, 1, inc, [set(C)])
    assert inc["C"] != C and out.rounded == 0


def test_rounding_off_rounds_nothing(monkeypatch, bed_group):
    y, X = bed_group
    calls = []
    monkeypatch.setattr(slopefree, "rounded_design", lambda *a: calls.append(a) or set())
    out = certify_slope_free(y, X, restarts=5, rounding=False)
    assert calls == [] and out.rounded == 0


def test_only_a_node_that_solved_the_relaxations_rounds(monkeypatch, bed_group):
    """An inherited node rounds nothing: its parent's point was rounded already."""
    y, X = bed_group
    real_size = slopefree.size_certificate
    relaxed = []

    def size(c, G, lam, m, cuts=(), fixes=(), **k):
        res = real_size(c, G, lam, m, cuts=cuts, fixes=fixes, **k)
        relaxed.append(k.get("relaxations", True) and not res.certified and res.u is not None)
        return res

    calls = []
    real_round = slopefree.rounded_design

    def spy(u, fixes, m):
        calls.append(len(fixes))
        return real_round(u, fixes, m)

    monkeypatch.setattr(slopefree, "size_certificate", size)
    monkeypatch.setattr(slopefree, "rounded_design", spy)
    certify_slope_free(y, X, restarts=5, clarabel_every=2)
    assert len(calls) == sum(relaxed) > 0
    assert all(d % 2 == 0 for d in calls)


# ---------------------------------------------------------------- properties


@SETTINGS
@given(u=st.lists(st.floats(0, 1), min_size=3, max_size=12), data=st.data())
def test_property_the_rounded_design_respects_the_fixings_and_dominates(u, data):
    u = np.array(u)
    p = len(u)
    fixed = data.draw(st.lists(st.integers(0, p - 1), unique=True, max_size=p - 1))
    fixes = [(j, data.draw(st.integers(0, 1))) for j in fixed]
    ones = sum(v for _, v in fixes)
    free = [j for j in range(p) if j not in dict(fixes)]
    m = data.draw(st.integers(ones, ones + len(free)))
    C = rounded_design(u, fixes, m)
    assert len(C) == m
    assert {j for j, v in fixes if v} <= C
    assert not {j for j, v in fixes if not v} & C
    taken, left = [j for j in free if j in C], [j for j in free if j not in C]
    assert all(u[a] >= u[b] for a in taken for b in left)


@SETTINGS
@given(seed=st.integers(0, 10_000), rounding=st.booleans(), every=st.integers(1, 3))
def test_property_rounding_certifies_the_enumerated_optimum(seed, rounding, every):
    """Rounding changes the threshold the later nodes face, never what is proved."""
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(slopefree, "PRICE_LIMIT", 0)
        y, X = panel(seed % 60, p=6)
        best, C = positive_optimum(y, X)
        if C is None:
            return
        out = certify_slope_free(y, X, restarts=3, rounding=rounding, clarabel_every=every)
        if out.certified:
            assert set(out.controls) == C
            assert float(out.incumbent) == pytest.approx(best, rel=1e-9)
        else:
            assert out.open
    finally:
        mp.undo()


# ---------------------------------------------------------------- edges


def test_every_free_control_in_is_the_whole_free_set():
    u = np.array([0.2, 0.3, 0.1])
    assert rounded_design(u, [(1, 0)], 2) == {0, 2}


def test_a_node_without_a_point_rounds_nothing(monkeypatch):
    """When the relaxations fail the node has no point, and nothing is rounded."""
    y, X = panel(0, p=8)
    D = ExactData(y, X)
    calls = []
    monkeypatch.setattr(slopefree, "rounded_design", lambda *a: calls.append(a) or set())

    def fail(c, G, lam, R):
        raise slopefree.cp.error.SolverError("injected")

    monkeypatch.setattr(slopefree, "solve", fail)
    monkeypatch.setattr(slopefree, "dual_multipliers", lambda *a, **k: (None, None))
    out = slopefree.SlopeFreeCertificate(incumbent=Q(0), controls=[], ratio=Q(0), sizes_after_hull=[])
    inc = {"lam": Q(10) ** -9, "C": set(), "every": 1, "round": True}
    slopefree._branch(D.c, D.G, out, 3, [], [], 0, 400, 0, 0, inc, [])
    assert calls == [] and out.rounded == 0


# ---------------------------------------------------------------- failures


@pytest.mark.parametrize("rounding", [0, 1, "yes", None])
def test_a_malformed_rounding_flag_is_refused(rounding):
    y, X = panel(0, p=4)
    with pytest.raises(MlsynthConfigError, match="rounding"):
        certify_slope_free(y, X, restarts=3, rounding=rounding)
