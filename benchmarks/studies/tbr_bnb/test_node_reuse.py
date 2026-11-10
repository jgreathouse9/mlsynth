"""Branching on the parent's relaxation point (#965).

At a node of the branch and bound that the first-order tiers leave open, the Shor
and full relaxations (Clarabel) are solved, and on the test bed they almost never
close it: on 21 branching groups of 20 simulated markets, none of the 127 nodes
that reached the full relaxation closed there. What the solve buys is the point the
next control is branched on (the one whose relaxed value is nearest a half), and
that point, from the dual's smoothing, doubles the tree. So a node solves the
relaxations only every ``clarabel_every`` levels and in between branches on its
parent's point. On 38 branching groups of 30 simulated units, every second level
took 35-40% fewer Clarabel solves for 5-11% more nodes, and no group certified a
different design."""
from fractions import Fraction as Q

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthConfigError

from . import slopefree
from .brute import geo_panel, panel, positive_optimum
from .certificate import ExactData
from .slopefree import CLARABEL_EVERY, certify_slope_free, objective, size_certificate
from .treatment import _split

SETTINGS = settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])


@pytest.fixture(autouse=True)
def _bound_every_node(monkeypatch):
    """Every node is bounded, so the search branches on a 26-control group."""
    monkeypatch.setattr(slopefree, "PRICE_LIMIT", 0)


@pytest.fixture(scope="module")
def bed_group():
    """30 simulated units, 20 periods, treatment (2, 14, 25, 28), group 22 of s30_k4:
    26 candidate controls, a group whose certificate branches (46 nodes, 16 full
    relaxations). The group used before, treatment (0, 4, 13, 17) of seed 22, needs
    one full relaxation once every incumbent is cut (test_incumbent_cuts.py)."""
    return _split(geo_panel(31, n=20, p=30), (2, 14, 25, 28))[:2]


def _count_solves(monkeypatch):
    real = slopefree.solve
    seen = []

    def solve(c, G, lam, R, solver=None):
        seen.append(R.level)
        return real(c, G, lam, R, solver=solver)

    monkeypatch.setattr(slopefree, "solve", solve)
    return seen


def _record_nodes(monkeypatch):
    """Each size certificate's (depth, relaxations solved) and result."""
    real = slopefree.size_certificate
    nodes = []

    def size(c, G, lam, m, cuts=(), fixes=(), **k):
        res = real(c, G, lam, m, cuts=cuts, fixes=fixes, **k)
        nodes.append((len(fixes), k.get("relaxations", True), res))
        return res

    monkeypatch.setattr(slopefree, "size_certificate", size)
    return nodes


# ---------------------------------------------------------------- smoke


def test_smoke_the_parents_point_saves_solves_and_keeps_the_design(monkeypatch, bed_group):
    y, X = bed_group
    seen = _count_solves(monkeypatch)
    every = certify_slope_free(y, X, restarts=5, clarabel_every=1)
    full_every = seen.count("full")
    seen.clear()
    second = certify_slope_free(y, X, restarts=5, clarabel_every=2)
    assert every.certified and second.certified
    assert (second.incumbent, second.controls) == (every.incumbent, every.controls)
    assert 0 < seen.count("full") < full_every


# ---------------------------------------------------------------- unit


def test_the_default_refreshes_every_second_level():
    assert CLARABEL_EVERY == 2


def test_the_relaxations_run_only_at_the_refresh_levels(monkeypatch, bed_group):
    y, X = bed_group
    nodes = _record_nodes(monkeypatch)
    out = certify_slope_free(y, X, restarts=5, clarabel_every=2)
    assert out.certified and any(d % 2 for d, _, _ in nodes)
    assert all(relax for d, relax, _ in nodes if d % 2 == 0)
    assert not any(relax for d, relax, _ in nodes if d % 2)


def test_every_node_solves_the_relaxations_at_one(monkeypatch, bed_group):
    y, X = bed_group
    nodes = _record_nodes(monkeypatch)
    certify_slope_free(y, X, restarts=5, clarabel_every=1)
    assert nodes and all(relax for _, relax, _ in nodes)


def test_a_size_without_relaxations_solves_nothing(monkeypatch, bed_group):
    """relaxations=False stops after the first-order tiers: no Clarabel solve, and an
    open result carries no point of its own."""
    y, X = bed_group
    D = ExactData(y, X)
    seen = _count_solves(monkeypatch)
    lam = Q(10) ** -9                    # a ratio below every design's: nothing closes
    res = size_certificate(D.c, D.G, lam, 5, fixes=[(0, 1)], relaxations=False)
    assert seen == [] and not res.certified and res.u is None and res.tier == "first-order"


def test_a_size_without_relaxations_still_certifies_by_the_dual():
    y, X = panel(0, p=6)
    D = ExactData(y, X)
    res = size_certificate(D.c, D.G, Q(10) ** 9, 3, fixes=[(0, 1)], relaxations=False)
    assert res.certified and res.tier in ("reuse", "dual")


def test_an_inherited_node_branches_on_its_parents_point(monkeypatch, bed_group):
    """At a node that skips the relaxations the branching control is the free control
    whose value in the parent's point is nearest a half."""
    y, X = bed_group
    real_branch = slopefree._branch
    seen = []

    def branch(c, G, out, m, cuts, fixes, *a, parent_u=None, **k):
        seen.append((list(fixes), None if parent_u is None else np.array(parent_u)))
        return real_branch(c, G, out, m, cuts, fixes, *a, parent_u=parent_u, **k)

    monkeypatch.setattr(slopefree, "_branch", branch)
    certify_slope_free(y, X, restarts=5, clarabel_every=2)
    by_fixes = {tuple(f): u for f, u in seen}
    checked = 0
    for f, u in seen:
        if len(f) == 2 and u is not None:          # a child of a depth-1 node, which inherited
            parent = tuple(f[:1])
            pu = by_fixes.get(parent)
            if pu is None:
                continue
            free = [j for j in range(len(u)) if j not in dict(f[:1])]
            assert f[1][0] == min(free, key=lambda j: abs(pu[j] - 0.5))
            checked += 1
    assert checked > 0


def test_a_node_without_a_parent_point_solves_the_relaxations(monkeypatch, bed_group):
    """When the parent had no point (its solve failed), the child refreshes, whatever
    its level."""
    y, X = bed_group
    nodes = _record_nodes(monkeypatch)
    real = slopefree.solve
    calls = []

    def solve(c, G, lam, R, solver=None):                   # the root's first full solve fails
        if R.level == "full":
            calls.append(1)
            if len(calls) == 1:
                raise slopefree.cp.error.SolverError("injected")
        return real(c, G, lam, R, solver=solver)

    monkeypatch.setattr(slopefree, "solve", solve)
    monkeypatch.setattr(slopefree, "dual_multipliers", lambda *a, **k: (None, None))
    D = ExactData(y, X)
    out = slopefree.SlopeFreeCertificate(incumbent=Q(0), controls=[], ratio=Q(0), sizes_after_hull=[])
    inc = {"lam": Q(10) ** -9, "C": set(), "every": 2}
    slopefree._branch(D.c, D.G, out, 5, [], [], 0, 400, 0, 1, inc, [])
    assert [relax for d, relax, _ in nodes if d == 1] == [True, True]


# ---------------------------------------------------------------- properties


@SETTINGS
@given(seed=st.integers(0, 10_000), every=st.integers(1, 4))
def test_property_any_refresh_rate_certifies_the_enumerated_optimum(seed, every):
    """The refresh rate changes which control is branched on, never what is proved."""
    import pytest as _pytest
    mp = _pytest.MonkeyPatch()
    try:
        mp.setattr(slopefree, "PRICE_LIMIT", 0)
        y, X = panel(seed % 60, p=6)
        best, C = positive_optimum(y, X)
        if C is None:
            return
        out = certify_slope_free(y, X, restarts=3, clarabel_every=every)
        if out.certified:
            assert set(out.controls) == C
            assert float(out.incumbent) == pytest.approx(best, rel=1e-9)
        else:
            assert out.open
    finally:
        mp.undo()


# ---------------------------------------------------------------- edges


def test_the_root_always_solves_the_relaxations(monkeypatch, bed_group):
    y, X = bed_group
    nodes = _record_nodes(monkeypatch)
    certify_slope_free(y, X, restarts=5, clarabel_every=99)
    assert all(relax for d, relax, _ in nodes if d == 0)
    assert any(not relax for d, relax, _ in nodes if d > 0)


def test_a_node_at_the_depth_limit_is_reported_open(monkeypatch, bed_group):
    y, X = bed_group
    out = certify_slope_free(y, X, restarts=5, clarabel_every=2, max_depth=1)
    assert not out.certified and out.open


# ---------------------------------------------------------------- failures


@pytest.mark.parametrize("every", [0, -1, 1.5, "2", True])
def test_a_malformed_refresh_rate_is_refused(every):
    y, X = panel(0, p=4)
    with pytest.raises(MlsynthConfigError, match="clarabel_every"):
        certify_slope_free(y, X, restarts=3, clarabel_every=every)
