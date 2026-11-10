"""The certified search returns the enumerated optimum, certified, on panels small
enough to enumerate; and its parts do what they say."""
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from .brute import panel, positive_optimum
from .certificate import ExactData
from .search import Certificate, Incumbent, SearchStats, certify, incumbent, pieces_to, run_node, stalled


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_smoke_certify_returns_a_certified_optimum(seed):
    y, X = panel(seed, p=5)
    out = certify(y, X, ratio=2.0, restarts=20)
    assert isinstance(out, Certificate) and out.certified
    best, C = positive_optimum(y, X)
    assert float(out.incumbent) == pytest.approx(best, rel=1e-9)
    assert set(out.controls) == C


@settings(max_examples=6, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10_000))
def test_the_certified_value_is_the_enumerated_positive_slope_optimum(seed):
    y, X = panel(seed, p=4, n=24)
    out = certify(y, X, ratio=2.0, restarts=10)
    best, _ = positive_optimum(y, X)
    assert out.certified and float(out.incumbent) == pytest.approx(best, rel=1e-9)


def test_the_stages_only_narrow():
    y, X = panel(3, p=5)
    out = certify(y, X, ratio=2.0, restarts=20)
    assert out.pairs >= out.after_hull >= out.after_shor
    assert out.search.open == [] and out.search.cert_failed == 0


def test_the_pieces_cover_zero_to_past_the_cap():
    pcs = pieces_to(0.5, Fraction(40), 1.25)
    assert pcs[0][0] == 0.0 and Fraction(pcs[-1][1]) >= 40
    assert all(a[1] == b[0] for a, b in zip(pcs, pcs[1:]))


def test_the_incumbent_search_finds_the_enumerated_optimum():
    y, X = panel(5, p=6)
    v, C = incumbent(y, X, restarts=30)
    best, Cb = positive_optimum(y, X)
    assert v == pytest.approx(best, rel=1e-12) and C == Cb


def test_the_stall_rule():
    assert not stalled(100, -0.5, 1.0)                      # early: keep going
    assert stalled(900, 0.97, 1.0)                          # 3% short at 900: branch
    assert not stalled(900, 0.995, 1.0)                     # 0.5% short at 900: keep going
    assert stalled(1700, 0.999, 1.0)                        # not pruned by 1700: branch


def test_a_fixing_beyond_the_size_is_infeasible_and_counted():
    y, X = panel(6, p=4)
    stats = SearchStats()
    inc = Incumbent(UB=Fraction(1), C={0}, cuts=[{0}], D=ExactData(y, X))
    run_node(y, X, None, inc, 1.0, 0.1, 1.0, 1, [(0, 1), (1, 1)], None, stats)
    assert stats.infeasible == 1 and stats.nodes == 0


# ---------------------------------------------------------------- edge and failure
def test_a_duplicated_geo_is_refused_for_want_of_a_slope_cap():
    y, X = panel(7, p=4)
    X = np.column_stack([X, X[:, 0]])
    with pytest.raises(ValueError, match="singular"):
        certify(y, X, ratio=2.0, restarts=5)


def test_one_candidate_control_is_certified():
    y, X = panel(8, p=1)
    X = X * np.sign(y @ X)                                  # a positive slope exists
    out = certify(y, X, ratio=2.0, restarts=3)
    assert out.certified and out.controls == [0]


def test_no_positive_slope_is_refused():
    y, X = panel(8, p=1)
    X = -X * np.sign(y @ X)                                 # the only slope is negative
    with pytest.raises(ValueError, match="positive slope"):
        certify(y, X, ratio=2.0, restarts=3)


def test_an_unreachable_node_branches_down_to_the_depth_limit(monkeypatch):
    """A target no bound can clear: the node branches on a geo at each level and
    its leaves are recorded open at the limit, each child warm-started. Adoption
    is switched off, since a rounded design would otherwise replace the
    unreachable incumbent and let the children prune."""
    from .lifts import Lift
    monkeypatch.setattr(Incumbent, "offer", lambda self, *a, **k: None)
    y, X = panel(9, p=4)
    stats = SearchStats()
    inc = Incumbent(UB=Fraction(10) ** 30, C={0}, cuts=[{0}], D=ExactData(y, X))
    run_node(y, X, Lift(y, X, "full"), inc, float(np.linalg.norm(y)) ** 2,
             0.1, 2.0, 2, [], None, stats, max_depth=1)
    assert stats.branched == 1 and len(stats.open) == 2 and stats.nodes == 3
    assert {fx[0][1] for *_, fx in stats.open} == {0, 1}


def test_a_failed_certificate_retries_then_branches(monkeypatch):
    """The double bound clears the target but the certificate does not: the node
    retries with more headroom, then branches instead of pruning."""
    from . import search
    from .lifts import Lift
    y, X = panel(10, p=4)
    _, C = incumbent(y, X, restarts=5)
    x = X[:, sorted(C)].sum(1); b = float((y @ x) / (x @ x))
    UB = ExactData(y, X).rss(C, b) * Fraction(1, 2)            # far below every bound
    monkeypatch.setattr(search, "joint_clears", lambda *a, **k: False)
    stats = SearchStats()
    inc = Incumbent(UB=UB, C=set(C), cuts=[set(C)], D=ExactData(y, X))
    run_node(y, X, Lift(y, X, "full"), inc, float(np.linalg.norm(y)) ** 2,
             0.05, 4.0, len(C), [], None, stats, max_depth=0)
    assert stats.cert_failed == 2 and stats.pruned == 0 and len(stats.open) == 1


# ---------------------------------------------------------------- the incumbent (#965, step 7)
from .hull import positive_rss  # noqa: E402


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10_000), restarts=st.integers(1, 3))
def test_the_incumbent_has_a_positive_slope(seed, restarts):
    """A design whose slope is not positive fails the correlation gate; its best
    positive-slope fit is the limit b -> 0+, an RSS of y'y, so no descent settles on it."""
    y, X = panel(seed, p=6)
    v, C = incumbent(y, X, restarts=restarts, seed=seed)
    x = X[:, sorted(C)].sum(1)
    assert y @ x > 0 and v == pytest.approx(positive_rss(y, X, C), rel=1e-12)


def test_the_positive_rss_of_a_negative_slope_design_is_the_total():
    y, X = panel(1, p=4)
    C = {0}
    if y @ X[:, 0] > 0:
        X = X.copy(); X[:, 0] = -X[:, 0]
    assert positive_rss(y, X, C) == pytest.approx(float(y @ y))


@settings(max_examples=12, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10_000))
def test_a_poor_starting_incumbent_is_improved_by_the_search(seed):
    """From one restart the incumbent can be far from the optimum; the search rounds
    each branched node's primal point to a design, adopts any that beats the
    incumbent with a no-good cut of its own, and still certifies the optimum."""
    y, X = panel(seed, p=6)
    out = certify(y, X, ratio=2.0, restarts=1, seed=seed)
    best, C = positive_optimum(y, X)
    assert out.certified and float(out.incumbent) == pytest.approx(best, rel=1e-9)


def _worst_positive(y, X):
    """The positive-slope control group with the largest RSS, a deliberately poor start."""
    from itertools import combinations
    from .hull import positive_rss
    worst, arg = -1.0, None
    for r in range(1, X.shape[1] + 1):
        for C in combinations(range(X.shape[1]), r):
            x = X[:, list(C)].sum(1)
            if y @ x > 0 and positive_rss(y, X, C) > worst:
                worst, arg = positive_rss(y, X, C), set(C)
    return worst, arg


@pytest.mark.parametrize("seed", range(4))
def test_the_search_replaces_a_poor_incumbent_and_certifies_the_optimum(seed, monkeypatch):
    """Started from the worst positive-slope design, the search rounds its nodes'
    primal points to designs, adopts better ones with cuts of their own, and ends
    at the enumerated optimum, certified."""
    from . import search
    y, X = panel(seed, p=5)
    monkeypatch.setattr(search, "incumbent", lambda y, X, **k: _worst_positive(y, X))
    out = certify(y, X, ratio=2.0, restarts=1)
    best, C = positive_optimum(y, X)
    assert out.search.adopted >= 1
    assert out.certified and float(out.incumbent) == pytest.approx(best, rel=1e-9) and set(out.controls) == C


def test_an_offered_design_is_adopted_only_when_it_is_better():
    y, X = panel(3, p=5)
    D = ExactData(y, X)
    best, C = positive_optimum(y, X)
    worst, W = _worst_positive(y, X)
    x = X[:, sorted(W)].sum(1)
    inc = Incumbent(UB=D.rss(W, float((y @ x) / (x @ x))), C=set(W), cuts=[set(W)], D=D)
    stats = SearchStats()
    inc.offer(y, X, set(W), stats)                           # itself: no change
    assert stats.adopted == 0 and inc.cuts == [set(W)]
    inc.offer(y, X, C, stats)                                # the optimum: adopted, with a cut
    assert stats.adopted == 1 and inc.C == C and inc.cuts == [set(W), C]
    inc.offer(y, X, set(W), stats)                           # worse than the incumbent: refused
    assert stats.adopted == 1 and inc.C == C
