"""The reference key's four gates on the slope-free search, handled lazily (#965).

For a fixed treatment group the RSS-best design also has the highest correlation,
so the reference key's optimum is the lowest-RSS design that passes the four gates.
certify_gated certifies the RSS-best design not yet excluded, tests its gates with
the library's own objective, and excludes it with a no-good cut when it fails.
Every claim is checked against enumeration with the same gates."""
from fractions import Fraction as Q
from itertools import combinations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthDataError

from .brute import gated_optimum, panel, panel_ar
from .certificate import ExactData
from .gated import certify_gated
from .slopefree import Rows, certify_slope_free

SETTINGS = settings(max_examples=12, deadline=None, suppress_health_check=[HealthCheck.too_slow])


@pytest.fixture(autouse=True)
def _bound_every_node(monkeypatch):
    """These tests are of the relaxation and its branching, so every node is
    bounded: on panels this small, pricing would close every size at its root
    (test_pricing.py checks that the two give the same answer)."""
    from . import slopefree
    monkeypatch.setattr(slopefree, "PRICE_LIMIT", 0)
N_TEST = 8


def _lifted(u):
    z = np.concatenate([[1], u]).astype(object)
    return np.array([[Q(int(a * b)) for b in z] for a in z], dtype=object)


# ---------------------------------------------------------------- smoke


def test_smoke_a_design_that_passes_every_gate():
    y, X = panel_ar(1)
    out = certify_gated(y, X, n_test=N_TEST, restarts=10)
    rss, C = gated_optimum(y, X, N_TEST)
    assert out.certified and out.passing and set(out.controls) == C
    assert float(out.rss) == pytest.approx(rss, rel=1e-9)


# ---------------------------------------------------------------- unit


@pytest.mark.parametrize("seed", [1, 7])
def test_failing_designs_are_excluded_in_order(seed):
    """On these panels the RSS-best design fails Durbin-Watson: it is excluded, and
    the next certified design is the next by RSS, until one passes."""
    y, X = panel_ar(seed)
    out = certify_gated(y, X, n_test=N_TEST, restarts=10)
    rss, C = gated_optimum(y, X, N_TEST)
    assert out.certified and out.passing and set(out.controls) == C
    assert len(out.rounds) >= 2 and not all(out.rounds[0]["gates"])
    assert [float(r["rss"]) for r in out.rounds] == sorted(float(r["rss"]) for r in out.rounds)


@pytest.mark.parametrize("seed", range(3))
def test_an_excluded_design_is_never_returned(seed):
    """The excluded optimum is cut from every size's relaxation, so in the full
    relaxation its own size closes in at most three nodes (1, 3 and 1 on these
    panels) where branching around it to the leaves takes 8 to 10. The Shor root
    carries the cut but not its products, and the size of seed 1 takes 5 nodes."""
    y, X = panel(seed, p=6)
    best = certify_slope_free(y, X, restarts=10, hull=False, shor_root=False)
    second = certify_slope_free(y, X, restarts=10, hull=False, exclude=[set(best.controls)], shor_root=False)
    assert second.certified and set(second.controls) != set(best.controls)
    assert second.incumbent >= best.incumbent
    assert second.sizes[len(best.controls)]["nodes"] <= 3


def test_a_cut_holds_at_every_size_for_every_other_design():
    """sum_C u - sum_notC u <= |C| - 1 excludes C and no other binary vector, at
    any size (Balas and Jeroslow 1972), so excluded designs are cut at every size."""
    y, X = panel(2, p=5)
    c = ExactData(y, X).c
    C = {0, 2}
    for m in range(1, 6):
        R = Rows(c, m, cuts=[C])
        for D in combinations(range(5), m):
            u = np.zeros(5, dtype=int); u[list(D)] = 1
            if c @ u.astype(object) >= 0:
                assert R.holds(_lifted(u)) == (set(D) != C)


# ---------------------------------------------------------------- properties


@SETTINGS
@given(seed=st.integers(0, 10_000))
def test_property_the_gated_optimum_is_the_enumerated_one(seed):
    y, X = panel_ar(seed % 200, p=5)
    rss, C = gated_optimum(y, X, N_TEST)
    out = certify_gated(y, X, n_test=N_TEST, restarts=5)
    assert out.certified
    if C is None:
        assert not out.passing and out.controls is None
    else:
        assert out.passing and set(out.controls) == C and float(out.rss) == pytest.approx(rss, rel=1e-9)


# ---------------------------------------------------------------- edges and failure


def test_no_design_passes():
    """Seed 2: every design fails a gate; the search ends certified with none."""
    y, X = panel_ar(2)
    assert gated_optimum(y, X, N_TEST)[1] is None
    out = certify_gated(y, X, n_test=N_TEST, restarts=10)
    assert out.certified and not out.passing and out.controls is None


def test_a_failed_correlation_gate_ends_the_search_at_once():
    """The RSS-best remaining design has the highest correlation of all remaining,
    so when it fails the 0.8 floor every other design does too."""
    y, X = panel_ar(0)
    out = certify_gated(y, X, n_test=N_TEST, restarts=10)
    assert len(out.rounds) == 1 and not out.rounds[0]["gates"][0]
    assert out.certified and not out.passing


def test_the_round_limit_leaves_the_search_uncertified():
    y, X = panel_ar(7)
    out = certify_gated(y, X, n_test=N_TEST, restarts=10, max_rounds=1)
    assert not out.certified and len(out.rounds) == 1


def test_a_ceiling_at_zero_ends_the_search_at_once():
    """No design has an RSS below zero, so a caller that needs one is answered,
    certified, before any design is scored."""
    y, X = panel_ar(1)
    out = certify_gated(y, X, n_test=N_TEST, restarts=10, ceiling=Q(0))
    assert out.certified and out.stopped and not out.passing and out.controls is None
    assert out.rounds == []


def test_a_ceiling_ends_the_search_at_the_first_round_above_it():
    """With the ceiling between the first two rounds' RSS, the first round is scored
    and fails, and the search ends at the second, certified."""
    y, X = panel_ar(7)
    full = certify_gated(y, X, n_test=N_TEST, restarts=10)
    assert len(full.rounds) >= 2 and not all(full.rounds[0]["gates"])
    mid = (full.rounds[0]["rss"] + full.rounds[1]["rss"]) / 2
    out = certify_gated(y, X, n_test=N_TEST, restarts=10, ceiling=mid)
    assert out.certified and out.stopped and not out.passing and len(out.rounds) == 1


def test_the_ceiling_reaches_every_rounds_relaxation(monkeypatch):
    """Each round's certificate is asked the narrower question, so a group that
    cannot reach the ceiling closes without its own optimum being proved."""
    from . import gated

    real = gated.certify_slope_free
    seen = []

    def spy(*a, ceiling=None, **kw):
        seen.append(ceiling)
        return real(*a, ceiling=ceiling, **kw)

    monkeypatch.setattr(gated, "certify_slope_free", spy)
    y, X = panel_ar(7)
    full = certify_gated(y, X, n_test=N_TEST, restarts=10)
    c = full.rss * 2
    certify_gated(y, X, n_test=N_TEST, restarts=10, ceiling=c)
    assert seen[-1] == c and len(seen) >= 2


def test_a_ceiling_above_the_passing_design_changes_nothing():
    y, X = panel_ar(7)
    full = certify_gated(y, X, n_test=N_TEST, restarts=10)
    out = certify_gated(y, X, n_test=N_TEST, restarts=10, ceiling=full.rss * 2)
    assert out.passing and not out.stopped and out.controls == full.controls


def test_an_a_a_window_too_long_for_the_scoring_window_is_refused():
    y, X = panel_ar(1, n=10)
    with pytest.raises(MlsynthDataError, match="A/A"):
        certify_gated(y, X, n_test=8, restarts=3)


def test_every_design_excluded_leaves_none():
    """Excluding every positive-slope design leaves nothing to certify."""
    y, X = panel(0, p=2)
    D = ExactData(y, X)
    designs = [set(C) for r in (1, 2) for C in combinations(range(2), r)
               if sum(D.c[j] for j in C) > 0]
    with pytest.raises(ValueError, match="no design"):
        certify_slope_free(y, X, restarts=3, exclude=designs)


def test_branching_never_adopts_an_excluded_leaf(monkeypatch):
    """With a poor incumbent and the optimum excluded, branching reaches the
    optimum's own leaf, which bypasses the relaxation and its cut; the search must
    still return the best design that is not excluded."""
    from . import search
    from .brute import positive_optimum
    y, X = panel(1, p=5)
    best, C = positive_optimum(y, X)
    D = ExactData(y, X)
    ratios = {}
    for r in range(1, 6):
        for T in combinations(range(5), r):
            s = sum(D.c[j] for j in T)
            if s > 0:
                ratios[frozenset(T)] = s * s / sum(D.G[i, j] for i in T for j in T)
    second = max((T for T in ratios if set(T) != C), key=lambda T: ratios[T])
    worst = min((T for T in ratios if len(T) == 1), key=lambda T: ratios[T])
    monkeypatch.setattr(search, "incumbent", lambda y, X, **k: (None, set(worst)))
    out = certify_slope_free(y, X, exclude=[C], hull=False, max_depth=12)
    assert out.certified and set(out.controls) == set(second)


def test_every_design_failing_a_later_gate_is_excluded_in_turn(monkeypatch):
    """When every design passes the correlation floor and fails Durbin-Watson, the
    search excludes them one by one until none is left, and ends certified with none."""
    from . import gated
    monkeypatch.setattr(gated, "gates_of", lambda y, X, C, n: ((True, True, True, False), None))
    y, X = panel(0, p=3)
    D = ExactData(y, X)
    n_positive = sum(1 for r in (1, 2, 3) for T in combinations(range(3), r) if sum(D.c[j] for j in T) > 0)
    out = certify_gated(y, X, n_test=N_TEST, restarts=3)
    assert out.certified and not out.passing and len(out.rounds) == n_positive
    assert len({tuple(r["controls"]) for r in out.rounds}) == n_positive


def test_another_value_error_is_not_read_as_no_design(monkeypatch):
    """Only NoDesignLeft ends the search with none; any other error propagates."""
    from . import gated

    def boom(*a, **k):
        raise ValueError("the certificate's matrix must be dyadic")

    monkeypatch.setattr(gated, "certify_slope_free", boom)
    y, X = panel_ar(1)
    with pytest.raises(ValueError, match="dyadic"):
        certify_gated(y, X, n_test=N_TEST)


def test_a_slope_free_certificate_left_open_leaves_the_round_uncertified(monkeypatch):
    """When the slope-free search ends with nodes open (here the depth limit at the root,
    the descent patched to find only the poorest design), the gated search stops
    uncertified, with no round recorded; the treatment search retries such a group at
    the final ceiling. GeoLift at five geos reached this from a 10-restart pass design."""
    from . import search

    y, X = panel(0, p=5)
    D = ExactData(y, X)
    worst = min((j for j in range(5) if D.c[j] > 0), key=lambda j: D.c[j] ** 2 / D.G[j, j])
    monkeypatch.setattr(search, "incumbent", lambda y, X, **k: (None, {worst}))
    out = certify_gated(y, X, n_test=8, restarts=3, hull=False, max_depth=0)
    assert not out.certified and out.rounds == [] and not out.passing
