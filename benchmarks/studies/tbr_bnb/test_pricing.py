"""A subtree priced design by design instead of bounded (#965).

A node of the slope-free search whose subtree holds few designs is cheaper to
price than to bound, as in MAREX's branch and bound (search_skeleton.
cheaper_to_price): each design's ratio (c'u)^2 / u'Gu is computed in doubles for
the whole subtree at once with a rigorous upper bound on its rounding, a design
is discarded only when that bound is at or below the incumbent's ratio, and every
design it cannot discard is evaluated exactly before it is adopted."""
from fractions import Fraction as Q
from itertools import combinations
from math import comb

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from .brute import panel, positive_optimum
from .certificate import ExactData
from .slopefree import (SlopeFreeCertificate, _branch, _ratio, certify_slope_free, price_subtree,
                        ratio_upper)

SETTINGS = settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def _subtree(p, m, fixes):
    ins = {j for j, v in fixes if v}; outs = {j for j, v in fixes if not v}
    free = [j for j in range(p) if j not in ins | outs]
    for S in combinations(free, m - len(ins)):
        yield ins | set(S)


def _best(c, G, p, m, fixes, lam, excluded=()):
    best = None
    for C in _subtree(p, m, fixes):
        cu, r = _ratio(c, G, C)
        if cu > 0 and r > lam and C not in excluded and (best is None or r > best[0]):
            best = (r, C)
    return best


def _run(c, G, m, fixes, lam, excluded=()):
    inc = {"lam": lam, "C": set()}
    out = SlopeFreeCertificate(incumbent=Q(0), controls=[], ratio=Q(0), sizes_after_hull=[])
    price_subtree(c, G, out, m, list(fixes), inc, list(excluded))
    return out, inc


# ---------------------------------------------------------------- smoke


def test_smoke_a_priced_node_finds_its_best_design():
    y, X = panel(0, p=6)
    D = ExactData(y, X)
    out, inc = _run(D.c, D.G, 3, [], Q(0))
    best = _best(D.c, D.G, 6, 3, [], Q(0))
    assert inc["C"] == best[1] and inc["lam"] == best[0]
    assert out.priced == comb(6, 3) and out.priced_nodes == 1


# ---------------------------------------------------------------- unit


@pytest.mark.parametrize("seed,m,fixes", [(1, 2, []), (2, 3, [(0, 1)]), (3, 3, [(1, 0), (4, 1)]),
                                          (4, 4, [(5, 0)])])
def test_a_priced_node_agrees_with_enumeration(seed, m, fixes):
    y, X = panel(seed, p=6)
    D = ExactData(y, X)
    inc_r, _ = (positive_optimum(y, X)[0], None)
    lam = Q(0)
    out, inc = _run(D.c, D.G, m, fixes, lam)
    best = _best(D.c, D.G, 6, m, fixes, lam)
    if best is None:
        assert inc["C"] == set()
    else:
        assert inc["C"] == best[1] and inc["lam"] == best[0]


def test_a_priced_node_keeps_an_incumbent_that_nothing_beats():
    y, X = panel(1, p=6)
    D = ExactData(y, X)
    top = _best(D.c, D.G, 6, 3, [], Q(0))
    out, inc = _run(D.c, D.G, 3, [], top[0])          # the subtree's own best is the incumbent
    assert inc["C"] == set() and inc["lam"] == top[0]
    assert out.exact <= 1                             # only the tie, at most, is evaluated exactly


def test_an_excluded_design_is_not_adopted_by_pricing():
    y, X = panel(2, p=6)
    D = ExactData(y, X)
    top = _best(D.c, D.G, 6, 3, [], Q(0))
    out, inc = _run(D.c, D.G, 3, [], Q(0), excluded=[top[1]])
    second = _best(D.c, D.G, 6, 3, [], Q(0), excluded=[top[1]])
    assert inc["C"] == second[1] and inc["C"] != top[1]


@given(seed=st.integers(0, 10_000), m=st.integers(1, 5), k=st.integers(0, 3))
@SETTINGS
def test_property_pricing_matches_enumeration(seed, m, k):
    y, X = panel(seed % 300, p=6)
    D = ExactData(y, X)
    rng = np.random.default_rng(seed)
    fixes = [(int(j), int(rng.integers(0, 2))) for j in rng.choice(6, size=k, replace=False)]
    ones = sum(v for _, v in fixes)
    if ones > m or k - ones > 6 - m:
        return
    lam = Q(0) if seed % 2 else _ratio(D.c, D.G, {int(rng.integers(0, 6))})[1]
    out, inc = _run(D.c, D.G, m, fixes, lam)
    best = _best(D.c, D.G, 6, m, fixes, lam)
    assert (inc["C"] == set()) if best is None else (inc["C"] == best[1] and inc["lam"] == best[0])


@given(scale=st.floats(1e-6, 1e9), seed=st.integers(0, 10_000))
@SETTINGS
def test_property_the_ratio_bound_is_above_the_exact_ratio(scale, seed):
    """The double bound is at or above every design's exact ratio, on data with
    heavy cancellation: c of both signs and nearly collinear columns."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=40)
    X = np.column_stack([base + 1e-7 * rng.normal(size=40) for _ in range(7)]) * scale
    y = (X[:, 0] - X[:, 1] + 1e-3 * rng.normal(size=40)) * 1.0
    D = ExactData(y, X)
    cf = np.array([float(v) for v in D.c]); Gf = np.array([[float(v) for v in r] for r in D.G])
    for C in combinations(range(7), 3):
        cu, r = _ratio(D.c, D.G, set(C))
        ub = ratio_upper(D.c, D.G, cf, Gf, set(), list(C))
        if cu > 0:
            assert Q(ub) >= r


def test_the_ratio_bound_has_no_positive_slope_where_the_sum_is_negative():
    y, X = panel(0, p=6)
    D = ExactData(y, X)
    cf = np.array([float(v) for v in D.c]); Gf = np.array([[float(v) for v in r] for r in D.G])
    neg = [j for j in range(6) if D.c[j] < 0]
    if neg:
        assert ratio_upper(D.c, D.G, cf, Gf, set(), neg[:1]) == -np.inf


# ---------------------------------------------------------------- price or bound


@pytest.mark.parametrize("seed", range(4))
def test_price_or_bound_gives_the_same_certified_answer(monkeypatch, seed):
    from . import slopefree

    y, X = panel(seed, p=6)
    best, C = positive_optimum(y, X)
    answers = []
    for limit in (0, 10 ** 9):                       # never price, price every subtree
        monkeypatch.setattr(slopefree, "PRICE_LIMIT", limit)
        out = certify_slope_free(y, X, restarts=5)
        answers.append(out)
        assert out.certified and set(out.controls) == C
        assert float(out.incumbent) == pytest.approx(best, rel=1e-9)
    assert answers[0].priced == 0 and answers[1].priced > 0


def test_a_subtree_above_the_limit_is_bounded(monkeypatch):
    from . import slopefree

    monkeypatch.setattr(slopefree, "PRICE_LIMIT", 5)
    y, X = panel(0, p=6)
    out = certify_slope_free(y, X, restarts=5, hull=False)
    # sizes 1 and 5 hold 6 designs each, above the limit; nothing at the roots is priced
    assert out.certified and all(d["nodes"] >= 1 for d in out.sizes.values())


def test_the_limit_is_where_pricing_costs_one_bound():
    from mlsynth.utils.search_skeleton import cheaper_to_price

    from . import slopefree

    L = slopefree.PRICE_LIMIT
    assert cheaper_to_price(L, slopefree.PRICE_SECONDS, slopefree.BOUND_SECONDS)
    assert not cheaper_to_price(L * 2 + 1, slopefree.PRICE_SECONDS, slopefree.BOUND_SECONDS)


# ---------------------------------------------------------------- the rounding terms


def _floats_of(c, G):
    return (np.array([float(v) for v in c]), np.array([[float(v) for v in r] for r in G]))


@given(a=st.integers(1, 10 ** 6), b=st.integers(1, 10 ** 6), big=st.integers(10 ** 7, 10 ** 9))
@SETTINGS
def test_property_the_bound_holds_where_the_numerator_cancels(a, b, big):
    """c'u = (big + a/3e5) + (-big + b/7e5) is about 1e-5 while its terms are about
    1e8, so each term's rounding (an ulp of 1e8 is 1.5e-8) is some 1e-3 of the sum:
    far beyond the 1e-6 inflation, so the sum's own error term has to carry it."""
    c = np.array([Q(big) + Q(a, 300000), -Q(big) + Q(b, 700000), Q(1)], dtype=object)
    G = np.array([[Q(1), Q(0), Q(0)], [Q(0), Q(1), Q(0)], [Q(0), Q(0), Q(1)]], dtype=object)
    cf, Gf = _floats_of(c, G)
    cu, r = _ratio(c, G, {0, 1})
    ub = ratio_upper(c, G, cf, Gf, set(), [0, 1])
    assert cu > 0 and Q(ub) >= r


@given(a=st.integers(1, 10 ** 6), b=st.integers(1, 10 ** 6), e=st.integers(1, 10 ** 6),
       big=st.integers(10 ** 7, 10 ** 9))
@SETTINGS
def test_property_the_bound_holds_where_the_gram_sum_cancels(a, b, e, big):
    """u'Gu = G00 + G11 + 2 G01 with G00, G11 about big and G01 about -big: the sum is
    about 1e-5 while its terms are about 1e8, so the Gram sum's own error term has to
    carry its rounding."""
    c = np.array([Q(1), Q(1), Q(0)], dtype=object)
    G = np.array([[Q(big) + Q(a, 300000), -Q(big) + Q(e, 500000), Q(0)],
                  [-Q(big) + Q(e, 500000), Q(big) + Q(b, 700000), Q(0)],
                  [Q(0), Q(0), Q(1)]], dtype=object)
    cf, Gf = _floats_of(c, G)
    cu, r = _ratio(c, G, {0, 1})
    ub = ratio_upper(c, G, cf, Gf, set(), [0, 1])
    assert Q(ub) >= r


def test_the_best_bound_is_evaluated_first():
    """From a weak incumbent every positive design is a candidate; taken best bound
    first, the first exact value already clears every later bound, so only a design
    or two is evaluated exactly, not the whole subtree."""
    y, X = panel(0, p=6)
    D = ExactData(y, X)
    positive = sum(1 for C in _subtree(6, 3, []) if _ratio(D.c, D.G, C)[0] > 0)
    out, inc = _run(D.c, D.G, 3, [], Q(0))
    assert positive >= 5 and out.exact <= 2
