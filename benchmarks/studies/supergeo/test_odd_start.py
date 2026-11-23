"""The starting design for an odd number of markets, and the safe window's size check.

With groups of 2 to 4 and an odd N a design holds an odd number of triples. A triple splits one market against two,
so it balances only where one total is near the sum of two others. The sorted start put its triple on the three
largest markets; on markets of similar size that triple costs about a market's total squared (658,491 of a start
costing 660,085 on a simulated 235-market panel), the heuristic left 160,540 of it in one triple, and the safe
window built from that incumbent needed 1.1e9 rows. The start is now the best design of one triple and pairs
(``ce._start``): for a fixed triple the rest pair as sorted neighbours (the exchange argument), so the start is the
minimum over every triple of its cost plus the sorted pairing of the rest. The window refuses before it allocates
(``ce.safe_window``).
"""
from itertools import combinations

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st, HealthCheck

from . import certify_exact as ce, dive, mitm_exact as mx, test_bcp


def _cost(z, design):
    return sum(dive._sc(z, tuple(g)) for g in design)


def _is_design(design, n, lo=2, hi=4):
    used = sorted(v for g in design for v in g)
    return used == list(range(n)) and all(lo <= len(g) <= hi for g in design)


def _sorted_rest(z, triple):
    """The triple, and the other markets paired as neighbours in sorted order."""
    rest = [int(v) for v in np.argsort(z, kind="stable") if int(v) not in triple]
    return [tuple(triple)] + [tuple(rest[i:i + 2]) for i in range(0, len(rest), 2)]


def _cheapest_triple(z):
    return min(combinations(range(len(z)), 3), key=lambda t: dive._sc(z, t))


# --- the start ----------------------------------------------------------------------------------------------------

def test_the_triple_goes_where_it_balances():
    """Smoke: six markets near 10 and one near 20. The sorted start's triple is the three largest (10.4, 10.5 and
    20.5, cost (20.5 - 20.9)^2 at best); the triple that balances is 20.5 against two near 10.25, and the start uses
    one such triple and pairs the rest."""
    z = np.array([10.0, 10.1, 10.2, 10.3, 10.4, 20.5, 10.5])
    s = ce._start(z, 2, 4)
    assert _is_design(s, 7)
    triples = [g for g in s if len(g) == 3]
    assert len(triples) == 1 and 5 in triples[0]
    assert _cost(z, s) < _cost(z, test_bcp._sorted_pairs(z))


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10 ** 9), m=st.integers(1, 6), kind=st.integers(0, 3))
def test_the_odd_start_is_the_best_design_of_one_triple_and_pairs(seed, m, kind):
    """Property, over odd counts from 3 to 13 and the study's four generators (integer totals with many ties among
    them): the start is a design of one triple and pairs whose cost equals the least, over every triple, of the
    triple's cost plus the sorted pairing of the rest; so it is never dearer than the sorted start (the triple at the
    top) or than the cheapest triple with the rest sorted."""
    n = 2 * m + 1
    z = test_bcp.gen(np.random.default_rng(seed), n, kind)
    s = ce._start(z, 2, 4)
    assert _is_design(s, n) and sum(len(g) == 3 for g in s) == 1 and all(len(g) <= 3 for g in s)
    c = _cost(z, s)
    best = min(_cost(z, _sorted_rest(z, t)) for t in combinations(range(n), 3))
    assert c == pytest.approx(best, rel=1e-12, abs=1e-12)
    assert c <= _cost(z, test_bcp._sorted_pairs(z)) * (1 + 1e-12) + 1e-12
    assert c <= _cost(z, _sorted_rest(z, _cheapest_triple(z))) * (1 + 1e-12) + 1e-12


@settings(max_examples=100, deadline=None)
@given(seed=st.integers(0, 10 ** 9), m=st.integers(1, 6))
def test_the_start_holds_the_cheapest_triple_on_similar_markets(seed, m):
    """Property: on markets in [100, 110] with one in [200, 220], a triple without the large market costs at least
    90^2 and so does the large market in a pair, while a triple holding it costs at most 20^2 plus at most 10^2 per
    pair. The start's triple holds the large market, and costs no more than the whole design built on the cheapest
    triple (brute force over every triple). The start's triple need not be the cheapest one: a dearer triple can
    leave a cheaper pairing (seed 94: 13.9 against 1.19, cheaper in total)."""
    rng = np.random.default_rng(seed)
    n = 2 * m + 1
    z = np.concatenate((rng.uniform(100, 110, n - 1), [rng.uniform(200, 220)]))
    s = ce._start(z, 2, 4)
    tri = [g for g in s if len(g) == 3][0]
    assert n - 1 in tri
    assert dive._sc(z, tuple(tri)) <= _cost(z, _sorted_rest(z, _cheapest_triple(z))) + 1e-9


@settings(max_examples=150, deadline=None)
@given(seed=st.integers(0, 10 ** 9), m=st.integers(1, 5))
def test_the_odd_start_on_totals_of_either_sign(seed, m):
    """Property: on totals of either sign (a centred panel), where the best singleton of a triple need not be its
    largest market, the start still equals the brute-force best design of one triple and pairs."""
    n = 2 * m + 1
    z = np.random.default_rng(seed).normal(0, 10, n)
    s = ce._start(z, 2, 4)
    assert _is_design(s, n)
    best = min(_cost(z, _sorted_rest(z, t)) for t in combinations(range(n), 3))
    assert _cost(z, s) == pytest.approx(best, rel=1e-12, abs=1e-12)


def test_an_even_count_keeps_the_sorted_pairs():
    """Unit: an even count has no triple, and the start is the sorted pairs as before."""
    z = np.random.default_rng(5).lognormal(0, 1, 10)
    assert ce._start(z, 2, 4) == test_bcp._sorted_pairs(z)


def test_three_and_five_markets():
    """Edge: three markets are one triple; five are a triple and a pair."""
    z3 = np.array([1.0, 2.0, 3.0])
    assert [sorted(g) for g in ce._start(z3, 2, 4)] == [[0, 1, 2]]
    z5 = np.array([1.0, 2.0, 3.0, 7.0, 7.5])
    s = ce._start(z5, 2, 4)
    assert _is_design(s, 5) and sorted(len(g) for g in s) == [2, 3]
    assert _cost(z5, s) == pytest.approx(dive._sc(z5, (0, 1, 2)) + dive._sc(z5, (3, 4)))   # 0 + 0.25: 3 = 1 + 2


def test_other_limits_keep_the_blocks():
    """Unit: with lo = 3 the start is the sorted markets cut into the fewest blocks, as before."""
    z = np.arange(9.0)
    s = ce._start(z, 3, 4)
    assert _is_design(s, 9, 3, 4) and [len(g) for g in s] == [3, 3, 3]


def test_the_certificate_still_proves_odd_instances():
    """Unit: from the new start, the staged certificate proves the brute-force optimum on odd counts of similar
    markets with one near double."""
    for seed in range(6):
        rng = np.random.default_rng(seed)
        z = np.concatenate((rng.uniform(100, 110, 8), [rng.uniform(200, 220)]))
        out = ce.certify_staged(z, heuristic_seconds=0.0)
        assert out["proved"] and out["opt"] == pytest.approx(test_bcp.brute(z), rel=1e-9, abs=1e-12)


# --- the window's size check --------------------------------------------------------------------------------------

def _root_inputs(n=12, seed=0):
    z = test_bcp.gen(np.random.default_rng(seed), n, 0)
    zr, scale, pi, tau = ce._root(z, 2, 4)
    return zr, pi, tau


def test_a_window_over_the_cap_is_refused_before_it_is_built(monkeypatch):
    """Failure: a window whose rows exceed the cap raises WindowTooLarge, a MemoryError naming the row count, the
    cap and the memory the rows need, and the rows are never allocated."""
    z, pi, tau = _root_inputs()
    full = len(ce.safe_window(z, pi, 2, 4, 1e30))
    def no_fill(*a, **k):
        raise AssertionError("the rows were allocated")
    monkeypatch.setattr(mx, "_window_fill", no_fill)
    with pytest.raises(ce.WindowTooLarge, match=r"rows.*cap 10.*GiB") as e:
        ce.safe_window(z, pi, 2, 4, 1e30, cap=10)
    assert isinstance(e.value, MemoryError) and e.value.rows >= full


def test_a_window_at_the_cap_is_built():
    """Unit: the cap is inclusive; a window of exactly the cap's rows (before de-duplication) is built and equals
    the uncapped one."""
    z, pi, tau = _root_inputs()
    rows = mx._window_counts(*ce._window_args(z, pi, 2, 4, 1e30)).sum()
    W = ce.safe_window(z, pi, 2, 4, 1e30, cap=int(rows))
    assert np.array_equal(W, ce.safe_window(z, pi, 2, 4, 1e30))


def test_the_certificate_names_the_incumbent_and_the_bound_when_the_window_is_refused(monkeypatch):
    """Failure: when the first window is refused, certify_staged raises WindowTooLarge with the incumbent's cost and
    the root bound in its message, the two numbers whose gap set the window's threshold."""
    z = test_bcp.gen(np.random.default_rng(3), 12, 0)
    monkeypatch.setattr(ce, "WINDOW_CAP", 5)
    poor = test_bcp._sorted_pairs(z)
    with pytest.raises(ce.WindowTooLarge, match=r"incumbent .* bound "):
        ce.certify_staged(z, incumbent=poor, heuristic_seconds=0.0)
