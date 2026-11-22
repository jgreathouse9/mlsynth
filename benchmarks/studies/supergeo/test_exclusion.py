"""Pair exclusion (exclusion.py): keeping k pairs of n markets and leaving the rest out.

Two answers to "which k pairs" (roadmap, the matching section):

* nested trimming, Trimmed Match Design's: form the full optimal pairing and drop its worst pair, then the next,
  so the k pairs kept are that pairing's k cheapest (``nested_trim``);
* the cardinality-optimal k pairs, re-paired at each k: for a cost of the difference of scalar totals, disjoint
  neighbour pairs in sorted order chosen by a dynamic program (``sorted_k_pairs``); for any cost, the blossom
  matcher with n - 2k dummy markets joined to every market at cost 0 (``matched_k_pairs``).

The tests check each against brute force over every k-matching, the two exact methods against each other on the
cases they share (CLAUDE.md invariant 8), the cardinality-optimal cost never above the nested one, the two equal
at k = 0 and k = n // 2, an instance where they differ, and the failures.
"""
from itertools import combinations

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from . import exclusion


def _brute_k(W, k):
    """The least cost of k disjoint pairs of range(len(W)), by recursion on the lowest market."""
    n = len(W)
    memo = {}

    def best(rest, j):
        if j == 0:
            return 0.0
        if len(rest) < 2 * j:
            return float("inf")
        key = (rest, j)
        if key in memo:
            return memo[key]
        i, others = rest[0], rest[1:]
        out = best(others, j)                                     # i left out
        for b in others:
            out = min(out, W[i][b] + best(tuple(x for x in others if x != b), j - 1))
        memo[key] = out
        return out
    return best(tuple(range(n)), k)


def _sym(rng, n, lo=0.0, hi=100.0):
    A = rng.uniform(lo, hi, (n, n))
    W = np.triu(A, 1)
    return W + W.T


def _cost(W, pairs):
    return float(sum(W[a, b] for a, b in pairs))


def _is_k_matching(pairs, n, k):
    used = [v for p in pairs for v in p]
    return len(pairs) == k and len(set(used)) == len(used) and all(0 <= v < n for v in used) and \
        all(a < b for a, b in pairs)


# --- nested trimming ----------------------------------------------------------------------------------------------

def test_nested_trim_keeps_the_cheapest_pairs_of_one_pairing():
    """Smoke: pairs of costs 4, 1 and 9 kept in that order: k = 0, 1, 2, 3 cost 0, 1, 5, 14."""
    W = np.zeros((6, 6))
    for (a, b), c in {(0, 1): 4.0, (2, 3): 1.0, (4, 5): 9.0}.items():
        W[a, b] = W[b, a] = c
    costs, order = exclusion.nested_trim(W, [(0, 1), (2, 3), (4, 5)])
    assert list(costs) == [0.0, 1.0, 5.0, 14.0] and order == [(2, 3), (0, 1), (4, 5)]


# --- the cardinality-optimal pairs --------------------------------------------------------------------------------

def test_the_sorted_program_pairs_neighbours():
    """Smoke: totals 0, 2, 3, 5: one pair is (2, 3) at cost 1; two pairs are (0, 2) and (3, 5) at cost 8."""
    costs, pairs = exclusion.sorted_k_pairs(np.array([0.0, 2.0, 3.0, 5.0]))
    assert list(costs) == [0.0, 1.0, 8.0]
    assert pairs[1] == [(1, 2)] and sorted(pairs[2]) == [(0, 1), (2, 3)]


@settings(max_examples=200, deadline=None)
@given(z=st.lists(st.integers(-20, 20), min_size=0, max_size=9).map(lambda v: np.array(v, dtype=float)))
def test_the_sorted_program_is_the_brute_force_optimum_at_every_k(z):
    """Property, over up to 9 integer totals (ties frequent): for every k from 0 to n // 2 the program's pairs are
    a k-matching of disjoint neighbours in sorted order and cost the least over every k-matching."""
    W = (z[:, None] - z[None, :]) ** 2
    costs, pairs = exclusion.sorted_k_pairs(z)
    assert len(costs) == len(z) // 2 + 1 == len(pairs)
    rank = np.empty(len(z), int)
    rank[np.lexsort((np.arange(len(z)), z))] = np.arange(len(z))
    for k in range(len(z) // 2 + 1):
        assert _is_k_matching(pairs[k], len(z), k)
        assert all(abs(rank[a] - rank[b]) == 1 for a, b in pairs[k])
        assert costs[k] == _cost(W, pairs[k]) == _brute_k(W, k)


@settings(max_examples=150, deadline=None)
@given(seed=st.integers(0, 10 ** 6), n=st.integers(0, 9), data=st.data())
def test_the_matcher_with_dummies_is_the_brute_force_optimum(seed, n, data):
    """Property, over general non-negative costs on up to 9 markets and every k: the matcher with n - 2k dummies
    returns a k-matching of the least cost over every k-matching."""
    W = _sym(np.random.default_rng(seed), n)
    k = data.draw(st.integers(0, n // 2))
    pairs, cost = exclusion.matched_k_pairs(W, k)
    assert _is_k_matching(pairs, n, k)
    assert cost == pytest.approx(_cost(W, pairs)) and cost == pytest.approx(_brute_k(W, k), rel=1e-9, abs=1e-9)


@settings(max_examples=60, deadline=None)
@given(seed=st.integers(0, 10 ** 6), n=st.integers(2, 40), data=st.data())
def test_the_two_exact_methods_agree_on_scalar_totals(seed, n, data):
    """Property (invariant 8, the two solvers on the cases they share): on up to 40 lognormal totals and any k,
    the sorted program's cost equals the matcher's on the same cost matrix."""
    z = np.random.default_rng(seed).lognormal(0, 1.2, n) * 100
    W = (z[:, None] - z[None, :]) ** 2
    k = data.draw(st.integers(0, n // 2))
    costs, _ = exclusion.sorted_k_pairs(z)
    assert exclusion.matched_k_pairs(W, k)[1] == pytest.approx(costs[k], rel=1e-9, abs=1e-6)


# --- the two answers compared -------------------------------------------------------------------------------------

@settings(max_examples=60, deadline=None)
@given(seed=st.integers(0, 10 ** 6), n=st.integers(0, 30))
def test_the_cardinality_optimal_cost_never_exceeds_the_nested_one(seed, n):
    """Property: at every k the re-paired optimum costs no more than nested trimming of the full optimal pairing,
    and the two are equal at k = 0 and k = n // 2 (nothing kept, and the full pairing itself)."""
    z = np.random.default_rng(seed).lognormal(0, 1.2, n) * 100
    W = (z[:, None] - z[None, :]) ** 2
    opt, pairs = exclusion.sorted_k_pairs(z)
    nested, _ = exclusion.nested_trim(W, pairs[n // 2])
    assert np.all(opt <= nested * (1 + 1e-12) + 1e-9)
    assert opt[0] == nested[0] == 0.0 and opt[-1] == pytest.approx(nested[-1], rel=1e-12)


def test_the_two_answers_differ_when_the_best_pair_is_not_in_the_full_pairing():
    """Unit: totals 0, 2, 3, 5 pair fully as (0, 2) and (3, 5); trimming one keeps a pair of cost 4, while the best
    single pair is (2, 3) at cost 1, which the full pairing does not contain."""
    z = np.array([0.0, 2.0, 3.0, 5.0])
    W = (z[:, None] - z[None, :]) ** 2
    opt, pairs = exclusion.sorted_k_pairs(z)
    nested, _ = exclusion.nested_trim(W, pairs[2])
    assert nested[1] == 4.0 and opt[1] == 1.0


# --- edges and failures -------------------------------------------------------------------------------------------

def test_no_markets_and_one_market():
    """Edge: no markets keep no pairs; one market keeps none either."""
    assert [list(x) for x in exclusion.sorted_k_pairs(np.zeros(0))] == [[0.0], [[]]]
    costs, pairs = exclusion.sorted_k_pairs(np.array([7.0]))
    assert list(costs) == [0.0] and pairs == [[]]
    assert exclusion.matched_k_pairs(np.zeros((1, 1)), 0) == ([], 0.0)


@pytest.mark.parametrize("k", [-1, 3, 1.5, True])
def test_a_k_outside_zero_to_n_over_two_is_rejected(k):
    """Failure: k must be an integer from 0 to n // 2; anything else raises ValueError naming k."""
    with pytest.raises(ValueError, match="k"):
        exclusion.matched_k_pairs(_sym(np.random.default_rng(0), 5), k)


def test_nested_trim_rejects_pairs_that_are_not_a_matching():
    """Failure: nested trimming starts from a pairing; repeated markets raise ValueError."""
    with pytest.raises(ValueError, match="matching"):
        exclusion.nested_trim(_sym(np.random.default_rng(1), 4), [(0, 1), (1, 2)])
