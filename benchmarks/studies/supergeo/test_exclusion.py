"""Pair exclusion (exclusion.py): keeping k pairs of n markets and leaving the rest out.

Two answers to "which k pairs" (roadmap, the matching section):

* nested trimming, Trimmed Match Design's: form the full optimal pairing and drop its worst pair, then the next,
  so the k pairs kept are that pairing's k cheapest (``nested_trim``);
* the cardinality-optimal k pairs, re-paired at each k: for a cost of the difference of scalar totals, disjoint
  neighbour pairs in sorted order chosen by a dynamic program (``sorted_k_pairs``); for any cost, the blossom
  matcher with n - 2k dummy markets joined to every market at cost 0 (``matched_k_pairs``).

A design built on them (``design_excluding``) takes the number of pairs to exclude, and markets excluded by hand,
and refuses an exclusion that leaves no design: one that excludes every pair, more pairs than the markets form, or
so many markets that fewer than ``min_pairs`` pairs (or none) can form.

The tests check each against brute force over every k-matching, the two exact methods against each other on the
cases they share (CLAUDE.md invariant 8), the cardinality-optimal cost never above the nested one, the two equal
at k = 0 and k = n // 2, an instance where they differ, and the failures.
"""
from itertools import combinations

import numpy as np
import pandas as pd
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
    """Property, over general costs on up to 9 markets, non-negative or signed, and every k: the matcher with
    n - 2k dummies returns exactly k pairs, of the least cost over every k-matching. Signed costs are where a
    dummy-dummy pair would pay, by freeing two markets to form another pair of negative cost."""
    lo = data.draw(st.sampled_from([0.0, -100.0]))
    W = _sym(np.random.default_rng(seed), n, lo=lo)
    k = data.draw(st.integers(0, n // 2))
    pairs, cost = exclusion.matched_k_pairs(W, k)
    assert _is_k_matching(pairs, n, k)
    assert cost == pytest.approx(_cost(W, pairs)) and cost == pytest.approx(_brute_k(W, k), rel=1e-9, abs=1e-9)


def test_the_matcher_keeps_k_pairs_when_pairs_cost_nothing():
    """Edge: six markets of equal totals, every pair at cost 0; k = 1 and k = 2 keep exactly that many pairs, as
    does the sorted program."""
    W = np.zeros((6, 6))
    for k in (0, 1, 2, 3):
        pairs, cost = exclusion.matched_k_pairs(W, k)
        assert _is_k_matching(pairs, 6, k) and cost == 0.0
        assert _is_k_matching(exclusion.sorted_k_pairs(np.ones(6))[1][k], 6, k)


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


# --- a design after exclusion -------------------------------------------------------------------------------------

def _check_design(d, n, excluded, kept):
    """The design keeps ``kept`` disjoint pairs, none touching a market excluded by hand, and reports every other
    market as excluded."""
    pairs = d["pairs"]
    used = sorted(v for p in pairs for v in p)
    assert _is_k_matching(pairs, n, kept)
    assert not set(used) & set(excluded)
    assert sorted(d["excluded_units"]) == sorted(set(range(n)) - set(used))
    assert set(excluded) <= set(d["excluded_units"])


def test_a_design_excluding_one_pair():
    """Smoke: totals 0, 2, 3, 5 with one pair excluded. Re-pairing keeps markets 1 and 2 (totals 2 and 3, cost 1),
    the best single pair. Nested trimming keeps one pair of the full pairing, whose two pairs tie at cost 4, and
    the tie goes to the first in sorted order: markets 0 and 1."""
    z = np.array([0.0, 2.0, 3.0, 5.0])
    opt = exclusion.design_excluding(totals=z, drop_pairs=1)
    assert opt["pairs"] == [(1, 2)] and opt["cost"] == 1.0 and opt["excluded_units"] == [0, 3]
    nested = exclusion.design_excluding(totals=z, drop_pairs=1, method="nested")
    assert nested["pairs"] == [(0, 1)] and nested["cost"] == 4.0 and nested["excluded_units"] == [2, 3]


def test_no_exclusion_is_the_full_optimal_pairing():
    """Unit: dropping no pairs gives the full optimal pairing under both methods, from totals and from costs."""
    z = np.random.default_rng(3).lognormal(0, 1, 10)
    W = (z[:, None] - z[None, :]) ** 2
    full = exclusion.sorted_k_pairs(z)[0][-1]
    for kw in (dict(totals=z), dict(costs=W)):
        for method in ("optimal", "nested"):
            d = exclusion.design_excluding(**kw, method=method)
            _check_design(d, 10, [], 5)
            assert d["cost"] == pytest.approx(full, rel=1e-12)


def test_markets_excluded_by_hand_are_never_paired():
    """Unit: markets excluded by hand are left out before pairing, and an odd remainder leaves one more out."""
    z = np.arange(9, dtype=float) ** 2
    d = exclusion.design_excluding(totals=z, excluded_units=[0, 4])
    _check_design(d, 9, [0, 4], 3)
    assert len(d["excluded_units"]) == 3


@settings(max_examples=150, deadline=None)
@given(seed=st.integers(0, 10 ** 6), n=st.integers(0, 9), data=st.data())
def test_a_design_exists_exactly_when_enough_pairs_remain(seed, n, data):
    """Property, over up to 9 markets, any markets excluded by hand, any number of pairs dropped (past every pair
    included) and min_pairs from 1 to 3: the design raises ValueError exactly when fewer than min_pairs pairs
    remain, and otherwise keeps (n' // 2 - drop) pairs of the n' markets left: the re-paired ones at the brute-force
    optimum over those markets, the nested ones at no lower cost, and those the cheapest pairs of a full pairing at
    the brute-force optimum."""
    rng = np.random.default_rng(seed)
    W = _sym(rng, n)
    excluded = sorted(data.draw(st.sets(st.integers(0, n - 1), max_size=n))) if n else []
    drop = data.draw(st.integers(0, n // 2 + 2))
    min_pairs = data.draw(st.integers(1, 3))
    rest = [i for i in range(n) if i not in excluded]
    kept = len(rest) // 2 - drop
    if kept < min_pairs:
        with pytest.raises(ValueError, match="pair"):
            exclusion.design_excluding(costs=W, drop_pairs=drop, excluded_units=excluded, min_pairs=min_pairs)
        return
    opt = exclusion.design_excluding(costs=W, drop_pairs=drop, excluded_units=excluded, min_pairs=min_pairs)
    nested = exclusion.design_excluding(costs=W, drop_pairs=drop, excluded_units=excluded, min_pairs=min_pairs,
                                        method="nested")
    for d in (opt, nested):
        _check_design(d, n, excluded, kept)
        assert d["cost"] == pytest.approx(_cost(W, d["pairs"]))
    sub = W[np.ix_(rest, rest)]
    assert opt["cost"] == pytest.approx(_brute_k(sub, kept), rel=1e-9, abs=1e-9)
    assert nested["cost"] >= opt["cost"] - 1e-9
    full = nested["full_pairing"]
    assert opt["full_pairing"] is None
    assert _cost(W, full) == pytest.approx(_brute_k(sub, len(rest) // 2), rel=1e-9, abs=1e-9)
    assert set(nested["pairs"]) <= set(full)
    worst_kept = max(W[a, b] for a, b in nested["pairs"])
    assert all(W[a, b] >= worst_kept for a, b in set(full) - set(nested["pairs"]))


# --- an exclusion that leaves no design ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["optimal", "nested"])
@pytest.mark.parametrize("kw", [dict(totals=np.arange(8.0)), dict(costs=(np.arange(8.0)[:, None] -
                                                                         np.arange(8.0)[None, :]) ** 2)])
def test_excluding_every_pair_is_refused(method, kw):
    """Failure: 8 markets form 4 pairs; excluding all 4 leaves no design, and the error says so and names both
    counts."""
    with pytest.raises(ValueError, match=r"every pair.*4 of 4|4 of 4.*every pair"):
        exclusion.design_excluding(**kw, drop_pairs=4, method=method)


def test_excluding_more_pairs_than_the_markets_form_is_refused():
    """Failure: 8 markets form 4 pairs; excluding 6 is refused with both counts named."""
    with pytest.raises(ValueError, match=r"6 pairs.*only 4"):
        exclusion.design_excluding(totals=np.arange(8.0), drop_pairs=6)


def test_excluding_below_min_pairs_is_refused():
    """Failure: 8 markets, 2 pairs dropped, leave 2 pairs, fewer than min_pairs = 3; at min_pairs = 2 the same
    exclusion is a design (the boundary)."""
    with pytest.raises(ValueError, match=r"leaves 2 pairs.*min_pairs=3"):
        exclusion.design_excluding(totals=np.arange(8.0), drop_pairs=2, min_pairs=3)
    d = exclusion.design_excluding(totals=np.arange(8.0), drop_pairs=2, min_pairs=2)
    _check_design(d, 8, [], 2)


@pytest.mark.parametrize("excluded", [list(range(6)), [0, 1, 2, 3, 4]])
def test_excluding_so_many_markets_that_no_pair_forms_is_refused(excluded):
    """Failure: of 6 markets, excluding all 6, or 5, leaves none or one, and no pair can form."""
    with pytest.raises(ValueError, match="no pair"):
        exclusion.design_excluding(totals=np.arange(6.0), excluded_units=excluded)


def test_excluding_markets_then_every_remaining_pair_is_refused():
    """Failure: excluding 1 of 6 markets leaves 5, which form 2 pairs; dropping both excludes every pair."""
    with pytest.raises(ValueError, match=r"every pair.*2 of 2|2 of 2.*every pair"):
        exclusion.design_excluding(totals=np.arange(6.0), excluded_units=[5], drop_pairs=2)


def test_excluding_so_many_markets_that_too_few_pairs_form_is_refused():
    """Failure: excluding 3 of 8 markets leaves 5, which form 2 pairs, fewer than min_pairs = 3, before any pair
    is dropped."""
    with pytest.raises(ValueError, match=r"5 markets.*2 pairs.*min_pairs=3"):
        exclusion.design_excluding(totals=np.arange(8.0), excluded_units=[0, 1, 2], min_pairs=3)


def test_an_empty_panel_has_no_design():
    """Edge: no markets, and one market, form no pair."""
    for z in (np.zeros(0), np.array([1.0])):
        with pytest.raises(ValueError, match="no pair"):
            exclusion.design_excluding(totals=z)


@pytest.mark.parametrize("kw, match", [
    (dict(drop_pairs=-1), "drop_pairs"),
    (dict(drop_pairs=1.5), "drop_pairs"),
    (dict(drop_pairs=True), "drop_pairs"),
    (dict(min_pairs=0), "min_pairs"),
    (dict(min_pairs=True), "min_pairs"),
    (dict(excluded_units=[1, 1]), "excluded_units"),
    (dict(excluded_units=[8]), "excluded_units"),
    (dict(excluded_units=[-1]), "excluded_units"),
    (dict(excluded_units=[True]), "excluded_units"),
    (dict(excluded_units=[0.5]), "excluded_units"),
    (dict(method="greedy"), "method"),
])
def test_invalid_arguments_are_rejected(kw, match):
    """Failure: each malformed argument raises ValueError naming it."""
    with pytest.raises(ValueError, match=match):
        exclusion.design_excluding(totals=np.arange(8.0), **kw)


def test_costs_or_totals_exactly_one():
    """Failure: the design reads either a cost matrix or scalar totals, not both and not neither; a cost matrix that
    is not square and symmetric is refused."""
    z = np.arange(4.0)
    W = (z[:, None] - z[None, :]) ** 2
    with pytest.raises(ValueError, match="one of"):
        exclusion.design_excluding()
    with pytest.raises(ValueError, match="one of"):
        exclusion.design_excluding(costs=W, totals=z)
    with pytest.raises(ValueError, match="square|symmetric"):
        exclusion.design_excluding(costs=np.triu(W))
    with pytest.raises(ValueError, match="square"):
        exclusion.design_excluding(costs=np.zeros((3, 4)))


def test_nonfinite_inputs_and_pairs_outside_the_matrix_are_rejected():
    """Failure: an infinite cost, a NaN total, a matrix of totals, a pair of a market with itself (a market in two
    pairs) and a pair past the matrix each raise ValueError naming the fault."""
    W = _sym(np.random.default_rng(2), 4)
    W[0, 1] = W[1, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        exclusion.matched_k_pairs(W, 1)
    for z in (np.array([0.0, np.nan, 1.0, 2.0]), np.zeros((2, 2))):
        with pytest.raises(ValueError, match="finite vector"):
            exclusion.design_excluding(totals=z)
    with pytest.raises(ValueError, match="appears in two pairs"):
        exclusion.nested_trim(_sym(np.random.default_rng(1), 4), [(1, 1)])
    for bad in ([(0, 4)], [(-1, 2)]):
        with pytest.raises(ValueError, match="matching of the 4 markets"):
            exclusion.nested_trim(_sym(np.random.default_rng(1), 4), bad)


# --- the comparison script ----------------------------------------------------------------------------------------

def _simulated_frame(n=12, seed=0):
    """A simulated daily panel over 2020 of n geos, lognormal levels with noise, in the example data's columns."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", "2020-12-29")
    level = rng.lognormal(3, 1, n)
    rows = [(g, d, level[g] * (1 + 0.1 * rng.standard_normal())) for g in range(n) for d in dates]
    return pd.DataFrame(rows, columns=["geo", "date", "response"])


def test_the_comparison_tables_on_a_simulated_panel():
    """Smoke and unit: on 12 simulated geos the in-sample table has a row per drop with nested never cheaper than
    re-paired under either objective, both equal with nothing dropped; the rolling table has one value per window,
    every one with the full pairing at 1 and finite."""
    from . import exclusion_tm as tm
    P = tm.panel(_simulated_frame())
    a = tm.in_sample(P)
    assert [r["drop"] for r in a["rows"]] == list(range(6))
    for r in a["rows"]:
        assert r["tot_nested"] >= r["tot_opt"] * (1 - 1e-12) and r["ss_nested"] >= r["ss_opt"] * (1 - 1e-9)
    assert a["rows"][0]["tot_new"] == a["rows"][0]["ss_new"] == 0
    assert a["rows"][0]["tot_nested"] == pytest.approx(a["rows"][0]["tot_opt"])
    assert a["rows"][0]["ss_nested"] == pytest.approx(a["rows"][0]["ss_opt"])
    out = tm.rolling(P, drops=(1, 2))
    for r in out.values():
        assert all(len(v) == len(tm.WINDOW_STARTS) for v in r.values())
        assert all(np.isfinite(v).all() and (np.asarray(v) >= 0).all() for v in r.values())


def test_the_ss_res_matrix_is_pangeos_pair_cost():
    """Unit: every entry of the ss_res matrix equals PANGEO's split_cost of the two markets' trajectories."""
    from mlsynth.utils.pangeo_helpers.parallelism import split_cost
    from . import exclusion_tm as tm
    Y = np.random.default_rng(4).lognormal(2, 1, (6, 30))
    W = tm.ss_res_costs(Y)
    for a, b in combinations(range(6), 2):
        assert W[a, b] == pytest.approx(split_cost(Y[a], Y[b]), rel=1e-9)
    assert np.array_equal(W, W.T) and np.all(np.diag(W) == 0)


def test_the_design_window_ends_before_the_evaluation_window():
    """Unit: with every geo's response equal to its day number, the design totals sum the days before the window
    and the evaluation totals the window's 28 days, so no day is in both."""
    from . import exclusion_tm as tm
    dates = pd.date_range("2020-01-01", "2020-12-29")
    df = pd.DataFrame([(g, d, float(i)) for g in range(2) for i, d in enumerate(dates)],
                      columns=["geo", "date", "response"])
    z, zt = tm.window_totals(tm.panel(df), "2020-02-01")
    assert list(z) == [float(sum(range(31)))] * 2
    assert list(zt) == [float(sum(range(31, 59)))] * 2
