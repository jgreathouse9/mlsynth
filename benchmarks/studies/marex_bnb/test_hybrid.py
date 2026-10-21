"""The ladder of the enumeration-versus-branch-and-bound RCA, as tests.

Incident: on GeoLift with 8 treated, the certified best-first search took
2,333 s and the library's exact backend 105 s, both returning the same design.
The decomposition: 33 lifts at 70.7 s against 76.9 million priced sets at
1.37 microseconds, a factor of 4.29e-7 in units times 5.18e7 in cost per unit,
which reproduces the observed 22.2.

The cause the search can act on is an omission: it bounded every node whose
subtree held more than 64 sets, a constant with no units. A lift costs as much
as pricing about 50 million sets, so every lift after the root was spent on a
subtree cheaper to enumerate than to bound. The invariant nobody stated:

    bound a node only when enumerating its subtree would cost more than
    ``kappa`` times bounding it.

These tests pin that rule (``should_bound``), the vectorised certified
enumeration a node falls back on (``certified_enumerate``), and the hybrid's
end-to-end behaviour.

From the repository root::

    python -m pytest benchmarks/studies/marex_bnb/test_hybrid.py -q -p no:cacheprovider
"""
import itertools
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from benchmarks.studies.marex_bnb.certify import Panel
from benchmarks.studies.marex_bnb.hybrid import (certified_enumerate,
                                                 should_bound)
from mlsynth.utils.marex_helpers.exact import side_fit


# --- rung 3: the cost rule ----------------------------------------------------

@given(sets=st.integers(0, 10**12), price=st.floats(1e-9, 1e-3),
       lift=st.floats(1e-3, 1e4), kappa=st.floats(1.0, 10.0))
def test_should_bound_exactly_when_enumeration_costs_more(sets, price, lift, kappa):
    assert should_bound(sets, price, lift, kappa) == (
        sets >= 2 and sets * price > kappa * lift)


@given(price=st.floats(1e-9, 1e-3), lift=st.floats(1e-3, 1e4),
       a=st.integers(0, 10**12), b=st.integers(0, 10**12))
def test_should_bound_is_monotone_in_the_subtree(price, lift, a, b):
    small, large = sorted((a, b))
    if should_bound(small, price, lift, 1.0):
        assert should_bound(large, price, lift, 1.0)


def test_a_lift_is_never_spent_below_break_even():
    # GeoLift, 8 treated, the measured costs: a 35 s lift is worth
    # 35 / 1.37e-6 = 25.5 million sets of pricing.
    assert not should_bound(15_380_937, 1.37e-6, 35.0, 1.0)     # C(39, 7)
    assert should_bound(76_904_685, 1.37e-6, 35.0, 1.0)         # C(40, 8)


# --- rung 3: the certified enumeration's bound --------------------------------

def _panel(seed, T, n):
    rng = np.random.default_rng(seed)
    B = rng.normal(size=(T, n)) + rng.normal(size=(T, 1))
    A = B.mean(axis=1)
    return B, A


@settings(max_examples=60, deadline=None)
@given(seed=st.integers(0, 10**6), T=st.integers(3, 30), n=st.integers(5, 9),
       m=st.integers(1, 3))
def test_enumeration_bound_never_exceeds_the_fit(seed, T, n, m):
    B, A = _panel(seed, T, n)
    pan = Panel(B - A[:, None])
    sets = np.array(list(itertools.combinations(range(n), m)))
    lbs = certified_enumerate(pan, sets)
    for S, lb in zip(sets, lbs):
        g, w = side_fit(B, A, list(S))
        assert Fraction(lb) <= pan.side_upper(list(S), w)


def test_enumeration_bound_is_the_closed_form_when_it_is_interior():
    # Three points around the origin: the affine optimum is inside the
    # triangle, so the closed form is the fit and the bound is tight.
    Bc = np.array([[1.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    Bc[2] = 0.5
    B = Bc + 3.0
    A = np.full(3, 3.0)
    pan = Panel(B - A[:, None])
    lb = certified_enumerate(pan, np.array([[0, 1, 2]]))[0]
    g, _ = side_fit(B, A, [0, 1, 2])
    assert lb <= g + 1e-15 and lb >= g * (1 - 1e-12)


def test_enumeration_bound_of_a_degenerate_set_is_zero_not_an_error():
    B = np.ones((4, 3))
    A = np.ones(4)
    pan = Panel(B - A[:, None])
    assert certified_enumerate(pan, np.array([[0, 1]]))[0] == 0.0


# --- rungs 0 and 4: the hybrid end to end -------------------------------------

def test_hybrid_below_break_even_never_lifts_and_finds_the_optimum():
    from benchmarks.studies.marex_bnb.bnb12 import objective, panels
    from benchmarks.studies.marex_bnb.hybrid import search
    for name, (B, A) in panels().items():
        n = B.shape[1]
        brute = {S: objective(B, A, S) for S in itertools.combinations(range(n), 3)}
        fstar = min(brute.values())
        res = search(B, A, 3, price_cost=1.0e-6, lift_cost=lambda K: 30.0)
        assert res["stats"]["lifts"] == 0, name
        assert res["set"] in [S for S, v in brute.items() if v <= fstar * (1 + 1e-12)]
        # f* is the floating-point fit; the certified values bracket the exact one
        assert float(res["lower"]) <= fstar * (1 + 1e-12)
        assert fstar <= float(res["hi"]) * (1 + 1e-12)
        assert float(res["gap"]) <= 1e-9 * fstar


def test_hybrid_above_break_even_lifts_and_still_finds_the_optimum():
    from benchmarks.studies.marex_bnb.bnb12 import objective, panels
    from benchmarks.studies.marex_bnb.hybrid import search
    B, A = panels()["factor"]
    n = B.shape[1]
    brute = {S: objective(B, A, S) for S in itertools.combinations(range(n), 3)}
    fstar = min(brute.values())
    # pricing made artificially dear, so bounding the root pays
    res = search(B, A, 3, price_cost=1.0, lift_cost=lambda K: 1.0)
    assert res["stats"]["lifts"] >= 1
    assert abs(float(res["hi"]) - fstar) <= 1e-9 * fstar
    assert float(res["gap"]) <= 1e-9 * fstar


def test_hybrid_rejects_a_kappa_below_one():
    from benchmarks.studies.marex_bnb.hybrid import search
    B, A = _panel(0, 10, 6)
    with pytest.raises(ValueError, match="kappa"):
        search(B, A, 2, kappa=0.5)


def test_hybrid_with_default_costs_enumerates_a_small_panel():
    from benchmarks.studies.marex_bnb.bnb12 import objective, panels
    from benchmarks.studies.marex_bnb.hybrid import default_lift_cost, search
    B, A = panels()["noise"]
    n = B.shape[1]
    fstar = min(objective(B, A, S) for S in itertools.combinations(range(n), 3))
    res = search(B, A, 3)
    assert res["stats"]["lifts"] == 0 and res["stats"]["price_cost"] > 0
    assert abs(float(res["hi"]) - fstar) <= 1e-9 * fstar
    assert default_lift_cost(116) == pytest.approx(210.0)


def test_hybrid_reports_an_exact_tie_as_unresolved():
    from benchmarks.studies.marex_bnb.bnb12 import panels
    from benchmarks.studies.marex_bnb.hybrid import search
    B, _ = panels()["factor"]
    B = B.copy()
    B[:, 11] = B[:, 4]                        # markets 4 and 11 identical
    res = search(B, B.mean(axis=1), 3, price_cost=1e-6, lift_cost=lambda K: 30.0)
    tied = {res["set"]} | set(res["unresolved"])
    assert {(0, 4, 10), (0, 10, 11)} <= tied


def test_hybrid_stopped_by_its_time_limit_keeps_a_valid_bound():
    from benchmarks.studies.marex_bnb.bnb12 import objective, panels
    from benchmarks.studies.marex_bnb.hybrid import search
    B, A = panels()["factor"]
    n = B.shape[1]
    fstar = min(objective(B, A, S) for S in itertools.combinations(range(n), 3))
    res = search(B, A, 3, price_cost=1.0, lift_cost=lambda K: 1.0, time_limit=1e-9)
    assert float(res["lower"]) <= fstar


def test_hybrid_skips_nodes_without_a_completion_and_logs_each_bound():
    from benchmarks.studies.marex_bnb.bnb12 import objective
    from benchmarks.studies.marex_bnb.hybrid import search
    B, A = _panel(3, 12, 5)
    fstar = min(objective(B, A, S) for S in itertools.combinations(range(5), 3))
    lines = []
    res = search(B, A, 3, price_cost=1.0, lift_cost=lambda K: 1e-3, log=lines.append)
    assert abs(float(res["hi"]) - fstar) <= 1e-9 * fstar
    assert lines and all("node" in line for line in lines)


@pytest.mark.parametrize("builder", ["trace_lift", "indicator_lift", "treated_lift"])
def test_a_lift_of_a_node_without_free_markets_is_refused(builder):
    # Such a node is one treated set, a leaf; its lift has no PSD block to
    # speak of, and building it indexed past the matrix.
    import benchmarks.studies.marex_bnb.certified_dnn as cd
    with pytest.raises(ValueError, match="free market"):
        getattr(cd, builder)(5, 3, [0, 1, 2], [3, 4], [])


def test_hybrid_never_bounds_a_single_set():
    from benchmarks.studies.marex_bnb.hybrid import should_bound
    assert not should_bound(1, 1.0, 1e-9, 1.0)
    assert not should_bound(0, 1.0, 1e-9, 1.0)


def test_hybrid_keeps_an_overlapping_set_that_does_not_improve():
    # Market 2 a copy of market 0: (0, 10, 11) and (2, 10, 11) tie exactly, and
    # the second priced has a certified interval reaching above the first's.
    from benchmarks.studies.marex_bnb.bnb12 import panels
    from benchmarks.studies.marex_bnb.hybrid import search
    B, _ = panels()["factor"]
    B = B.copy()
    B[:, 2] = B[:, 0]
    res = search(B, B.mean(axis=1), 3, price_cost=1e-6, lift_cost=lambda K: 30.0)
    assert res["set"] == (0, 10, 11)
    assert res["unresolved"] == [(2, 10, 11)]
    assert all(isinstance(j, int) for j in res["set"])


@settings(max_examples=40, deadline=None)
@given(seed=st.integers(0, 10**6), T=st.integers(4, 30), n=st.integers(5, 9),
       m=st.integers(2, 3))
def test_enumeration_bound_is_as_tight_as_the_library_closed_form(seed, T, n, m):
    # The certificate gives up only rounding against the library's pricing.
    from mlsynth.utils.marex_helpers.exact import treated_fit_bound
    B, A = _panel(seed, T, n)
    pan = Panel(B - A[:, None])
    sets = np.array(list(itertools.combinations(range(n), m)))
    ours = certified_enumerate(pan, sets)
    lib = treated_fit_bound(B, A, sets)
    scale = np.max(np.abs(pan.Bc)) ** 2
    assert np.all(ours >= lib - 1e-10 * scale * (1 + lib / scale))


def test_enumeration_reads_the_gram_not_the_data():
    # Pricing must not rebuild each set's Gram block from the T x m data:
    # with the panel's Gram cached, a deliberately poisoned data matrix must
    # not change the bounds (only the cached Gram and its error bound are used).
    B, A = _panel(1, 12, 6)
    pan = Panel(B - A[:, None])
    sets = np.array(list(itertools.combinations(range(6), 2)))
    before = certified_enumerate(pan, sets)
    pan.Bc = np.full_like(pan.Bc, np.nan)
    after = certified_enumerate(pan, sets)
    assert np.array_equal(before, after)


def test_enumeration_bound_holds_on_a_hostile_collinear_panel():
    # Three nearly collinear markets of norm 1e6 and one far market, T = 200:
    # the Gram's own rounding (gamma_T |B|'|B|) exceeds every other slack here,
    # so a certificate that trusts the floating-point Gram rises above the fit
    # of the pair (0, 1) (201.0993 against an exact 201.0803).
    rng = np.random.default_rng(22)
    T, L = 200, 1e6
    v = rng.normal(size=T)
    u = rng.normal(size=T)
    Bc = np.column_stack([L * v + u, -L * v + u * 1.0000001,
                          0.5 * L * v + u * 0.9999999 + 1e-3 * rng.normal(size=T),
                          rng.normal(size=T) * L])
    pan = Panel(Bc)
    B, A = Bc, np.zeros(T)
    for m in (2, 3):
        sets = np.array(list(itertools.combinations(range(4), m)))
        for S, lb in zip(sets, certified_enumerate(pan, sets)):
            g, w = side_fit(B, A, list(S))
            assert Fraction(lb) <= pan.side_upper(list(S), w), S
