"""Checks of the greedy-incumbent study's claims on panels small enough to price
every treated set.

- The instrument with no seed is the library's search, count for count.
- The greedy incumbent is an admissible design of a size searched, and its exact
  upper bound is at least the optimum's exact lower bound: a valid bound.
- Any admissible seed leaves the walk unchanged: the walk reaches exactly the
  sets whose bound is under the optimum, and returns the optimum.
- Adding the complement's closed-form bound keeps the bound under every set's
  objective, and the search under it returns the optimum.
"""
import itertools

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

import mlsynth.utils.marex_helpers.exact as ex
from mlsynth.utils.marex_helpers.diagnostics import SearchReport
from mlsynth.utils.marex_helpers.formulation import carries_weight
from mlsynth.utils.syndes_helpers.exact import SubsetFilter

from . import greedy as gr
from . import search as sr

PROPERTY = settings(max_examples=25, deadline=None,
                    suppress_health_check=[HealthCheck.too_slow])


def _panel(seed, n, T=12):
    """Two common factors and market noise; the target is the markets' mean."""
    rng = np.random.default_rng(seed)
    f = np.cumsum(rng.normal(size=(T, 2)), axis=0)
    B = f @ rng.uniform(0.4, 1.6, (2, n)) + rng.normal(size=(T, n))
    return B, B.mean(axis=1)


def _optimum(B, A, sizes):
    """The least objective over admissible treated sets of the sizes, by brute force."""
    n = B.shape[1]
    best = (np.inf, None)
    for m in sizes:
        for S in itertools.combinations(range(n), m):
            g, w = ex.side_fit(B, A, list(S))
            if not carries_weight(w):
                continue
            h, _ = ex.side_fit(B, A, [j for j in range(n) if j not in S])
            best = min(best, (g + h, S))
    return best


def _bounds_below(B, A, sizes, value):
    """How many candidates the library's treated bound puts under ``value`` plus
    the search's margin, priced in the search's own blocks."""
    n = B.shape[1]
    total = 0
    cut = value + ex.PRUNE_MARGIN * (1.0 + abs(value))
    for block in ex._chunks(n, sizes, SubsetFilter()):
        lb = ex.treated_fit_bound(B, A, block) + ex.control_fit_floor(B, A)
        total += int((lb < cut).sum())
    return total


@pytest.mark.parametrize("seed, n, sizes", [(0, 8, [3]), (1, 9, [2, 3, 4]),
                                            (2, 10, [5]), (3, 7, list(range(1, 7)))])
def test_the_unseeded_instrument_is_the_library(seed, n, sizes):
    B, A = _panel(seed, n)
    report = SearchReport()
    lib = ex.search_cluster(B, A, sizes, filt=SubsetFilter(), report=report)[0]
    counts = report.freeze("enumeration").counts
    got = sr.run_search(B, A, sizes)
    assert got["value"] == lib.value and got["treated"] == list(lib.treated)
    assert got["fitted"] == counts.fitted
    assert got["walked"] == (counts.fitted + counts.cleared_by_treated_fit
                             + counts.unweighted + counts.without_control)


def test_the_greedy_incumbent_is_exact_at_one_market():
    """At one treated market the forward pass prices every market."""
    B, A = _panel(4, 9)
    design, _, evaluated = gr.incumbent(B, A, [1])
    assert design.value == pytest.approx(_optimum(B, A, [1])[0], rel=1e-12)
    assert evaluated == 9


@given(st.integers(0, 10 ** 6), st.integers(5, 9), st.data())
@PROPERTY
def test_the_greedy_incumbent_is_admissible_and_bounds_the_optimum(seed, n, data):
    B, A = _panel(seed, n)
    lo = data.draw(st.integers(1, n - 1))
    hi = data.draw(st.integers(lo, n - 1))
    sizes = list(range(lo, hi + 1))
    design, _, _ = gr.incumbent(B, A, sizes)
    value, S = _optimum(B, A, sizes)
    if design is None:                    # no admissible set on either path
        return
    assert len(design.treated) in sizes and carries_weight(design.w)
    assert list(design.control) == [j for j in range(n) if j not in design.treated]
    best = gr.as_design(B, A, S)
    assert gr.bracket(B, A, design)[1] >= gr.bracket(B, A, best)[0]
    assert design.value >= value * (1 - 1e-9)


@given(st.integers(0, 10 ** 6), st.integers(5, 10), st.data())
@PROPERTY
def test_any_seed_leaves_the_walk_unchanged(seed, n, data):
    """Every candidate fits the buffer here (at most C(10, 5) = 252), so the
    walk is in bound order throughout and the theorem applies as stated."""
    B, A = _panel(seed, n)
    m = data.draw(st.integers(1, n - 1))
    value, S = _optimum(B, A, [m])
    rivals = [T for T in itertools.combinations(range(n), m)
              if carries_weight(ex.side_fit(B, A, list(T))[1])]
    seed_set = data.draw(st.sampled_from(rivals))
    plain = sr.run_search(B, A, [m])
    seeded = sr.run_search(B, A, [m], seed=gr.as_design(B, A, seed_set))
    assert not plain["second_pass"] and not seeded["second_pass"]
    assert seeded["value"] == pytest.approx(value, rel=1e-12)
    assert plain["value"] == pytest.approx(value, rel=1e-12)
    assert seeded["walked"] == plain["walked"] == _bounds_below(B, A, [m], value)
    assert seeded["fitted"] <= plain["fitted"]


def test_a_seed_under_the_optimum_ends_the_search_at_once():
    """The seed is applied: a value no design attains clears every candidate,
    and the search returns it. This is why a seed must be a design, evaluated:
    the theorem needs the cutoff never to fall under the optimum."""
    B, A = _panel(6, 9)
    value, S = _optimum(B, A, [4])
    fake = gr.as_design(B, A, S)
    fake = ex.ClusterDesign(value=0.5 * value, treated=fake.treated,
                            control=fake.control, w=fake.w, v=fake.v)
    got = sr.run_search(B, A, [4], seed=fake)
    assert got["value"] == 0.5 * value
    assert got["walked"] < sr.run_search(B, A, [4])["walked"]


@given(st.integers(0, 10 ** 6), st.integers(4, 9))
@PROPERTY
def test_the_two_sided_bound_is_under_every_objective(seed, n):
    B, A = _panel(seed, n)
    for m in range(1, n):
        sets = np.array(list(itertools.combinations(range(n), m)), dtype=np.int64)
        lb = sr.two_sided_bound(B, A, sets)
        for S, bound in zip(sets, lb):
            C = [j for j in range(n) if j not in S]
            f = ex.side_fit(B, A, list(S))[0] + ex.side_fit(B, A, C)[0]
            assert bound <= f * (1 + 1e-9) + 1e-12


@given(st.integers(0, 10 ** 6), st.integers(5, 9), st.data())
@PROPERTY
def test_the_two_sided_search_returns_the_optimum(seed, n, data):
    B, A = _panel(seed, n)
    lo = data.draw(st.integers(1, n - 1))
    sizes = list(range(lo, data.draw(st.integers(lo, n - 1)) + 1))
    value, _ = _optimum(B, A, sizes)
    got = sr.run_search(B, A, sizes, two_sided=True)
    assert got["value"] == pytest.approx(value, rel=1e-12)
    assert got["walked"] <= sr.run_search(B, A, sizes)["walked"]


def test_the_instrument_restores_the_library():
    B, A = _panel(5, 8)
    sr.run_search(B, A, [4], seed=gr.as_design(B, A, (0, 1, 2, 3)), two_sided=True)
    assert ex._TopK is sr.LIBRARY_TOPK
    assert ex.treated_fit_bound is sr.LIBRARY_BOUND
    assert ex.control_fit_floor is sr.LIBRARY_FLOOR
