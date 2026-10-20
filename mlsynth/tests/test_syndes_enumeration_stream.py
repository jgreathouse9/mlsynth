"""The two-way and one-way enumerations stream candidates as arrays.

Contract: the stream emits exactly the admissible treated sets the walk of
``SearchSpace.iter_subsets`` filtered by ``SubsetFilter.accepts`` emits, in the
same order, in blocks of at most ``chunk`` rows. The order is what breaks ties
between designs of equal value, so it is part of the contract, not an
implementation detail. ``SubsetFilter.accepts_rows`` is ``accepts`` applied to
every row of a block.
"""
from itertools import combinations

import numpy as np
from hypothesis import given, settings, strategies as st

from mlsynth.utils.syndes_helpers.exact import (
    SubsetFilter,
    _search_space,
    _stream_from_space,
)
from mlsynth.utils.syndes_helpers.restrictions import DesignRestrictions


def _reference(N, K, filt):
    space = _search_space(N, K, filt)
    return [s for s in space.iter_subsets() if filt.accepts(s)]


def _streamed(N, K, filt, chunk):
    blocks = list(_stream_from_space(_search_space(N, K, filt), filt, chunk))
    assert all(1 <= len(b) <= chunk for b in blocks)
    return [tuple(int(i) for i in row) for b in blocks for row in b]


@st.composite
def _filters(draw):
    N = draw(st.integers(3, 12))
    K = draw(st.integers(1, N - 1))
    units = list(range(N))
    forced = draw(st.lists(st.sampled_from(units), max_size=2, unique=True))
    rest = [u for u in units if u not in forced]
    forbidden = draw(st.lists(st.sampled_from(rest), max_size=3, unique=True)) if rest else []
    pairs = draw(st.lists(st.tuples(st.sampled_from(units), st.sampled_from(units))
                          .filter(lambda p: p[0] != p[1]), max_size=3))
    use_costs = draw(st.booleans())
    costs = np.array(draw(st.lists(st.floats(0, 3), min_size=N, max_size=N))) if use_costs else None
    budget = float(draw(st.floats(0.5, 10))) if use_costs else None
    n_sets = draw(st.integers(0, 2))
    rng = np.random.default_rng(draw(st.integers(0, 2**31 - 1)))
    nogood = [np.sort(rng.choice(N, K, replace=False)) for _ in range(n_sets)]
    strata = []
    if draw(st.booleans()) and N >= 4:
        members = sorted(rng.choice(N, N // 2, replace=False).tolist())
        strata = [(members, int(draw(st.integers(0, 1))), None)]
    rules = DesignRestrictions(forced_in=forced, forbidden=forbidden,
                               conflict_pairs=pairs, strata=strata)
    return N, K, SubsetFilter.build(rules, costs, budget, nogood, N=N)


@settings(max_examples=300, deadline=None)
@given(_filters(), st.integers(1, 40))
def test_the_stream_is_the_walk_in_order(problem, chunk):
    N, K, filt = problem
    assert _streamed(N, K, filt, chunk) == _reference(N, K, filt)


@settings(max_examples=300, deadline=None)
@given(_filters())
def test_accepts_rows_is_accepts_on_every_row(problem):
    N, K, filt = problem
    rows = np.array(list(combinations(range(N), K)), dtype=np.int64)
    expected = np.array([filt.accepts(tuple(int(i) for i in r)) for r in rows])
    np.testing.assert_array_equal(filt.accepts_rows(rows), expected)


def test_the_unrestricted_stream_is_combinations():
    filt = SubsetFilter.build(None, None, None, None, N=9)
    assert _streamed(9, 4, filt, 7) == list(combinations(range(9), 4))


def test_an_empty_space_streams_nothing():
    rules = DesignRestrictions(forced_in=[0, 1, 2])
    filt = SubsetFilter.build(rules, None, None, None, N=6)
    assert _streamed(6, 2, filt, 5) == []


def test_every_design_rejected_streams_nothing():
    filt = SubsetFilter.build(None, np.ones(6), 1.5, None, N=6)
    assert _streamed(6, 2, filt, 5) == []


def test_forced_units_only_stream_one_set():
    rules = DesignRestrictions(forced_in=[1, 4])
    filt = SubsetFilter.build(rules, None, None, None, N=6)
    assert _streamed(6, 2, filt, 5) == [(1, 4)]


@settings(max_examples=100, deadline=None)
@given(_filters(), st.integers(1, 40), st.integers(1, 12))
def test_split_tables_keep_the_walks_order(problem, chunk, table_rows):
    """Tables larger than the cache limit are split on their first element; the
    order must survive every depth of splitting."""
    import mlsynth.utils.syndes_helpers.exact as exact
    N, K, filt = problem
    saved = exact.TABLE_ROWS
    exact.TABLE_ROWS = table_rows
    exact._combination_table.cache_clear()
    try:
        assert _streamed(N, K, filt, chunk) == _reference(N, K, filt)
    finally:
        exact.TABLE_ROWS = saved
        exact._combination_table.cache_clear()
