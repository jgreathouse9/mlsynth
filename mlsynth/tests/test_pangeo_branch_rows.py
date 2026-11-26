"""A branch's shifted costs are computed for the groups its search reads, not the whole pool.

Each first-level branch of the re-optimised search (:func:`columns._iter_branches`) builds rigorous lower bounds on
its groups' shifted costs (cost lower bound plus the duals of the cliques a group is in, :func:`columns._shifted`).
They were computed over the whole pool, 4.6 million groups on a 40-geo Q = 3 arm, where the branch's search reads
only the groups left to it and its fixed group: 89 ms of the branch's 105 ms outside its LP, about 135 s of a 687 s
proof. They are now computed for those rows only, to the same doubles, with minus infinity elsewhere, a lower bound
on any shifted cost.
"""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from mlsynth.utils.pangeo_helpers import columns, cover


def _panel(seed, n, T=12, levels=5.0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, T)).cumsum(axis=1) + rng.normal(size=(n, 1)) * levels


def _setup(seed, n, Q, T=12, mult=1.0):
    pool = columns.score_columns(np.arange(n), _panel(seed, n, T) * mult, Q)
    k = columns.min_groups(n, Q, 1)
    return pool, k, columns.root_lp(pool, k)


def _cliques(pool, rng, count):
    """Random cliques of 2..6 pool rows (whether they truly share a unit does not matter to _shifted)."""
    out = []
    for _ in range(count):
        size = int(rng.integers(2, min(6, len(pool)) + 1))
        out.append(np.sort(rng.choice(len(pool), size=size, replace=False)).astype(np.int64))
    return out


def _same(a, b):
    """Bitwise equality of two float arrays (so -0.0 and 0.0 differ, and NaN never matches); a row lookup is
    compared as the full array it stands for."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return a.shape == b.shape and np.array_equal(a.view(np.int64), b.view(np.int64))


# --- smoke --------------------------------------------------------------------------------------------------------

def test_smoke_shifted_costs_over_rows_are_the_full_ones_there_and_minus_infinity_elsewhere():
    pool, _, _ = _setup(4, 10, 2)
    rng = np.random.default_rng(0)
    cl = _cliques(pool, rng, 8)
    mu = rng.uniform(0, 1, size=len(cl))
    rows = np.sort(rng.choice(len(pool), size=len(pool) // 3, replace=False))
    full, part = columns._shifted(pool, cl, mu), columns._shifted(pool, cl, mu, rows=rows)
    assert part.shape == full.shape and _same(part[rows], full[rows])
    out = np.ones(len(pool), bool)
    out[rows] = False
    assert np.all(part[out] == -np.inf)


# --- unit ---------------------------------------------------------------------------------------------------------

def test_a_row_lookup_returns_its_values_on_its_rows_and_minus_infinity_elsewhere():
    look = columns._Rows(np.array([7, 2, 5]), np.array([0.7, 0.2, 0.5]), 10)
    assert len(look) == 10 and look.shape == (10,)
    assert np.array_equal(look[np.array([2, 5, 7])], [0.2, 0.5, 0.7])
    assert np.array_equal(look[np.array([0, 2, 9])], [-np.inf, 0.2, -np.inf])
    mask = np.zeros(10, bool)
    mask[[5, 6]] = True
    assert np.array_equal(look[mask], [0.5, -np.inf])
    assert float(look[5]) == 0.5 and float(look[4]) == -np.inf
    full = np.full(10, -np.inf)
    full[[7, 2, 5]] = [0.7, 0.2, 0.5]
    assert np.array_equal(np.asarray(look), full)


def test_each_branch_holds_only_its_own_rows():
    """Unit: a branch's reduced costs cover the groups left to it and its shifted costs those and its fixed group,
    with no array over the whole pool."""
    pool, k, root = _setup(7, 14, 2)
    keep = np.arange(len(pool))
    cl = columns._separate_cliques(pool, keep, k)
    seen = 0
    for j, _, left, duals in columns._iter_branches(pool, root, keep, k, cl):
        if duals is None:
            continue
        assert isinstance(duals.reduced_costs, columns._Rows)
        assert np.array_equal(duals.reduced_costs.rows, np.sort(left))
        if duals.cost_lo is not None:
            seen += 1
            assert isinstance(duals.cost_lo, columns._Rows)
            assert np.array_equal(duals.cost_lo.rows, np.sort(np.append(left, j)))
    assert seen > 0


def test_the_rounding_allowance_counts_cliques_over_the_whole_pool():
    """Unit: the rounding allowance is set by the most cliques any group is in, even a group outside the rows, so
    the rows get the full computation's doubles."""
    pool, _, _ = _setup(4, 10, 2)
    outside, inside = 0, 1
    cl = [np.array([outside, 5 + i], np.int64) for i in range(6)] + [np.array([inside, 2], np.int64)]
    mu = np.full(len(cl), 0.3)
    full = columns._shifted(pool, cl, mu)
    part = columns._shifted(pool, cl, mu, rows=np.array([inside]))
    assert _same(part[[inside]], full[[inside]])


def test_each_branch_carries_its_groups_and_its_fixed_groups_shifted_costs():
    """Unit: a branch's duals hold the whole-pool shifted costs on the groups left to it and on its fixed group,
    and minus infinity elsewhere."""
    pool, k, root = _setup(7, 14, 2)
    keep = np.arange(len(pool))
    cl = columns._separate_cliques(pool, keep, k)
    seen = 0
    for j, _, left, duals in columns._iter_branches(pool, root, keep, k, cl):
        if duals is None or duals.cost_lo is None:
            continue
        seen += 1
        rows = np.append(left, j)
        full = columns._shifted(pool, duals.cliques, duals.mu)
        assert _same(duals.cost_lo[rows], full[rows])
        rest = np.ones(len(pool), bool)
        rest[rows] = False
        assert np.all(duals.cost_lo[rest] == -np.inf)
    assert seen > 0


def test_the_branch_bounds_are_unchanged():
    """Unit: every branch's rigorous bound and reduced costs on its groups are what the whole-pool shifted costs
    give."""
    pool, k, root = _setup(7, 14, 2)
    keep = np.arange(len(pool))
    cl = columns._separate_cliques(pool, keep, k)
    new = list(columns._iter_branches(pool, root, keep, k, cl))
    real = columns._shifted
    columns._shifted = lambda pool_, cliques, mu, rows=None: real(pool_, cliques, mu)
    try:
        old = list(columns._iter_branches(pool, root, keep, k, cl))
    finally:
        columns._shifted = real
    assert len(new) == len(old) > 1
    for (j, b, left, d), (j2, b2, left2, d2) in zip(new, old):
        assert (j, b) == (j2, b2) and np.array_equal(left, left2)
        if d is not None:
            assert _same(d.reduced_costs[left], d2.reduced_costs[left])


# --- edge cases ---------------------------------------------------------------------------------------------------

def test_a_row_lookup_with_no_rows_is_minus_infinity_everywhere():
    look = columns._Rows(np.zeros(0, np.int64), np.zeros(0), 4)
    assert np.all(look[np.arange(4)] == -np.inf) and look[np.zeros(0, np.int64)].size == 0


def test_no_rows_gives_minus_infinity_everywhere_and_all_rows_the_full_computation():
    pool, _, _ = _setup(4, 10, 2)
    rng = np.random.default_rng(1)
    cl = _cliques(pool, rng, 5)
    mu = rng.uniform(0, 1, size=len(cl))
    assert np.all(np.asarray(columns._shifted(pool, cl, mu, rows=np.zeros(0, np.int64))) == -np.inf)
    assert _same(columns._shifted(pool, cl, mu, rows=np.arange(len(pool))), columns._shifted(pool, cl, mu))


def test_no_cliques_leaves_the_shifted_costs_unset():
    """Edge: a branch whose residual LP keeps no clique carries no shifted costs, as before."""
    pool, k, root = _setup(7, 14, 2)
    for _, _, _, duals in columns._iter_branches(pool, root, np.arange(len(pool)), k, ()):
        if duals is not None:
            assert duals.cost_lo is None


def test_rows_on_an_ill_conditioned_pool_get_the_full_doubles():
    """Edge: on a pool with costs up to 9e9 (paths scaled by 1e4; the scorer centres them, so levels alone cancel)
    and clique duals near 1e-8, the restricted computation gives the full one's doubles."""
    pool, _, _ = _setup(5, 10, 2, mult=1e4)
    assert float(np.abs(pool.cost_lo).max()) > 1e9
    rng = np.random.default_rng(2)
    cl = _cliques(pool, rng, 10)
    mu = rng.uniform(0, 1e-8, size=len(cl))
    rows = np.sort(rng.choice(len(pool), size=len(pool) // 2, replace=False))
    assert _same(columns._shifted(pool, cl, mu, rows=rows)[rows], columns._shifted(pool, cl, mu)[rows])


# --- property -----------------------------------------------------------------------------------------------------

@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**6), n=st.integers(4, 10), Q=st.integers(1, 3), ncl=st.integers(0, 12),
       frac=st.floats(0.0, 1.0), scale=st.sampled_from([1e-12, 1.0, 1e8]), mult=st.sampled_from([1.0, 1e4]))
def test_property_restricted_shifted_costs_are_the_full_ones_on_the_rows(seed, n, Q, ncl, frac, scale, mult):
    """Property: over pools, cliques, dual scales (1e-12 to 1e8), path scales (costs up to 1e9) and row sets, the restricted shifted
    costs are the full computation's doubles on the rows and minus infinity elsewhere."""
    if not columns.feasible(n, Q, 1):
        return
    pool, _, _ = _setup(seed, n, Q, T=8, mult=mult)
    rng = np.random.default_rng(seed)
    cl = _cliques(pool, rng, ncl)
    mu = rng.uniform(0, scale, size=len(cl))
    rows = np.flatnonzero(rng.random(len(pool)) < frac)
    full, part = columns._shifted(pool, cl, mu), columns._shifted(pool, cl, mu, rows=rows)
    assert _same(part[rows], full[rows])
    rest = np.ones(len(pool), bool)
    rest[rows] = False
    assert np.all(part[rest] == -np.inf)


@settings(max_examples=60, deadline=None)
@given(size=st.integers(0, 60), data=st.data())
def test_property_a_row_lookup_is_the_full_array_it_stands_for(size, data):
    """Property: over sizes, row sets in any order and queries (index arrays with repeats, masks), a row lookup
    returns what the full array, minus infinity off its rows, returns."""
    rows = np.array(data.draw(st.lists(st.integers(0, max(size - 1, 0)), unique=True, max_size=size)) if size else [],
                    dtype=np.int64)
    values = np.array(data.draw(st.lists(st.floats(-1e9, 1e9), min_size=rows.size, max_size=rows.size)), dtype=float)
    look = columns._Rows(rows, values, size)
    full = np.full(size, -np.inf)
    full[rows] = values
    idx = np.array(data.draw(st.lists(st.integers(0, max(size - 1, 0)), max_size=20)) if size else [], dtype=np.int64)
    assert np.array_equal(look[idx], full[idx])
    mask = np.array(data.draw(st.lists(st.booleans(), min_size=size, max_size=size)), dtype=bool)
    assert np.array_equal(look[mask], full[mask])
    assert np.array_equal(np.asarray(look), full)


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**6), n=st.integers(4, 12), Q=st.integers(1, 3), limit=st.sampled_from([7, 10**7]),
       incumbent=st.sampled_from([np.inf, 1e3]))
def test_property_the_reoptimised_search_is_unchanged(seed, n, Q, limit, incumbent):
    """Property: over arms, caps and incumbents, the re-optimised search with branch-restricted shifted costs
    returns what it returns with whole-pool ones: status, design, both bounds and node count."""
    if not columns.feasible(n, Q, 1):
        return
    pool, k, root = _setup(seed, n, Q, T=8)
    keep = np.arange(len(pool))
    new = columns._search_reopt(pool, root, keep, k, Q, incumbent, limit)
    real = columns._shifted
    columns._shifted = lambda pool_, cliques, mu, rows=None: real(pool_, cliques, mu)
    try:
        old = columns._search_reopt(pool, root, keep, k, Q, incumbent, limit)
    finally:
        columns._shifted = real
    assert (new.status, new.nodes, new.upper, new.lower) == (old.status, old.nodes, old.upper, old.lower)
    assert np.array_equal(new.rows, old.rows)
