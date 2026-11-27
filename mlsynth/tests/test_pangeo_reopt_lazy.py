"""The re-optimised proof search holds one first-level branch at a time.

:func:`columns._search_reopt` branches once on the unit with the fewest kept groups and searches each branch with its
residual LP's duals. Each branch's duals hold two arrays over the whole pool (rigorous reduced costs and shifted
costs), 74 MB at the 4.6 million groups of a 40-geo Q = 3 arm. The branches were built as a list, every LP solved
and every branch's duals held before the first search: on that arm, from its optimum, with 1,519 branches, the
search passed 14 GB and the kernel killed it about 190 branches in. The branches now come from a generator, so a
branch's duals are freed once its search ends and the LPs solved are those of the branches reached.
"""
from __future__ import annotations

import gc
import weakref

import numpy as np
from hypothesis import HealthCheck, example, given, settings, strategies as st

from mlsynth.utils.pangeo_helpers import columns, cover


def _panel(seed, n, T=12, levels=5.0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, T)).cumsum(axis=1) + rng.normal(size=(n, 1)) * levels


def _setup(seed, n, Q, T=12):
    pool = columns.score_columns(np.arange(n), _panel(seed, n, T), Q)
    k = columns.min_groups(n, Q, 1)
    return pool, k, columns.root_lp(pool, k)


def _watch_duals(monkeypatch):
    """Spy on cover.search: at each call, collect garbage and record how many earlier branches' duals are still
    alive."""
    refs, alive, real = [], [], cover.search

    def rec(pool, duals, keep, k, Q, best, limit, fixed=(), **kw):
        gc.collect()
        alive.append(sum(r() is not None for r in refs))
        if duals is not None:                       # a fixed group covering the arm has no residual LP
            refs.append(weakref.ref(duals))
        return real(pool, duals, keep, k, Q, best, limit, fixed=fixed, **kw)
    monkeypatch.setattr(cover, "search", rec)
    return alive


def _count_lps(monkeypatch):
    calls, real = [], columns.solve_boxed_lp

    def rec(*a, **kw):
        calls.append(1)
        return real(*a, **kw)
    monkeypatch.setattr(columns, "solve_boxed_lp", rec)
    return calls


# --- smoke --------------------------------------------------------------------------------------------------------

def test_smoke_the_branches_come_from_a_generator_with_the_lists_contents():
    """Smoke: _iter_branches is a generator, and its branches are _branches' (same rows, bounds, groups left and
    reduced costs)."""
    pool, k, root = _setup(4, 12, 2)
    keep = np.arange(len(pool))
    gen = columns._iter_branches(pool, root, keep, k, ())
    assert iter(gen) is gen and not isinstance(gen, list)
    lazy, eager = list(gen), columns._branches(pool, root, keep, k, ())
    assert len(lazy) == len(eager) > 1
    for (j, b, left, d), (j2, b2, left2, d2) in zip(lazy, eager):
        assert j == j2 and b == b2 and np.array_equal(left, left2)
        assert (d is None) == (d2 is None)
        if d is not None:
            assert np.array_equal(d.reduced_costs, d2.reduced_costs)


# --- unit ---------------------------------------------------------------------------------------------------------

def test_a_branchs_duals_are_freed_before_the_next_branch_is_searched(monkeypatch):
    """Unit: when a branch is searched, no earlier branch's duals are alive."""
    pool, k, root = _setup(4, 14, 2)
    alive = _watch_duals(monkeypatch)
    res = columns._search_reopt(pool, root, np.arange(len(pool)), k, 2, np.inf, 10**7)
    assert res.status == "proved" and len(alive) > 2
    assert alive == [0] * len(alive)


def test_a_search_stopped_by_its_cap_solves_no_lp_past_the_branch_it_stopped_in(monkeypatch):
    """Unit: with a cap the first branch's search reaches, the search solves the parent LP and the first branch's,
    not every branch's."""
    pool, k, root = _setup(4, 14, 2)
    keep = np.arange(len(pool))
    calls = _count_lps(monkeypatch)
    branches = len(columns._branches(pool, root, keep, k, columns._separate_cliques(pool, keep, k)))
    calls.clear()
    res = columns._search_reopt(pool, root, keep, k, 2, np.inf, 2)
    assert res.status == "node_limit" and branches > 2
    assert len(calls) == 2                                  # the parent LP and the first branch's


# --- edge cases ---------------------------------------------------------------------------------------------------

def test_no_kept_groups_yields_no_branch():
    pool, k, root = _setup(4, 8, 2)
    assert list(columns._iter_branches(pool, root, np.zeros(0, np.int64), k, ())) == []
    assert columns._branches(pool, root, np.zeros(0, np.int64), k, ()) == []


def test_a_node_limit_of_zero_solves_at_most_the_first_branch(monkeypatch):
    pool, k, root = _setup(8, 14, 2)
    calls = _count_lps(monkeypatch)
    res = columns._search_reopt(pool, root, np.arange(len(pool)), k, 2, np.inf, 0)
    assert res.status == "node_limit" and res.nodes == 0 and len(calls) <= 2


# --- property -----------------------------------------------------------------------------------------------------

def _eager(real):
    return lambda *a, **kw: iter(list(real(*a, **kw)))


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**6), n=st.integers(4, 12), Q=st.integers(1, 3),
       limit=st.sampled_from([0, 1, 7, 50, 10**7]), incumbent=st.sampled_from([np.inf, 1e3, 0.0]))
@example(seed=2115, n=4, Q=2, limit=7, incumbent=np.inf)     # a group covering the arm, searched after a branch
def test_property_the_lazy_search_is_the_eager_search_holding_one_branch(seed, n, Q, limit, incumbent):
    """Property: over arms, sizes, node caps and incumbents, the search over branches drawn one at a time returns
    what it returns over the branches built in full first (status, design, bounds, nodes), and no earlier branch's
    duals are alive when a branch is searched."""
    if not columns.feasible(n, Q, 1):
        return
    pool, k, root = _setup(seed, n, Q, T=8)
    keep = np.arange(len(pool))
    real_iter, real_search = columns._iter_branches, cover.search
    refs, alive = [], []

    def rec(pool_, duals, keep_, k_, Q_, best, lim, fixed=(), **kw):
        gc.collect()
        alive.append(sum(r() is not None for r in refs))
        if duals is not None:                       # a fixed group covering the arm has no residual LP
            refs.append(weakref.ref(duals))
        return real_search(pool_, duals, keep_, k_, Q_, best, lim, fixed=fixed, **kw)
    cover.search = rec
    try:
        lazy = columns._search_reopt(pool, root, keep, k, Q, incumbent, limit)
    finally:
        cover.search = real_search
    columns._iter_branches = _eager(real_iter)
    try:
        eager = columns._search_reopt(pool, root, keep, k, Q, incumbent, limit)
    finally:
        columns._iter_branches = real_iter
    assert (lazy.status, lazy.nodes, lazy.upper, lazy.lower) == (eager.status, eager.nodes, eager.upper, eager.lower)
    assert np.array_equal(lazy.rows, eager.rows)
    assert all(a == 0 for a in alive)


# --- the generator holds no earlier branch -------------------------------------------------------------------------

def _draw_watching(monkeypatch, pool, root, keep, k):
    """Draw every branch of _iter_branches as _search_reopt does (the loop names rebound at each draw); after each
    draw, collect garbage and count the earlier branches' duals and child LP results still alive. Returns the
    counts per draw and whether each branch had duals."""
    results, real = [], columns.solve_boxed_lp

    def rec(*a, **kw):
        out = real(*a, **kw)
        if kw.get("basis") is not None:              # a child's LP; the parent's is held for the whole level
            results.append(weakref.ref(out))
        return out
    monkeypatch.setattr(columns, "solve_boxed_lp", rec)
    duals_refs, counts, had = [], [], []
    seen_results = 0
    for j, bound, left, duals in columns._iter_branches(pool, root, keep, k, ()):
        mine = results[seen_results:]                 # this draw's child LP, if it solved one
        gc.collect()
        counts.append((sum(r() is not None for r in duals_refs),
                       sum(r() is not None for r in results[:seen_results])))
        seen_results = len(results)
        had.append(duals is not None)
        if duals is not None:
            duals_refs.append(weakref.ref(duals))
        del mine
    return counts, had


def test_a_branch_with_no_residual_lp_holds_no_earlier_branchs_duals(monkeypatch):
    """Unit: the group that covers the whole arm is drawn after branches with duals and leaves the generator before
    any LP; when it is drawn, no earlier branch's duals or LP result are alive."""
    pool, k, root = _setup(2115, 4, 2, T=8)
    counts, had = _draw_watching(monkeypatch, pool, root, np.arange(len(pool)), k)
    assert any(had[i] and not had[i + 1] for i in range(len(had) - 1)), "a branch with duals, then one without"
    assert all(c == (0, 0) for c in counts), f"earlier branches alive at a draw: {counts}"


def test_a_branch_with_no_group_left_holds_no_earlier_branch(monkeypatch):
    """Edge: with the kept groups cut so that some fixed group leaves units no kept group fits, that branch leaves
    the generator before any LP, and holds nothing of the branch before it."""
    pool, k, root = _setup(5, 8, 2)
    rng = np.random.default_rng(0)
    for _ in range(40):
        keep = np.flatnonzero(rng.random(len(pool)) < 0.35)
        if keep.size == 0:
            continue
        gen_rows = list(columns._iter_branches(pool, root, keep, k, ()))
        if any(r[3] is not None for r in gen_rows) and any(r[2].size == 0 for r in gen_rows):
            break
    else:  # pragma: no cover - the fixed draws above contain such a subset
        raise AssertionError("no kept subset with an empty branch")
    del gen_rows
    counts, had = _draw_watching(monkeypatch, pool, root, keep, k)
    assert all(c == (0, 0) for c in counts), f"earlier branches alive at a draw: {counts}"


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow,
                                                                 HealthCheck.function_scoped_fixture])
@given(seed=st.integers(0, 10**6), n=st.integers(4, 10), Q=st.integers(1, 3), share=st.sampled_from([1.0, 0.6, 0.3]))
@example(seed=2115, n=4, Q=2, share=1.0)
def test_property_no_draw_holds_an_earlier_branch(monkeypatch, seed, n, Q, share):
    """Property: over arms and kept subsets, when any branch is drawn no earlier branch's duals or child LP result
    are alive, whichever way the branch before it left the generator."""
    if not columns.feasible(n, Q, 1):
        return
    pool, k, root = _setup(seed, n, Q, T=8)
    rng = np.random.default_rng(seed)
    keep = np.arange(len(pool)) if share == 1.0 else np.flatnonzero(rng.random(len(pool)) < share)
    counts, _ = _draw_watching(monkeypatch, pool, root, keep, k)
    assert all(c == (0, 0) for c in counts), f"earlier branches alive at a draw: {counts}"
