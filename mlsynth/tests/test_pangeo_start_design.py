"""The start design widens its window until it finds a design.

:func:`cover.start_design` searches the ``START_PER_UNIT`` n groups of least reduced cost for a first design. At
Q = 4 that window can hold no exact cover: on a 24-geo seasonal arm its 960 groups cover every unit, and a search of
614 nodes proves no design with at least k groups among them. The quick search then started with no incumbent, kept
all 1.27 million groups and took 666 s. The start design now widens its window ``START_GROWTH``-fold and searches
again whenever a window holds no design, up to the whole pool; four times wider, that arm's start found a design 3.4%
above the optimum in 0.28 s.
"""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from mlsynth.utils.pangeo_helpers import columns, cover


def _panel(seed, n, T=12, levels=5.0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, T)).cumsum(axis=1) + rng.normal(size=(n, 1)) * levels


def _setup(seed, n, Q):
    pool = columns.score_columns(np.arange(n), _panel(seed, n), Q)
    k = columns.min_groups(n, Q, 1)
    return pool, k, columns.root_lp(pool, k)


def _spy(monkeypatch):
    seen, real = [], cover.search

    def rec(pool, root, keep, k, Q, best_up, node_limit, fixed=(), **kw):
        out = real(pool, root, keep, k, Q, best_up, node_limit, fixed=fixed, **kw)
        seen.append((len(keep), out))
        return out
    monkeypatch.setattr(cover, "search", rec)
    return seen


def _is_design(pool, rows, n, k):
    units = sorted(int(u) for j in rows for u in pool.members[j] if u >= 0)
    return units == list(range(n)) and len(rows) >= k


# --- smoke --------------------------------------------------------------------------------------------------------

def test_smoke_a_window_too_small_for_a_design_is_widened_until_one_is_found(monkeypatch):
    """Smoke: with a first window of one group per unit, a 12-unit Q = 3 arm's start finds no design there, widens,
    and returns an exact cover with at least k groups."""
    n, Q = 12, 3
    pool, k, root = _setup(1, n, Q)
    monkeypatch.setattr(cover, "START_PER_UNIT", 1)
    seen = _spy(monkeypatch)
    res = cover.start_design(pool, root, k, Q)
    assert len(seen) >= 2 and seen[0][1].rows.size == 0           # the first window held no design
    assert res.rows.size and _is_design(pool, res.rows, n, k)


# --- unit ---------------------------------------------------------------------------------------------------------

def test_each_window_is_start_growth_times_the_last_until_a_design(monkeypatch):
    """Unit: the windows hold at least START_PER_UNIT n groups, then START_GROWTH times that, and so on (never past
    the pool; ties at the cut can only add groups), and the search stops at the first window that holds a design."""
    n, Q = 12, 3
    pool, k, root = _setup(1, n, Q)
    monkeypatch.setattr(cover, "START_PER_UNIT", 1)
    seen = _spy(monkeypatch)
    cover.start_design(pool, root, k, Q)
    want = [min(len(pool), n * cover.START_GROWTH ** i) for i in range(len(seen))]
    assert all(w <= s <= len(pool) for (s, _), w in zip(seen, want))   # ties at the cut can only add groups
    assert all(out.rows.size == 0 for _, out in seen[:-1]) and seen[-1][1].rows.size > 0


def test_a_window_that_holds_a_design_is_searched_once(monkeypatch):
    """Unit: a first window that holds a design is the only one searched, as before."""
    n, Q = 12, 2
    pool, k, root = _setup(2, n, Q)
    seen = _spy(monkeypatch)
    res = cover.start_design(pool, root, k, Q)
    assert len(seen) == 1 and res.rows.size


def test_the_start_reports_the_nodes_of_every_window_it_searched(monkeypatch):
    n, Q = 12, 3
    pool, k, root = _setup(1, n, Q)
    monkeypatch.setattr(cover, "START_PER_UNIT", 1)
    seen = _spy(monkeypatch)
    res = cover.start_design(pool, root, k, Q)
    assert res.nodes == sum(out.nodes for _, out in seen)


def test_the_start_reports_every_window_through_the_progress_array(monkeypatch):
    n, Q = 12, 3
    pool, k, root = _setup(1, n, Q)
    monkeypatch.setattr(cover, "START_PER_UNIT", 1)
    seen = _spy(monkeypatch)
    prog = cover.new_progress()
    res = cover.start_design(pool, root, k, Q, progress=prog)
    assert prog[cover.P_SEARCHES] == len(seen) and prog[cover.P_TOTAL] == res.nodes


# --- edge cases ---------------------------------------------------------------------------------------------------

def test_with_no_design_in_the_pool_the_last_window_is_the_whole_pool(monkeypatch):
    """Edge: asking for more groups than the arm can hold leaves no design anywhere; the start widens to the whole
    pool, searches it, and returns no design."""
    n, Q = 8, 2
    pool, _, root = _setup(3, n, Q)
    monkeypatch.setattr(cover, "START_PER_UNIT", 1)
    seen = _spy(monkeypatch)
    res = cover.start_design(pool, root, n // 2 + 1, Q)
    assert res.rows.size == 0
    assert seen[-1][0] == len(pool) and len(seen) >= 2


# --- property -----------------------------------------------------------------------------------------------------

@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**6), n=st.integers(4, 12), Q=st.integers(1, 4), per=st.integers(1, 40))
def test_property_the_start_finds_a_design_whenever_the_pool_holds_one(seed, n, Q, per):
    """Property: over panels, sizes and first windows, the start returns an exact cover with at least k groups
    whenever the arm has one (columns.feasible), its windows grow START_GROWTH-fold up to the pool, and its node
    count is the sum of its searches'."""
    if not columns.feasible(n, Q, 1):
        return
    pool, k, root = _setup(seed, n, Q)
    seen, real = [], cover.search
    saved = cover.START_PER_UNIT

    def rec(pool_, root_, keep, k_, Q_, best_up, node_limit, fixed=(), **kw):
        out = real(pool_, root_, keep, k_, Q_, best_up, node_limit, fixed=fixed, **kw)
        seen.append((len(keep), out))
        return out
    cover.search, cover.START_PER_UNIT = rec, per
    try:
        res = cover.start_design(pool, root, k, Q)
    finally:
        cover.search, cover.START_PER_UNIT = real, saved
    assert res.rows.size and _is_design(pool, res.rows, n, k)
    want = [min(len(pool), per * n * cover.START_GROWTH ** i) for i in range(len(seen))]
    assert all(w <= s <= len(pool) for (s, _), w in zip(seen, want))
    assert res.nodes == sum(out.nodes for _, out in seen)
