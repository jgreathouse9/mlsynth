"""The cover search's progress counters, readable while it runs.

:func:`mlsynth.utils.pangeo_helpers.cover.search` takes an optional ``progress`` array of ``cover.PROGRESS_SLOTS``
doubles (:func:`cover.new_progress`). The compiled search writes into it every ``cover.PROGRESS_EVERY`` nodes and
when it ends, and releases the GIL while it runs, so another thread can read the counters as
the search goes. The search prints nothing and its result does not depend on the array: the counters are numbers a
caller may watch, not output. :func:`columns.solve_exact` passes one array to every search it runs.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.pangeo_helpers import columns, cover

P = cover


def _panel(seed, n, T=12, levels=5.0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, T)).cumsum(axis=1) + rng.normal(size=(n, 1)) * levels


def _setup(seed, n, Q, T=12):
    Y = _panel(seed, n, T)
    idx = np.arange(n)
    pool = columns.score_columns(idx, Y, Q)
    k = columns.min_groups(n, Q, 1)
    root = columns.root_lp(pool, k)
    return Y, idx, pool, k, root


def _same(a, b):
    assert a.status == b.status and a.nodes == b.nodes
    assert np.array_equal(a.rows, b.rows)
    assert a.upper == b.upper or (np.isinf(a.upper) and np.isinf(b.upper))
    assert a.lower == b.lower or (np.isinf(a.lower) and np.isinf(b.lower))


def _final_is_the_result(prog, res, searches=1, total=None):
    assert prog[P.P_DONE] == 1.0
    assert prog[P.P_SEARCHES] == searches
    assert prog[P.P_NODES] == res.nodes
    assert prog[P.P_TOTAL] == (res.nodes if total is None else total)
    assert prog[P.P_INCUMBENT] == res.upper or (np.isinf(prog[P.P_INCUMBENT]) and np.isinf(res.upper))
    assert 0.0 <= prog[P.P_ROOT_DONE] <= prog[P.P_ROOT_TOTAL]


# --- smoke --------------------------------------------------------------------------------------------------------

def test_smoke_a_search_given_a_progress_array_reports_its_counts():
    """Smoke: a search on a 12-unit arm at Q = 2 fills the array with its final counts."""
    _, _, pool, k, root = _setup(1, 12, 2)
    prog = cover.new_progress()
    res = cover.search(pool, root, np.arange(len(pool)), k, 2, np.inf, 10_000_000, progress=prog)
    assert res.status == "proved" and res.nodes > 0
    _final_is_the_result(prog, res)


def test_a_new_progress_array_is_zero_with_no_incumbent():
    prog = cover.new_progress()
    assert prog.shape == (cover.PROGRESS_SLOTS,) and prog.dtype == np.float64
    assert prog[P.P_SEARCHES] == prog[P.P_NODES] == prog[P.P_TOTAL] == prog[P.P_DONE] == 0.0
    assert np.isinf(prog[P.P_INCUMBENT]) and np.isinf(prog[P.P_LOWEST])


# --- unit: the counters are the result's, and accumulate ----------------------------------------------------------

@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**6), n=st.integers(2, 11), Q=st.integers(1, 3),
       limit=st.sampled_from([1, 7, 200, 5_000, 10_000_000]), incumbent=st.booleans())
def test_property_the_array_changes_nothing_and_ends_holding_the_result(seed, n, Q, limit, incumbent):
    """Property: over panels, sizes, caps and incumbents, a search with the array returns the same result, bit for
    bit, as one without, and the array ends holding that result's nodes, incumbent and status."""
    if not columns.feasible(n, Q, 1):
        return
    _, _, pool, k, root = _setup(seed, n, Q)
    keep = np.arange(len(pool))
    best = np.inf
    if incumbent:
        start = cover.start_design(pool, root, k, Q)
        best = start.upper if start.rows.size else np.inf
    plain = cover.search(pool, root, keep, k, Q, best, limit)
    prog = cover.new_progress()
    watched = cover.search(pool, root, keep, k, Q, best, limit, progress=prog)
    _same(plain, watched)
    _final_is_the_result(prog, watched)


def test_the_totals_accumulate_over_the_searches_that_share_an_array():
    """Unit: two searches on one array: the search count is 2, the total is the sum of their nodes, and the
    per-search counts are the second search's."""
    _, _, pool, k, root = _setup(2, 12, 2)
    keep = np.arange(len(pool))
    prog = cover.new_progress()
    a = cover.search(pool, root, keep, k, 2, np.inf, 10_000_000, progress=prog)
    b = cover.search(pool, root, keep, k, 2, a.upper, 10_000_000, progress=prog)
    _final_is_the_result(prog, b, searches=2, total=a.nodes + b.nodes)


def test_the_start_design_reports_through_the_array():
    _, _, pool, k, root = _setup(3, 12, 2)
    prog = cover.new_progress()
    res = cover.start_design(pool, root, k, 2, progress=prog)
    _final_is_the_result(prog, res)


def test_solve_exact_reports_every_search_it_ran_through_one_array():
    """Unit: an arm whose quick search is capped and whose proof runs after the first MIP: the array's total is the
    solve's ``search_nodes`` (the fixed group of each re-optimised branch counts as a node), and the design and its
    diagnostics are those of the solve without the array."""
    Y, idx, _, _, _ = _setup(4, 12, 3)
    plain_c, plain_d = columns.solve_exact(idx, Y, 3, search_node_limit=20)
    assert "second_search" in plain_d                                  # the arm reaches the proof after the MIP
    prog = cover.new_progress()
    c, d = columns.solve_exact(idx, Y, 3, search_node_limit=20, progress=prog)
    assert [sorted(x["members"]) for x in c] == [sorted(x["members"]) for x in plain_c]
    for key in ("objective_value", "search", "search_nodes", "mip_solves", "search_start", "second_search"):
        assert d[key] == plain_d[key]
    assert prog[P.P_DONE] == 1.0
    assert prog[P.P_TOTAL] == d["search_nodes"] > 0
    assert prog[P.P_SEARCHES] >= 2


def test_the_reoptimised_search_counts_each_branch_fixed_group_as_a_node():
    """Unit: the re-optimised search's node count includes each searched branch's fixed group, and so does the
    array's total."""
    _, _, pool, k, root = _setup(5, 12, 2)
    keep = np.arange(len(pool))
    prog = cover.new_progress()
    res = columns._search_reopt(pool, root, keep, k, 2, np.inf, 10_000_000, progress=prog)
    assert prog[P.P_TOTAL] == res.nodes > 0
    assert prog[P.P_SEARCHES] >= 1


# --- the counters move while the search runs ----------------------------------------------------------------------

def test_the_counters_can_be_read_from_another_thread_while_the_search_runs():
    """Unit: a second search on an array a first search already used runs on a thread while this one polls the
    array. Some reads land mid-search (the GIL is released) and see a node count strictly between 0 and the final
    count, and a total that is the first search's nodes plus the node count; within the search the node count never
    falls, the incumbent never rises, and the root has started at least one of its branches and at most all."""
    _, _, pool, k, root = _setup(6, 16, 3)
    keep = np.arange(len(pool))
    prog = cover.new_progress()
    first = cover.search(pool, root, keep, k, 3, np.inf, 1000, progress=prog)
    out = {}
    worker = threading.Thread(target=lambda: out.setdefault(
        "res", cover.search(pool, root, keep, k, 3, np.inf, 400_000, progress=prog)))
    snaps = []
    worker.start()
    while worker.is_alive():
        snaps.append(prog.copy())
        time.sleep(0.0005)
    worker.join()
    res = out["res"]
    assert res.nodes > 4 * cover.PROGRESS_EVERY
    live = [s for s in snaps if s[P.P_SEARCHES] == 2.0]
    mid = [s for s in live if s[P.P_DONE] == 0.0 and 0 < s[P.P_NODES] < res.nodes]
    assert len(mid) >= 2, "no read landed while the search ran"
    assert all(s[P.P_TOTAL] == first.nodes + s[P.P_NODES] for s in mid)
    assert all(1 <= s[P.P_ROOT_DONE] <= s[P.P_ROOT_TOTAL] for s in mid)
    assert all(s[P.P_DEPTH] >= 0 for s in mid)
    nodes = [s[P.P_NODES] for s in live]
    inc = [s[P.P_INCUMBENT] for s in live]
    assert all(a <= b for a, b in zip(nodes, nodes[1:]))
    assert all(a >= b for a, b in zip(inc, inc[1:]))
    _final_is_the_result(prog, res, searches=2, total=first.nodes + res.nodes)


def test_a_capped_search_ends_with_its_counts():
    """Unit: a search stopped by its cap a little past two write intervals ends holding its own count, the cap plus
    the node that crossed it."""
    _, _, pool, k, root = _setup(6, 16, 3)
    keep = np.arange(len(pool))
    limit = 2 * cover.PROGRESS_EVERY + 17
    prog = cover.new_progress()
    res = cover.search(pool, root, keep, k, 3, np.inf, limit, progress=prog)
    assert res.status == "node_limit" and res.nodes == limit + 1
    _final_is_the_result(prog, res)


# --- edge cases ---------------------------------------------------------------------------------------------------

def test_a_search_with_nothing_to_search_still_reports_it_ran():
    """Edge: no groups to search, and fixed groups that are already the design: the search count rises, the node
    count is 0 and the search is done."""
    _, _, pool, k, root = _setup(7, 6, 2)
    prog = cover.new_progress()
    res = cover.search(pool, root, np.zeros(0, np.int64), k, 2, np.inf, 100, progress=prog)
    _final_is_the_result(prog, res)
    design = cover.start_design(pool, root, k, 2).rows
    res2 = cover.search(pool, root, np.zeros(0, np.int64), k, 2, np.inf, 100, fixed=design, progress=prog)
    _final_is_the_result(prog, res2, searches=2, total=res.nodes + res2.nodes)


def test_a_search_with_a_unit_no_group_covers_reports_it_ran():
    """Edge: kept groups that leave a unit uncovered end at the root with 0 nodes."""
    _, _, pool, k, root = _setup(8, 8, 2)
    unit0 = np.flatnonzero((pool.members == 0).any(axis=1))
    keep = np.setdiff1d(np.arange(len(pool)), unit0)
    prog = cover.new_progress()
    res = cover.search(pool, root, keep, k, 2, np.inf, 1000, progress=prog)
    assert res.nodes == 0 and res.rows.size == 0
    _final_is_the_result(prog, res)


# --- failure: a malformed array is refused before the search -------------------------------------------------------

@pytest.mark.parametrize("bad", [
    np.zeros(cover.PROGRESS_SLOTS, dtype=np.int64),
    np.zeros(cover.PROGRESS_SLOTS - 1),
    np.zeros((cover.PROGRESS_SLOTS, 2)),
    np.zeros(2 * cover.PROGRESS_SLOTS)[::2],
    [0.0] * cover.PROGRESS_SLOTS,
])
def test_a_malformed_progress_array_is_refused(bad):
    _, _, pool, k, root = _setup(9, 8, 2)
    with pytest.raises(MlsynthConfigError, match="progress"):
        cover.search(pool, root, np.arange(len(pool)), k, 2, np.inf, 100, progress=bad)


def test_a_read_only_progress_array_is_refused():
    _, _, pool, k, root = _setup(9, 8, 2)
    prog = cover.new_progress()
    prog.setflags(write=False)
    with pytest.raises(MlsynthConfigError, match="progress"):
        cover.search(pool, root, np.arange(len(pool)), k, 2, np.inf, 100, progress=prog)
