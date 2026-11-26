"""The proof after the first MIP gets a budget that finishes the hard arms, and its plain stage a cap of its own.

The second search (:func:`columns._search_gated`) runs the plain search for as many nodes as the children's columns,
then the re-optimised search with the rest of ``search_node_limit``. At a cap of 1.5 million nodes, a 40-geo Q = 3
seasonal arm stopped there and went to the HiGHS MIP loop, whose fifth MIP passed 1,800 s. Given 200 million, the
second search proved the arm in 1,082 s with no further MIP, but its plain stage took 16.6 million of those nodes
(273 s) and proved nothing: the children's columns, the plain stage's budget, grow with the arm. The plain stage is
now capped at ``PLAIN_STAGE_LIMIT``, the 1.5 million nodes every earlier arm's plain stage ran within.
"""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from mlsynth.utils.pangeo_helpers import columns, cover


def _panel(seed, n, T=12, levels=5.0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, T)).cumsum(axis=1) + rng.normal(size=(n, 1)) * levels


def _setup(seed, n, Q, T=12):
    pool = columns.score_columns(np.arange(n), _panel(seed, n, T), Q)
    k = columns.min_groups(n, Q, 1)
    return pool, k, columns.root_lp(pool, k)


def _optimum(pool, root, k, Q):
    """The plain search from the root's duals over every group, run to its proof: an oracle that shares no code
    with the gate or the child LPs."""
    res = cover.search(pool, root, np.arange(len(pool)), k, Q, np.inf, 10**9)
    assert res.status == "proved"
    return float(pool.cost[res.rows].sum())


def _spy(monkeypatch):
    limits, real = [], cover.search

    def rec(pool, root, keep, k, Q, best, limit, fixed=(), **kw):
        if not fixed:
            limits.append(int(limit))
        return real(pool, root, keep, k, Q, best, limit, fixed=fixed, **kw)
    monkeypatch.setattr(cover, "search", rec)
    return limits


# --- smoke --------------------------------------------------------------------------------------------------------

def test_smoke_a_capped_plain_stage_hands_over_and_the_search_proves_the_optimum(monkeypatch):
    """Smoke: with the plain stage capped at 5 nodes, the gated search re-optimises and proves the optimum."""
    pool, k, root = _setup(9, 14, 2)
    monkeypatch.setattr(cover, "PLAIN_STAGE_LIMIT", 5)
    res = columns._search_gated(pool, root, np.arange(len(pool)), k, 2, np.inf, 10**7)
    assert res.status == "proved" and res.stage == "reoptimised"
    assert float(pool.cost[res.rows].sum()) == pytest.approx(_optimum(pool, root, k, 2), rel=1e-9)


# --- unit ---------------------------------------------------------------------------------------------------------

def test_the_plain_stage_stops_at_its_cap_when_the_childrens_columns_exceed_it(monkeypatch):
    pool, k, root = _setup(9, 14, 2)
    keep = np.arange(len(pool))
    assert columns._reopt_budget(pool, keep) > 5
    monkeypatch.setattr(cover, "PLAIN_STAGE_LIMIT", 5)
    limits = _spy(monkeypatch)
    columns._search_gated(pool, root, keep, k, 2, np.inf, 10**7)
    assert limits[0] == 5


def test_under_the_cap_the_plain_stage_still_gets_the_childrens_columns(monkeypatch):
    pool, k, root = _setup(9, 14, 2)
    keep = np.arange(len(pool))
    monkeypatch.setattr(cover, "PLAIN_STAGE_LIMIT", 10**9)
    limits = _spy(monkeypatch)
    columns._search_gated(pool, root, keep, k, 2, np.inf, 10**7)
    assert limits[0] == columns._reopt_budget(pool, keep)


def test_the_default_budgets_hold_the_40_geo_q3_proof():
    """Unit: the default proof budget holds the 66.7 million nodes the 40-geo Q = 3 arm's second search placed
    with its plain stage uncapped, and the plain stage's cap is the 1.5 million every earlier arm's plain stage ran
    within."""
    assert cover.SEARCH_NODE_LIMIT >= 66_739_755
    assert cover.PLAIN_STAGE_LIMIT == 1_500_000 < cover.SEARCH_NODE_LIMIT


def test_the_second_search_in_solve_exact_gets_the_default_budget(monkeypatch):
    """Unit: solve_exact hands the proof after the quick search the default node budget."""
    monkeypatch.setattr(cover, "START_NODE_LIMIT", 0)
    monkeypatch.setattr(cover, "SEARCH_QUICK_LIMIT", 1)
    seen, real = [], columns._search_from

    def rec(pool, root, k, Q, limit, z_up, rows, **kw):
        seen.append(int(limit))
        return real(pool, root, k, Q, limit, z_up, rows, **kw)
    monkeypatch.setattr(columns, "_search_from", rec)
    _, d = columns.solve_exact(np.arange(14), _panel(13, 14), 3)
    assert seen == [cover.SEARCH_NODE_LIMIT] and d["search"] == "proved"


# --- edge cases ---------------------------------------------------------------------------------------------------

def test_a_plain_cap_of_zero_goes_straight_to_reoptimisation(monkeypatch):
    pool, k, root = _setup(9, 14, 2)
    monkeypatch.setattr(cover, "PLAIN_STAGE_LIMIT", 0)
    limits = _spy(monkeypatch)
    res = columns._search_gated(pool, root, np.arange(len(pool)), k, 2, np.inf, 10**7)
    assert limits == [] and res.status == "proved" and res.stage == "reoptimised"


def test_a_node_limit_under_the_plain_cap_still_bounds_the_plain_stage(monkeypatch):
    pool, k, root = _setup(8, 14, 2)
    monkeypatch.setattr(cover, "PLAIN_STAGE_LIMIT", 10**9)
    limits = _spy(monkeypatch)
    res = columns._search_gated(pool, root, np.arange(len(pool)), k, 2, np.inf, 5)
    assert limits == [5] and res.status == "node_limit" and res.nodes <= 5 + 1


# --- property -----------------------------------------------------------------------------------------------------

@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**6), n=st.integers(4, 12), Q=st.integers(1, 3), cap=st.integers(0, 400),
       limit=st.sampled_from([0, 3, 60, 10**7]))
def test_property_the_plain_stage_is_the_least_of_the_three_budgets(seed, n, Q, cap, limit):
    """Property: over arms, plain caps and node limits, the plain stage gets the least of the children's columns,
    the node limit and the plain cap; the search places at most the limit (plus the node that crosses it); and a
    search that proves returns the optimum."""
    if not columns.feasible(n, Q, 1):
        return
    pool, k, root = _setup(seed, n, Q, T=8)
    keep = np.arange(len(pool))
    limits, real, saved = [], cover.search, cover.PLAIN_STAGE_LIMIT

    def rec(pool_, root_, keep_, k_, Q_, best, lim, fixed=(), **kw):
        if not fixed:
            limits.append(int(lim))
        return real(pool_, root_, keep_, k_, Q_, best, lim, fixed=fixed, **kw)
    cover.search, cover.PLAIN_STAGE_LIMIT = rec, cap
    try:
        res = columns._search_gated(pool, root, keep, k, Q, np.inf, limit)
    finally:
        cover.search, cover.PLAIN_STAGE_LIMIT = real, saved
    want = min(columns._reopt_budget(pool, keep, cap=limit), limit, cap)
    assert limits[:1] == ([want] if want > 0 else [])
    assert res.nodes <= limit + 1
    if res.status == "proved":
        assert float(pool.cost[res.rows].sum()) == pytest.approx(_optimum(pool, root, k, Q), rel=1e-9, abs=1e-12)
