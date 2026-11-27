"""PANGEO's proof is the volume-bound tree (:mod:`mlsynth.utils.pangeo_helpers.vtree`).

:func:`columns._search_from`, the proof from the better of the quick search's and the first window's designs, runs
the tree over the groups that design's upper bound cannot exclude, with the design as the incumbent. Its nodes are
capped at the smaller of ``search_node_limit`` and ``cover.VOLUME_NODE_LIMIT``; a capped proof is followed by the
iterative cutoff (test_pangeo_certify.py), or with ``certify_node_limit=0`` leaves the MIP loop to run as before. The oracle is the plain cover search from the root's duals over every group, run to its proof, which
``test_pangeo_cover.py`` checks against the full MIP.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from mlsynth.utils.pangeo_helpers import columns, cover, mip, vtree


def _panel(seed, n, T=12, levels=5.0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, T)).cumsum(axis=1) + rng.normal(size=(n, 1)) * levels


def _optimum(idx, Y, Q, min_pairs=1):
    n = len(idx)
    pool = columns.score_columns(idx, Y, Q)
    k = columns.min_groups(n, Q, min_pairs)
    res = cover.search(pool, columns.root_lp(pool, k), np.arange(len(pool)), k, Q, np.inf, 10**9)
    assert res.status == "proved"
    return float(pool.cost[res.rows].sum())


class _Spy:
    """Records the tree's calls (keep, incumbent, node limit, result) and each MIP solve."""

    def __init__(self, set_attr):
        self.trees, self.mips = [], []
        real = vtree.search

        def tree(pool, root, keep, k, best_up, node_limit, **kw):
            out = real(pool, root, keep, k, best_up, node_limit, **kw)
            self.trees.append(dict(keep=np.asarray(keep), best_up=best_up, limit=node_limit, out=out))
            return out

        def solve(cands, idx, **kw):
            self.mips.append(1)
            return mip.solve_partition(cands, idx, **kw)
        set_attr(vtree, "search", tree)
        self.solve = solve


# --- smoke --------------------------------------------------------------------------------------------------------

def test_smoke_a_capped_quick_search_is_proved_by_the_volume_tree(monkeypatch):
    """Smoke: on a 14-unit Q = 3 arm whose quick search stops at its cap, the proof is the volume tree, it proves the
    optimum, and no MIP runs."""
    n, Q = 14, 3
    Y, idx = _panel(2, n), np.arange(n)
    monkeypatch.setattr(cover, "SEARCH_QUICK_LIMIT", 30)
    spy = _Spy(monkeypatch.setattr)
    _, d = columns.solve_exact(idx, Y, Q, solve=spy.solve)
    assert len(spy.trees) == 1 and spy.mips == []
    assert (d["search"], d["second_search"], d["mip_solves"]) == ("proved", "volume", 0)
    assert d["objective_value"] == pytest.approx(_optimum(idx, Y, Q), rel=1e-9)
    assert d["dual_bound"] <= d["objective_value"] * (1 + 1e-12) + 1e-12


# --- unit ---------------------------------------------------------------------------------------------------------

def test_the_tree_searches_the_groups_the_incumbent_cannot_exclude(monkeypatch):
    n, Q = 14, 3
    Y, idx = _panel(2, n), np.arange(n)
    monkeypatch.setattr(cover, "SEARCH_QUICK_LIMIT", 30)
    spy = _Spy(monkeypatch.setattr)
    columns.solve_exact(idx, Y, Q, solve=spy.solve)
    pool = columns.score_columns(idx, Y, Q)
    root = columns.root_lp(pool, columns.min_groups(n, Q, 1))
    t, = spy.trees
    z_up = Fraction(t["best_up"])
    assert np.array_equal(t["keep"], np.flatnonzero(root.reduced_costs <= columns._up(z_up - Fraction(root.bound))))


def test_the_tree_is_held_to_the_volume_node_limit(monkeypatch):
    n, Q = 14, 3
    Y, idx = _panel(2, n), np.arange(n)
    monkeypatch.setattr(cover, "SEARCH_QUICK_LIMIT", 30)
    monkeypatch.setattr(cover, "VOLUME_NODE_LIMIT", 123)
    spy = _Spy(monkeypatch.setattr)
    columns.solve_exact(idx, Y, Q, solve=spy.solve)
    assert spy.trees[0]["limit"] == 123


def test_the_tree_is_held_to_the_search_node_limit(monkeypatch):
    n, Q = 14, 3
    Y, idx = _panel(2, n), np.arange(n)
    monkeypatch.setattr(cover, "SEARCH_QUICK_LIMIT", 30)
    monkeypatch.setattr(cover, "WINDOW_NODE_LIMIT", 0)
    spy = _Spy(monkeypatch.setattr)
    columns.solve_exact(idx, Y, Q, solve=spy.solve, search_node_limit=77)
    assert spy.trees[0]["limit"] == 77


def test_a_capped_tree_leaves_the_mip_loop_to_prove(monkeypatch):
    """A tree stopped by its cap reports ``node_limit``; with the certify budget at 0 the MIP loop then runs as
    before and reaches the optimum (the iterative cutoff that otherwise follows is test_pangeo_certify.py's)."""
    n, Q = 14, 3
    Y, idx = _panel(2, n), np.arange(n)
    monkeypatch.setattr(cover, "SEARCH_QUICK_LIMIT", 30)
    monkeypatch.setattr(cover, "VOLUME_NODE_LIMIT", 1)
    spy = _Spy(monkeypatch.setattr)
    _, d = columns.solve_exact(idx, Y, Q, solve=spy.solve, certify_node_limit=0)
    assert spy.trees[0]["out"].status == "node_limit" and len(spy.mips) >= 1
    assert d["search"] == "node_limit" and d["second_search"] == "volume"
    assert d["objective_value"] == pytest.approx(_optimum(idx, Y, Q), rel=1e-9)


def test_the_proof_reports_through_the_progress_array(monkeypatch):
    n, Q = 14, 3
    Y, idx = _panel(2, n), np.arange(n)
    monkeypatch.setattr(cover, "SEARCH_QUICK_LIMIT", 30)
    prog = cover.new_progress()
    spy = _Spy(monkeypatch.setattr)
    _, d = columns.solve_exact(idx, Y, Q, solve=spy.solve, progress=prog)
    assert prog[cover.P_TOTAL] == d["search_nodes"]
    assert prog[cover.P_NODES] == spy.trees[-1]["out"].nodes and prog[cover.P_DONE] == 1.0


def test_the_search_from_result_names_the_volume_stage():
    n, Q = 12, 2
    Y, idx = _panel(3, n), np.arange(n)
    pool = columns.score_columns(idx, Y, Q)
    k = columns.min_groups(n, Q, 1)
    root = columns.root_lp(pool, k)
    rows = np.array([columns.column_index(g, n) for g in columns.seed_partition(n)])
    z_up = sum((Fraction(float(pool.cost_up[j])) for j in rows), Fraction(0))
    found = columns._search_from(pool, root, k, Q, 10**9, z_up, rows)
    assert found["stage"] == "volume" and found["search"] == "proved"
    assert float(pool.cost[found["rows"]].sum()) == pytest.approx(_optimum(idx, Y, Q), rel=1e-9)


# --- edges --------------------------------------------------------------------------------------------------------

def test_an_incumbent_at_the_optimum_is_returned_when_the_tree_finds_nothing_cheaper():
    n, Q = 12, 2
    Y, idx = _panel(4, n), np.arange(n)
    pool = columns.score_columns(idx, Y, Q)
    k = columns.min_groups(n, Q, 1)
    root = columns.root_lp(pool, k)
    best = cover.search(pool, root, np.arange(len(pool)), k, Q, np.inf, 10**9).rows
    z_up = sum((Fraction(float(pool.cost_up[j])) for j in best), Fraction(0))
    found = columns._search_from(pool, root, k, Q, 10**9, z_up, best)
    assert found["search"] == "proved" and sorted(found["rows"]) == sorted(best)


# --- property -----------------------------------------------------------------------------------------------------

@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**6), n=st.integers(8, 13), Q=st.integers(2, 3), quick=st.integers(1, 60),
       vlimit=st.sampled_from([1, 50, 10**7]))
def test_property_the_solve_is_exact_whatever_caps_the_proof(seed, n, Q, quick, vlimit):
    """Property: over arms, quick caps and proof caps, the solve reaches the optimum, and a proof that did not prove
    leaves the result to the MIP loop."""
    if not columns.feasible(n, Q, 1):
        return
    Y, idx = _panel(seed, n), np.arange(n)
    saved = (cover.SEARCH_QUICK_LIMIT, cover.VOLUME_NODE_LIMIT)
    cover.SEARCH_QUICK_LIMIT, cover.VOLUME_NODE_LIMIT = quick, vlimit
    try:
        _, d = columns.solve_exact(idx, Y, Q)
    finally:
        cover.SEARCH_QUICK_LIMIT, cover.VOLUME_NODE_LIMIT = saved
    assert d["objective_value"] == pytest.approx(_optimum(idx, Y, Q), rel=1e-9)
    if d.get("second_search") == "volume" and d["search"] == "node_limit":
        assert d["mip_solves"] >= 1
