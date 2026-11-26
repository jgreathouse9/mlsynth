"""The MIP loop's margin comes from the best design held, not only from the MIPs' designs.

:func:`columns.solve_exact` widens the MIP's window round by round until every group a cheaper design could use is in
it: a group whose reduced-cost lower bound exceeds z - L is in no design cheaper than z. The loop took z from its own
MIPs' designs. When the first MIP is held to too few groups (#1166) its design is far above the designs the cover
searches already hold, so the loop kept widening windows the searches' design had already closed: on a 24-geo Q = 4
seasonal arm, four MIPs and 190 s after the searches held a design within 3% of the optimum.

The loop now takes z as the least of its MIPs' designs and the searches' (the quick search's, and the proof after the
first MIP's when it stops at its cap). Its exit is unchanged in meaning: when the window holds every group with
reduced cost at most z - L, the last MIP's design is the optimum.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from mlsynth.utils.pangeo_helpers import columns, cover, mip


def _panel(seed, n, T=12, levels=5.0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, T)).cumsum(axis=1) + rng.normal(size=(n, 1)) * levels


def _up(pool, rows):
    return sum((Fraction(float(pool.cost_up[j])) for j in rows), Fraction(0))


class _Watch:
    """Spies on the quick search, the proof after the first MIP and the MIP, recording what each returned."""

    def __init__(self, set_attr):
        self.quick, self.second, self.windows = [], [], []
        rq, rf = columns._search, columns._search_from

        def q(*a, **k):
            out = rq(*a, **k)
            self.quick.append(out)
            return out

        def f(*a, **k):
            out = rf(*a, **k)
            self.second.append(out)
            return out
        set_attr(columns, "_search", q)
        set_attr(columns, "_search_from", f)

    def solve(self, cands, idx, **kw):
        self.windows.append(len(cands))
        return mip.solve_partition(cands, idx, **kw)

    def held(self, pool):
        """The least upper bound among the searches' designs, None when they hold none."""
        ups = [_up(pool, r["rows"]) for r in self.quick + self.second if len(r["rows"])]
        return min(ups) if ups else None


def _poor_first_window(set_attr):
    """A first MIP window of one group and the seed groups: the first MIP #1166 reports."""
    set_attr(columns, "FIRST_PER_UNIT", 0)
    set_attr(columns, "FIRST_MIN", 1)


def _admitted(pool, root, z, n):
    """The groups a design of upper bound z admits (reduced cost at most z - L), and the seed groups."""
    seed = {columns.column_index(g, n) for g in columns.seed_partition(n)}
    inside = set(np.flatnonzero(root.reduced_costs <= columns._up(z - Fraction(root.bound))).tolist())
    return len(inside | seed)


# --- smoke --------------------------------------------------------------------------------------------------------

def test_smoke_the_loop_stops_widening_past_the_searches_design(monkeypatch):
    """Smoke: a 16-unit Q = 3 arm whose quick search holds the optimum and whose first MIP is held to one group and
    the seed groups. Every MIP after the first is held within the groups the searches' design admits (from its own
    designs the loop's sixth window held 1,032 groups), and the design is the optimum."""
    n, Q = 16, 3
    Y, idx = _panel(9, n), np.arange(n)
    _poor_first_window(monkeypatch.setattr)
    w = _Watch(monkeypatch.setattr)
    chosen, d = columns.solve_exact(idx, Y, Q, search_node_limit=30, solve=w.solve)
    pool = columns.score_columns(idx, Y, Q)
    root = columns.root_lp(pool, columns.min_groups(n, Q, 1))
    held = w.held(pool)
    assert held is not None and w.second and w.second[0]["search"] == "node_limit"
    cap = _admitted(pool, root, held, n)
    assert all(size <= cap for size in w.windows[1:]) and cap < 1032
    _, d_ref = columns.solve_exact(idx, Y, Q)
    assert d["objective_value"] == pytest.approx(d_ref["objective_value"], rel=1e-9)


# --- unit ---------------------------------------------------------------------------------------------------------

def test_the_second_searchs_design_holds_the_margin_when_the_quick_search_found_none(monkeypatch):
    """Unit: a start and quick search that find no design leave the second search's design to set the margin: every
    MIP after the second search is held within the groups that design admits."""
    n, Q = 14, 3
    Y, idx = _panel(9, n), np.arange(n)
    _poor_first_window(monkeypatch.setattr)
    monkeypatch.setattr(cover, "START_NODE_LIMIT", 0)
    w = _Watch(monkeypatch.setattr)
    chosen, d = columns.solve_exact(idx, Y, Q, search_node_limit=1, solve=w.solve)
    pool = columns.score_columns(idx, Y, Q)
    root = columns.root_lp(pool, columns.min_groups(n, Q, 1))
    assert len(w.quick[0]["rows"]) == 0                         # the quick search held no design
    assert w.second and len(w.second[0]["rows"]) and w.second[0]["search"] == "node_limit"
    cap = _admitted(pool, root, _up(pool, w.second[0]["rows"]), n)
    assert all(size <= cap for size in w.windows[1:])
    _, d_ref = columns.solve_exact(idx, Y, Q)
    assert d["objective_value"] == pytest.approx(d_ref["objective_value"], rel=1e-9)


def test_the_margin_is_unchanged_when_no_search_ran():
    """Unit: with the search off (search_node_limit=0) the loop runs on its MIPs' designs alone and reaches the
    optimum, as before."""
    n, Q = 12, 2
    Y, idx = _panel(4, n), np.arange(n)
    c, d = columns.solve_exact(idx, Y, Q, search_node_limit=0)
    _, d_ref = columns.solve_exact(idx, Y, Q)
    assert d["search"] == "off" and d["mip_solves"] >= 1
    assert d["objective_value"] == pytest.approx(d_ref["objective_value"], rel=1e-9)


# --- property -----------------------------------------------------------------------------------------------------

@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**6), n=st.integers(6, 14), Q=st.integers(2, 3), limit=st.sampled_from([1, 5, 30]))
def test_property_every_window_after_the_first_is_within_the_held_designs_margin_and_the_solve_is_exact(
        seed, n, Q, limit):
    """Property: over panels, sizes and caps, with the first MIP held to one group and the seed groups, every MIP
    after the first holds at most the groups the best design the searches hold admits, and the solve reaches the
    optimum with every unit covered once."""
    if not columns.feasible(n, Q, 1):
        return
    Y, idx = _panel(seed, n), np.arange(n)
    saved = []

    def set_attr(obj, name, value):
        saved.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)
    try:
        _poor_first_window(set_attr)
        w = _Watch(set_attr)
        chosen, d = columns.solve_exact(idx, Y, Q, search_node_limit=limit, solve=w.solve)
    finally:
        for obj, name, value in reversed(saved):
            setattr(obj, name, value)
    pool = columns.score_columns(idx, Y, Q)
    root = columns.root_lp(pool, columns.min_groups(n, Q, 1))
    held = w.held(pool)
    if held is not None and len(w.windows) > 1:
        cap = _admitted(pool, root, held, n)
        assert all(size <= cap for size in w.windows[1:])
    _, d_ref = columns.solve_exact(idx, Y, Q)
    assert d["objective_value"] == pytest.approx(d_ref["objective_value"], rel=1e-9, abs=1e-12)
    assert sorted(int(u) for c in chosen for u in c["members"]) == list(range(n))
