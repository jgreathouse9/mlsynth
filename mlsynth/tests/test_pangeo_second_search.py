"""The proof after the first MIP: its incumbent, and the cost of its gate.

When the quick cover search stops at its cap, :func:`columns.solve_exact` runs the first MIP and then searches again
from an incumbent. That incumbent is the better of the two designs it holds: the quick search's and the first MIP's.
A first MIP held to too few groups can return a design far above the quick search's (on a 36-geo Q = 3 seasonal arm,
920.1 against 1.104), and searching from it keeps nearly every group.

The gated search's plain-stage budget (:func:`columns._reopt_budget`) counts the columns re-optimisation would solve,
branch by branch; it is used only up to the node limit, so it stops counting there. Counted in full it costs the
number of first-level branches times the kept groups: over an hour on that arm.
"""
from __future__ import annotations

import time
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from mlsynth.utils.pangeo_helpers import columns, cover, mip


def _panel(seed, n, T=12, levels=5.0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, T)).cumsum(axis=1) + rng.normal(size=(n, 1)) * levels


def _rows_up(pool, rows):
    return sum((Fraction(float(pool.cost_up[j])) for j in rows), Fraction(0))


def _rows_of(chosen, n):
    return [columns.column_index(tuple(sorted(int(u) for u in c["members"])), n) for c in chosen]


def _recording_mip():
    """The real MIP, recording each solve's design."""
    designs = []

    def solve(cands, idx, **kw):
        out = mip.solve_partition(cands, idx, **kw)
        designs.append(out[0])
        return out
    return solve, designs


def _poor_first_window(set_attr):
    """A first MIP window of one group plus the seed groups, so the first MIP covers the arm only with seed pairs and
    returns the optimum over those, far above the arm's optimum: the first MIP #1166 reports."""
    set_attr(columns, "FIRST_PER_UNIT", 0)
    set_attr(columns, "FIRST_MIN", 1)


def _spy_gated(monkeypatch):
    seen, real = [], columns._search_gated

    def rec(pool, root, keep, k, Q, best_up, node_limit, **kw):
        seen.append((len(keep), best_up))
        return real(pool, root, keep, k, Q, best_up, node_limit, **kw)
    monkeypatch.setattr(columns, "_search_gated", rec)
    return seen


def _spy_quick(monkeypatch):
    seen, real = [], columns._search

    def rec(*a, **kw):
        out = real(*a, **kw)
        seen.append(out)
        return out
    monkeypatch.setattr(columns, "_search", rec)
    return seen


# --- the incumbent of the proof after the first MIP ---------------------------------------------------------------

def test_smoke_a_poor_first_mip_does_not_set_the_second_searchs_incumbent(monkeypatch):
    """Smoke: on a 14-unit Q = 3 arm the quick search stops at its cap holding a design of cost about 4.5, and a
    first MIP held to one group and the seed groups returns a design ten times dearer. The second search starts from the quick search's design,
    keeps only the groups its margin admits, and the solve returns the optimum."""
    n, Q = 14, 3
    Y, idx = _panel(2, n), np.arange(n)
    _poor_first_window(monkeypatch.setattr)
    solve, designs = _recording_mip()
    gated, quick = _spy_gated(monkeypatch), _spy_quick(monkeypatch)
    chosen, d = columns.solve_exact(idx, Y, Q, search_node_limit=30, solve=solve)
    pool = columns.score_columns(idx, Y, Q)
    root = columns.root_lp(pool, columns.min_groups(n, Q, 1))
    q_up = _rows_up(pool, quick[0]["rows"])
    mip_up = _rows_up(pool, _rows_of(designs[0], n))
    assert q_up < mip_up / 10                                         # the first MIP is the poor one
    (kept, best_up), = gated
    assert best_up == columns._up(q_up)
    assert kept == int((root.reduced_costs <= columns._up(q_up - Fraction(root.bound))).sum()) < len(pool)
    _, d_ref = columns.solve_exact(idx, Y, Q)                      # the optimum, cross-checked in test_pangeo_cover
    assert d["objective_value"] == pytest.approx(d_ref["objective_value"], rel=1e-9)


def test_the_incumbent_is_the_design_of_least_upper_bound():
    """Unit: of two designs, the incumbent is the one whose rigorous upper bound (the sum of its groups' cost_up, in
    rationals) is the smaller, in either order, with that bound."""
    n, Q = 10, 2
    pool = columns.score_columns(np.arange(n), _panel(7, n), Q)
    root = columns.root_lp(pool, columns.min_groups(n, Q, 1))
    good = cover.start_design(pool, root, columns.min_groups(n, Q, 1), Q).rows
    poor = np.array([columns.column_index(g, n) for g in columns.seed_partition(n)])
    assert _rows_up(pool, good) < _rows_up(pool, poor)
    for first, second in ((good, poor), (poor, good)):
        z, rows = columns._incumbent(pool, [first, second])
        assert z == _rows_up(pool, good) and np.array_equal(rows, good)


def test_an_empty_design_is_never_the_incumbent_and_a_tie_keeps_the_first():
    n, Q = 10, 2
    pool = columns.score_columns(np.arange(n), _panel(7, n), Q)
    poor = np.array([columns.column_index(g, n) for g in columns.seed_partition(n)])
    z, rows = columns._incumbent(pool, [poor, np.zeros(0, np.int64)])
    assert z == _rows_up(pool, poor) and np.array_equal(rows, poor)
    z, rows = columns._incumbent(pool, [np.zeros(0, np.int64), poor])
    assert np.array_equal(rows, poor)
    twin = poor[::-1].copy()                                          # the same design, its groups in another order
    z, rows = columns._incumbent(pool, [poor, twin])
    assert rows is poor


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**6), n=st.integers(6, 13), Q=st.integers(2, 3), limit=st.sampled_from([5, 20, 60]))
def test_property_the_second_search_starts_from_the_better_design_and_the_solve_is_exact(seed, n, Q, limit):
    """Property: over panels, sizes and caps, with a first MIP held to one group and the seed groups, the second
    search's incumbent is the better of the quick search's and that MIP's designs, and the solve reaches the
    optimum."""
    if not columns.feasible(n, Q, 1):
        return
    Y, idx = _panel(seed, n), np.arange(n)
    solve, designs = _recording_mip()
    gated, quick = [], []
    real_g, real_q = columns._search_gated, columns._search
    saved = (columns.FIRST_PER_UNIT, columns.FIRST_MIN)
    _poor_first_window(setattr)

    def rec_g(pool, root, keep, k, Q_, best_up, node_limit, **kw):
        gated.append(best_up)
        return real_g(pool, root, keep, k, Q_, best_up, node_limit, **kw)

    def rec_q(*a, **kw):
        out = real_q(*a, **kw)
        quick.append(out)
        return out
    columns._search_gated, columns._search = rec_g, rec_q
    try:
        chosen, d = columns.solve_exact(idx, Y, Q, search_node_limit=limit, solve=solve)
    finally:
        columns._search_gated, columns._search = real_g, real_q
        columns.FIRST_PER_UNIT, columns.FIRST_MIN = saved
    pool = columns.score_columns(idx, Y, Q)
    if gated:
        mip_up = _rows_up(pool, _rows_of(designs[0], n))
        q_rows = quick[0]["rows"]
        want = min(mip_up, _rows_up(pool, q_rows)) if len(q_rows) else mip_up
        assert gated[0] == columns._up(want)
    _, d_ref = columns.solve_exact(idx, Y, Q)
    assert d["objective_value"] == pytest.approx(d_ref["objective_value"], rel=1e-9, abs=1e-12)
    covered = sorted(int(u) for c in chosen for u in c["members"])
    assert covered == list(range(n))


# --- the plain stage's budget, counted only up to its cap ----------------------------------------------------------

@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**6), n=st.integers(2, 10), Q=st.integers(1, 3), frac=st.floats(0.0, 1.0),
       cap=st.integers(0, 3000))
def test_property_the_capped_budget_is_the_budget_up_to_its_cap(seed, n, Q, frac, cap):
    """Property: over pools, kept subsets and caps, the capped budget is the smaller of the full count and the cap."""
    Y = _panel(seed, n)
    pool = columns.score_columns(np.arange(n), Y, Q)
    rng = np.random.default_rng(seed)
    keep = np.flatnonzero(rng.uniform(size=len(pool)) < frac)
    full = columns._reopt_budget(pool, keep)
    assert columns._reopt_budget(pool, keep, cap=cap) == min(full, cap)


def test_the_capped_budget_stops_counting_at_its_cap():
    """Unit: over every group of a 20-unit Q = 3 arm (60,439 groups, thousands of first-level branches) the budget
    capped at 1,000 is 1,000 and takes well under a second; counted in full it visits every branch."""
    n, Q = 20, 3
    pool = columns.score_columns(np.arange(n), _panel(3, n), Q)
    keep = np.arange(len(pool))
    t = time.perf_counter()
    b = columns._reopt_budget(pool, keep, cap=1000)
    assert b == 1000
    assert time.perf_counter() - t < 0.5


def test_a_cap_of_zero_or_no_kept_group_gives_a_zero_budget():
    pool = columns.score_columns(np.arange(8), _panel(4, 8), 2)
    assert columns._reopt_budget(pool, np.arange(len(pool)), cap=0) == 0
    assert columns._reopt_budget(pool, np.zeros(0, np.int64), cap=100) == 0


def test_the_gated_search_counts_its_budget_only_up_to_its_node_limit(monkeypatch):
    """Unit: the gated search asks for the budget capped at its node limit."""
    n, Q = 12, 2
    pool = columns.score_columns(np.arange(n), _panel(5, n), Q)
    k = columns.min_groups(n, Q, 1)
    root = columns.root_lp(pool, k)
    seen, real = [], columns._reopt_budget

    def rec(pool_, keep_, cap=None):
        seen.append(cap)
        return real(pool_, keep_, cap=cap)
    monkeypatch.setattr(columns, "_reopt_budget", rec)
    columns._search_gated(pool, root, np.arange(len(pool)), k, Q, np.inf, 777)
    assert seen == [777]
