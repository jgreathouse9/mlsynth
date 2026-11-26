"""The proof starts from the quick search's design, before any MIP.

When the quick cover search stops at its cap holding a design, :func:`columns.solve_exact` ran the first MIP over a
window of the groups of least reduced cost and the seed partition, then the proof search from the better of the two
designs. That window holds near-duplicate groups that cannot sit together, so the first MIP covered most of the arm
with seed groups: 550.7 on a 40-geo Q = 3 seasonal arm and 920.1 on a 36-geo one, where the quick search held 1.34517
and 1.104 (0.15% and 0% above the optima). The proof always started from the quick design, and the first MIP was 7 s
spent before it. The proof now runs from the quick design first; the MIP loop runs only when it is capped, and the
proof after the first MIP only when the quick search found no design.
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


def _rows_up(pool, rows):
    return sum((Fraction(float(pool.cost_up[j])) for j in rows), Fraction(0))


def _optimum(idx, Y, Q):
    """The full MIP over every candidate group: an oracle that shares no code with the search."""
    pool = columns.score_columns(idx, Y, Q)
    _, d = mip.solve_partition(pool.candidates(), idx, min_pairs=1, return_diagnostics=True)
    return d["objective_value"]


class _Log:
    """Records, in order, the quick search's result, each proof search's incumbent and each MIP solve."""

    def __init__(self, set_attr):
        self.events, self.quick = [], []
        real_q, real_g = columns._search, columns._search_gated

        def quick(*a, **kw):
            out = real_q(*a, **kw)
            self.quick.append(out)
            self.events.append(("quick", None))
            return out

        def gated(pool, root, keep, k, Q, best_up, node_limit, **kw):
            self.events.append(("proof", best_up))
            return real_g(pool, root, keep, k, Q, best_up, node_limit, **kw)

        def solve(cands, idx, **kw):
            self.events.append(("mip", None))
            return mip.solve_partition(cands, idx, **kw)
        set_attr(columns, "_search", quick)
        set_attr(columns, "_search_gated", gated)
        self.solve = solve

    def kinds(self):
        return [e for e, _ in self.events]


# --- smoke --------------------------------------------------------------------------------------------------------

def test_smoke_a_capped_quick_search_with_a_design_is_proved_without_a_mip(monkeypatch):
    """Smoke: on a 14-unit Q = 3 arm whose quick search stops at its cap holding a design, the proof runs from that
    design, proves it, and no MIP is solved."""
    n, Q = 14, 3
    Y, idx = _panel(2, n), np.arange(n)
    monkeypatch.setattr(cover, "SEARCH_QUICK_LIMIT", 30)
    log = _Log(monkeypatch.setattr)
    chosen, d = columns.solve_exact(idx, Y, Q, solve=log.solve)
    assert log.quick[0]["search"] == "node_limit" and len(log.quick[0]["rows"])
    assert log.kinds() == ["quick", "proof"]
    assert (d["search"], d["search_start"], d["mip_solves"], d["backend"]) == ("proved", "quick_design", 0, "search")
    assert d["objective_value"] == pytest.approx(_optimum(idx, Y, Q), rel=1e-9)


# --- unit ---------------------------------------------------------------------------------------------------------

def test_the_proof_starts_from_the_quick_designs_rigorous_upper_bound(monkeypatch):
    n, Q = 14, 3
    Y, idx = _panel(2, n), np.arange(n)
    monkeypatch.setattr(cover, "SEARCH_QUICK_LIMIT", 30)
    log = _Log(monkeypatch.setattr)
    columns.solve_exact(idx, Y, Q, solve=log.solve)
    pool = columns.score_columns(idx, Y, Q)
    (_, best_up), = [e for e in log.events if e[0] == "proof"]
    assert best_up == columns._up(_rows_up(pool, log.quick[0]["rows"]))


def test_a_capped_proof_from_the_quick_design_leaves_the_mip_loop_and_is_not_repeated(monkeypatch):
    """Unit: when the proof from the quick design stops at its cap, the MIP loop runs and reaches the optimum, and
    the proof is not run again after the first MIP."""
    n, Q = 14, 3
    Y, idx = _panel(2, n), np.arange(n)
    log = _Log(monkeypatch.setattr)
    chosen, d = columns.solve_exact(idx, Y, Q, search_node_limit=30, solve=log.solve)
    kinds = log.kinds()
    assert kinds[:3] == ["quick", "proof", "mip"] and kinds.count("proof") == 1
    assert d["search"] == "node_limit" and d["mip_solves"] >= 1 and d["second_search"] is not None
    assert d["objective_value"] == pytest.approx(_optimum(idx, Y, Q), rel=1e-9)


def test_with_no_quick_design_the_first_mip_comes_before_the_proof(monkeypatch):
    """Unit: a quick search that stops before finding any design leaves the first MIP to supply the proof's
    incumbent, as before."""
    n, Q = 14, 3
    Y, idx = _panel(13, n), np.arange(n)
    monkeypatch.setattr(cover, "START_NODE_LIMIT", 0)
    monkeypatch.setattr(cover, "SEARCH_QUICK_LIMIT", 1)
    log = _Log(monkeypatch.setattr)
    _, d = columns.solve_exact(idx, Y, Q, solve=log.solve)
    assert len(log.quick[0]["rows"]) == 0
    assert log.kinds()[:3] == ["quick", "mip", "proof"]
    assert d["search_start"] == "first_mip" and d["objective_value"] == pytest.approx(_optimum(idx, Y, Q), rel=1e-9)


# --- edge cases ---------------------------------------------------------------------------------------------------

def test_a_quick_search_that_proves_runs_no_second_proof(monkeypatch):
    n, Q = 10, 2
    Y, idx = _panel(4, n), np.arange(n)
    log = _Log(monkeypatch.setattr)
    _, d = columns.solve_exact(idx, Y, Q, solve=log.solve)
    assert log.kinds() == ["quick"] and d["search_start"] == "staged" and d["mip_solves"] == 0


def test_a_named_backend_still_gets_its_mip_and_no_search(monkeypatch):
    n, Q = 10, 2
    Y, idx = _panel(4, n), np.arange(n)
    log = _Log(monkeypatch.setattr)
    _, d = columns.solve_exact(idx, Y, Q, backend="highspy", solve=log.solve)
    assert "quick" not in log.kinds() and "proof" not in log.kinds() and d["mip_solves"] >= 1


# --- property -----------------------------------------------------------------------------------------------------

@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**6), n=st.integers(6, 13), Q=st.integers(2, 3), quick=st.integers(1, 60),
       limit=st.sampled_from([30, 10**7]))
def test_property_the_first_proof_starts_from_the_quick_design_and_the_solve_is_exact(seed, n, Q, quick, limit):
    """Property: over arms, quick caps and node limits, the solve reaches the optimum; the proof runs at most once;
    when the quick search held a design the proof starts from it before any MIP; and a solve the quick design's
    proof finished solves no MIP."""
    if not columns.feasible(n, Q, 1):
        return
    Y, idx = _panel(seed, n), np.arange(n)
    saved = (columns._search, columns._search_gated, cover.SEARCH_QUICK_LIMIT)
    log = _Log(setattr)
    cover.SEARCH_QUICK_LIMIT = quick
    try:
        chosen, d = columns.solve_exact(idx, Y, Q, search_node_limit=limit, solve=log.solve)
    finally:
        columns._search, columns._search_gated, cover.SEARCH_QUICK_LIMIT = saved
    assert d["objective_value"] == pytest.approx(_optimum(idx, Y, Q), rel=1e-9, abs=1e-12)
    kinds = log.kinds()
    assert kinds.count("proof") <= 1
    q = log.quick[0] if log.quick else None
    if q is not None and q["search"] == "node_limit" and len(q["rows"]):
        pool = columns.score_columns(idx, Y, Q)
        assert kinds[1] == "proof" and log.events[1][1] == columns._up(_rows_up(pool, q["rows"]))
    if d.get("search_start") == "quick_design":
        assert "mip" not in kinds and d["mip_solves"] == 0
    assert sorted(int(u) for c in chosen for u in c["members"]) == list(range(n))
