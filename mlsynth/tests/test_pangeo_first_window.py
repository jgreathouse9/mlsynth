"""The first window is solved by the exact-cover search, not a MIP, and the proof starts from the better design.

When the quick cover search stops at its cap, :func:`columns.solve_exact` ran the first HiGHS MIP over a window: the
``FIRST_PER_UNIT`` n groups of least reduced cost and the seed partition. Where the window holds the optimum, that MIP
gave the proof its incumbent: at Q = 2 the quick search's design was 8.7% above the optimum on a 50-geo seasonal arm
and the window's best was the optimum. Skipping the MIP and proving from the quick design took that arm from 34 s to
335 s (99 million nodes against 4.6 million). Where the window holds groups that cannot sit together (Q = 3), the
MIP returned 550.7 or 920.1 against a quick design at or within 0.15% of the optimum.

The cover search, re-optimised (:func:`columns._search_gated`), over the same window and from the quick search's
design, finds the window's best design or proves the window holds nothing better: the same designs as HiGHS on four
arms, in 3.4 s against 11.6 s (50 geos, Q = 2), 0.43 s against 5.8 s (40 geos, Q = 2), and under 0.01 s against 4.1
and 7.1 s (36 and 40 geos, Q = 3). The proof then starts from the better of the window's design and the quick
search's; the MIP loop runs only when that proof stops at its cap, and the proof is not run again after it.
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


def _first_window(pool, n, root):
    """The first MIP's window, as solve_exact forms it."""
    rc = root.reduced_costs
    count = min(len(pool), max(columns.FIRST_MIN, columns.FIRST_PER_UNIT * n))
    delta = float(np.partition(rc, count - 1)[count - 1])
    seed = np.array([columns.column_index(g, n) for g in columns.seed_partition(n)])
    return np.union1d(np.flatnonzero(rc <= delta), seed)


class _Log:
    """Records, in order, the quick search, the window search (its groups, incumbent, limit and result), the proof's
    incumbent and stage, and each MIP solve."""

    def __init__(self, set_attr, quick_rows=None):
        self.events, self.quick, self.windows, self.proofs = [], [], [], []
        real_q, real_w, real_f = columns._search, columns._search_window, columns._search_from

        def quick(*a, **kw):
            out = real_q(*a, **kw)
            if quick_rows is not None and out["search"] != "proved":   # a capped quick search holding a poorer design
                out = dict(out, rows=quick_rows(out))
            self.quick.append(out)
            self.events.append("quick")
            return out

        def window(pool, root, keep, k, Q, best_up, node_limit, **kw):
            out = real_w(pool, root, keep, k, Q, best_up, node_limit, **kw)
            self.windows.append(dict(keep=np.asarray(keep), best_up=best_up, limit=node_limit, out=out))
            self.events.append("window")
            return out

        def proof(pool, root, k, Q, node_limit, z_up, rows, **kw):
            out = real_f(pool, root, k, Q, node_limit, z_up, rows, **kw)
            self.proofs.append(dict(z_up=z_up, rows=np.asarray(rows), out=out))
            self.events.append("proof")
            return out

        def solve(cands, idx, **kw):
            self.events.append("mip")
            return mip.solve_partition(cands, idx, **kw)
        set_attr(columns, "_search", quick)
        set_attr(columns, "_search_window", window)
        set_attr(columns, "_search_from", proof)
        self.solve = solve


def _seed_rows(n):
    return lambda out: np.array([columns.column_index(g, n) for g in columns.seed_partition(n)], dtype=np.int64)


# --- smoke --------------------------------------------------------------------------------------------------------

def test_smoke_a_capped_quick_search_goes_to_the_window_search_then_the_proof_and_no_mip(monkeypatch):
    """Smoke: on a 14-unit Q = 3 arm whose quick search stops at its cap, the window is searched, the proof runs from
    the better design and proves it, and no MIP is solved."""
    n, Q = 14, 3
    Y, idx = _panel(2, n), np.arange(n)
    monkeypatch.setattr(cover, "SEARCH_QUICK_LIMIT", 30)
    log = _Log(monkeypatch.setattr)
    chosen, d = columns.solve_exact(idx, Y, Q, solve=log.solve)
    assert log.quick[0]["search"] == "node_limit"
    assert log.events == ["quick", "window", "proof"]
    assert (d["search"], d["mip_solves"], d["backend"]) == ("proved", 0, "search")
    assert d["second_search"] == log.proofs[0]["out"]["stage"]
    assert d["objective_value"] == pytest.approx(_optimum(idx, Y, Q), rel=1e-9)


# --- unit ---------------------------------------------------------------------------------------------------------

def test_the_window_is_the_first_mips_window_searched_from_the_quick_design(monkeypatch):
    n, Q = 14, 3
    Y, idx = _panel(2, n), np.arange(n)
    monkeypatch.setattr(cover, "SEARCH_QUICK_LIMIT", 30)
    log = _Log(monkeypatch.setattr)
    columns.solve_exact(idx, Y, Q, solve=log.solve)
    pool = columns.score_columns(idx, Y, Q)
    root = columns.root_lp(pool, columns.min_groups(n, Q, 1))
    w, = log.windows
    assert np.array_equal(np.sort(w["keep"]), _first_window(pool, n, root))
    assert w["best_up"] == columns._up(_rows_up(pool, log.quick[0]["rows"]))


def test_a_better_window_design_is_the_proofs_incumbent(monkeypatch):
    """Unit: when the quick search holds a poor design (here the seed partition) and the window holds a better one,
    the proof starts from the window's design and says so."""
    n, Q = 14, 3
    Y, idx = _panel(2, n), np.arange(n)
    monkeypatch.setattr(cover, "SEARCH_QUICK_LIMIT", 30)
    log = _Log(monkeypatch.setattr, quick_rows=_seed_rows(n))
    _, d = columns.solve_exact(idx, Y, Q, solve=log.solve)
    pool = columns.score_columns(idx, Y, Q)
    w_rows = log.windows[0]["out"].rows
    assert w_rows.size and _rows_up(pool, w_rows) < _rows_up(pool, log.quick[0]["rows"])
    assert log.proofs[0]["z_up"] == _rows_up(pool, w_rows)
    assert d["search_start"] == "first_window" and d["objective_value"] == pytest.approx(_optimum(idx, Y, Q), rel=1e-9)


def test_a_quick_design_the_window_does_not_beat_is_the_proofs_incumbent(monkeypatch):
    n, Q = 14, 3
    Y, idx = _panel(2, n), np.arange(n)
    monkeypatch.setattr(cover, "SEARCH_QUICK_LIMIT", 30)
    log = _Log(monkeypatch.setattr)
    _, d = columns.solve_exact(idx, Y, Q, solve=log.solve)
    pool = columns.score_columns(idx, Y, Q)
    assert log.windows[0]["out"].rows.size == 0                       # the window held nothing better
    assert log.proofs[0]["z_up"] == _rows_up(pool, log.quick[0]["rows"]) and d["search_start"] == "quick_design"


def test_the_window_search_is_held_to_the_search_node_limit(monkeypatch):
    n, Q = 14, 3
    Y, idx = _panel(2, n), np.arange(n)
    log = _Log(monkeypatch.setattr)
    columns.solve_exact(idx, Y, Q, search_node_limit=30, solve=log.solve)
    assert log.windows[0]["limit"] == 30


def test_a_capped_proof_leaves_the_mip_loop_and_is_not_repeated(monkeypatch):
    """Unit: when the proof stops at its cap, the MIP loop runs and reaches the optimum, and the proof is not run
    again after the first MIP."""
    n, Q = 14, 3
    Y, idx = _panel(2, n), np.arange(n)
    log = _Log(monkeypatch.setattr)
    _, d = columns.solve_exact(idx, Y, Q, search_node_limit=30, solve=log.solve)
    assert log.events[:4] == ["quick", "window", "proof", "mip"] and log.events.count("proof") == 1
    assert d["search"] == "node_limit" and d["mip_solves"] >= 1
    assert d["objective_value"] == pytest.approx(_optimum(idx, Y, Q), rel=1e-9)


# --- edge cases ---------------------------------------------------------------------------------------------------

def test_with_no_quick_design_the_window_search_supplies_one(monkeypatch):
    """Edge: a quick search that stops before holding any design leaves the window search, from no incumbent, to
    find one (the window holds the seed partition), and the proof runs from it with no MIP."""
    n, Q = 14, 3
    Y, idx = _panel(13, n), np.arange(n)
    monkeypatch.setattr(cover, "START_NODE_LIMIT", 0)
    monkeypatch.setattr(cover, "SEARCH_QUICK_LIMIT", 1)
    log = _Log(monkeypatch.setattr)
    _, d = columns.solve_exact(idx, Y, Q, solve=log.solve)
    assert len(log.quick[0]["rows"]) == 0 and log.windows[0]["best_up"] == np.inf
    assert log.events == ["quick", "window", "proof"] and d["search_start"] == "first_window"
    assert d["objective_value"] == pytest.approx(_optimum(idx, Y, Q), rel=1e-9)


def test_a_quick_search_that_proves_runs_no_window(monkeypatch):
    n, Q = 10, 2
    Y, idx = _panel(4, n), np.arange(n)
    log = _Log(monkeypatch.setattr)
    _, d = columns.solve_exact(idx, Y, Q, solve=log.solve)
    assert log.events == ["quick"] and d["search_start"] == "staged" and d["mip_solves"] == 0


def test_a_named_backend_gets_its_mip_and_no_search(monkeypatch):
    n, Q = 10, 2
    Y, idx = _panel(4, n), np.arange(n)
    log = _Log(monkeypatch.setattr)
    _, d = columns.solve_exact(idx, Y, Q, backend="highspy", solve=log.solve)
    assert set(log.events) == {"mip"} and d["mip_solves"] >= 1


# --- property -----------------------------------------------------------------------------------------------------

@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**6), n=st.integers(6, 13), Q=st.integers(2, 3), quick=st.integers(1, 60),
       limit=st.sampled_from([30, 10**7]), poor=st.booleans())
def test_property_the_proof_starts_from_the_better_design_and_the_solve_is_exact(seed, n, Q, quick, limit, poor):
    """Property: over arms, quick caps, node limits and quick designs (the search's, or the seed partition), the
    solve reaches the optimum; the window is searched once, before any proof or MIP; the proof runs at most once,
    from the least upper bound of the window's and the quick search's designs; and a proof that runs before a MIP
    stopped at its cap."""
    if not columns.feasible(n, Q, 1):
        return
    Y, idx = _panel(seed, n), np.arange(n)
    saved = (columns._search, columns._search_window, columns._search_from, cover.SEARCH_QUICK_LIMIT)
    log = _Log(setattr, quick_rows=_seed_rows(n) if poor else None)
    cover.SEARCH_QUICK_LIMIT = quick
    try:
        chosen, d = columns.solve_exact(idx, Y, Q, search_node_limit=limit, solve=log.solve)
    finally:
        columns._search, columns._search_window, columns._search_from, cover.SEARCH_QUICK_LIMIT = saved
    assert d["objective_value"] == pytest.approx(_optimum(idx, Y, Q), rel=1e-9, abs=1e-12)
    assert sorted(int(u) for c in chosen for u in c["members"]) == list(range(n))
    ev = log.events
    assert ev.count("window") <= 1 and ev.count("proof") <= 1
    if "window" in ev:
        assert ev.index("window") == 1 and ("mip" not in ev or ev.index("mip") > ev.index("window"))
    if "proof" in ev:
        pool = columns.score_columns(idx, Y, Q)
        designs = [r for r in (log.windows[0]["out"].rows, np.asarray(log.quick[0]["rows"], np.int64)) if len(r)]
        if ev.index("proof") < (ev.index("mip") if "mip" in ev else len(ev)):
            assert log.proofs[0]["z_up"] == min(_rows_up(pool, r) for r in designs)
    if "mip" in ev and "proof" in ev and ev.index("proof") < ev.index("mip"):
        assert log.proofs[0]["out"]["search"] == "node_limit"
