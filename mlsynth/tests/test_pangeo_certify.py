"""PANGEO's certified gap: the iterative cutoff after a capped proof (:func:`columns._certify`).

When the proof from the best design (:func:`columns._search_from`) stops at its cap, rounds of the volume tree search
below the cutoffs ``L (1 + CERTIFY_STEP)^i`` (L the root bound), the last capped at the incumbent's upper bound. A
round that ends with no design proves every design costs at least its cutoff (or the tree's lowest leaf bound, when
lower); a round that ends with a design proves that design optimal. When the rounds' node budget
(``certify_node_limit``) runs out, :func:`columns.solve_exact` returns the best design with the highest bound proved,
reported as a certified gap, and no MIP runs. ``certify_node_limit=0`` leaves a capped proof to the MIP loop, as
before. The oracle is the plain cover search from the root's duals over every group, run to its proof.
"""
from __future__ import annotations

import warnings
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from mlsynth.exceptions import MlsynthConfigError
from mlsynth.utils.pangeo_helpers import columns, cover, mip, vtree
from mlsynth.utils.pangeo_helpers.structures import MipSolveReport, PangeoCertifiedGapWarning


def _panel(seed, n, T=12, levels=5.0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, T)).cumsum(axis=1) + rng.normal(size=(n, 1)) * levels


def _arm(seed, n, Q):
    Y, idx = _panel(seed, n), np.arange(n)
    pool = columns.score_columns(idx, Y, Q)
    k = columns.min_groups(n, Q, 1)
    root = columns.root_lp(pool, k)
    return Y, idx, pool, k, root


def _optimum(pool, root, k, Q):
    res = cover.search(pool, root, np.arange(len(pool)), k, Q, np.inf, 10**9)
    assert res.status == "proved"
    return float(pool.cost[res.rows].sum())


def _seed_design(pool, n):
    rows = np.array([columns.column_index(g, n) for g in columns.seed_partition(n)], dtype=np.int64)
    return rows, sum((Fraction(float(pool.cost_up[j])) for j in rows), Fraction(0))


def _far(root, z_up):
    """The scripted tests need the incumbent at least five steps above the root bound, so no early round is last."""
    assert z_up > Fraction(root.bound) * (1 + Fraction(cover.CERTIFY_STEP)) ** 5


class _Trees:
    """Records every tree the rounds run (keep, incumbent bound, node limit, result); ``script`` replaces the tree
    with a list of outcomes, one a call: ``("none", status, lower)`` for no design or ``("found", status, lower)``
    for the first kept group's design (any rows stand for a design here)."""

    def __init__(self, set_attr, script=None):
        self.calls, real, self.script = [], vtree.search, list(script) if script is not None else None

        def tree(pool, root, keep, k, best_up, node_limit, **kw):
            if self.script is None:
                out = real(pool, root, keep, k, best_up, node_limit, **kw)
            else:
                kind, status, lower = self.script.pop(0)
                rows = np.asarray(keep[:1], dtype=np.int64) if kind == "found" else np.zeros(0, np.int64)
                up = float(pool.cost_up[rows].sum()) if rows.size else float(best_up)
                out = cover.CoverSearch(status, rows, up, float(lower), 3, stage="volume")
            self.calls.append(dict(keep=np.asarray(keep), best_up=best_up, limit=int(node_limit), out=out))
            return out
        set_attr(vtree, "search", tree)


# --- smoke --------------------------------------------------------------------------------------------------------

def test_smoke_the_rounds_prove_the_optimum_from_a_poor_design():
    """Smoke: from the seed design of a 12-unit Q = 2 arm, the rounds end proved with the optimum and a rigorous
    lower bound at or below it."""
    n, Q = 12, 2
    _, _, pool, k, root = _arm(3, n, Q)
    rows, z_up = _seed_design(pool, n)
    out = columns._certify(pool, root, k, Q, z_up, rows, 10**8)
    z = _optimum(pool, root, k, Q)
    assert out["search"] == "proved" and out["stage"] == "certify" and out["rounds"] >= 1
    assert float(pool.cost[out["rows"]].sum()) == pytest.approx(z, rel=1e-9)
    assert out["lower"] <= z * (1 + 1e-12) and np.isfinite(out["lower"]) and out["nodes"] > 0


def test_smoke_solve_exact_certifies_a_capped_proof_without_a_mip(monkeypatch):
    """Smoke: a 14-unit Q = 3 arm whose quick search and proof both stop at their caps is proved by the rounds, and
    no MIP runs."""
    n, Q = 14, 3
    Y, idx, pool, k, root = _arm(2, n, Q)
    monkeypatch.setattr(cover, "SEARCH_QUICK_LIMIT", 30)
    monkeypatch.setattr(cover, "VOLUME_NODE_LIMIT", 1)
    mips = []
    _, d = columns.solve_exact(idx, Y, Q, solve=lambda c, i, **kw: mips.append(1) or mip.solve_partition(c, i, **kw))
    assert mips == [] and d["mip_solves"] == 0 and d["search"] == "proved" and d["backend"] == "search"
    assert d["certify_rounds"] >= 1 and d["status"] == "optimal"
    assert d["objective_value"] == pytest.approx(_optimum(pool, root, k, Q), rel=1e-9)


# --- unit: the rounds ---------------------------------------------------------------------------------------------

def test_each_round_searches_below_its_cutoff_over_the_groups_that_cutoff_keeps(monkeypatch):
    """The i-th round's incumbent bound is the least double at or above L (1 + step)^i, and its groups are those
    whose reduced cost is at most that cutoff less L."""
    n, Q = 12, 2
    _, _, pool, k, root = _arm(3, n, Q)
    rows, z_up = _seed_design(pool, n)
    _far(root, z_up)
    L = Fraction(root.bound)
    trees = _Trees(monkeypatch.setattr, script=[("none", "proved", 1e300)] * 2 + [("none", "node_limit", 0.0)])
    columns._certify(pool, root, k, Q, z_up, rows, 10**6)
    step = Fraction(cover.CERTIFY_STEP)
    for i, call in enumerate(trees.calls[:3], start=1):
        U = L * (1 + step) ** i
        assert call["best_up"] == columns._up(U)
        assert np.array_equal(call["keep"], np.flatnonzero(root.reduced_costs <= columns._up(U - L)))


def test_no_round_searches_above_the_incumbent_and_the_last_is_capped_at_it(monkeypatch):
    """From a poor design through the real tree, every round's bound is at most the incumbent's upper bound, and the
    round that proves is the one capped at it or one that found a cheaper design."""
    n, Q = 12, 2
    _, _, pool, k, root = _arm(5, n, Q)
    rows, z_up = _seed_design(pool, n)
    trees = _Trees(monkeypatch.setattr)
    out = columns._certify(pool, root, k, Q, z_up, rows, 10**8)
    assert out["search"] == "proved" and all(c["best_up"] <= columns._up(z_up) for c in trees.calls)
    last = trees.calls[-1]
    assert last["best_up"] == columns._up(z_up) or last["out"].rows.size > 0


def test_a_round_with_no_design_proves_the_lesser_of_its_cutoff_and_its_lowest_leaf(monkeypatch):
    """A proved round with no design bounds every design below by its cutoff, or by the tree's lowest leaf bound
    when that is lower; a later round stopped by its cap leaves that bound standing."""
    n, Q = 12, 2
    _, _, pool, k, root = _arm(3, n, Q)
    rows, z_up = _seed_design(pool, n)
    _far(root, z_up)
    L = Fraction(root.bound)
    U1 = L * (1 + Fraction(cover.CERTIFY_STEP))
    leaf = float(L) + 0.25 * float(U1 - L)                 # below the first cutoff
    _Trees(monkeypatch.setattr, script=[("none", "proved", leaf), ("none", "node_limit", 0.0)])
    out = columns._certify(pool, root, k, Q, z_up, rows, 10**6)
    assert out["search"] == "node_limit" and out["lower"] == columns._down(Fraction(leaf))
    assert sorted(out["rows"]) == sorted(rows) and out["rounds"] == 2


def test_a_proved_round_with_no_design_and_a_high_leaf_proves_its_cutoff(monkeypatch):
    n, Q = 12, 2
    _, _, pool, k, root = _arm(3, n, Q)
    rows, z_up = _seed_design(pool, n)
    _far(root, z_up)
    L = Fraction(root.bound)
    U1 = L * (1 + Fraction(cover.CERTIFY_STEP))
    _Trees(monkeypatch.setattr, script=[("none", "proved", 1e300), ("none", "node_limit", 0.0)])
    out = columns._certify(pool, root, k, Q, z_up, rows, 10**6)
    assert out["lower"] == columns._down(U1) and out["cutoff"] == columns._down(U1)


def test_a_round_that_finds_and_proves_a_design_ends_the_rounds_with_it(monkeypatch):
    n, Q = 12, 2
    _, _, pool, k, root = _arm(3, n, Q)
    rows, z_up = _seed_design(pool, n)
    _far(root, z_up)
    trees = _Trees(monkeypatch.setattr, script=[("none", "proved", 1e300), ("found", "proved", 0.0)])
    out = columns._certify(pool, root, k, Q, z_up, rows, 10**6)
    assert out["search"] == "proved" and len(trees.calls) == 2
    assert np.array_equal(out["rows"], trees.calls[1]["out"].rows) and out["lower"] == 0.0


def test_a_design_found_by_a_round_its_cap_stopped_is_kept_with_the_earlier_bound(monkeypatch):
    n, Q = 12, 2
    _, _, pool, k, root = _arm(3, n, Q)
    rows, z_up = _seed_design(pool, n)
    _far(root, z_up)
    L = Fraction(root.bound)
    U1 = L * (1 + Fraction(cover.CERTIFY_STEP))
    trees = _Trees(monkeypatch.setattr, script=[("none", "proved", 1e300), ("found", "node_limit", 0.0)])
    out = columns._certify(pool, root, k, Q, z_up, rows, 10**6)
    assert out["search"] == "node_limit" and np.array_equal(out["rows"], trees.calls[1]["out"].rows)
    assert out["lower"] == columns._down(U1)


def test_the_rounds_share_one_node_budget(monkeypatch):
    """Each round gets what the earlier rounds left of the budget, and the rounds together place at most it."""
    n, Q = 12, 2
    _, _, pool, k, root = _arm(3, n, Q)
    rows, z_up = _seed_design(pool, n)
    _far(root, z_up)
    trees = _Trees(monkeypatch.setattr, script=[("none", "proved", 1e300)] * 3 + [("none", "node_limit", 0.0)])
    out = columns._certify(pool, root, k, Q, z_up, rows, 100)
    assert [c["limit"] for c in trees.calls] == [100, 97, 94, 91]
    assert out["nodes"] == 12 and out["search"] == "node_limit"


# --- edges --------------------------------------------------------------------------------------------------------

def test_a_budget_of_zero_runs_no_round_and_proves_the_root_bound(monkeypatch):
    n, Q = 12, 2
    _, _, pool, k, root = _arm(3, n, Q)
    rows, z_up = _seed_design(pool, n)
    trees = _Trees(monkeypatch.setattr)
    out = columns._certify(pool, root, k, Q, z_up, rows, 0)
    assert trees.calls == [] and out["search"] == "node_limit" and out["rounds"] == 0
    assert out["lower"] == columns._down(Fraction(root.bound)) and sorted(out["rows"]) == sorted(rows)


def test_an_incumbent_within_one_step_of_the_root_bound_takes_a_single_round(monkeypatch):
    """When the incumbent's bound is below the first cutoff, the only round is the plain proof below it."""
    n, Q = 12, 2
    _, _, pool, k, root = _arm(3, n, Q)
    best = cover.search(pool, root, np.arange(len(pool)), k, Q, np.inf, 10**9).rows
    z_up = sum((Fraction(float(pool.cost_up[j])) for j in best), Fraction(0))
    trees = _Trees(monkeypatch.setattr)
    monkeypatch.setattr(cover, "CERTIFY_STEP", float(z_up / Fraction(root.bound)))     # one step past the optimum
    out = columns._certify(pool, root, k, Q, z_up, best, 10**8)
    assert len(trees.calls) == 1 and trees.calls[0]["best_up"] == columns._up(z_up) and out["search"] == "proved"


# --- solve_exact: wiring and reporting ----------------------------------------------------------------------------

def test_a_certify_budget_of_zero_leaves_a_capped_proof_to_the_mip_loop(monkeypatch):
    n, Q = 14, 3
    Y, idx, pool, k, root = _arm(2, n, Q)
    monkeypatch.setattr(cover, "SEARCH_QUICK_LIMIT", 30)
    monkeypatch.setattr(cover, "VOLUME_NODE_LIMIT", 1)
    _, d = columns.solve_exact(idx, Y, Q, certify_node_limit=0)
    assert d["search"] == "node_limit" and d["mip_solves"] >= 1 and "certify_rounds" not in d
    assert d["objective_value"] == pytest.approx(_optimum(pool, root, k, Q), rel=1e-9)


def test_a_budget_that_runs_out_returns_the_design_with_its_certified_gap_and_warns(monkeypatch):
    """With a budget too small to finish, the design comes back without a MIP, reported as a certified gap: the
    dual bound at or below the optimum, the gap the distance from the design's upper bound to it, and a warning."""
    n, Q = 14, 3
    Y, idx, pool, k, root = _arm(2, n, Q)
    monkeypatch.setattr(cover, "SEARCH_QUICK_LIMIT", 30)
    monkeypatch.setattr(cover, "VOLUME_NODE_LIMIT", 1)
    mips = []
    with pytest.warns(PangeoCertifiedGapWarning, match="certified"):
        _, d = columns.solve_exact(idx, Y, Q, certify_node_limit=1,
                                   solve=lambda c, i, **kw: mips.append(1) or mip.solve_partition(c, i, **kw))
    z = _optimum(pool, root, k, Q)
    assert mips == [] and d["mip_solves"] == 0 and d["search"] == "certified" and d["status"] == "certified_gap"
    assert d["dual_bound"] <= z * (1 + 1e-12) <= d["objective_value"] * (1 + 1e-12) + 1e-300
    assert d["mip_gap"] == pytest.approx((d["objective_upper"] - d["dual_bound"]) / abs(d["objective_value"]))
    assert d["mip_gap"] >= 0 and d["certify_rounds"] >= 1
    report = MipSolveReport.from_diagnostics({**d, "solver_name": d["solver_name"]})
    assert report.search == "certified" and report.status == "certified_gap" and report.dual_bound == d["dual_bound"]


def test_the_default_certify_budget_is_the_module_constant(monkeypatch):
    n, Q = 14, 3
    Y, idx, *_ = _arm(2, n, Q)
    monkeypatch.setattr(cover, "SEARCH_QUICK_LIMIT", 30)
    monkeypatch.setattr(cover, "VOLUME_NODE_LIMIT", 1)
    seen = []
    real = columns._certify
    monkeypatch.setattr(columns, "_certify", lambda *a, **kw: seen.append(a[6]) or real(*a, **kw))
    columns.solve_exact(idx, Y, Q)
    assert seen == [cover.CERTIFY_NODE_LIMIT]


def _first_mip_setup(monkeypatch):
    """No quick or window design, so the proof starts from the first MIP's design, and the proof is capped."""
    monkeypatch.setattr(cover, "START_NODE_LIMIT", 0)
    monkeypatch.setattr(cover, "WINDOW_NODE_LIMIT", 0)
    monkeypatch.setattr(cover, "SEARCH_QUICK_LIMIT", 1)
    monkeypatch.setattr(cover, "VOLUME_NODE_LIMIT", 1)
    monkeypatch.setattr(columns, "FIRST_PER_UNIT", 0)
    monkeypatch.setattr(columns, "FIRST_MIN", 1)


def test_a_capped_proof_from_the_first_mips_design_is_certified_too(monkeypatch):
    """When the proof starts from the first MIP's design and stops at its cap, the rounds follow it, prove the
    optimum, and the MIP loop stops at that one MIP."""
    n, Q = 14, 3
    Y, idx, pool, k, root = _arm(13, n, Q)
    _first_mip_setup(monkeypatch)
    _, d = columns.solve_exact(idx, Y, Q)
    assert d["search_start"] == "first_mip" and d["search"] == "proved" and d["mip_solves"] == 1
    assert d["certify_rounds"] >= 1
    assert d["objective_value"] == pytest.approx(_optimum(pool, root, k, Q), rel=1e-9)


def test_a_certified_gap_from_the_first_mips_design_counts_that_mip(monkeypatch):
    n, Q = 14, 3
    Y, idx, pool, k, root = _arm(13, n, Q)
    _first_mip_setup(monkeypatch)
    with pytest.warns(PangeoCertifiedGapWarning):
        _, d = columns.solve_exact(idx, Y, Q, certify_node_limit=1)
    assert d["search"] == "certified" and d["status"] == "certified_gap" and d["mip_solves"] == 1
    assert d["dual_bound"] <= _optimum(pool, root, k, Q) * (1 + 1e-12)


# --- failures -----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [-1, 2.5, True, "10"])
def test_an_invalid_certify_budget_is_refused_by_name(bad):
    n, Q = 12, 2
    Y, idx, *_ = _arm(3, n, Q)
    with pytest.raises(MlsynthConfigError, match="certify_node_limit"):
        columns.solve_exact(idx, Y, Q, certify_node_limit=bad)


def test_the_config_refuses_a_negative_certify_budget():
    from mlsynth.utils.pangeo_helpers import make_seasonal_sales_panel
    from mlsynth.utils.pangeo_helpers.config import PANGEOConfig
    df = make_seasonal_sales_panel(units_per_arm=6, arms=("A",), T=20, seed=0)
    with pytest.raises(Exception, match="certify_node_limit"):
        PANGEOConfig(df=df, outcome="sales", arm="arm", unitid="unit", time="time", certify_node_limit=-1)
    cfg = PANGEOConfig(df=df, outcome="sales", arm="arm", unitid="unit", time="time")
    assert cfg.certify_node_limit == cover.CERTIFY_NODE_LIMIT


# --- properties ---------------------------------------------------------------------------------------------------

@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**6), n=st.integers(8, 13), Q=st.integers(2, 3), quick=st.integers(1, 60),
       vlimit=st.sampled_from([1, 50, 10**7]), budget=st.sampled_from([0, 1, 5, 50, 10**7]))
def test_property_the_design_and_its_bound_bracket_the_optimum_whatever_the_caps(seed, n, Q, quick, vlimit, budget):
    """Property: over arms and every cap, the dual bound is at or below the optimum and the design at or above it;
    a proof (search "proved") returns the optimum; a certified gap returns without a MIP and a non-negative gap; and
    only a certify budget of 0 leaves a capped proof to the MIP loop."""
    if not columns.feasible(n, Q, 1):
        return
    Y, idx = _panel(seed, n), np.arange(n)
    pool = columns.score_columns(idx, Y, Q)
    k = columns.min_groups(n, Q, 1)
    z = _optimum(pool, columns.root_lp(pool, k), k, Q)
    saved = (cover.SEARCH_QUICK_LIMIT, cover.VOLUME_NODE_LIMIT)
    cover.SEARCH_QUICK_LIMIT, cover.VOLUME_NODE_LIMIT = quick, vlimit
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PangeoCertifiedGapWarning)
            _, d = columns.solve_exact(idx, Y, Q, certify_node_limit=budget)
    finally:
        cover.SEARCH_QUICK_LIMIT, cover.VOLUME_NODE_LIMIT = saved
    assert d["objective_value"] >= z * (1 - 1e-9)
    if d.get("dual_bound") is not None:
        assert d["dual_bound"] <= z * (1 + 1e-9) + 1e-12
    if d["search"] == "proved" or d["mip_solves"] >= 1:
        assert d["objective_value"] == pytest.approx(z, rel=1e-9)
    if d["search"] == "certified":
        assert d["mip_solves"] == 0 and d["mip_gap"] >= 0 and budget > 0
    if d["mip_solves"] >= 1 and d.get("second_search") == "volume":
        assert budget == 0


@settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**6), n=st.integers(8, 13), Q=st.integers(2, 3),
       b1=st.integers(0, 400), extra=st.integers(0, 4000))
def test_property_a_larger_budget_never_proves_less(seed, n, Q, b1, extra):
    """Property: the rounds are deterministic, so a larger budget repeats every round a smaller one finished and
    proves a bound at least as high, never above the optimum."""
    if not columns.feasible(n, Q, 1):
        return
    Y, idx = _panel(seed, n), np.arange(n)
    pool = columns.score_columns(idx, Y, Q)
    k = columns.min_groups(n, Q, 1)
    root = columns.root_lp(pool, k)
    rows, z_up = _seed_design(pool, n)
    a = columns._certify(pool, root, k, Q, z_up, rows, b1)
    b = columns._certify(pool, root, k, Q, z_up, rows, b1 + extra)
    z = _optimum(pool, root, k, Q)
    assert a["lower"] <= b["lower"] <= z * (1 + 1e-9) + 1e-12
    assert float(pool.cost[b["rows"]].sum()) <= float(pool.cost[a["rows"]].sum()) * (1 + 1e-12) + 1e-300
