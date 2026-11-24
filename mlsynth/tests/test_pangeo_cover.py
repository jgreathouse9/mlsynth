"""PANGEO's exact path proved by an exact-cover search before any MIP.

After the groups are scored and the root LP gives duals, :mod:`mlsynth.utils.pangeo_helpers.cover` searches the
exact covers of the arm depth first, pruning a partial design when its Lagrangian bound (the root duals, the
cardinality row and the uncovered units) reaches the best design's upper bound. A search that finishes under its node
cap proves its design optimal, with a certified gap, and no MIP runs. A search stopped by the cap hands its best design
to the MIP loop of :func:`columns.solve_exact` as part of the first MIP's groups.

The oracle is the full MIP over every candidate group (``mip.solve_partition``), as in ``test_pangeo_columns.py``.
"""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from mlsynth.exceptions import MlsynthConfigError, MlsynthEstimationError
from mlsynth.utils.pangeo_helpers import columns, cover, mip
from mlsynth.utils.pangeo_helpers.parallelism import enumerate_candidate_pairs


def _panel(seed, n, T=12, levels=5.0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, T)).cumsum(axis=1) + rng.normal(size=(n, 1)) * levels


def _reference(idx, Y, Q, min_pairs=1, **kw):
    cands = enumerate_candidate_pairs(idx, Y, Q, **kw)
    return mip.solve_partition(cands, idx, min_pairs=min_pairs, return_diagnostics=True)


def _assert_design(chosen, idx, Q, k):
    covered = sorted(int(u) for c in chosen for u in c["members"])
    assert covered == sorted(int(u) for u in idx)                       # every unit exactly once
    assert len(chosen) >= k
    for c in chosen:
        assert 1 <= len(c["side_a"]) <= Q and 1 <= len(c["side_b"]) <= Q


def _close(a, b):
    return a == pytest.approx(b, rel=1e-9, abs=1e-12 * max(1.0, abs(b)))


# --- smoke --------------------------------------------------------------------------------------------------------

def test_smoke_the_search_proves_the_optimum_without_a_mip():
    """Smoke: a 12-unit arm at Q = 2 is solved by the search alone, at the full MIP's optimum."""
    Y = _panel(1, 12)
    idx = np.arange(12)
    calls = []

    def spy(*a, **k):
        calls.append(1)
        return mip.solve_partition(*a, **k)

    chosen, d = columns.solve_exact(idx, Y, 2, solve=spy)
    _, d_ref = _reference(idx, Y, 2)
    _assert_design(chosen, idx, 2, columns.min_groups(12, 2, 1))
    assert calls == []
    assert d["search"] == "proved" and d["mip_solves"] == 0 and d["backend"] == "search"
    assert _close(d["objective_value"], d_ref["objective_value"])
    assert d["search_nodes"] >= 1


# --- unit: the certificate ------------------------------------------------------------------------------------------

def test_the_reported_bounds_bracket_the_optimum():
    """Unit: the dual bound is at most the optimum, the design's upper bound at least it, and the certified gap is
    the gap between them, of the order of rounding."""
    Y = _panel(2, 14)
    idx = np.arange(14)
    _, d = columns.solve_exact(idx, Y, 3)
    _, d_ref = _reference(idx, Y, 3)
    z = d_ref["objective_value"]
    tol = 1e-9 * max(1.0, abs(z))
    assert d["search"] == "proved"
    assert d["dual_bound"] <= z + tol
    assert d["objective_upper"] >= z - tol
    assert d["dual_bound"] <= d["objective_value"] <= d["objective_upper"]
    assert d["mip_gap"] == pytest.approx((d["objective_upper"] - d["dual_bound"]) / d["objective_value"])
    assert d["mip_gap"] <= 1e-12
    assert d["lagrangian_bound"] <= d["dual_bound"]


def test_the_design_cost_is_the_sum_of_its_groups():
    Y = _panel(3, 10)
    chosen, d = columns.solve_exact(np.arange(10), Y, 2)
    assert d["objective_value"] == pytest.approx(sum(c["score"] for c in chosen), rel=1e-12)
    assert d["n_selected_pairs"] == len(chosen)


def test_the_search_honours_min_pairs():
    """Unit: the cardinality row k = max(min_pairs, ceil(n / 2Q)) holds in the search's design."""
    Y = _panel(4, 12)
    idx = np.arange(12)
    chosen, d = columns.solve_exact(idx, Y, 3, min_pairs=5)
    _, d_ref = _reference(idx, Y, 3, min_pairs=5)
    assert d["search"] == "proved" and len(chosen) >= 5
    assert _close(d["objective_value"], d_ref["objective_value"])


def test_a_capped_start_still_ends_in_a_proof(monkeypatch):
    """Unit: the start is a capped search over the cheapest groups by reduced cost; with no start at all the proof
    search finds the optimum on its own."""
    monkeypatch.setattr(cover, "START_NODE_LIMIT", 0)
    Y = _panel(5, 12)
    idx = np.arange(12)
    _, d = columns.solve_exact(idx, Y, 2)
    _, d_ref = _reference(idx, Y, 2)
    assert d["search"] == "proved" and _close(d["objective_value"], d_ref["objective_value"])


# --- unit: the node cap and the fallback --------------------------------------------------------------------------

def test_the_node_cap_falls_back_to_the_mip():
    """Unit: a search stopped by its node cap reports it, and the MIP loop finds the optimum."""
    Y = _panel(6, 12)
    idx = np.arange(12)
    calls = []

    def spy(*a, **k):
        calls.append(1)
        return mip.solve_partition(*a, **k)

    _, d = columns.solve_exact(idx, Y, 2, search_node_limit=1, solve=spy)
    _, d_ref = _reference(idx, Y, 2)
    assert d["search"] == "node_limit" and d["mip_solves"] == len(calls) >= 1
    assert d["backend"] != "search"
    assert _close(d["objective_value"], d_ref["objective_value"])


def test_the_fallback_mip_holds_the_searchs_design(monkeypatch):
    """Unit: the first MIP after a capped search holds the search's best design, so it starts from it. The first
    MIP's own window is cut to one group, so the design can reach it only as the search's."""
    monkeypatch.setattr(columns, "FIRST_PER_UNIT", 0)
    monkeypatch.setattr(columns, "FIRST_MIN", 1)
    Y = _panel(7, 14)
    idx = np.arange(14)
    pool = columns.score_columns(idx, Y, 3)
    root = columns.root_lp(pool, columns.min_groups(14, 3, 1))
    start = cover.start_design(pool, root, columns.min_groups(14, 3, 1), 3)
    assert start.rows.size > 0
    want = {tuple(int(u) for u in pool.unit_indices[pool.members[j][pool.members[j] >= 0]]) for j in start.rows}
    seen = []

    def spy(cands, ix, **kw):
        seen.append({tuple(int(u) for u in c["members"]) for c in cands})
        return mip.solve_partition(cands, ix, **kw)

    columns.solve_exact(idx, Y, 3, search_node_limit=1, solve=spy)
    seed = {tuple(int(u) for u in idx[list(g)]) for g in columns.seed_partition(14)}
    assert len(want - seed) >= 2          # more than the window's one group: the design came in as the search's
    assert want <= seen[0]


def test_a_node_limit_of_zero_is_the_mip_path():
    """Unit: search_node_limit=0 skips the search and reports it."""
    Y = _panel(8, 10)
    _, d = columns.solve_exact(np.arange(10), Y, 2, search_node_limit=0)
    assert d["search"] == "off" and d["mip_solves"] >= 1 and d["search_nodes"] == 0


def test_the_default_node_limit_is_read_at_call_time(monkeypatch):
    monkeypatch.setattr(cover, "SEARCH_NODE_LIMIT", 0)
    _, d = columns.solve_exact(np.arange(10), _panel(9, 10), 2)
    assert d["search"] == "off"


def test_without_numba_the_search_is_skipped(monkeypatch):
    """Unit: the search is compiled code; without numba (the optional ``fast`` extra) the MIP path runs."""
    monkeypatch.setattr(cover, "_NUMBA", False)
    _, d = columns.solve_exact(np.arange(10), _panel(10, 10), 2)
    assert d["search"] == "unavailable" and d["mip_solves"] >= 1


# --- unit: the search alone -----------------------------------------------------------------------------------------

def test_search_returns_a_valid_cover_of_the_kept_groups():
    """Unit: the rows returned are a design over the groups searched, and its upper bound is the sum of theirs."""
    Y = _panel(11, 12)
    idx = np.arange(12)
    pool = columns.score_columns(idx, Y, 2)
    k = columns.min_groups(12, 2, 1)
    root = columns.root_lp(pool, k)
    res = cover.search(pool, root, np.arange(len(pool)), k, 2, np.inf, 10**7)
    assert res.status == "proved" and res.rows.size >= k
    units = np.concatenate([pool.members[j][pool.members[j] >= 0] for j in res.rows])
    assert sorted(units.tolist()) == list(range(12))
    assert res.upper >= float(pool.cost_up[res.rows].sum()) * (1 - 1e-15)
    assert res.lower <= res.upper


def test_search_with_an_unbeatable_incumbent_finds_no_design():
    """Edge: an incumbent bound below every design leaves the search with nothing better; it still proves that."""
    Y = _panel(12, 8)
    pool = columns.score_columns(np.arange(8), Y, 2)
    k = columns.min_groups(8, 2, 1)
    root = columns.root_lp(pool, k)
    res = cover.search(pool, root, np.arange(len(pool)), k, 2, -1.0, 10**7)
    assert res.status == "proved" and res.rows.size == 0


def test_search_reports_the_node_cap():
    Y = _panel(13, 12)
    pool = columns.score_columns(np.arange(12), Y, 2)
    k = columns.min_groups(12, 2, 1)
    root = columns.root_lp(pool, k)
    res = cover.search(pool, root, np.arange(len(pool)), k, 2, np.inf, 1)
    assert res.status == "node_limit" and res.nodes == 2


def test_search_over_groups_that_miss_a_unit_has_no_design():
    """Edge: a unit no kept group covers ends the search at the root."""
    Y = _panel(14, 8)
    pool = columns.score_columns(np.arange(8), Y, 2)
    k = columns.min_groups(8, 2, 1)
    root = columns.root_lp(pool, k)
    keep = np.flatnonzero(~(pool.members == 0).any(axis=1))         # no group holds unit 0
    res = cover.search(pool, root, keep, k, 2, np.inf, 10**7)
    assert res.status == "proved" and res.rows.size == 0 and res.nodes == 0



def test_search_over_no_groups_proves_nothing_better_than_the_incumbent():
    """Edge: with no groups to search the search is over at once; its bound is the incumbent's."""
    Y = _panel(21, 8)
    pool = columns.score_columns(np.arange(8), Y, 2)
    k = columns.min_groups(8, 2, 1)
    root = columns.root_lp(pool, k)
    res = cover.search(pool, root, np.array([], dtype=int), k, 2, 3.5, 10)
    assert res.status == "proved" and res.rows.size == 0 and res.nodes == 0
    assert res.upper == res.lower == 3.5

# --- edge -----------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("n,Q", [(2, 1), (3, 2), (4, 2), (5, 5), (6, 3), (7, 2)])
def test_tiny_arms(n, Q):
    Y = _panel(15, n)
    idx = np.arange(n)
    chosen, d = columns.solve_exact(idx, Y, Q)
    _, d_ref = _reference(idx, Y, Q)
    _assert_design(chosen, idx, Q, columns.min_groups(n, Q, 1))
    assert d["search"] == "proved" and _close(d["objective_value"], d_ref["objective_value"])


def test_identical_units_give_a_design_of_cost_zero_and_an_absolute_gap():
    """Edge: every group costs 0, up to bounds of the order of rounding; the gap relative to a cost of 0 has no
    meaning, so it is the absolute gap, as the MIP path reports it."""
    Y = np.tile(np.arange(8.0), (9, 1))
    chosen, d = columns.solve_exact(np.arange(9), Y, 2)
    _assert_design(chosen, np.arange(9), 2, columns.min_groups(9, 2, 1))
    assert d["search"] == "proved" and d["objective_value"] == 0.0
    assert d["mip_gap"] == d["objective_upper"] - d["dual_bound"] <= 1e-20


def test_units_past_64_use_more_than_one_mask_word():
    """Edge: the uncovered set is a bit mask of ceil(n / 64) words; at 66 units pairs cover words 0 and 1."""
    Y = _panel(16, 66, T=8)
    idx = np.arange(66)
    chosen, d = columns.solve_exact(idx, Y, 1)
    _, d_ref = _reference(idx, Y, 1)
    _assert_design(chosen, idx, 1, columns.min_groups(66, 1, 1))
    assert _close(d["objective_value"], d_ref["objective_value"])


# --- failure --------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [-1, 2.5, "many"])
def test_an_invalid_node_limit_is_refused(bad):
    with pytest.raises(MlsynthConfigError, match="search_node_limit"):
        columns.solve_exact(np.arange(8), _panel(17, 8), 2, search_node_limit=bad)


@pytest.mark.parametrize("n,Q,min_pairs,match", [(1, 2, 1, "no admissible"), (5, 1, 1, "odd"),
                                                  (6, 2, 4, "min_pairs")])
def test_no_design_is_still_reported_as_the_mip_reports_it(n, Q, min_pairs, match):
    with pytest.raises(MlsynthEstimationError, match=match):
        columns.solve_exact(np.arange(n), _panel(0, max(n, 2))[:n], Q, min_pairs=min_pairs)


@pytest.mark.parametrize("backend", ["cvxpy", "highspy"])
def test_a_backend_named_explicitly_gets_its_mip(backend):
    """Unit: the search runs only with backend="auto"; naming a MIP backend asks for that MIP."""
    pytest.importorskip("highspy") if backend == "highspy" else None
    _, d = columns.solve_exact(np.arange(10), _panel(19, 10), 2, backend=backend)
    assert d["search"] == "off" and d["backend"] == backend and d["mip_solves"] >= 1


def test_an_unknown_backend_is_still_refused():
    with pytest.raises(MlsynthConfigError):
        columns.solve_exact(np.arange(8), _panel(20, 8), 2, backend="gurobi-ish")


# --- the report -----------------------------------------------------------------------------------------------------

def test_the_report_carries_the_search():
    from mlsynth.utils.pangeo_helpers.structures import MipSolveReport

    _, d = columns.solve_exact(np.arange(10), _panel(18, 10), 2)
    rep = MipSolveReport.from_diagnostics(d)
    assert rep.backend == "search" and rep.status == "optimal" and rep.solver_name == "exact-cover search"
    assert rep.search == "proved" and rep.search_nodes == d["search_nodes"]
    assert rep.proven_gap == d["mip_gap"] and rep.dual_bound == d["dual_bound"]


def test_a_report_from_a_full_mip_leaves_the_search_fields_none():
    from mlsynth.utils.pangeo_helpers.structures import MipSolveReport

    _, d = _reference(np.arange(6), _panel(0, 6), 2)
    rep = MipSolveReport.from_diagnostics(dict(d, backend=d.get("backend", "highspy")))
    assert rep.search is None and rep.search_nodes is None


def test_the_pipeline_reports_the_search():
    from mlsynth.utils.pangeo_helpers import make_seasonal_sales_panel, prepare_pangeo_inputs, run_pangeo

    panel = make_seasonal_sales_panel(units_per_arm=10, arms=("A", "B"), T=30, seed=4)
    inp = prepare_pangeo_inputs(df=panel, outcome="sales", arm="arm", unitid="unit", time="time")
    res = run_pangeo(inp, max_supergeo_size=2, fast=False, compute_power=False)
    nE = res.metadata["n_estimation"]
    for arm, idx in inp.arm_units.items():
        rep = res.mip_reports[arm]
        _, d_ref = _reference(np.asarray(idx), inp.Y[:, :nE], 2)
        assert rep.search == "proved" and rep.backend == "search"
        assert _close(rep.objective_value, d_ref["objective_value"])


# --- property -------------------------------------------------------------------------------------------------------

@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**6), n=st.integers(2, 11), Q=st.integers(1, 3), min_pairs=st.integers(1, 4),
       cap=st.sampled_from([0, 1, 50, 10**7]), objective=st.sampled_from(["ss_res", "r2", "supergeo_total"]))
def test_property_every_node_cap_gives_the_full_mips_optimum(seed, n, Q, min_pairs, cap, objective):
    """Property: over arms, sizes, cardinality rows, objectives and node caps (off, tiny, small, ample), the design is
    the full MIP's optimum; a proved search brackets it with its bounds, and a capped one falls back."""
    Y = _panel(seed, n, 8)
    idx = np.arange(n)
    if not columns.feasible(n, Q, min_pairs):
        with pytest.raises(MlsynthEstimationError):
            columns.solve_exact(idx, Y, Q, min_pairs=min_pairs, objective=objective, search_node_limit=cap)
        return
    _, d_ref = _reference(idx, Y, Q, min_pairs, objective=objective)
    chosen, d = columns.solve_exact(idx, Y, Q, min_pairs=min_pairs, objective=objective, search_node_limit=cap)
    _assert_design(chosen, idx, Q, columns.min_groups(n, Q, min_pairs))
    z = d_ref["objective_value"]
    assert _close(d["objective_value"], z)
    assert d["search"] in ({"off"} if cap == 0 else {"proved", "node_limit"})
    if d["search"] == "proved":
        tol = 1e-9 * max(1.0, abs(z))
        assert d["dual_bound"] <= z + tol and d["objective_upper"] >= z - tol and d["mip_solves"] == 0
    else:
        assert d["mip_solves"] >= 1
