"""PANGEO's exact path without full enumeration in Python: every group scored in compiled code through a quadratic
form, the root LP bound with a cardinality cut, and every group whose reduced cost proves it out of every design
cheaper than the best one found left out of the MIP.

The oracle is the path this replaces, ``parallelism.enumerate_candidate_pairs`` and ``mip.solve_partition`` over
every candidate group: the scorer has to give the same groups, splits and scores, and the solver the same optimum.
"""
from __future__ import annotations

import itertools
from math import comb

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from mlsynth.exceptions import MlsynthEstimationError
from mlsynth.utils.pangeo_helpers import columns, mip
from mlsynth.utils.pangeo_helpers.parallelism import enumerate_candidate_pairs

OBJECTIVES = ("ss_res", "r2", "weighted", "supergeo_total")


def _panel(seed, n, T=12, levels=5.0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, T)).cumsum(axis=1) + rng.normal(size=(n, 1)) * levels


def _kwargs(objective, T, n, seed, cov=False, unit_weights=False, cov_weights=False):
    rng = np.random.default_rng(seed + 1)
    kw = dict(objective=objective)
    if objective == "weighted":
        kw["weights"] = rng.uniform(0.2, 2.0, T)
    if cov:
        kw["cov"] = rng.normal(size=(n, 3))
        kw["cov_scales"] = rng.uniform(0.5, 2.0, 3)
        if cov_weights:
            kw["cov_weights"] = rng.uniform(0.0, 2.0, 3)
    if unit_weights:
        kw["unit_weights"] = rng.uniform(0.5, 3.0, n)
    return kw


def _reference(idx, Y, Q, min_pairs=1, **kw):
    cands = enumerate_candidate_pairs(idx, Y, Q, **kw)
    chosen, d = mip.solve_partition(cands, idx, min_pairs=min_pairs, return_diagnostics=True)
    return chosen, d


def _assert_cover(chosen, idx, Q):
    covered = [int(u) for c in chosen for u in c["members"]]
    assert sorted(covered) == sorted(int(u) for u in idx)
    for c in chosen:
        assert 1 <= len(c["side_a"]) <= Q and 1 <= len(c["side_b"]) <= Q
        assert sorted(list(c["side_a"]) + list(c["side_b"])) == sorted(int(u) for u in c["members"])


# --- smoke --------------------------------------------------------------------------------------------------------

def test_smoke_scores_every_group_and_solves():
    """Smoke: a 10-unit arm at Q = 2 scores every group finitely and returns an exact cover with diagnostics."""
    Y = _panel(0, 10)
    idx = np.arange(10)
    pool = columns.score_columns(idx, Y, 2)
    assert len(pool) == columns.count_columns(10, 2) == sum(comb(10, m) for m in (2, 3, 4))
    assert np.isfinite(pool.cost).all()
    chosen, d = columns.solve_exact(idx, Y, 2)
    _assert_cover(chosen, idx, 2)
    assert d["path"] == "exact_mip" and d["n_candidate_pairs"] == len(pool)
    assert d["objective_value"] == pytest.approx(sum(c["score"] for c in chosen))


# --- the scorer ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("objective", OBJECTIVES)
@pytest.mark.parametrize("cov,unit_weights,cov_weights", [(False, False, False), (True, False, False),
                                                           (True, True, True), (False, True, False)])
def test_scores_match_the_enumeration(objective, cov, unit_weights, cov_weights):
    """Unit: the same groups in the same order, the same split, the same score as best_split, for every objective,
    with and without covariates and unit weights."""
    n, T, Q = 8, 10, 3
    Y = _panel(3, n, T)
    idx = np.arange(n)
    kw = _kwargs(objective, T, n, 3, cov, unit_weights, cov_weights)
    ref = enumerate_candidate_pairs(idx, Y, Q, **kw)
    got = columns.score_columns(idx, Y, Q, **kw).candidates()
    assert len(got) == len(ref)
    for g, r in zip(got, ref):
        assert list(g["members"]) == list(r["members"])
        assert g["score"] == pytest.approx(r["score"], rel=1e-9, abs=1e-12)
        assert list(g["side_a"]) == list(r["side_a"]) and list(g["side_b"]) == list(r["side_b"])


def test_scores_use_the_arm_rows_of_a_wider_panel():
    """Unit: unit indices pick rows of the whole panel, as in the enumeration; members are those indices."""
    Y = _panel(4, 12)
    idx = np.array([1, 4, 5, 8, 9, 11])
    uw = np.random.default_rng(0).uniform(0.5, 2, 12)
    ref = enumerate_candidate_pairs(idx, Y, 2, unit_weights=uw)
    got = columns.score_columns(idx, Y, 2, unit_weights=uw).candidates()
    assert [list(g["members"]) for g in got] == [list(r["members"]) for r in ref]
    assert [g["score"] for g in got] == pytest.approx([r["score"] for r in ref], rel=1e-9)


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**6), n=st.integers(2, 8), Q=st.integers(1, 4), T=st.integers(2, 15),
       objective=st.sampled_from(OBJECTIVES), cov=st.booleans(), uw=st.booleans(), cw=st.booleans(),
       scale=st.sampled_from([1e-3, 1.0, 1e4]))
def test_property_scores_match_the_enumeration(seed, n, Q, T, objective, cov, uw, cw, scale):
    """Property: over panels, sizes, objectives, covariates and unit weights the scorer is the enumeration."""
    Y = _panel(seed, n, T) * scale
    idx = np.arange(n)
    kw = _kwargs(objective, T, n, seed, cov, uw, cw)
    ref = enumerate_candidate_pairs(idx, Y, Q, **kw)
    got = columns.score_columns(idx, Y, Q, **kw).candidates()
    assert [list(g["members"]) for g in got] == [list(r["members"]) for r in ref]
    for g, r in zip(got, ref):
        assert g["score"] == pytest.approx(r["score"], rel=1e-8, abs=1e-10 * max(1.0, scale) ** 2)


def test_r2_falls_back_to_ss_res_on_a_flat_half():
    """Edge: a half with no variation has ss_tot 0; r2 then scores the residual SS, as split_cost does."""
    Y = np.vstack([np.full(6, 3.0), np.arange(6.0), np.arange(6.0) ** 1.5, np.full(6, 1.0)])
    ref = enumerate_candidate_pairs(np.arange(4), Y, 2, objective="r2")
    got = columns.score_columns(np.arange(4), Y, 2, objective="r2").candidates()
    assert [g["score"] for g in got] == pytest.approx([r["score"] for r in ref], rel=1e-9, abs=1e-12)


def test_weighted_needs_weights():
    """Failure: objective='weighted' without weights raises, as split_cost does."""
    with pytest.raises(ValueError, match="weights"):
        columns.score_columns(np.arange(4), _panel(0, 4), 2, objective="weighted")


def test_unknown_objective_raises():
    """Failure: an objective the scorer does not know is reported."""
    with pytest.raises(ValueError, match="objective"):
        columns.score_columns(np.arange(4), _panel(0, 4), 2, objective="variance")


def test_non_finite_scores_are_dropped_like_the_enumeration():
    """Edge: a unit with a NaN path gives NaN scores; those groups are dropped, as the enumeration drops them."""
    Y = _panel(1, 5)
    Y[2, 3] = np.nan
    ref = enumerate_candidate_pairs(np.arange(5), Y, 2)
    got = columns.score_columns(np.arange(5), Y, 2).candidates()
    assert [list(g["members"]) for g in got] == [list(r["members"]) for r in ref]


@pytest.mark.parametrize("n,Q", [(1, 1), (2, 1), (3, 1), (5, 2), (6, 3), (4, 9)])
def test_count_columns(n, Q):
    """Unit: the count is the number of subsets of size 2 to min(2Q, n)."""
    assert columns.count_columns(n, Q) == sum(comb(n, m) for m in range(2, min(2 * Q, n) + 1))


def test_too_many_columns_is_reported(monkeypatch):
    """Failure: an arm whose groups would not fit in memory raises before allocating, naming the fast path."""
    monkeypatch.setattr(columns, "MAX_COLUMNS", 100)
    with pytest.raises(MlsynthEstimationError, match="fast=True"):
        columns.score_columns(np.arange(10), _panel(0, 10), 2)


# --- the index of a group ---------------------------------------------------------------------------------------

@settings(max_examples=100, deadline=None)
@given(n=st.integers(2, 12), Q=st.integers(1, 4), data=st.data())
def test_property_column_index_is_the_enumeration_order(n, Q, data):
    """Property: column_index gives a group's row in the pool, which is itertools' order by size then lexicographic."""
    hi = min(2 * Q, n)
    m = data.draw(st.integers(2, hi))
    g = tuple(sorted(data.draw(st.lists(st.integers(0, n - 1), min_size=m, max_size=m, unique=True))))
    order = [c for k in range(2, hi + 1) for c in itertools.combinations(range(n), k)]
    assert columns.column_index(g, n) == order.index(g)


# --- feasibility and the starting partition ---------------------------------------------------------------------

@pytest.mark.parametrize("n,Q,min_pairs,ok", [(1, 2, 1, False), (2, 1, 1, True), (3, 1, 1, False), (3, 2, 1, True),
                                               (7, 3, 3, True), (7, 3, 4, False), (8, 1, 4, True),
                                               (8, 1, 5, False)])
def test_feasible(n, Q, min_pairs, ok):
    """Unit: an exact cover exists iff n >= 2, n is even when Q = 1, and min_pairs <= n // 2."""
    assert columns.feasible(n, Q, min_pairs) is ok


@settings(max_examples=100, deadline=None)
@given(n=st.integers(2, 40), Q=st.integers(1, 5))
def test_property_seed_partition_is_a_design(n, Q):
    """Property: when a design exists, the seed partition is one: an exact cover by groups each splittable into
    halves of at most Q, with n // 2 groups, the most any design has."""
    if not columns.feasible(n, Q, 1):
        return
    groups = columns.seed_partition(n)
    assert sorted(u for g in groups for u in g) == list(range(n))
    assert all(2 <= len(g) <= min(3, 2 * Q) for g in groups)
    assert len(groups) == n // 2


@pytest.mark.parametrize("n,Q,min_pairs,k", [(20, 3, 1, 4), (18, 3, 1, 3), (20, 2, 1, 5), (7, 3, 3, 3),
                                              (5, 9, 1, 1)])
def test_min_groups(n, Q, min_pairs, k):
    """Unit: every design has at least ceil(n / 2Q) groups (the Chvatal-Gomory rounding of the cover rows), and at
    least min_pairs."""
    assert columns.min_groups(n, Q, min_pairs) == k


# --- the bound ----------------------------------------------------------------------------------------------------

def _brute_opt(pool, n, k):
    best = np.inf
    rows = [tuple(int(p) for p in r if p >= 0) for r in pool.members]
    by_first = {}
    for j, r in enumerate(rows):
        by_first.setdefault(r[0], []).append(j)

    def go(left, acc, count):
        nonlocal best
        if acc >= best:
            return
        if not left:
            if count >= k:
                best = acc
            return
        for j in by_first.get(min(left), []):
            if set(rows[j]) <= left:
                go(left - set(rows[j]), acc + pool.cost[j], count + 1)

    go(set(range(n)), 0.0, 0)
    return best


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**6), n=st.integers(2, 8), Q=st.integers(1, 3), min_pairs=st.integers(1, 4),
       spread=st.floats(0.0, 3.0))
def test_property_lagrangian_bound_holds_for_any_duals(seed, n, Q, min_pairs, spread):
    """Property: for any duals, with sigma clipped at 0, the Lagrangian bound is at most the cost of every design.
    The bound is what licenses leaving a column out, so it must hold off the LP optimum too."""
    if not columns.feasible(n, Q, min_pairs):
        return
    Y = _panel(seed, n, 8)
    pool = columns.score_columns(np.arange(n), Y, Q)
    k = columns.min_groups(n, Q, min_pairs)
    opt = _brute_opt(pool, n, k)
    rng = np.random.default_rng(seed)
    pi = rng.normal(size=n) * spread * float(np.median(pool.cost))
    sigma = float(rng.normal() * spread * float(np.median(pool.cost)))
    bound, rc = columns.lagrangian_bound(pool, pi, sigma, k)
    assert bound <= opt + 1e-9 * max(1.0, abs(opt))
    assert np.allclose(rc, pool.cost - pi[pool.members].sum(axis=1, where=pool.members >= 0)
                       - max(sigma, 0.0), atol=1e-9 * max(1.0, abs(pool.cost).max()))


@pytest.mark.parametrize("n,Q", [(8, 3), (10, 2), (9, 2), (12, 3)])
def test_root_lp_is_the_full_lp_with_the_cut(n, Q):
    """Unit: column generation ends at the LP optimum over every column with the cardinality row, and its
    Lagrangian bound equals that LP value."""
    import scipy.sparse as sp
    from scipy.optimize import linprog

    Y = _panel(7, n, 10)
    pool = columns.score_columns(np.arange(n), Y, Q)
    k = columns.min_groups(n, Q, 1)
    N = len(pool)
    r, c = np.nonzero(pool.members >= 0)
    A = sp.csr_matrix((np.ones(r.size), (pool.members[r, c], r)), shape=(n, N))
    full = linprog(pool.cost, A_eq=A, b_eq=np.ones(n), A_ub=-np.ones((1, N)), b_ub=[-k], bounds=(0, 1),
                   method="highs")
    root = columns.root_lp(pool, k)
    assert root.lp_value == pytest.approx(full.fun, rel=1e-7, abs=1e-9)
    assert root.bound == pytest.approx(full.fun, rel=1e-7, abs=1e-9)
    assert root.pool_size <= N and root.rounds >= 1
    assert (root.reduced_costs >= -1e-7 * max(1.0, abs(full.fun))).all()


def test_the_cut_raises_the_bound_when_n_is_not_a_multiple_of_2q():
    """Unit: 20 units in groups of at most 6 need four groups; the LP without the cut uses 20/6 of the cheap
    6-groups, and the cut closes part of that gap (measured 0.796 -> 1.086 on the seasonal panel)."""
    from mlsynth.utils.pangeo_helpers import make_seasonal_sales_panel, prepare_pangeo_inputs

    panel = make_seasonal_sales_panel(units_per_arm=20, arms=("A",), T=60, seed=0)
    inp = prepare_pangeo_inputs(df=panel, outcome="sales", arm="arm", unitid="unit", time="time")
    idx = np.asarray(inp.arm_units["A"])
    pool = columns.score_columns(idx, inp.Y[:, :42], 3)
    with_cut = columns.root_lp(pool, 4)
    without = columns.root_lp(pool, 1)
    assert with_cut.bound > without.bound * 1.2


# --- the solver ---------------------------------------------------------------------------------------------------

@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**6), n=st.integers(2, 9), Q=st.integers(1, 3), min_pairs=st.integers(1, 5),
       objective=st.sampled_from(OBJECTIVES), cov=st.booleans())
def test_property_same_optimum_as_the_full_mip(seed, n, Q, min_pairs, objective, cov):
    """Property: the optimum is the full MIP's, and a case with no design raises from both."""
    T = 8
    Y = _panel(seed, n, T)
    idx = np.arange(n) + 3
    Yfull = np.vstack([np.zeros((3, T)), Y])
    kw = _kwargs(objective, T, n + 3, seed, cov=cov)
    if not columns.feasible(n, Q, min_pairs):
        with pytest.raises(MlsynthEstimationError):
            _reference(idx, Yfull, Q, min_pairs, **kw)
        with pytest.raises(MlsynthEstimationError):
            columns.solve_exact(idx, Yfull, Q, min_pairs=min_pairs, **kw)
        return
    _, d_ref = _reference(idx, Yfull, Q, min_pairs, **kw)
    chosen, d = columns.solve_exact(idx, Yfull, Q, min_pairs=min_pairs, **kw)
    _assert_cover(chosen, idx, Q)
    assert len(chosen) >= min_pairs
    z = d_ref["objective_value"]
    assert d["objective_value"] == pytest.approx(z, rel=1e-9, abs=1e-12 * max(1.0, abs(z)))
    assert d["lagrangian_bound"] <= d["objective_value"] + 1e-9 * max(1.0, abs(z))


def test_columns_are_eliminated():
    """Unit: on a 16-unit arm at Q = 3 the MIP sees a small share of the groups and finds the full MIP's optimum."""
    Y = _panel(11, 16, 20)
    idx = np.arange(16)
    _, d_ref = _reference(idx, Y, 3)
    chosen, d = columns.solve_exact(idx, Y, 3)
    assert d["objective_value"] == pytest.approx(d_ref["objective_value"], rel=1e-9)
    assert d["n_candidate_pairs"] == columns.count_columns(16, 3)
    assert d["columns_solved"] < d["n_candidate_pairs"] / 10
    assert d["mip_solves"] >= 1 and d["lp_rounds"] >= 1


def test_a_poor_first_design_is_followed_by_the_proof(monkeypatch):
    """Unit: with a first restricted MIP of only the seed partition's columns and the cheapest few, its design is
    not the optimum; the later solves over more columns find it."""
    monkeypatch.setattr(columns, "FIRST_PER_UNIT", 0)
    monkeypatch.setattr(columns, "FIRST_MIN", 1)
    Y = _panel(12, 12, 15)
    idx = np.arange(12)
    _, d_ref = _reference(idx, Y, 2)
    _, d = columns.solve_exact(idx, Y, 2)
    assert d["mip_solves"] >= 2
    assert d["objective_value"] == pytest.approx(d_ref["objective_value"], rel=1e-9)


def test_a_poor_first_design_does_not_open_every_group(monkeypatch):
    """Unit: a first design far above the optimum makes z - L large; the next MIP takes at most GROWTH times as many
    groups, so the last MIP holds at most GROWTH times the groups within z* - L of the bound (and the seed design's),
    not every group within the first design's margin. Measured: 24 geos at Q = 3 handed 189,950 of 190,026 groups to
    the second MIP, which took 795 s, when the first design came from little more than the seed design."""
    monkeypatch.setattr(columns, "FIRST_PER_UNIT", 0)
    monkeypatch.setattr(columns, "FIRST_MIN", 1)
    Y = _panel(3, 12, 15)
    idx = np.arange(12)
    calls = []

    def spy(cands, ix, **kw):
        out = mip.solve_partition(cands, ix, **kw)
        calls.append(out[1]["objective_value"])
        return out

    _, d = columns.solve_exact(idx, Y, 2, solve=spy)
    pool = columns.score_columns(idx, Y, 2)
    root = columns.root_lp(pool, columns.min_groups(12, 2, 1))
    tol = 1e-9 * max(1.0, float(pool.cost.max()))
    need = int((root.reduced_costs <= d["objective_value"] - root.bound + tol).sum())
    first_margin = int((root.reduced_costs <= calls[0] - root.bound + tol).sum())
    cap = columns.GROWTH * need + len(columns.seed_partition(12))
    assert first_margin > cap                     # the first design's margin would break the cap
    assert d["columns_solved"] <= cap


def test_the_cardinality_cut_reaches_the_mip():
    """Unit: the MIP is given min_pairs = max(min_pairs, ceil(n / 2Q)), a valid row that speeds the proof."""
    seen = []

    def spy(cands, idx, **kw):
        seen.append(kw["min_pairs"])
        return mip.solve_partition(cands, idx, **kw)

    columns.solve_exact(np.arange(10), _panel(2, 10), 2, solve=spy)
    assert seen and set(seen) == {3}


def test_the_solve_passes_backend_and_gaps():
    """Unit: the backend and the gaps reach every restricted MIP."""
    seen = []

    def spy(cands, idx, **kw):
        seen.append((kw["backend"], kw["mip_rel_gap"], kw["mip_abs_gap"], kw["return_diagnostics"]))
        return mip.solve_partition(cands, idx, **kw)

    columns.solve_exact(np.arange(8), _panel(2, 8), 2, backend="cvxpy", mip_rel_gap=0.01, mip_abs_gap=0.5,
                        solve=spy)
    assert seen and set(seen) == {("cvxpy", 0.01, 0.5, True)}


def test_a_gap_returns_a_design_within_it():
    """Unit: with a relative gap the design costs at most (1 + gap) times the optimum, and the dual bound reported
    is at most the optimum."""
    Y = _panel(5, 14, 12)
    idx = np.arange(14)
    _, d_ref = _reference(idx, Y, 3)
    z = d_ref["objective_value"]
    _, d = columns.solve_exact(idx, Y, 3, mip_rel_gap=0.05)
    assert z - 1e-9 * z <= d["objective_value"] <= 1.05 * z + 1e-9 * z
    if d["dual_bound"] is not None:
        assert d["dual_bound"] <= z + 1e-9 * z


@pytest.mark.parametrize("n,Q", [(2, 1), (3, 2), (4, 2), (5, 5), (6, 3)])
def test_tiny_arms(n, Q):
    """Edge: one or two groups, and Q at least n; the optimum is the full MIP's."""
    Y = _panel(9, n)
    idx = np.arange(n)
    _, d_ref = _reference(idx, Y, Q)
    chosen, d = columns.solve_exact(idx, Y, Q)
    _assert_cover(chosen, idx, Q)
    assert d["objective_value"] == pytest.approx(d_ref["objective_value"], rel=1e-9, abs=1e-12)


def test_identical_units():
    """Edge: every unit the same path gives every group cost 0; the solver returns a design of cost 0."""
    Y = np.tile(np.arange(8.0), (9, 1))
    chosen, d = columns.solve_exact(np.arange(9), Y, 2)
    _assert_cover(chosen, np.arange(9), 2)
    assert d["objective_value"] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("n,Q,min_pairs,match", [(1, 2, 1, "no admissible"), (5, 1, 1, "odd"),
                                                  (6, 2, 4, "min_pairs")])
def test_no_design_is_reported_as_the_mip_reports_it(n, Q, min_pairs, match):
    """Failure: a case with no design raises the MIP path's error and its reason."""
    with pytest.raises(MlsynthEstimationError, match=match):
        columns.solve_exact(np.arange(n), _panel(0, max(n, 2))[:n], Q, min_pairs=min_pairs)


def test_solver_errors_propagate():
    """Failure: an error from the MIP is not swallowed."""
    def broken(*a, **k):
        raise MlsynthEstimationError("PANGEO partition MIP did not solve (status=Solve error)")

    with pytest.raises(MlsynthEstimationError, match="Solve error"):
        columns.solve_exact(np.arange(8), _panel(0, 8), 2, solve=broken)


# --- the pipeline -------------------------------------------------------------------------------------------------

def _seasonal(n, Q=2, **kw):
    from mlsynth.utils.pangeo_helpers import make_seasonal_sales_panel, prepare_pangeo_inputs, run_pangeo

    panel = make_seasonal_sales_panel(units_per_arm=n, arms=("A", "B"), T=30, seed=4)
    inp = prepare_pangeo_inputs(df=panel, outcome="sales", arm="arm", unitid="unit", time="time")
    return inp, run_pangeo(inp, max_supergeo_size=Q, fast=False, compute_power=False, **kw)


def test_the_pipeline_designs_at_the_full_mips_optimum_and_reports_it():
    """Unit: fast=False at Q = 2 designs each arm at the full MIP's optimum over the estimation window, and the
    report counts the groups scored, the groups the MIP saw and the bound."""
    inp, res = _seasonal(10)
    nE = res.metadata["n_estimation"]
    for arm, idx in inp.arm_units.items():
        d = res.metadata["solver_diagnostics"][arm]
        _, d_ref = _reference(np.asarray(idx), inp.Y[:, :nE], 2)
        assert d["objective_value"] == pytest.approx(d_ref["objective_value"], rel=1e-9)
        rep = res.mip_reports[arm]
        assert rep.objective_value == pytest.approx(d["objective_value"])
        assert rep.columns_scored == columns.count_columns(len(idx), 2)
        assert rep.columns_solved is not None and rep.columns_solved <= rep.columns_scored
        assert rep.lagrangian_bound is not None and rep.lagrangian_bound <= rep.objective_value + 1e-9


def test_the_pipeline_solves_through_its_solve_partition(monkeypatch):
    """Unit: the restricted MIPs go through pipeline.solve_partition, so a backend or a patch there reaches them."""
    from mlsynth.utils.pangeo_helpers import pipeline as ppipe

    calls = []
    real = ppipe.solve_partition
    monkeypatch.setattr(ppipe, "solve_partition", lambda *a, **k: calls.append(k) or real(*a, **k))
    _seasonal(8)
    assert len(calls) >= 2


def test_the_report_fields_default_to_none():
    """Unit: a report built from the diagnostics of a full MIP (no elimination) leaves the new fields None."""
    from mlsynth.utils.pangeo_helpers.structures import MipSolveReport

    _, d = _reference(np.arange(6), _panel(0, 6), 2)
    d["backend"] = d.get("backend", "highspy")
    rep = MipSolveReport.from_diagnostics(d)
    assert rep.columns_scored is None and rep.columns_solved is None and rep.lagrangian_bound is None


def test_on_ties_the_split_attains_the_minimum():
    """Unit: with duplicated units many splits tie in exact arithmetic, and rounding decides which one each
    computation returns. The split returned costs the minimum, recomputed from its halves by split_cost."""
    from mlsynth.utils.pangeo_helpers.parallelism import split_cost

    base = _panel(8, 3, 10)
    Y = np.vstack([base, base, base[:1]])
    ref = enumerate_candidate_pairs(np.arange(7), Y, 3)
    got = columns.score_columns(np.arange(7), Y, 3).candidates()
    scale = float(np.abs(Y - Y.mean(axis=1, keepdims=True)).max()) ** 2
    for g, r in zip(got, ref):
        own = split_cost(Y[g["side_a"]].mean(axis=0), Y[g["side_b"]].mean(axis=0))
        assert len(g["side_a"]) <= 3 and len(g["side_b"]) <= 3
        assert own == pytest.approx(r["score"], abs=1e-12 * scale)


def test_a_costlier_later_design_does_not_replace_the_best(monkeypatch):
    """Unit: a MIP that returns a dearer design than the best so far (a gap lets it) leaves the cheaper design in
    place, and the last MIP's dual bound, which holds every design cheaper than the best, is the one reported. The
    stub returns the seed design whenever the MIP does not improve on the best design so far."""
    monkeypatch.setattr(columns, "FIRST_PER_UNIT", 0)
    monkeypatch.setattr(columns, "FIRST_MIN", 1)
    Y = _panel(3, 12, 15)
    idx = np.arange(12)
    real, returned = [], []

    def stub(cands, ix, **kw):
        chosen, d = mip.solve_partition(cands, ix, **kw)
        better = not real or d["objective_value"] < min(real) - 1e-9
        real.append(d["objective_value"])
        if not better:
            seed = [c for c in cands if tuple(int(u) for u in c["members"]) in set(columns.seed_partition(12))]
            chosen = seed
            d = dict(d, objective_value=float(sum(c["score"] for c in seed)), dual_bound=1e-6)
        returned.append(d)
        return chosen, d

    _, d_ref = _reference(idx, Y, 2)
    chosen, d = columns.solve_exact(idx, Y, 2, solve=stub)
    assert returned[-1]["objective_value"] > min(real)   # the last MIP returned the dearer design
    assert d["objective_value"] == pytest.approx(d_ref["objective_value"], rel=1e-9)
    assert d["objective_value"] == pytest.approx(sum(c["score"] for c in chosen))
    assert d["dual_bound"] == 1e-6
    assert d["mip_solves"] == len(returned)
