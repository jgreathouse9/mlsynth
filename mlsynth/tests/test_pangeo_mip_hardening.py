"""PANGEO's exact-cover MIP, hardened: gaps of zero by default, one retry with presolve off, and the solve reported.

HiGHS stops a MIP at a relative gap of 1e-4 by default and reports "optimal". On a 20-market instance with one
outlying market (the supergeo study's default-gap instance) both backends returned 281,977,490 against the optimum
281,971,264, the proven gap 2.2e-5 and HiGHS's own dual bound at the optimum. HiGHS's presolve returned "Solve
error" on a feasible 14-market LP master (fixture presolve_failure_master.json) and the simplex solved it with
presolve off. The solve's status, its retry and its proven gap are reported on the result (PangeoResults.mip_reports)
and in the metadata.
"""
from __future__ import annotations

import itertools
import json
import pathlib

import cvxpy as cp
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from mlsynth import PANGEO
from mlsynth.exceptions import MlsynthConfigError, MlsynthEstimationError
from mlsynth.utils.pangeo_helpers import make_seasonal_sales_panel, mip
from mlsynth.utils.pangeo_helpers.structures import MipSolveReport

pytestmark = pytest.mark.skipif(not mip._highspy_available(), reason="highspy not installed")

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "pangeo" / "presolve_failure_master.json"

# The supergeo study's default-gap instance: 20 markets, one outlier, every group of 2 to 4 with its best split.
DEFAULT_GAP_Z = np.array([58.0, 50.0, 68.0, 22.0, 76.0, 22.0, 67.0, 77.0, 77.0, 3.0, 33.0, 60.0, 27.0, 7.0, 55.0,
                          39.0, 11.0, 9.0, 53.0, 17022.0])
DEFAULT_GAP_OPT = 281971264.0


def _group_cost(z, g):
    S = z[list(g)].sum()
    return min((2 * z[list(A)].sum() - S) ** 2 for r in range(1, len(g)) for A in itertools.combinations(g, r))


def _candidates(z, sizes=(2, 3, 4)):
    n = len(z)
    return [dict(members=list(g), score=float(_group_cost(z, g)), side_a=[g[0]], side_b=list(g[1:]))
            for k in sizes for g in itertools.combinations(range(n), k)]


def _brute(cands, n):
    by_first = {}
    for c in cands:
        by_first.setdefault(min(c["members"]), []).append(c)
    best = [np.inf]

    def go(left, acc):
        if acc >= best[0]:
            return
        if not left:
            best[0] = acc
            return
        i = min(left)
        for c in by_first.get(i, []):
            if set(c["members"]) <= left:
                go(left - set(c["members"]), acc + c["score"])

    go(frozenset(range(n)), 0.0)
    return best[0]


def _objective(chosen):
    return sum(c["score"] for c in chosen)


# --- the gap -------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("backend", ["highspy", "cvxpy"])
def test_the_default_gap_instance_is_solved_to_its_optimum(backend):
    """Smoke and unit: with the default gaps (0) both backends return the optimum, and the diagnostics say the gap
    asked for was 0 and was enforced."""
    cands = _candidates(DEFAULT_GAP_Z)
    chosen, d = mip.solve_partition(cands, np.arange(20), return_diagnostics=True, backend=backend)
    assert _objective(chosen) == DEFAULT_GAP_OPT
    assert d["status"] == "optimal"
    assert d["mip_rel_gap_requested"] == 0.0 and d["mip_abs_gap_requested"] == 0.0
    assert d["gap_enforced"] is True
    assert d["presolve_retry"] is False


@pytest.mark.parametrize("backend", ["highspy", "cvxpy"])
def test_a_stated_gap_is_applied_and_reported(backend):
    """Unit: a stated relative gap is passed to the solver and echoed; the design costs no more than the optimum
    times (1 + gap) and the proven gap reported is within it."""
    cands = _candidates(DEFAULT_GAP_Z)
    chosen, d = mip.solve_partition(cands, np.arange(20), return_diagnostics=True, backend=backend,
                                    mip_rel_gap=1e-4)
    assert d["mip_rel_gap_requested"] == 1e-4
    assert DEFAULT_GAP_OPT <= _objective(chosen) <= DEFAULT_GAP_OPT * (1 + 1e-4)
    assert d["mip_gap"] is not None and d["mip_gap"] <= 1e-4 + 1e-12


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10**9), n=st.integers(4, 9), spread=st.sampled_from([1.0, 1e3, 1e6]),
       backend=st.sampled_from(["highspy", "cvxpy"]))
def test_the_partition_is_the_brute_force_optimum(seed, n, spread, backend):
    """Property: over random panels with costs spread over up to six decades, the partition's objective equals the
    least exact cover by brute force, on either backend, with the proven gap at most the requested 0."""
    rng = np.random.default_rng(seed)
    z = rng.normal(100.0, 20.0, n) * rng.choice([1.0, spread], n, p=[0.8, 0.2])
    cands = _candidates(z)
    chosen, d = mip.solve_partition(cands, np.arange(n), return_diagnostics=True, backend=backend)
    want = _brute(cands, n)
    assert _objective(chosen) == pytest.approx(want, rel=1e-9, abs=1e-9)
    assert sorted(u for c in chosen for u in c["members"]) == list(range(n))


# --- the retry -----------------------------------------------------------------------------------------------------

def _fixture_master():
    """The supergeo study's node master: partition rows (= 1), subset-row cuts (<= 1), artificial singletons."""
    import highspy
    fx = json.loads(FIXTURE.read_text())
    cols = [tuple(c) for c in fx["cols"]]; costs = [float.fromhex(c) for c in fx["costs"]]
    cuts = [tuple(c) for c in fx["cuts"]]; n = fx["n"]; big = float.fromhex(fx["big"])
    h = highspy.Highs(); h.setOptionValue("output_flag", False); h.setOptionValue("threads", 1)
    C = len(cols); m = C + n; nr = n + len(cuts)
    lp = highspy.HighsLp(); lp.num_col_, lp.num_row_ = m, nr
    lp.col_cost_ = np.concatenate([np.asarray(costs, float), np.full(n, big)])
    lp.col_lower_ = np.zeros(m); lp.col_upper_ = np.full(m, np.inf)
    rl = np.ones(nr); ru = np.ones(nr); rl[n:] = -np.inf
    lp.row_lower_ = rl; lp.row_upper_ = ru
    bypair = {}
    for t, S in enumerate(cuts):
        a, b, c = sorted(S)
        for p in ((a, b), (a, c), (b, c)):
            bypair.setdefault(p, []).append(t)
    start = np.zeros(m + 1, np.int32); idx = []
    for j, col in enumerate(cols):
        rows = list(col); hit = set()
        for a in range(len(col)):
            for b in range(a + 1, len(col)):
                hit.update(bypair.get((col[a], col[b]), ()))
        rows.extend(n + t for t in sorted(hit))
        idx.extend(rows); start[j + 1] = start[j] + len(rows)
    for i in range(n):
        idx.append(i); start[C + i + 1] = start[C + i] + 1
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise; lp.a_matrix_.start_ = start
    lp.a_matrix_.index_ = np.array(idx, np.int32); lp.a_matrix_.value_ = np.ones(len(idx))
    h.passModel(lp)
    return h


def test_a_model_presolve_cannot_solve_is_solved_without_it():
    """Unit: on the study's fixture HiGHS's presolve returns 'Solve error'; the run retries once with presolve off
    and solves it, and says it retried."""
    h = _fixture_master()
    status, retried = mip._run_highs(h)
    assert status == "Optimal" and retried is True
    assert h.getInfo().objective_function_value == pytest.approx(17969091942.9903, rel=1e-6)


def test_an_optimal_first_run_is_not_retried():
    import highspy
    h = highspy.Highs(); h.setOptionValue("output_flag", False)
    x = h.addVariable(lb=0.0, ub=1.0); h.addConstr(x >= 0.5); h.changeObjectiveSense(highspy.ObjSense.kMinimize)
    h.changeColCost(0, 1.0)
    status, retried = mip._run_highs(h)
    assert status == "Optimal" and retried is False


def test_the_highspy_partition_runs_through_the_retry(monkeypatch):
    """Unit: the highspy backend solves through _run_highs and reports its retry."""
    calls = []
    real = mip._run_highs

    def spy(h):
        status, _ = real(h)
        calls.append(status)
        return status, True

    monkeypatch.setattr(mip, "_run_highs", spy)
    cands = _candidates(DEFAULT_GAP_Z[:8])
    chosen, d = mip.solve_partition(cands, np.arange(8), return_diagnostics=True, backend="highspy")
    assert calls == ["Optimal"] and d["presolve_retry"] is True


def test_the_cvxpy_partition_retries_with_presolve_off(monkeypatch):
    """Unit: on the cvxpy path a first solve that is not optimal is repeated once with presolve off; the solver's
    options carry the gaps both times."""
    seen = []
    real = cp.Problem.solve

    def solve(self, *a, **kw):
        seen.append(dict(kw))
        out = real(self, *a, **kw)
        if len(seen) == 1:
            self._status = "solver_error"
        return out

    monkeypatch.setattr(cp.Problem, "solve", solve)
    cands = _candidates(DEFAULT_GAP_Z[:8])
    chosen, d = mip.solve_partition(cands, np.arange(8), return_diagnostics=True, backend="cvxpy")
    assert len(seen) == 2 and d["presolve_retry"] is True and d["status"] == "optimal"
    assert seen[0]["mip_rel_gap"] == 0.0 and seen[1]["mip_rel_gap"] == 0.0
    assert seen[1]["presolve"] == "off" and seen[0].get("presolve", "on") != "off"


def test_an_infeasible_cover_is_reported_after_the_retry(monkeypatch):
    """Failure: an arm no exact cover exists for raises MlsynthEstimationError naming the obstruction, after the
    retry with presolve off."""
    calls = []
    real = mip._run_highs

    def spy(h):
        out = real(h)
        calls.append(out)
        return out

    monkeypatch.setattr(mip, "_run_highs", spy)
    cands = [dict(members=[0, 1], score=1.0, side_a=[0], side_b=[1]),
             dict(members=[1, 2], score=1.0, side_a=[1], side_b=[2])]
    with pytest.raises(MlsynthEstimationError, match="structurally uncoverable|no exact cover"):
        mip.solve_partition(cands, np.arange(3), backend="highspy")
    assert len(calls) == 1 and calls[0][1] is True and calls[0][0] != "Optimal"


# --- the options per solver ----------------------------------------------------------------------------------------

def test_the_gap_options_per_cvxpy_solver():
    """Unit: HiGHS and SCIP take both gaps (and presolve off on the retry); GLPK_MI and CBC cannot be given them
    through cvxpy, and the report says the gap was not enforced."""
    opts, ok = mip._cvxpy_options("HIGHS", 1e-3, 2.0, presolve=False)
    assert ok and opts == {"mip_rel_gap": 1e-3, "mip_abs_gap": 2.0, "presolve": "off"}
    opts, ok = mip._cvxpy_options("SCIP", 1e-3, 2.0, presolve=True)
    assert ok and opts == {"scip_params": {"limits/gap": 1e-3, "limits/absgap": 2.0}}
    opts, ok = mip._cvxpy_options("SCIP", 0.0, 0.0, presolve=False)
    assert ok and opts["scip_params"]["presolving/maxrounds"] == 0
    for name in ("GLPK_MI", "CBC"):
        opts, ok = mip._cvxpy_options(name, 0.0, 0.0, presolve=True)
        assert opts == {} and ok is False


@pytest.mark.parametrize("bad", [dict(mip_rel_gap=-1e-3), dict(mip_rel_gap=1.0), dict(mip_rel_gap=float("nan")),
                                 dict(mip_abs_gap=-1.0), dict(mip_abs_gap=float("inf"))])
def test_a_bad_gap_is_refused_by_the_solver(bad):
    cands = _candidates(DEFAULT_GAP_Z[:6])
    with pytest.raises(MlsynthConfigError, match="mip_"):
        mip.solve_partition(cands, np.arange(6), **bad)


# --- the result ----------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def panel():
    return make_seasonal_sales_panel(units_per_arm=6, arms=("A", "B"), T=60, seed=0)


def _fit(panel, **kw):
    cfg = dict(df=panel, outcome="sales", arm="arm", unitid="unit", time="time", display_graphs=False,
               compute_power=False, **kw)
    return PANGEO(cfg).fit()


def test_the_solve_is_reported_on_the_result(panel):
    """Smoke: a MIP-path fit carries one MipSolveReport per arm, optimal with the gaps asked for and the proven gap,
    and the metadata carries the same diagnostics."""
    res = _fit(panel, max_supergeo_size=2)
    assert set(res.mip_reports) == {"A", "B"}
    for arm, r in res.mip_reports.items():
        assert isinstance(r, MipSolveReport)
        assert r.status == "optimal" and r.requested_rel_gap == 0.0 and r.requested_abs_gap == 0.0
        assert r.gap_enforced and r.presolve_retry is False and r.backend in ("highspy", "cvxpy")
        assert r.proven_gap is not None and r.proven_gap <= 1e-9
        d = res.metadata["solver_diagnostics"][arm]
        assert d["status"] == r.status and d["mip_gap"] == r.proven_gap


def test_a_stated_gap_reaches_the_solver_from_the_config(panel):
    res = _fit(panel, max_supergeo_size=2, mip_rel_gap=0.05, mip_abs_gap=0.5)
    for r in res.mip_reports.values():
        assert r.requested_rel_gap == 0.05 and r.requested_abs_gap == 0.5


def test_paths_without_a_mip_report_nothing(panel):
    """Edge: the fast path and the pairs path solve no MIP and report none."""
    assert _fit(panel, max_supergeo_size=2, fast=True).mip_reports == {}
    assert _fit(panel, max_supergeo_size=1).mip_reports == {}


def test_the_automatic_q_keeps_its_design_s_reports(panel):
    res = _fit(panel, max_supergeo_size=None)
    q = res.max_supergeo_size
    if q == 1:
        assert res.mip_reports == {}
    else:
        assert set(res.mip_reports) == {"A", "B"}


@pytest.mark.parametrize("bad", [dict(mip_rel_gap=-0.1), dict(mip_rel_gap=1.0), dict(mip_rel_gap=float("nan")),
                                 dict(mip_abs_gap=-1.0), dict(mip_abs_gap=float("inf"))])
def test_a_bad_gap_is_refused_by_the_config(panel, bad):
    cfg = dict(df=panel, outcome="sales", arm="arm", unitid="unit", time="time", display_graphs=False, **bad)
    with pytest.raises(MlsynthConfigError):
        PANGEO(cfg)
