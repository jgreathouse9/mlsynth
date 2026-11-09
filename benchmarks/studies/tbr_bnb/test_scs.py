"""The relaxation by SCS (#965).

At 100 geos Clarabel's time is the 97 x 97 cone's factorization at every
interior-point iteration, on the face or off it (README, "The relaxation on its
face"). SCS (O'Donoghue, Chu, Parikh and Boyd 2016) splits the cone from the rows
(ADMM on the homogeneous self-dual embedding), so an iteration is one projection
onto the cone and one sparse solve. Its multipliers go to the same exact
certificate, which holds for any multipliers, so the bound is valid whatever SCS
returns; only its tightness depends on the solve. SCS solves in the full space:
on the face the auxiliary rows slowed it (results/face.log). The face's
certificate still applies, through ``bound``. A solve SCS does not finish is
solved again by Clarabel, and both solves are reported."""
import itertools
from fractions import Fraction as Q

import cvxpy as cp
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.exceptions import MlsynthConfigError

from . import slopefree
from .brute import geo_panel, panel, positive_optimum
from .certificate import ExactData
from .progress import reporting
from .slopefree import Rows, bound, certify_slope_free, objective, solve
from .treatment import _split

SETTINGS = settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])


@pytest.fixture(autouse=True)
def _bound_every_node(monkeypatch):
    monkeypatch.setattr(slopefree, "PRICE_LIMIT", 0)


def _designs(p, m, fixes):
    fx = dict(fixes)
    for C in itertools.combinations(range(p), m):
        if all((j in C) == bool(v) for j, v in fx.items()):
            yield C


def _pinned():
    y, X = panel(2, p=6)
    D = ExactData(y, X)
    return D, slopefree.dyadic_below(D.yy / 2)


def _bed():
    return _split(geo_panel(66, n=20, p=15), (0, 9, 10))[:2]


def _spy_solves(monkeypatch):
    """Every cvxpy solve's keyword arguments, in order."""
    seen = []
    real = cp.Problem.solve
    monkeypatch.setattr(cp.Problem, "solve", lambda self, *a, **k: seen.append(k) or real(self, *a, **k))
    return seen


# ---------------------------------------------------------------- smoke


def test_smoke_a_bed_group_certifies_the_same_design_with_scs():
    y, X = _bed()
    a = certify_slope_free(y, X, restarts=5, solver="scs")
    b = certify_slope_free(y, X, restarts=5, solver="clarabel")
    assert a.certified and b.certified and (a.incumbent, a.controls) == (b.incumbent, b.controls)


def test_smoke_scs_solves_a_size():
    D, lam = _pinned()
    s = solve(D.c, D.G, lam, Rows(D.c, 3, level="full"), solver="scs")
    assert s.solver == "scs" and s.status == "optimal" and np.isfinite(s.value) and s.Z.shape == (7, 7)


# ---------------------------------------------------------------- unit


def test_clarabel_is_the_default():
    assert slopefree.SOLVER == "clarabel"
    D, lam = _pinned()
    assert solve(D.c, D.G, lam, Rows(D.c, 3, level="shor")).solver == "clarabel"


def test_scs_is_called_with_its_tolerances(monkeypatch):
    seen = _spy_solves(monkeypatch)
    D, lam = _pinned()
    solve(D.c, D.G, lam, Rows(D.c, 3, level="shor"), solver="scs")
    assert [k["solver"] for k in seen] == ["SCS"]
    assert seen[0]["eps_abs"] == seen[0]["eps_rel"] == slopefree.SCS_EPS == 1e-6


def test_scs_matches_clarabel_in_the_full_space():
    D, lam = _pinned()
    for level in ("shor", "full"):
        a = solve(D.c, D.G, lam, Rows(D.c, 3, level=level), solver="scs")
        b = solve(D.c, D.G, lam, Rows(D.c, 3, level=level), solver="clarabel")
        assert abs(a.value - b.value) * a.scale <= 2e-4 * a.scale


def test_scs_solves_off_the_face_and_the_face_still_certifies(monkeypatch):
    """The face's rows slow SCS (results/face.log), so it solves in the full space;
    the rows keep their face, and bound() still takes the face's certificate."""
    called = []
    monkeypatch.setattr(slopefree, "_solve_on_face", lambda *a, **k: called.append(1))
    D, lam = _pinned()
    R = Rows(D.c, 3, level="shor", face=True)
    s = solve(D.c, D.G, lam, R, solver="scs")
    assert not s.face and called == [] and R.face is not None
    A = objective(D.c, D.G, lam)
    assert bound(A, R, s.y_eq, s.y_in, 3) <= bound(A, Rows(D.c, 3, level="shor"), s.y_eq, s.y_in, 3)


def test_the_solve_event_names_its_solver():
    D, lam = _pinned()
    events = []
    with reporting(progress=events.append):
        solve(D.c, D.G, lam, Rows(D.c, 3, level="shor"), solver="scs")
        solve(D.c, D.G, lam, Rows(D.c, 3, level="shor"), solver="clarabel")
        solve(D.c, D.G, lam, Rows(D.c, 3, level="shor", face=True), solver="clarabel")
    assert [e.fields["solver"] for e in events if e.kind == "solve"] == ["scs", "clarabel", "clarabel"]


@pytest.mark.parametrize("shor", [False, True])
def test_the_size_certificate_passes_its_solver(monkeypatch, shor):
    """To every level it solves: the Shor level only when asked for (SHOR_TIER)."""
    seen = []
    real = slopefree.solve

    def spy(c, G, lam, R, solver=None):
        seen.append((R.level, solver))
        return real(c, G, lam, R, solver=solver)

    monkeypatch.setattr(slopefree, "solve", spy)
    D, lam = _pinned()
    slopefree.size_certificate(D.c, D.G, lam, 3, schedules=(), solver="scs", shor=shor)
    assert seen and all(s == "scs" for _, s in seen)
    assert ("shor" in {lv for lv, _ in seen}) is shor


def test_the_solver_reaches_every_solve_of_the_search(monkeypatch):
    seen = _spy_solves(monkeypatch)
    y, X = _bed()
    certify_slope_free(y, X, restarts=5, solver="scs")
    assert seen and all(k["solver"] == "SCS" for k in seen)


def test_the_group_screen_gets_the_solver(monkeypatch):
    """Against a ceiling the search first tries one certificate for every size
    (group_certificate); it solves with the search's solver."""
    seen = []
    real = slopefree.group_certificate

    def spy(*a, solver=None, **k):
        seen.append(solver)
        return real(*a, solver=solver, **k)

    monkeypatch.setattr(slopefree, "group_certificate", spy)
    y, X = _bed()
    first = certify_slope_free(y, X, restarts=5)
    certify_slope_free(y, X, restarts=5, ceiling=first.incumbent * Q(99, 100), solver="scs")
    assert seen == ["scs"]


def test_the_group_rows_solve_with_scs(monkeypatch):
    seen = _spy_solves(monkeypatch)
    D, _ = _pinned()
    best, _ = positive_optimum(*panel(2, p=6))
    res = slopefree.group_certificate(D.c, D.G, slopefree.dyadic_below(best * 1.5), schedules=(), solver="scs")
    assert [k["solver"] for k in seen] == ["SCS"] and res.tier in ("full", "open")


@pytest.mark.parametrize("status, answered", [
    ("optimal", True), ("optimal_inaccurate", True), ("infeasible", True), ("infeasible_inaccurate", True),
    ("unbounded", False), ("unbounded_inaccurate", False), ("user_limit", False), ("solver_error", False),
    ("infeasible_or_unbounded", False)])
def test_which_scs_statuses_are_an_answer(status, answered):
    """An answer is a point with multipliers, or a reported infeasibility; a limit,
    an error or an unbounded relaxation (impossible: Z is bounded by its trace) is
    solved again by Clarabel."""
    assert slopefree._scs_answered(status) is answered


# ---------------------------------------------------------------- properties


@SETTINGS
@given(seed=st.integers(0, 10_000), p=st.sampled_from([5, 6]), data=st.data())
def test_property_scs_certifies_its_own_value(seed, p, data):
    """The exact bound at SCS's multipliers is its value to its tolerance: over 700
    solves of panels with six controls the gap is at most 4.7e-5 of the objective's
    scale (median 8e-9), so 2e-4 holds with room. A node whose fixings leave no design
    with c'u >= 0 is reported infeasible with no point."""
    y, X = panel(seed % 60, p=p)
    D = ExactData(y, X)
    best, _ = positive_optimum(y, X)
    lam = slopefree.dyadic_below(best * 0.99 if best > 0 else D.yy / 2)
    m = data.draw(st.integers(2, p - 2))
    j = data.draw(st.integers(0, p - 1))
    fixes = data.draw(st.sampled_from([(), ((j, 0),), ((j, 1),)]))
    feasible = any(sum(D.c[i] for i in C) >= 0 for C in _designs(p, m, fixes))
    A = objective(D.c, D.G, lam)
    for level in ("shor", "full"):
        R = Rows(D.c, m, fixes=fixes, level=level, face=data.draw(st.booleans()))
        s = solve(D.c, D.G, lam, R, solver="scs")
        if s.solver == "clarabel":       # SCS did not answer: the fallback's own tests cover it
            continue
        if not feasible:
            assert s.status.startswith("infeasible") and s.Z is None and s.value == -np.inf
            continue
        v = s.value * s.scale
        b = float(bound(A, R, s.y_eq, s.y_in, m))
        assert b >= v - 2e-4 * s.scale and abs(b - v) <= 2e-4 * s.scale, (level, s.status, b, v, s.scale)


@settings(max_examples=10, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10_000))
def test_property_scs_certifies_the_enumerated_optimum(seed):
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(slopefree, "PRICE_LIMIT", 0)
        y, X = panel(seed % 60, p=7)
        best, C = positive_optimum(y, X)
        if C is None:
            return
        out = certify_slope_free(y, X, restarts=3, solver="scs")
        if out.certified:
            assert set(out.controls) == C
            assert float(out.incumbent) == pytest.approx(best, rel=1e-9)
        else:
            assert out.open
    finally:
        mp.undo()


# ---------------------------------------------------------------- edges


def test_an_infeasible_node_is_reported_by_scs():
    y, X = panel(17, p=6)
    D = ExactData(y, X)
    best, _ = positive_optimum(y, X)
    lam = slopefree.dyadic_below(best * 0.99)
    for level in ("shor", "full"):
        s = solve(D.c, D.G, lam, Rows(D.c, 3, fixes=((0, 0),), level=level), solver="scs")
        assert s.solver == "scs" and s.status.startswith("infeasible") and s.Z is None and s.value == -np.inf


def test_a_forked_worker_solves_with_scs_after_the_parent_has():
    from concurrent.futures import TimeoutError as Timeout
    import os
    from . import treatment

    D, lam = _pinned()
    args = (D.c, D.G, lam, Rows(D.c, 3, level="full"))
    solve(*args, solver="scs")
    pool = treatment._pool(1)
    try:
        assert pool.submit(_status_by_scs, *args).result(timeout=120) == "optimal"
    except Timeout:                       # pragma: no cover - reached only if SCS hangs in a fork
        for q in list(pool._processes):
            os.kill(q, 9)
        pytest.fail("a worker forked after the parent's SCS solve never returned")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _status_by_scs(c, G, lam, R):
    return solve(c, G, lam, R, solver="scs").status


# ---------------------------------------------------------------- failures


def _scs_fails(monkeypatch, how):
    """SCS fails as ``how`` says; Clarabel runs as itself."""
    real = cp.Problem.solve

    def fake(self, *a, **k):
        if k.get("solver") == "SCS":
            if how == "error":
                raise cp.error.SolverError("injected")
            out = real(self, *a, **k)
            self._status = how
            return out
        return real(self, *a, **k)

    monkeypatch.setattr(cp.Problem, "solve", fake)


@pytest.mark.parametrize("how", ["error", "user_limit"])
def test_a_solve_scs_does_not_answer_is_solved_by_clarabel_and_both_are_reported(monkeypatch, how):
    _scs_fails(monkeypatch, how)
    D, lam = _pinned()
    events = []
    with reporting(progress=events.append):
        s = solve(D.c, D.G, lam, Rows(D.c, 3, level="shor"), solver="scs")
    want = solve(D.c, D.G, lam, Rows(D.c, 3, level="shor"), solver="clarabel")
    got = [(e.fields["solver"], e.fields["status"]) for e in events if e.kind == "solve"]
    assert got == [("scs", how), ("clarabel", want.status)]
    assert s.solver == "clarabel" and s.value == pytest.approx(want.value, rel=1e-9)


def test_the_fallback_solves_on_the_face_when_the_rows_have_one(monkeypatch):
    _scs_fails(monkeypatch, "error")
    D, lam = _pinned()
    s = solve(D.c, D.G, lam, Rows(D.c, 3, level="shor", face=True), solver="scs")
    assert s.solver == "clarabel" and s.face


def test_when_both_solvers_fail_the_error_is_raised_and_both_are_reported(monkeypatch):
    def fail(self, *a, **k):
        raise cp.error.SolverError("injected " + k.get("solver", ""))

    monkeypatch.setattr(cp.Problem, "solve", fail)
    D, lam = _pinned()
    events = []
    with reporting(progress=events.append):
        with pytest.raises(cp.error.SolverError, match="injected CLARABEL"):
            solve(D.c, D.G, lam, Rows(D.c, 3, level="shor"), solver="scs")
    assert [(e.fields["solver"], e.fields["status"]) for e in events] == [("scs", "error"), ("clarabel", "error")]


@pytest.mark.parametrize("bad", ["SCS", "gurobi", "", None, 1, True])
def test_a_malformed_solver_is_refused(monkeypatch, bad):
    """Before the search starts: the descent is never run on a bad flag."""
    from . import search

    def no_descent(*a, **k):
        raise AssertionError("the search started before the solver was checked")

    y, X = panel(0, p=4)
    D, lam = _pinned()
    with monkeypatch.context() as mp:
        mp.setattr(search, "incumbent", no_descent)
        with pytest.raises(MlsynthConfigError, match="solver"):
            certify_slope_free(y, X, restarts=3, solver=bad)
    if bad is not None:                   # None is SOLVER in size_certificate() and solve()
        with pytest.raises(MlsynthConfigError, match="solver"):
            slopefree.size_certificate(D.c, D.G, lam, 3, solver=bad)
        with pytest.raises(MlsynthConfigError, match="solver"):
            solve(D.c, D.G, lam, Rows(D.c, 3, level="shor"), solver=bad)
