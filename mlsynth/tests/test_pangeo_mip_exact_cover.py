"""PANGEO's partition MIP returns an exact cover of the arm or raises (#1169).

A unit that no candidate group holds has an exact-cover row with no variables, ``0 = 1``, so the program has no
solution. cvxpy's SCIP interface drops every row without variables (``scip_conif.add_model_lin_constr``: "Ignore
empty constraints"), whatever its right-hand side, and SCIP then reports the remaining program optimal: before this
fix ``solve_partition`` returned a design that left the unit out. HiGHS, through cvxpy or ``highspy``, keeps the row
and reports the program infeasible.

The MIP now checks the candidates before any solve (each unit in at least one group) and the design after it (each
unit in exactly one chosen group, and no unit outside the arm), so the answer does not depend on how a solver treats
a row it was given. Each test runs under every MILP path installed: ``highspy``, and cvxpy with HiGHS and with SCIP.
"""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from mlsynth.exceptions import MlsynthEstimationError
from mlsynth.utils.pangeo_helpers import mip

cp = pytest.importorskip("cvxpy")


def _group(members, score=1.0):
    members = np.asarray(members)
    half = max(len(members) // 2, 1)
    return {"members": members, "score": float(score), "side_a": members[:half], "side_b": members[half:]}


PATHS = ["highspy", "cvxpy:HIGHS", "cvxpy:SCIP"]


@pytest.fixture(params=PATHS)
def path(request, monkeypatch):
    """One MILP path: the highspy backend, or the cvxpy backend held to one solver."""
    name = request.param
    if name == "highspy":
        if not mip._highspy_available():
            pytest.skip("highspy is not installed")
        return "highspy"
    solver = name.split(":")[1]
    if solver not in cp.installed_solvers():
        pytest.skip(f"cvxpy has no {solver}")
    monkeypatch.setattr(mip, "_MILP_SOLVERS", (solver,))
    return "cvxpy"


def _covers_exactly(chosen, units):
    seen = sorted(int(u) for c in chosen for u in c["members"])
    return seen == sorted(int(u) for u in units)


# --- smoke --------------------------------------------------------------------------------------------------------

def test_smoke_a_feasible_arm_is_covered_once(path):
    cands = [_group([0, 1], 4.0), _group([2, 3], 1.0), _group([0, 2], 1.0), _group([1, 3], 1.0)]
    chosen, d = mip.solve_partition(cands, np.arange(4), backend=path, return_diagnostics=True)
    assert _covers_exactly(chosen, np.arange(4))
    assert d["objective_value"] == pytest.approx(2.0)


# --- the defect: a unit in no group ---------------------------------------------------------------------------------

def test_a_unit_in_no_group_is_refused_before_any_solve(path, monkeypatch):
    """Failure: four units, one triple holding three of them. The only obstruction is the unit no group holds; the
    MIP raises, names it, and no solver runs. Before the fix cvxpy with SCIP returned the triple as the design."""
    ran = []
    monkeypatch.setattr(cp.Problem, "solve", lambda *a, **kw: ran.append("cvxpy"))
    monkeypatch.setattr(mip, "_run_highs", lambda h: ran.append("highs"))
    with pytest.raises(MlsynthEstimationError, match=r"no exact cover.*structurally uncoverable: indices \[2\]"):
        mip.solve_partition([_group([0, 1, 3], 3.0)], np.arange(4), backend=path)
    assert ran == []


def test_the_scip_path_drops_an_empty_row_so_the_check_cannot_be_left_to_it():
    """Unit: the reason for the check. cvxpy with SCIP reports a program whose only infeasible row has no variables
    optimal; cvxpy with SciPy's HiGHS reports it infeasible. If a cvxpy release keeps the row, this test fails and
    the docstring above is stale, not the MIP."""
    if "SCIP" not in cp.installed_solvers():
        pytest.skip("cvxpy has no SCIP")
    x = cp.Variable(1, boolean=True)
    p = cp.Problem(cp.Minimize(3 * x), [np.array([[1.0], [1.0], [0.0], [1.0]]) @ x == 1])
    p.solve(solver=cp.SCIP)
    assert p.status == "optimal"
    p.solve(solver=cp.SCIPY)
    assert p.status == "infeasible"


def test_a_design_that_is_not_an_exact_cover_is_refused_after_the_solve(monkeypatch):
    """Failure: the second guard. With the candidate check switched off, the SCIP path returns its design for the
    arm above, and the check of the design refuses it instead of returning it."""
    if "SCIP" not in cp.installed_solvers():
        pytest.skip("cvxpy has no SCIP")
    monkeypatch.setattr(mip, "_MILP_SOLVERS", ("SCIP",))
    monkeypatch.setattr(mip, "_require_coverable", lambda cands, units: None)
    with pytest.raises(MlsynthEstimationError, match=r"not an exact cover.*\[2\] in no chosen group"):
        mip.solve_partition([_group([0, 1, 3], 3.0)], np.arange(4), backend="cvxpy")


@pytest.mark.parametrize("backend", ["highspy", "cvxpy"])
def test_a_solver_that_reports_success_without_a_cover_is_refused(backend, monkeypatch):
    """Failure: the design check on each backend. A solver that says it solved a feasible arm but hands back no
    groups (HiGHS stubbed to report Optimal without running; cvxpy's solve stubbed to leave x unset) gets its
    design refused, not returned as an empty design."""
    if backend == "highspy" and not mip._highspy_available():
        pytest.skip("highspy is not installed")
    monkeypatch.setattr(mip, "_run_highs", lambda h: ("Optimal", False))
    monkeypatch.setattr(cp.Problem, "solve", lambda self, *a, **kw: None)
    monkeypatch.setattr(cp.Problem, "status", property(lambda self: "optimal"))
    cands = [_group([0, 1]), _group([2, 3])]
    with pytest.raises(MlsynthEstimationError, match=r"not an exact cover.*\[0, 1, 2, 3\] in no chosen group"):
        mip.solve_partition(cands, np.arange(4), backend=backend)


# --- unit: the two checks -------------------------------------------------------------------------------------------

def test_the_candidate_check_names_every_uncoverable_unit():
    with pytest.raises(MlsynthEstimationError, match=r"structurally uncoverable: indices \[5, 9\]"):
        mip._require_coverable([_group([1, 3]), _group([3, 7])], np.array([1, 3, 5, 7, 9]))
    mip._require_coverable([_group([1, 3]), _group([5, 7])], np.array([1, 3, 5, 7]))   # every unit held: no error


@pytest.mark.parametrize("chosen,why", [
    ([[0, 1]], r"\[2, 3\] in no chosen group"),
    ([[0, 1], [1, 2, 3]], r"\[1\] in more than one chosen group"),
    ([[0, 1], [2, 3, 8]], r"\[8\] outside the arm"),
])
def test_the_design_check_names_what_is_wrong(chosen, why):
    with pytest.raises(MlsynthEstimationError, match="not an exact cover.*" + why):
        mip._check_exact_cover([_group(c) for c in chosen], np.arange(4))


def test_the_design_check_accepts_an_exact_cover():
    mip._check_exact_cover([_group([2, 0]), _group([3, 1])], np.arange(4))


# --- edge -----------------------------------------------------------------------------------------------------------

def test_units_with_gaps_in_their_indices(path):
    """Edge: unit indices are labels, not positions; an uncoverable one is named by its label."""
    units = np.array([3, 7, 11, 20])
    chosen = mip.solve_partition([_group([3, 11]), _group([7, 20]), _group([3, 7])], units, backend=path)
    assert _covers_exactly(chosen, units)
    with pytest.raises(MlsynthEstimationError, match=r"indices \[20\]"):
        mip.solve_partition([_group([3, 7, 11])], units, backend=path)


def test_one_uncoverable_unit_among_many(path):
    cands = [_group([2 * i, 2 * i + 1]) for i in range(4)]            # units 0..7 paired; unit 8 in none
    with pytest.raises(MlsynthEstimationError, match=r"indices \[8\]"):
        mip.solve_partition(cands, np.arange(9), backend=path)


def test_no_candidates_still_raises_its_own_error(path):
    with pytest.raises(MlsynthEstimationError, match="no admissible supergeo pairs"):
        mip.solve_partition([], np.arange(4), backend=path)


# --- property -------------------------------------------------------------------------------------------------------

@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow,
                                                                  HealthCheck.function_scoped_fixture])
@given(data=st.data(), n=st.integers(2, 8), dropped=st.integers(0, 2), min_pairs=st.integers(1, 3))
def test_property_a_returned_design_is_an_exact_cover(path, data, n, dropped, min_pairs):
    """Property: over arms, candidate sets (groups of two to four units, with up to two units left out of every
    group) and cardinality rows, the MIP either raises MlsynthEstimationError or returns an exact cover of the arm
    with at least ``min_pairs`` groups and its cost; when a unit is in no group it raises, naming that unit."""
    units = np.arange(n)
    gone = set(data.draw(st.lists(st.sampled_from(range(n)), max_size=dropped, unique=True)))
    live = [u for u in range(n) if u not in gone]
    groups = data.draw(st.lists(st.lists(st.sampled_from(live), min_size=2, max_size=4, unique=True),
                                min_size=1, max_size=12)) if len(live) >= 2 else []
    if not groups:
        return
    cands = [_group(sorted(g), data.draw(st.floats(0.0, 10.0))) for g in groups]   # subnormal costs too (#1170)
    held = {u for g in groups for u in g}
    missing = sorted(set(range(n)) - held)
    try:
        chosen, d = mip.solve_partition(cands, units, min_pairs=min_pairs, backend=path, return_diagnostics=True)
    except MlsynthEstimationError as exc:
        if missing:
            assert f"indices {missing[:10]}" in str(exc)
        return
    assert not missing
    assert _covers_exactly(chosen, units) and len(chosen) >= min_pairs
    assert d["objective_value"] == pytest.approx(sum(c["score"] for c in chosen), abs=1e-9)
