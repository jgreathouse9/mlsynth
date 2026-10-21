"""SYNDES's MIP path says what SCIP found (#805).

SCIP stops at a time or gap limit with an incumbent, and cvxpy reports both as
``optimal_inaccurate``, the same status as a solve that converged. The design now
carries a typed :class:`SYNDESSolveOutcome` -- SCIP's own status, its dual bound
in the objective's units, the relative gap -- and a solve stopped at a limit
warns. The weights are refitted for the rounded assignment, so the objective is
the one the returned weights attain.
"""

import warnings

import numpy as np
import pytest

from mlsynth.exceptions import MlsynthEstimationError
from mlsynth.utils.syndes_helpers import optimization as opt
from mlsynth.utils.syndes_helpers.optimization import (SYNDESSolveWarning,
                                                        _gap,
                                                        _rounded_assignment,
                                                        solve_synthetic_design,
                                                        solve_synthetic_design_pool)
from mlsynth.utils.syndes_helpers.structures import SYNDESSolveOutcome

pytest.importorskip("pyscipopt")

MODES = ("global_2way", "global_equal_weights", "per_unit")


def _panel(N=8, T=12, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((T, 3)) @ rng.standard_normal((3, N))
            + 0.3 * rng.standard_normal((T, N)))


def _objective(Y, d):
    """The mode's objective, recomputed from the weights the design returns."""
    T = Y.shape[0]
    lam = d.lambda_value
    if d.mode == "per_unit":
        S = np.flatnonzero(d.assignment)
        Q = np.asarray(d.treated_weights)
        resid = Y[:, S] - Y @ Q[S].T
        return float((resid ** 2).sum() / (len(S) * T) + lam / len(S) * (Q ** 2).sum())
    tw, cw = np.asarray(d.treated_weights), np.asarray(d.control_weights)
    r = Y @ (tw - cw)
    return float(r @ r / T + lam * (tw @ tw + cw @ cw))


# --------------------------------------------------------------------------- #
# smoke
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", MODES)
def test_a_solve_carries_its_outcome(mode):
    d = solve_synthetic_design(_panel(), 2, mode=mode)
    out = d.solve_outcome
    assert isinstance(out, SYNDESSolveOutcome)
    assert out.status == "optimal" and not out.stopped_at_limit
    assert out.objective == d.objective_value and np.isfinite(out.dual_bound)


# --------------------------------------------------------------------------- #
# the objective is the returned weights'
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", MODES)
def test_the_objective_is_what_the_returned_weights_attain(mode):
    Y = _panel()
    d = solve_synthetic_design(Y, 2, mode=mode)
    assert d.objective_value == pytest.approx(_objective(Y, d), rel=1e-6)


@pytest.mark.parametrize("mode", MODES)
def test_the_refit_is_no_worse_than_scips_point(mode):
    d = solve_synthetic_design(_panel(seed=1), 3, mode=mode)
    out = d.solve_outcome
    assert out.objective <= out.solver_objective * (1 + 1e-6) + 1e-12


def test_one_way_control_weights_are_a_simplex():
    d = solve_synthetic_design(_panel(seed=2), 3, mode="global_equal_weights")
    cw = np.asarray(d.control_weights)
    assert (cw >= 0).all() and cw.sum() == pytest.approx(1.0, abs=1e-12)
    assert cw[np.asarray(d.assignment) == 1].max() == 0.0


@pytest.mark.parametrize("mode", MODES)
def test_the_bound_is_in_the_objectives_units(mode):
    """At a proven optimum the gap is zero. The one-way objective carries the
    constant ``lam / K`` that SCIP's canonical bound omits; a bound without it
    would show a gap of ``lam / K`` over the objective."""
    d = solve_synthetic_design(_panel(seed=3), 2, mode=mode)
    out = d.solve_outcome
    assert out.dual_bound <= out.objective * (1 + 1e-6) + 1e-9
    assert abs(out.gap) <= 1e-4


# --------------------------------------------------------------------------- #
# a binding limit is reported
# --------------------------------------------------------------------------- #


def _large(seed=3, T=30, N=40):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((T, 4)) @ rng.standard_normal((4, N))
            + 0.3 * rng.standard_normal((T, N)))


def test_a_solve_stopped_by_its_time_limit_warns_and_says_so():
    with pytest.warns(SYNDESSolveWarning, match="time limit") as rec:
        d = solve_synthetic_design(_large(), 5, mode="global_equal_weights",
                                   time_limit=2.0)
    out = d.solve_outcome
    assert out.status == "timelimit" and out.stopped_at_limit
    assert out.gap > 0 and out.dual_bound < out.objective
    assert f"{out.gap:.3g}" in str(rec[0].message)


def test_a_converged_solve_is_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error", SYNDESSolveWarning)
        solve_synthetic_design(_panel(), 2, mode="global_2way")


# --------------------------------------------------------------------------- #
# units
# --------------------------------------------------------------------------- #


def test_the_rounding_accepts_binaries_to_scips_feasibility_tolerance():
    np.testing.assert_array_equal(
        _rounded_assignment(np.array([1 - 1e-7, 2e-7, 1.0, 0.0])), [1, 0, 1, 0])


def test_a_fractional_assignment_is_refused():
    with pytest.raises(MlsynthEstimationError, match="not integral"):
        _rounded_assignment(np.array([1.0, 0.3, 0.0]))


@pytest.mark.parametrize("obj,bound,expected", [
    (2.0, 1.0, 0.5), (1.0, 1.0, 0.0), (0.0, 0.0, 0.0), (1.0, 2.0, -0.5),
    (-1.0, -2.0, 0.5),
])
def test_the_gap_is_relative_to_the_larger_of_the_two(obj, bound, expected):
    assert _gap(obj, bound) == pytest.approx(expected)


def test_the_gap_does_not_depend_on_units():
    for s in (1e-9, 1.0, 1e9):
        assert _gap(2.0 * s, 1.5 * s) == pytest.approx(0.25, rel=1e-12)


# --------------------------------------------------------------------------- #
# failures are reported
# --------------------------------------------------------------------------- #


def test_a_failed_refit_raises(monkeypatch):
    def broken(*a, **k):
        raise MlsynthEstimationError("SYNDES refit for the rounded assignment failed: "
                                     "solver_error")
    monkeypatch.setattr(opt, "_refit", broken)
    with pytest.raises(MlsynthEstimationError, match="refit"):
        solve_synthetic_design(_panel(), 2, mode="global_2way")


def test_the_pool_does_not_read_a_failure_as_an_exhausted_region(monkeypatch):
    def broken(*a, **k):
        raise MlsynthEstimationError("SYNDES refit for the rounded assignment failed")
    monkeypatch.setattr(opt, "_refit", broken)
    with pytest.raises(MlsynthEstimationError, match="refit"):
        solve_synthetic_design_pool(_panel(), 2, top_K=2, mode="global_2way")


def test_the_pool_still_stops_when_the_region_is_exhausted():
    """N = 3, K = 2: three designs exist, so a pool of five returns three."""
    pool = solve_synthetic_design_pool(_panel(N=3), 2, top_K=5, mode="global_2way")
    assert len(pool) == 3


def test_control_weights_outside_the_simplex_are_refused():
    from mlsynth.utils.syndes_helpers.optimization import _simplex_within_tolerance
    treated = np.array([True, False, False])
    np.testing.assert_allclose(
        _simplex_within_tolerance(np.array([1e-9, 0.6, 0.4 - 2e-9]), treated),
        [0.0, 0.6 / (1 - 2e-9), (0.4 - 2e-9) / (1 - 2e-9)])
    for bad in ([0.2, 0.4, 0.4], [0.0, 1.1, -0.1], [0.0, 0.5, 0.4]):
        with pytest.raises(MlsynthEstimationError, match="simplex"):
            _simplex_within_tolerance(np.array(bad), treated)


@pytest.mark.parametrize("behaviour", ["raise", "no_value"])
def test_a_refit_that_fails_in_any_way_raises_the_estimation_error(monkeypatch, behaviour):
    import cvxpy as cp
    orig = cp.Problem.solve

    def fake(self, *a, **kw):
        if kw.get("solver") != cp.CLARABEL:
            return orig(self, *a, **kw)
        if behaviour == "raise":
            raise RuntimeError("refit solver gone")
        self._status = "optimal"
        return None
    monkeypatch.setattr(cp.Problem, "solve", fake)
    with pytest.raises(MlsynthEstimationError, match="refit"):
        solve_synthetic_design(_panel(), 2, mode="global_2way")


# --------------------------------------------------------------------------- #
# another solver: cvxpy's status, and the limits that did not reach it
# --------------------------------------------------------------------------- #


def test_a_solver_without_scips_bounds_reports_what_it_can():
    """ECOS_BB solves the mixed-integer SOCP but takes no SCIP limits and gives
    no dual bound; the outcome says so."""
    d = solve_synthetic_design(_panel(N=5, T=8), 2, mode="global_2way",
                               solver="ECOS_BB", gap_limit=0.05, time_limit=10.0)
    out = d.solve_outcome
    assert out.solver_status == out.status
    assert out.dual_bound is None and out.gap is None and out.solve_time is None
    assert not out.limits_applied and not out.stopped_at_limit


def test_limits_reach_scip_only():
    from mlsynth.utils.syndes_helpers.optimization import _scip_limits
    assert _scip_limits("SCIP", 0.05, 60.0) == (
        {"limits/gap": 0.05, "limits/time": 60.0}, True)
    assert _scip_limits("ECOS_BB", 0.05, None) == ({}, False)
    assert _scip_limits("ECOS_BB", None, None) == ({}, True)


def test_a_fractional_assignment_from_the_solver_is_refused(monkeypatch):
    """The solve path reads the solver's assignment through the integrality
    check, not by rounding at one half."""
    import mlsynth.utils.miqp_accel as accel
    real = accel.solve_warm_cut

    def fractional(problem, D, **kw):
        out = real(problem, D, **kw)
        v = np.asarray(D.value, dtype=float).copy()
        v[np.argmax(v)] = 0.6
        D.save_value(v)
        return out
    monkeypatch.setattr(accel, "solve_warm_cut", fractional)
    with pytest.raises(MlsynthEstimationError, match="not integral"):
        solve_synthetic_design(_panel(), 2, mode="global_2way")


def test_the_accelerator_entry_is_recorded_only_when_asked_for():
    """Every SCIP solve now runs through ``solve_warm_cut``; ``raw_results`` names
    the accelerator only when a warm start or a cut was supplied."""
    Y = _panel()
    plain = solve_synthetic_design(Y, 2, mode="global_2way")
    assert "accel" not in plain.raw_results
    warm = solve_synthetic_design(Y, 2, mode="global_2way",
                                  warm_start_D=np.asarray(plain.assignment))
    assert warm.raw_results["accel"]["status"] == "optimal"
    assert warm.solve_outcome.status == "optimal"
