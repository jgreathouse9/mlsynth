"""Clarabel's settings in slopefree.solve (#965, task 40).

Every Clarabel solve of a relaxation runs on one thread (test_parallel.py says why)
and without iterative refinement. The certificate is built from the multipliers
(fbound.certificate, rule 26 of agents/agents_solvers.md), so it is a valid bound
at any multipliers: refinement buys accuracy in the point that the bound does not
read. These tests pin the settings on both of Clarabel's paths, leave SCS's alone,
and check that the bound stays valid at the multipliers the settings give."""
from fractions import Fraction as Q
from itertools import combinations

import cvxpy as cp
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mlsynth.utils.tbr_helpers.design.exact import slopefree as sf
from .brute import geo_panel, panel, positive_optimum
from mlsynth.utils.tbr_helpers.design.exact.certificate import ExactData

SETTINGS = settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def _record(monkeypatch):
    seen = []
    real = cp.Problem.solve
    monkeypatch.setattr(cp.Problem, "solve", lambda self, *a, **k: seen.append(k) or real(self, *a, **k))
    return seen


def _exact(y, X):
    D = ExactData(y, X)
    return D, D.c, D.G


def _ratio(c, G, C):
    cc = sum((c[j] for j in C), Q(0))
    return cc * cc / sum((G[i][j] for i in C for j in C), Q(0))


def _designs(p, m):
    for C in combinations(range(p), m):
        u = np.zeros(p, dtype=int)
        u[list(C)] = 1
        yield set(C), u


def _clarabel_calls(seen):
    return [k for k in seen if k.get("solver") in ("CLARABEL", cp.CLARABEL)]


# ---------------------------------------------------------------- smoke


def test_smoke_a_relaxation_solves_under_the_settings():
    y, X = panel(0, p=6)
    D, c, G = _exact(y, X)
    lam = sf.dyadic_below(_ratio(c, G, positive_optimum(y, X)[1]) * Q(9, 10))
    sol = sf.solve(c, G, lam, sf.Rows(c, 3))
    assert sol.status in ("optimal", "optimal_inaccurate")
    assert np.isfinite(sol.value)


# ---------------------------------------------------------------- unit


def test_the_settings_are_one_thread_and_no_refinement():
    assert sf.CLARABEL_SETTINGS == {"max_threads": 1, "iterative_refinement_enable": False}


def test_the_full_space_solve_takes_the_settings(monkeypatch):
    y, X = panel(1, p=6)
    D, c, G = _exact(y, X)
    lam = sf.dyadic_below(_ratio(c, G, positive_optimum(y, X)[1]) * Q(9, 10))
    seen = _record(monkeypatch)
    R = sf.Rows(c, 3)
    assert R.face is None
    sf.solve(c, G, lam, R, solver="clarabel")
    calls = _clarabel_calls(seen)
    assert calls and all(k.get("max_threads") == 1 and k.get("iterative_refinement_enable") is False for k in calls)


def test_the_face_solve_takes_the_settings(monkeypatch):
    M = geo_panel(11, n=20, p=9)
    D = ExactData(M[:, 0], M[:, 1:])
    seen = _record(monkeypatch)
    s = sf.solve(D.c, D.G, sf.dyadic_below(D.yy / 2), sf.Rows(D.c, 3, fixes=((0, 1),), level="shor", face=True))
    calls = _clarabel_calls(seen)
    assert s.face and calls
    assert all(k.get("max_threads") == 1 and k.get("iterative_refinement_enable") is False for k in calls)


def test_scs_is_not_given_clarabels_settings(monkeypatch):
    """SCS rejects keyword arguments it does not know, so Clarabel's stay off its call."""
    y, X = panel(2, p=6)
    D, c, G = _exact(y, X)
    lam = sf.dyadic_below(_ratio(c, G, positive_optimum(y, X)[1]) * Q(9, 10))
    seen = _record(monkeypatch)
    sf.solve(c, G, lam, sf.Rows(c, 3), solver="scs")
    scs = [k for k in seen if k.get("solver") in ("SCS", cp.SCS)]
    assert scs and all("iterative_refinement_enable" not in k and "max_threads" not in k for k in scs)


@pytest.mark.parametrize("seed", range(3))
def test_the_bound_holds_and_the_value_is_close(seed):
    """At the multipliers Clarabel returns without refinement the certificate is at
    least every design's value, and within 1e-3 of the solver's own value relative to
    the larger of the two and y'y: the multipliers are less accurate, the bound is
    still a bound."""
    y, X = panel(seed, p=6)
    D, c, G = _exact(y, X)
    lam = sf.dyadic_below(_ratio(c, G, positive_optimum(y, X)[1]) * Q(9, 10))
    A = sf.objective(c, G, lam)
    for m in (2, 3):
        R = sf.Rows(c, m)
        sol = sf.solve(c, G, lam, R)
        b = sf.bound(A, R, sol.y_eq, sol.y_in, m)
        best = max(sf.design_value(A, u) for C, u in _designs(6, m) if c @ u.astype(object) >= 0)
        assert b >= best
        assert abs(float(b) - sol.value * sol.scale) <= 1e-3 * max(abs(float(b)), float(D.yy))


@given(seed=st.integers(0, 10_000), m=st.integers(1, 4), shrink=st.fractions(Q(1, 2), Q(1)))
@SETTINGS
def test_property_the_bound_is_valid_at_any_panel(seed, m, shrink):
    """Over random panels, sizes and lam below the optimum's ratio, the certificate at
    the multipliers of a solve without refinement is at least every design's value."""
    y, X = panel(seed, p=6)
    D, c, G = _exact(y, X)
    lam = sf.dyadic_below(_ratio(c, G, positive_optimum(y, X)[1]) * shrink)
    A = sf.objective(c, G, lam)
    R = sf.Rows(c, m)
    sol = sf.solve(c, G, lam, R)
    b = sf.bound(A, R, sol.y_eq, sol.y_in, m)
    best = max(sf.design_value(A, u) for C, u in _designs(6, m) if c @ u.astype(object) >= 0)
    assert b is None or b >= best


# ---------------------------------------------------------------- edge


def test_a_single_control_size_solves_under_the_settings():
    y, X = panel(3, p=6)
    D, c, G = _exact(y, X)
    lam = sf.dyadic_below(_ratio(c, G, positive_optimum(y, X)[1]) * Q(9, 10))
    sol = sf.solve(c, G, lam, sf.Rows(c, 1))
    assert sol.status in ("optimal", "optimal_inaccurate", "infeasible", "infeasible_inaccurate")


# ---------------------------------------------------------------- failure


def test_a_solver_error_under_the_settings_is_raised(monkeypatch):
    """The settings change nothing about failure: a SolverError is reported and raised."""
    y, X = panel(4, p=6)
    D, c, G = _exact(y, X)
    lam = sf.dyadic_below(_ratio(c, G, positive_optimum(y, X)[1]) * Q(9, 10))

    def broken(self, *a, **k):
        raise cp.error.SolverError("Solver 'CLARABEL' failed.")

    monkeypatch.setattr(cp.Problem, "solve", broken)
    with pytest.raises(cp.error.SolverError):
        sf.solve(c, G, lam, sf.Rows(c, 3), solver="clarabel")
