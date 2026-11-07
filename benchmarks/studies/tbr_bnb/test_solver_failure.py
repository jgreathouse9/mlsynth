"""A solver failure in the full relaxation is no bound (#965).

On a simulated panel of 100 geos at four treated geos, Clarabel raised SolverError
on the full relaxation of one size while the search branched on group
(4, 27, 33, 58), and the error ended the whole treatment search. The Shor tier
already read a failure as no certificate and handed the size on. The full tier now
does the same: the size is left open, with the last round's relaxation point when
one was solved, and the branch and bound branches on it like any size whose
certificate does not close."""
from fractions import Fraction as Q

import cvxpy as cp
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from . import slopefree
from .brute import geo_panel, panel, positive_optimum, treatment_optimum
from .certificate import ExactData
from .slopefree import certify_slope_free, size_certificate
from .treatment import certify_treatment

SETTINGS = settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
N_TEST = 8


@pytest.fixture(autouse=True)
def _bound_every_node(monkeypatch):
    """Every node is bounded, so the full relaxation is reached (test_slopefree.py)."""
    monkeypatch.setattr(slopefree, "PRICE_LIMIT", 0)


def _fail_full(monkeypatch, which=None, error=None):
    """The full relaxation's solves raise; ``which`` the call numbers that do (1 the
    first), every one when None. The Shor tier's solves run as they are."""
    real = slopefree.solve
    calls = []

    def solve(c, G, lam, R):
        if R.level == "full":
            calls.append(1)
            if which is None or len(calls) in which:
                raise (error or cp.error.SolverError("injected"))
        return real(c, G, lam, R)

    monkeypatch.setattr(slopefree, "solve", solve)
    return calls


def _no_cheap_tier(monkeypatch):
    """The dual tier finds nothing and the Shor solve fails, so the full tier decides."""
    monkeypatch.setattr(slopefree, "dual_multipliers", lambda *a, **k: (None, None))
    real = slopefree.solve

    def solve(c, G, lam, R):
        if R.level == "shor":
            raise cp.error.SolverError("injected")
        return real(c, G, lam, R)

    monkeypatch.setattr(slopefree, "solve", solve)


def _size_inputs(seed=0, p=6, m=3):
    y, X = panel(seed, p=p)
    D = ExactData(y, X)
    return D.c, D.G, Q(0), m


# ---------------------------------------------------------------- smoke


def test_smoke_a_failed_full_solve_leaves_the_size_open(monkeypatch):
    _no_cheap_tier(monkeypatch)
    _fail_full(monkeypatch)
    c, G, lam, m = _size_inputs()
    res = size_certificate(c, G, lam, m)
    assert not res.certified and res.bound is None and res.u is None


# ---------------------------------------------------------------- unit


def test_a_later_failed_round_keeps_the_last_solved_point(monkeypatch):
    """Round 0 is solved and does not close; round 1 fails. The size is open, and
    branching gets round 0's relaxation point."""
    _no_cheap_tier(monkeypatch)
    monkeypatch.setattr(slopefree, "bound", lambda *a, **k: Q(1))
    seen, points = [], []
    real_solve = slopefree.solve

    def solve(c, G, lam, R):
        if R.level == "full":
            seen.append(1)
            if len(seen) == 2:
                raise cp.error.SolverError("injected")
            sol = real_solve(c, G, lam, R)
            points.append(np.asarray(sol.Z)[0, 1:].copy())
            return sol
        return real_solve(c, G, lam, R)

    monkeypatch.setattr(slopefree, "solve", solve)
    tris = iter([[(1.0, 0, (0, 1, 2))], [(1.0, 0, (0, 1, 3))]])
    monkeypatch.setattr(slopefree, "separate_triangles", lambda u, U, k=400: next(tris))
    c, G, lam, m = _size_inputs()
    res = size_certificate(c, G, lam, m)
    assert not res.certified and res.bound is None and res.rounds == 1 and res.triangles == 1
    assert res.u is not None and np.array_equal(res.u, points[0])


def test_every_full_solve_failing_still_certifies_by_branching(monkeypatch):
    """With no full relaxation ever solved, every size branches down to its designs,
    which are evaluated exactly: the optimum is still certified."""
    _no_cheap_tier(monkeypatch)
    calls = _fail_full(monkeypatch)
    y, X = panel(0, p=5)
    best, C = positive_optimum(y, X)
    out = certify_slope_free(y, X, hull=False, restarts=3, group_screen=False)
    assert calls and out.nodes > len(out.sizes)          # sizes failed and were branched
    assert out.certified and set(out.controls) == C
    assert float(out.incumbent) == pytest.approx(best, rel=1e-9)


def test_a_failure_at_the_depth_limit_is_reported_open(monkeypatch):
    """Nothing is pruned on a failure: at the depth limit the size is left open and
    the search reports it uncertified."""
    _no_cheap_tier(monkeypatch)
    calls = _fail_full(monkeypatch)
    y, X = panel(0, p=6)
    out = certify_slope_free(y, X, hull=False, max_depth=0, restarts=3, group_screen=False)
    assert calls and not out.certified and out.open


def test_the_treatment_search_survives_a_failed_solve(monkeypatch):
    """The incident: one failed solve inside one group's certification no longer ends
    the treatment search, and the design is the key's optimum."""
    _no_cheap_tier(monkeypatch)
    calls = _fail_full(monkeypatch, which={1, 3, 7})
    M = geo_panel(5)
    out = certify_treatment(M, 2, n_test=N_TEST, restarts=3)
    key, T, C = treatment_optimum(M, 2, N_TEST)
    assert len(calls) >= 7 and out.certified
    assert tuple(out.key) == pytest.approx(tuple(key), rel=1e-12)
    assert set(out.treatment) == T and set(out.controls) == C


# ---------------------------------------------------------------- properties


@given(seed=st.integers(0, 10_000), which=st.sets(st.integers(1, 40), max_size=12))
@SETTINGS
def test_property_failed_solves_never_certify_a_wrong_design(seed, which):
    import pytest as _pytest
    mp = _pytest.MonkeyPatch()
    try:
        mp.setattr(slopefree, "PRICE_LIMIT", 0)
        _no_cheap_tier(mp)
        _fail_full(mp, which=which)
        y, X = panel(seed, p=5)
        best, C = positive_optimum(y, X)
        if C is None:
            return
        out = certify_slope_free(y, X, hull=False, restarts=3, group_screen=False)
        if out.certified:
            assert float(out.incumbent) == pytest.approx(best, rel=1e-9)
        else:
            assert out.open
    finally:
        mp.undo()


# ---------------------------------------------------------------- edge


def test_no_failure_leaves_the_certificate_as_it_was(monkeypatch):
    calls = _fail_full(monkeypatch, which=set())
    y, X = panel(0, p=6)
    out = certify_slope_free(y, X, restarts=10)
    best, C = positive_optimum(y, X)
    assert out.certified and set(out.controls) == C


# ---------------------------------------------------------------- failure


def test_an_error_that_is_not_the_solvers_is_raised(monkeypatch):
    """Only the solver's failure reads as no bound; any other error is a defect and
    reaches the caller."""
    _no_cheap_tier(monkeypatch)
    _fail_full(monkeypatch, error=ValueError("not a solver failure"))
    c, G, lam, m = _size_inputs()
    with pytest.raises(ValueError, match="not a solver failure"):
        size_certificate(c, G, lam, m)
