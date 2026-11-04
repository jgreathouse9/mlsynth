"""Two solver settings measured on the GeoLift groups (#965): the node solver's
L-BFGS-B memory and the Shor stage's Clarabel tolerance. Each is pinned at its
measured value, and each is checked to reach the solver it sets."""
import numpy as np
import pytest

from . import dual, search
from .brute import panel, positive_optimum
from .lifts import Lift
from .oracle import Oracle


def test_the_lbfgs_memory_is_the_measured_one():
    """10 correction pairs: on the four groups the same nodes, prunes and branches as
    20, at 70-87% of the time; 5 leaves the hardest group uncertified
    (results/solver_settings.log). A change here needs that measurement redone."""
    assert dual.LBFGS_MEMORY == 10


def test_the_shor_tolerance_is_the_measured_one():
    """1e-5: the same Shor survivors as the default 1e-8 on the four groups, at
    about 89% of the time; 1e-4 keeps 18 pairs on the hardest group against 15."""
    assert search.SHOR_TOL == 1e-5


def test_the_memory_reaches_l_bfgs_b(monkeypatch):
    seen = []
    real = dual.minimize

    def spy(*a, **k):
        seen.append(k["options"]["maxcor"])
        return real(*a, **k)

    monkeypatch.setattr(dual, "minimize", spy)
    y, X = panel(4, p=6, n=30)
    dual.smoothed_lbfgs(Oracle(y, X, 0.4, 0.9, 3), np.zeros(Oracle(y, X, 0.4, 0.9, 3).dim), maxfun=30)
    assert seen and set(seen) == {dual.LBFGS_MEMORY}


def test_the_tolerance_reaches_clarabel(monkeypatch):
    """Lift.piece passes a tolerance to all three of Clarabel's stopping tests, and
    the default leaves Clarabel's own settings alone."""
    calls = []
    y, X = panel(1, p=5, n=30)
    L = Lift(y, X, "shor")
    real = L.prob.solve

    def spy(**kw):
        calls.append(kw)
        return real(**kw)

    monkeypatch.setattr(L.prob, "solve", spy)
    L.piece(0.3, 0.9, 2, tol=1e-5)
    L.piece(0.3, 0.9, 2)
    assert calls[0]["tol_gap_abs"] == calls[0]["tol_gap_rel"] == calls[0]["tol_feas"] == 1e-5
    assert not any(k.startswith("tol") for k in calls[1])


def test_the_shor_stage_solves_at_the_measured_tolerance(monkeypatch):
    tols = []
    real = Lift.piece

    def spy(self, *a, **k):
        tols.append(k.get("tol"))
        return real(self, *a, **k)

    monkeypatch.setattr(Lift, "piece", spy)
    y, X = panel(0)
    search.certify(y, X, restarts=10)
    assert tols and set(tols) == {search.SHOR_TOL}


@pytest.mark.parametrize("seed", range(3))
def test_a_looser_shor_solve_is_still_the_lift_to_its_accuracy(seed):
    """The tolerance moves the reported value by at most the solver's accuracy; the
    stage prunes only on the exact certificate, so this changes no decision."""
    y, X = panel(seed)
    tight = Lift(y, X, "shor").piece(0.3, 0.9, 3)
    loose = Lift(y, X, "shor").piece(0.3, 0.9, 3, tol=1e-5)
    assert loose == pytest.approx(tight, rel=1e-4)


@pytest.mark.parametrize("seed", range(3))
def test_certify_still_reaches_the_enumerated_optimum(seed):
    y, X = panel(seed)
    out = search.certify(y, X, restarts=10)
    best, _ = positive_optimum(y, X)
    assert out.certified and float(out.incumbent) == pytest.approx(best, rel=1e-9)
