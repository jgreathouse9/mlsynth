"""Robustness of SCMO's simplex weight solver to a failed cvxpy solve.

Regression coverage for the case where the primary solver (OSQP) terminates
without a primal solution, leaving ``w.value is None``. Previously
``simplex_weights`` fed that ``None`` straight into ``np.clip`` and crashed
with an opaque ``TypeError`` (``'>=' not supported between instances of
'NoneType' and 'float'`` on the current NumPy). It must instead fall back to a
robust solver, and raise a translated ``MlsynthEstimationError`` only if every
solver fails.
"""
from __future__ import annotations

import cvxpy as cp
import numpy as np
import pytest

from mlsynth.exceptions import MlsynthEstimationError
from mlsynth.utils.scmo_helpers import solvers


def _problem():
    """A small, well-posed matching problem (treated is a convex mix of donors)."""
    Z_donors = np.array(
        [[1.0, 0.0, 0.0, 1.0],
         [0.0, 1.0, 0.0, 1.0],
         [0.0, 0.0, 1.0, 1.0]]
    )
    # treated = 0.5 * donor0 + 0.5 * donor1 -> a genuine simplex target
    Z_treated = 0.5 * Z_donors[0] + 0.5 * Z_donors[1]
    return Z_treated, Z_donors


def test_simplex_weights_smoke_returns_valid_simplex():
    Z_treated, Z_donors = _problem()
    w = solvers.simplex_weights(Z_treated, Z_donors)
    assert w.shape == (Z_donors.shape[0],)
    assert np.all(w >= -1e-9)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)


def test_simplex_weights_falls_back_when_primary_solver_returns_none(monkeypatch):
    """OSQP returning no primal value must trigger the CLARABEL fallback rather
    than crash in ``np.clip``."""
    Z_treated, Z_donors = _problem()
    real_solve = cp.Problem.solve
    calls = {"solvers": []}

    def fake_solve(self, *args, **kwargs):
        solver = kwargs.get("solver")
        calls["solvers"].append(solver)
        if solver == cp.OSQP:
            # Simulate OSQP terminating without setting a primal solution:
            # skip the real solve so ``w.value`` stays None.
            return None
        return real_solve(self, *args, **kwargs)

    monkeypatch.setattr(cp.Problem, "solve", fake_solve)
    w = solvers.simplex_weights(Z_treated, Z_donors)

    # OSQP was attempted first, then the CLARABEL fallback.
    assert cp.OSQP in calls["solvers"]
    assert cp.CLARABEL in calls["solvers"]
    assert np.all(w >= -1e-9)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)


def test_simplex_weights_raises_translated_error_when_all_solvers_fail(monkeypatch):
    """When no solver returns a solution, the failure is reported as a
    translated ``MlsynthEstimationError`` -- not a raw ``TypeError`` from
    ``np.clip(None, ...)``."""
    Z_treated, Z_donors = _problem()
    monkeypatch.setattr(cp.Problem, "solve", lambda self, *a, **k: None)
    with pytest.raises(MlsynthEstimationError):
        solvers.simplex_weights(Z_treated, Z_donors)
