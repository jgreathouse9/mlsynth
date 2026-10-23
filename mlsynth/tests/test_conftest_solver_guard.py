"""The optional-solver guard skips only what a missing solver explains (#813).

``conftest.py`` turns a failure into a skip when the solver it needs is absent:
directly, when the message says so, and indirectly, when a MIP helper's search
swallowed the missing solver and reported "found no feasible design". The
indirect case must stay a failure wherever SCIP, the solver those helpers use,
is installed, whatever else is missing; otherwise a real regression in SYNDES,
MAREX, PANGEO or SCEXP reads as a skip.
"""
import importlib.util
from itertools import combinations
from pathlib import Path

import cvxpy
import numpy as np
import pytest

from mlsynth.exceptions import MlsynthEstimationError
from mlsynth.utils.syndes_helpers.exact import solve_synthetic_design_exact

_spec = importlib.util.spec_from_file_location(
    "_mlsynth_tests_conftest", Path(__file__).with_name("conftest.py"))
guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(guard)


def _no_feasible_design():
    """A real "found no feasible design" error, raised inside syndes_helpers."""
    rng = np.random.default_rng(0)
    Y = rng.standard_normal((10, 4))
    every = [np.array(s) for s in combinations(range(4), 2)]
    with pytest.raises(MlsynthEstimationError, match="found no feasible design") as info:
        solve_synthetic_design_exact(Y, 2, forbidden_sets=every)
    return info


def _installed(monkeypatch, names):
    monkeypatch.setattr(cvxpy, "installed_solvers", lambda: list(names))


def test_a_swallowed_failure_with_scip_installed_stays_a_failure(monkeypatch):
    info = _no_feasible_design()
    _installed(monkeypatch, ["SCIP", "CLARABEL", "HIGHS"])
    assert guard._is_missing_solver(info) is None


def test_a_swallowed_failure_without_scip_is_a_skip(monkeypatch):
    info = _no_feasible_design()
    _installed(monkeypatch, ["CLARABEL", "HIGHS"])
    assert guard._is_missing_solver(info) is not None


def test_a_direct_missing_solver_message_is_always_a_skip(monkeypatch):
    _installed(monkeypatch, ["SCIP", "CLARABEL"])
    with pytest.raises(MlsynthEstimationError) as info:
        try:
            raise cvxpy.error.SolverError("The solver MOSEK is not installed.")
        except cvxpy.error.SolverError as exc:
            raise MlsynthEstimationError("solve failed") from exc
    assert guard._is_missing_solver(info) is not None


def test_an_unrelated_failure_is_never_a_skip(monkeypatch):
    _installed(monkeypatch, ["CLARABEL"])
    with pytest.raises(ValueError) as info:
        raise ValueError("weights do not sum to one")
    assert guard._is_missing_solver(info) is None
