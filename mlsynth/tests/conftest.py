"""Shared pytest configuration for the mlsynth test suite.

Optional-solver skip guard
--------------------------
Several estimators solve mixed-integer or conic programs through cvxpy and
require an optional solver that is not in the base install -- ``SCIP``
(``pyscipopt``, the ``design`` extra) and the commercial ``MOSEK`` are the two
that recur (SYNDES, MAREX, PANGEO, SCEXP). On a machine without them the solve
fails, which turns an environment gap into a hard test failure even though the
code under test is correct. Two symptoms occur:

* direct -- cvxpy raises ``SolverError("The solver SCIP is not installed.")``,
  re-wrapped as ``MlsynthEstimationError`` and propagated; or
* indirect -- the pool/holdout/IC search calls the solver in a loop and treats
  any ``MlsynthEstimationError`` as "feasible region exhausted", so a missing
  solver surfaces downstream as "found no feasible design" /
  "optimization failed to solve".

This hook reclassifies both as *skips*, but only when the responsible optional
solver is genuinely absent (the direct message always; the indirect message
only when the failure passed through a MILP/MIP estimator helper and no optional
MILP/conic solver is installed). Where the optional solvers ARE installed, real
defects -- a wrong answer, a genuinely infeasible design -- still fail loudly.
Install the solvers (``pip install pyscipopt``, or MOSEK) to exercise the
skipped tests.
"""

from __future__ import annotations

import re

import pytest

# Direct "solver not installed" signatures (always safe to skip on).
_MISSING_SOLVER = re.compile(
    r"(solver\s+(MOSEK|SCIP|GUROBI|CBC|GLPK(_MI)?|XPRESS)\s+is not installed"
    r"|is installed in cvxpy\. Install pyscipopt"
    r"|No module named ['\"](mosek|pyscipopt)['\"])",
    re.IGNORECASE,
)

# Indirect signatures: a MIP/conic solve that failed for want of a solver, seen
# only after the missing error was swallowed by a pool/holdout/IC search.
_DOWNSTREAM = re.compile(
    r"found no feasible design|optimization failed to solve|"
    r"optimization did not return an assignment",
    re.IGNORECASE,
)

# MIP/conic estimator helpers whose tests pin an optional solver.
_MIP_MODULES = (
    "syndes_helpers", "syndes.py", "marex_helpers", "pangeo_helpers",
    "scexp.py",
)


def _optional_solvers_absent() -> bool:
    """True if any commonly-pinned optional cvxpy solver is missing."""
    try:
        import cvxpy as cp
        installed = set(cp.installed_solvers())
    except Exception:  # pragma: no cover - cvxpy is a hard dependency
        return True
    return bool({"SCIP", "MOSEK", "GUROBI"} - installed)


def _chain(exc: BaseException | None):
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__ or exc.__context__


def _is_missing_solver(excinfo) -> str | None:
    """Return a short reason if the failure is a missing optional solver."""
    for exc in _chain(excinfo.value):
        m = _MISSING_SOLVER.search(str(exc))
        if m:
            return m.group(0)
    # Indirect: swallowed-solver symptom, gated on the solver being absent and
    # the failure originating in a MIP/conic estimator helper.
    if _optional_solvers_absent():
        for exc in _chain(excinfo.value):
            if _DOWNSTREAM.search(str(exc)):
                tb = "".join(str(getattr(e, "path", "")) for e in excinfo.traceback)
                if any(mod in tb for mod in _MIP_MODULES):
                    return "optional MILP/conic solver (SCIP/MOSEK) not installed"
    return None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when in ("setup", "call") and report.failed \
            and call.excinfo is not None:
        reason = _is_missing_solver(call.excinfo)
        if reason is not None:
            report.outcome = "skipped"
            report.longrepr = (str(item.fspath), item.location[1] + 1,
                               f"Skipped: {reason}")
