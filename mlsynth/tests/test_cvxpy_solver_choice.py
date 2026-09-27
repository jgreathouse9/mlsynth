"""Library code names the solver it wants; it never leaves the choice to cvxpy.

``cvxpy.Problem.solve()`` with no ``solver=`` dispatches to the highest-ranked
*installed* solver, and an installed solver is not a working one. MOSEK outranks
every open-source solver and raises

    mosek.Error: rescode.err_missing_license_file(1008): License cannot be
    located. The default search path is ':/root/mosek/mosek.lic:'.

when it has no licence file. ``scmrelax`` depends on ``mosek``, so installing it
alongside mlsynth is enough to break any unqualified solve -- for a user as much
as for this suite, which is how the defect was found: ``ssdid_w``,
``ssdid_lambda`` and ``ssdid_est`` were the only four failures of their kind.

The two sites are ports of Tian, Lee and Panchenko's ``functions_ssdid.py``
(``benchmarks/reference/.cache/spatial_SDID/``), which solves the same two
programs at its lines 71 and 127 with ``problem.solve(verbose=False)`` -- the
defect came in with the port. ``.github/workflows/benchmarks.yml`` already keeps
``scmrelax`` out of the spsydid job for that reason, and it has to keep doing so:
the reference is the authority the cross-validation compares against, so its
solver choice is not ours to edit. Fixing the port does not retire that job
split, and the split never protected library callers anyway.

Naming the solver cannot move a pinned number here. Both SSDID programs are
strictly convex with one equality constraint, so the optimum is a point and no
tie-break is at stake; and the jobs that produced the pins had no MOSEK
installed, so cvxpy was already choosing CLARABEL.

Two faults, and the tests below separate them. The first is the unqualified
call. The second is that ``mosek.Error`` is not a ``cvxpy.error.SolverError``,
so it escaped the ``except`` clause that exists to translate solver failures
into ``MlsynthEstimationError`` -- the raw vendor exception reached the caller,
naming neither the estimator nor the contract.
"""
from __future__ import annotations

import ast
import pathlib
from typing import Dict, List, Tuple

import numpy as np
import pytest

cvxpy = pytest.importorskip("cvxpy")

from mlsynth.exceptions import MlsynthEstimationError
from mlsynth.utils.helperutils import ssdid_lambda, ssdid_w

ROOT = pathlib.Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def ssdid_data() -> dict:
    """The shape ``test_helperutils`` uses: ``T = 10``, ``J = 3``, ``a = 5``."""
    T, J = 10, 3
    rng = np.random.default_rng(0)
    return {
        "treated_y": np.arange(T, dtype=float) * 2.0,
        "donor_matrix": (np.array([np.arange(T) + i * 0.5 for i in range(J)],
                                  dtype=float).T + rng.random((T, J)) * 0.1),
        "a": 5, "k_horizon": 2, "eta": 0.1, "J": J,
    }


# --------------------------------------------------------------------------- #
# 1. the invariant nobody had asserted
# --------------------------------------------------------------------------- #
# Read with an AST and not a grep. A ``grep`` for ``.solve()`` over this library
# returned thirteen sites and missed two of the four this sweep finds, which is
# the lesson ``tools/simplex_qp_audit.py`` opens with: syntax is not the program.
#
# Two rules the first version of this sweep got wrong, both of them false
# positives, and both recorded here because a later edit will reintroduce them.
#
# ``osqp.OSQP()`` and ``clarabel.DefaultSolver(...)`` expose ``.solve()`` too.
# They take no ``solver`` keyword and are configured at construction, so
# matching on the method name alone makes the check unsatisfiable. Only names
# bound to a ``cp.Problem(...)`` are in scope.
#
# A solver passed through ``**kwargs`` is passed. MAREX builds
# ``kw = {"solver": eff_solver, ...}`` and SYNDES
# ``solve_kwargs = {"solver": solver, ...}``, then unpack them; reading the call
# alone reports both as unqualified. The dict literal is resolved where it is
# bound in the same function, and a call whose unpacking cannot be resolved is
# reported as unreadable instead of as a fault -- a distinction the audit makes
# for the same reason.
def _enclosing(tree: ast.AST, lineno: int) -> ast.AST:
    """The innermost function containing ``lineno``, or the module."""
    best = tree
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        end = getattr(node, "end_lineno", None)
        if node.lineno <= lineno and end is not None and lineno <= end:
            if best is tree or node.lineno >= getattr(best, "lineno", -1):
                best = node
    return best


def _problem_names(scope: ast.AST) -> set:
    """Names bound to a ``cp.Problem(...)`` / ``cvxpy.Problem(...)`` call."""
    bound = set()
    for node in ast.walk(scope):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        func = node.value.func
        if not (isinstance(func, ast.Attribute) and func.attr == "Problem"):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                bound.add(target.id)
            elif isinstance(target, ast.Attribute):
                bound.add(target.attr)
    return bound


def _problem_attrs(scope: ast.AST) -> set:
    """Attribute names bound to a ``cp.Problem(...)`` anywhere in ``scope``."""
    out = set()
    for node in ast.walk(scope):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        func = node.value.func
        if isinstance(func, ast.Attribute) and func.attr == "Problem":
            for target in node.targets:
                if isinstance(target, ast.Attribute):
                    out.add(target.attr)
    return out


def _dicts_naming_a_solver(scope: ast.AST) -> set:
    """Names bound in ``scope`` to a dict literal carrying a ``"solver"`` key.

    ``AnnAssign`` counts. SYNDES writes ``solve_kwargs: dict = {"solver": ...}``
    and reading only ``Assign`` filed it as unreadable -- a site the check has no
    power over, which is the one outcome this sweep must not produce silently.
    """
    out = set()
    for node in ast.walk(scope):
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        else:
            continue
        if not isinstance(value, ast.Dict):
            continue
        keys = {k.value for k in value.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)}
        if "solver" not in keys:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                out.add(target.id)
    return out


def _receiver_is_a_cvxpy_problem(recv: ast.AST, local: set, attrs: set) -> bool:
    """A local name is resolved in its own function; an attribute is not.

    ``self.problem = cp.Problem(...)`` in ``__init__`` is solved from another
    method, so an attribute receiver has to be resolved module-wide. A bare name
    must not be, or one function's cvxpy problem answers for another function's
    ``osqp`` object bound to the same name.
    """
    if isinstance(recv, ast.Call):                    # cp.Problem(...).solve()
        return (isinstance(recv.func, ast.Attribute)
                and recv.func.attr == "Problem")
    if isinstance(recv, ast.Name):
        return recv.id in local
    if isinstance(recv, ast.Attribute):               # prob.problem.solve()
        return recv.attr in attrs
    return False


def _solves(path: pathlib.Path) -> List[Tuple[int, str, str]]:
    """``(lineno, receiver, status)`` per cvxpy solve, status not ``"named"``."""
    tree = ast.parse(path.read_text())
    attrs = _problem_attrs(tree)
    out = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "solve"):
            continue
        scope = _enclosing(tree, node.lineno)
        if not _receiver_is_a_cvxpy_problem(node.func.value,
                                            _problem_names(scope), attrs):
            continue
        if any(kw.arg == "solver" for kw in node.keywords):
            continue
        unpacked = [kw.value for kw in node.keywords if kw.arg is None]
        if unpacked:
            resolvable = _dicts_naming_a_solver(scope)
            if all(isinstance(v, ast.Name) and v.id in resolvable
                   for v in unpacked):
                continue
            status = "unreadable"
        else:
            status = "missing"
        out.append((node.lineno, ast.unparse(node.func.value), status))
    return out


def _library_sources() -> List[pathlib.Path]:
    return [p for p in sorted(ROOT.rglob("*.py")) if "tests" not in p.parts]


def _sweep(status: str) -> Dict[str, List[Tuple[int, str]]]:
    found: Dict[str, List[Tuple[int, str]]] = {}
    for path in _library_sources():
        hits = [(ln, recv) for ln, recv, st in _solves(path) if st == status]
        if hits:
            found[str(path.relative_to(ROOT))] = hits
    return found


def test_no_library_cvxpy_problem_leaves_the_solver_to_cvxpy():
    """The bottom of the ladder: nothing asserted that the choice was made.

    A solve without ``solver=`` is a bug whose symptom depends on what else is
    installed, which is why it survived -- the suite is green on any machine
    without MOSEK.
    """
    offenders = _sweep("missing")
    assert not offenders, (
        "these cvxpy problems let cvxpy rank the solvers, so the solver they "
        "get depends on what is installed:\n"
        + "\n".join(f"  {f}:{ln}  {recv}.solve()"
                    for f, hits in offenders.items() for ln, recv in hits))


def test_no_cvxpy_solve_hides_its_solver_behind_an_unresolvable_unpacking():
    """Pinned as a count, not waived. A ``**kwargs`` the sweep cannot resolve is
    a site where this check has no power, so a new one is a change to review and
    not a pass."""
    assert _sweep("unreadable") == {}


def test_the_sweep_can_see_an_unqualified_solve(tmp_path):
    """The check has power: a solve with no ``solver=`` is reported, and the
    same call with one is not. Without this the test above passes vacuously the
    moment the AST walk stops matching."""
    probe = tmp_path / "probe.py"
    probe.write_text("import cvxpy as cp\n"
                     "def f(x):\n"
                     "    prob = cp.Problem(cp.Minimize(x))\n"
                     "    prob.solve()\n")
    assert _solves(probe) == [(4, "prob", "missing")]
    probe.write_text("import cvxpy as cp\n"
                     "def f(x):\n"
                     "    prob = cp.Problem(cp.Minimize(x))\n"
                     "    prob.solve(solver=cp.CLARABEL)\n")
    assert _solves(probe) == []


def test_the_sweep_ignores_solvers_that_take_no_solver_keyword(tmp_path):
    """``osqp`` and ``clarabel`` expose ``.solve()`` and are configured at
    construction. Reporting them would make the check unsatisfiable."""
    probe = tmp_path / "probe.py"
    probe.write_text("import osqp\n"
                     "def f():\n"
                     "    m = osqp.OSQP()\n"
                     "    return m.solve()\n")
    assert _solves(probe) == []


def test_the_sweep_reads_a_solver_passed_through_a_dict(tmp_path):
    """MAREX's and SYNDES's shape: the keyword is in a dict literal bound in the
    same function. A dict the sweep cannot resolve is unreadable, not a fault."""
    probe = tmp_path / "probe.py"
    probe.write_text("import cvxpy as cp\n"
                     "def f(x):\n"
                     "    prob = cp.Problem(cp.Minimize(x))\n"
                     "    kw = {'solver': cp.SCIP, 'verbose': False}\n"
                     "    prob.solve(**kw)\n")
    assert _solves(probe) == []
    probe.write_text("import cvxpy as cp\n"
                     "def f(x, **outer):\n"
                     "    prob = cp.Problem(cp.Minimize(x))\n"
                     "    prob.solve(**outer)\n")
    assert _solves(probe) == [(4, "prob", "unreadable")]


def test_the_sweep_does_not_cross_function_boundaries(tmp_path):
    """One module, two functions, one cvxpy problem and one osqp object bound to
    the same name. Keeping the last assignment over the whole module reports the
    osqp solve; the audit records this as a mistake it made."""
    probe = tmp_path / "probe.py"
    probe.write_text("import cvxpy as cp\n"
                     "import osqp\n"
                     "def a(x):\n"
                     "    prob = cp.Problem(cp.Minimize(x))\n"
                     "    prob.solve(solver=cp.CLARABEL)\n"
                     "def b():\n"
                     "    prob = osqp.OSQP()\n"
                     "    return prob.solve()\n")
    assert _solves(probe) == []


# --------------------------------------------------------------------------- #
# 2. the two call sites, at the level of what they ask for
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("call", ["w", "lambda"])
def test_ssdid_asks_for_clarabel_by_name(call, ssdid_data, monkeypatch):
    """Asserted on the keyword and not on the outcome, so the test keeps its
    power on a machine where the default would have worked anyway."""
    seen: Dict[str, object] = {}
    real = cvxpy.Problem.solve

    def spy(self, *args, **kwargs):
        seen.update(kwargs)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(cvxpy.Problem, "solve", spy)
    d = ssdid_data
    if call == "w":
        ssdid_w(d["treated_y"], d["donor_matrix"], d["a"], d["k_horizon"], d["eta"])
    else:
        ssdid_lambda(d["treated_y"], d["donor_matrix"], d["a"], d["k_horizon"],
                     d["eta"])
    assert seen.get("solver") == cvxpy.CLARABEL


@pytest.mark.parametrize("call", ["w", "lambda"])
def test_ssdid_runs_with_mosek_installed(call, ssdid_data):
    """The symptom, end to end. This is the test the four SSDID failures were:
    it passes on a machine with no MOSEK whatever the code does, so it is the
    test above that carries the contract."""
    d = ssdid_data
    if call == "w":
        weights, intercept = ssdid_w(d["treated_y"], d["donor_matrix"], d["a"],
                                     d["k_horizon"], d["eta"])
        assert weights.shape == (d["J"],)
    else:
        weights, intercept = ssdid_lambda(d["treated_y"], d["donor_matrix"],
                                          d["a"], d["k_horizon"], d["eta"])
        assert weights.shape == (d["a"],)
    assert np.all(np.isfinite(weights)) and np.isfinite(intercept)


# --------------------------------------------------------------------------- #
# 3. the second fault -- a solver failure that cvxpy does not wrap
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("func,name", [(ssdid_w, "ssdid_w"),
                                       (ssdid_lambda, "ssdid_lambda")])
def test_a_vendor_solver_exception_is_translated(func, name, ssdid_data,
                                                 monkeypatch):
    """``mosek.Error`` is not a ``cvxpy.error.SolverError``, so the ``except``
    clause meant to translate solver failures did not catch it and the raw
    vendor exception reached the caller. Any exception out of ``solve`` is a
    failed solve and is reported as one."""
    class VendorError(Exception):
        pass

    def boom(self, *args, **kwargs):
        raise VendorError("License cannot be located.")

    monkeypatch.setattr(cvxpy.Problem, "solve", boom)
    d = ssdid_data
    with pytest.raises(MlsynthEstimationError,
                       match=f"CVXPY solver failed in {name}"):
        func(d["treated_y"], d["donor_matrix"], d["a"], d["k_horizon"], d["eta"])


# --------------------------------------------------------------------------- #
# 4. a named solver that is None is not a named solver
# --------------------------------------------------------------------------- #
# The sweep above reads the call. It cannot read the *value*, and
# ``problem.solve(solver=None)`` behaves exactly like ``problem.solve()``: cvxpy
# ranks the installed solvers and takes the highest. Three sites in the library
# pass a variable whose default is ``None``, and each guards it differently:
#
#   marex_helpers/optimization.py     rebinds ``solver = solver or cp.CLARABEL``
#   masc_helpers/estimation.py        returns early when ``solver is None``
#   spcd_helpers/weights_exact.py     did not guard it at all
#
# A static check that tries to prove which of those is safe gets it wrong: an
# earlier version of this sweep reported all three, and two were guarded. So the
# value is checked by running the call instead, which has no false positives and
# covers only the paths it reaches -- a limit the test states.
def _reject_none_solver(monkeypatch):
    """Make ``Problem.solve`` raise on a ``None`` solver."""
    real = cvxpy.Problem.solve

    def guarded(self, *args, **kwargs):
        if "solver" in kwargs and kwargs["solver"] is None:
            raise AssertionError(
                "solve() was given solver=None, which lets cvxpy rank the "
                "installed solvers and pick mosek"
            )
        return real(self, *args, **kwargs)

    monkeypatch.setattr(cvxpy.Problem, "solve", guarded)


def test_spcd_exact_weights_names_a_solver(monkeypatch):
    """The site that was not guarded. It is also the one the suite noticed:
    ``test_spcd.py``'s three exact-weight failures were this call reaching mosek.
    """
    pytest.importorskip("cvxpy")
    from mlsynth.utils.spcd_helpers.weights_exact import exact_weights

    _reject_none_solver(monkeypatch)
    rng = np.random.default_rng(0)
    Y_pre = rng.normal(size=(24, 8)) + 10.0
    # y_star is the sign vector splitting the units into two non-empty groups.
    y_star = np.array([1.0] * 3 + [-1.0] * 5)
    w = exact_weights(Y_pre, y_star, sigma=0.1)
    assert w.shape == (8,) and np.all(np.isfinite(w))


def test_an_explicit_solver_still_overrides_the_default():
    """The default is a default, not a lock."""
    pytest.importorskip("cvxpy")
    from mlsynth.utils.spcd_helpers.weights_exact import exact_weights

    rng = np.random.default_rng(1)
    Y_pre = rng.normal(size=(24, 8)) + 10.0
    y_star = np.array([1.0] * 3 + [-1.0] * 5)
    a = exact_weights(Y_pre, y_star, sigma=0.1)
    b = exact_weights(Y_pre, y_star, sigma=0.1, solver=cvxpy.SCS)
    np.testing.assert_allclose(np.asarray(a, float), np.asarray(b, float),
                               rtol=1e-4, atol=1e-5)
