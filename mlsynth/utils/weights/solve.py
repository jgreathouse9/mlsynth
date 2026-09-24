"""One entry point for the library's weight programs.

Fourteen call sites build a quadratic program over a polyhedron in weight
space, each canonicalising its own constraints and each reporting -- or not
reporting -- its own diagnostics. They differ in which polyhedron they name and
agree on everything else, so the polyhedron becomes data and the dispatch
becomes a table.

Everything reduces to four exact primitives. A free intercept is profiled out
by centring, since for fixed ``w`` the optimal shift is the residual mean; a
ridge penalty is absorbed by stacking ``sqrt(ridge) * I`` under the design.
Both leave the constraint set untouched, so the table only has to answer the
question of which polyhedron, and each answer is a method that terminates
finitely at the exact minimiser.

The refusals are deliberate. Cressie-Read members below ``gamma = 1`` are
exponential-cone and belong to a conic solver; a capped simplex is polyhedral
but has no exact primitive here yet. Both raise and name themselves instead of
being approximated, on the same reasoning that keeps ``extra="forbid"`` on the
configs.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import lsq_linear, nnls

from mlsynth.exceptions import MlsynthConfigError, MlsynthEstimationError
from mlsynth.utils.bilevel.active_set import solve_simplex_qp

from .solution import WeightSolution
from .spec import WeightConstraint, WeightObjective

#: Residual below which the KKT certificate is accepted as optimal.
KKT_TOL: float = 1e-7
#: Weight below which a donor is off the support.
SUPPORT_TOL: float = 1e-9

ConstraintShape = Tuple[bool, bool, bool]
Backend = Callable[[np.ndarray, np.ndarray, WeightConstraint, Optional[np.ndarray]], Tuple[np.ndarray, str]]


# ---------------------------------------------------------------------------
# The four primitives
# ---------------------------------------------------------------------------
def _simplex(B, A, con, warm_start):
    """Exact primal active set over ``{w >= 0, 1'w = 1}`` (Dostal 2009, ch. 6)."""
    return solve_simplex_qp(B, A, warm_start=warm_start), "active-set"


def _cone(B, A, con, warm_start):
    """Lawson-Hanson non-negative least squares over ``{w >= 0}``."""
    w, _ = nnls(B, A)
    return w, "nnls"


def _affine(B, A, con, warm_start):
    """Equality-constrained least squares, solved on the null space of ``1'``.

    Writing ``w = 1_J/J + Z v`` with ``Z`` an orthonormal basis of
    ``{d : 1'd = 0}`` turns the constrained problem into an unconstrained one
    in ``v``, which a rank-revealing least squares answers exactly even when the
    donor Gram is singular.
    """
    n = B.shape[1]
    if n == 1:
        return np.ones(1), "closed-form"
    Z = np.linalg.svd(np.ones((1, n)))[2][1:].T
    w0 = np.full(n, 1.0 / n)
    v, *_ = np.linalg.lstsq(B @ Z, A - B @ w0, rcond=None)
    return w0 + Z @ v, "null-space"


def _free(B, A, con, warm_start):
    """Ordinary least squares, minimum-norm on a rank-deficient design."""
    w, *_ = np.linalg.lstsq(B, A, rcond=None)
    return w, "lstsq"


def _box(B, A, con, warm_start):
    """Bounded-variable least squares, the two-sided Lawson-Hanson."""
    lower = 0.0 if con.nonneg else -np.inf
    method = "bvls" if B.shape[0] >= B.shape[1] else "trf"
    res = lsq_linear(B, A, bounds=(lower, con.upper), method=method, tol=1e-12)
    return np.asarray(res.x, dtype=float), method


#: Dispatch by polyhedron. The key is ``(nonneg, sum_to_one, capped)``; a shape
#: absent from the table is refused by name in :func:`solve_weights`.
_BACKENDS: Dict[ConstraintShape, Backend] = {
    (True, True, False): _simplex,
    (True, False, False): _cone,
    (False, True, False): _affine,
    (False, False, False): _free,
    (True, False, True): _box,
    (False, False, True): _box,
}


# ---------------------------------------------------------------------------
# The certificate
# ---------------------------------------------------------------------------
def kkt_residual(
    B: np.ndarray,
    A: np.ndarray,
    weights: np.ndarray,
    intercept: float,
    constraint: WeightConstraint,
    objective: WeightObjective,
    *,
    tol: float = 1e-8,
) -> float:
    """Scale-free KKT violation at ``(weights, intercept)``.

    Every polyhedron here has non-empty relative interior, so Slater's condition
    holds and the KKT conditions are both necessary and sufficient for a global
    minimum of a convex quadratic (Boyd and Vandenberghe 2004, section 5.5.3).
    Recomputing the residual from the returned point makes the claim of
    optimality independent of the backend that produced it.

    The returned number is the largest of the primal violations and the
    complementarity-adjusted stationarity violations, divided by a curvature
    scale, so it is invariant to a common rescaling of ``B`` and ``A``.

    That scale has to be invariant under everything the program is invariant
    under, or it measures the data instead of the violation. Donor series in a
    panel sit at a common level -- Basque GDP per capita near 5, cigarette sales
    near 130 -- and under ``sum(w) == 1`` adding a constant to every donor and to
    the target changes nothing about the solution. A scale read off ``B'A``
    moves with that level, so on a level-dominated panel it divides a genuine
    stationarity violation down into the noise and certifies a point that is not
    optimal. The scale used is the largest diagonal entry of the Hessian after
    projecting out the directions the constraints annihilate, which moves the way
    the violation does.
    """
    B = np.asarray(B, dtype=float)
    A = np.asarray(A, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    n = w.size

    resid = A - B @ w - intercept
    grad = -2.0 * (B.T @ resid)
    if objective.ridge > 0.0:
        grad = grad + 2.0 * objective.ridge * (w - objective.target(n))
    core = B
    if constraint.sum_to_one:
        core = core - core.mean(axis=1, keepdims=True)
    if constraint.intercept:
        core = core - core.mean(axis=0, keepdims=True)
    scale = 2.0 * float(np.einsum("ij,ij->j", core, core).max(initial=0.0))
    scale = max(scale + 2.0 * objective.ridge, 1e-300)

    at_lower = np.zeros(n, dtype=bool) if not constraint.nonneg else w <= tol
    at_upper = (
        np.zeros(n, dtype=bool)
        if constraint.upper is None
        else w >= constraint.upper - tol
    )
    interior = ~(at_lower | at_upper)

    # The equality multiplier is free; take the value the interior coordinates
    # agree on, which is exactly where stationarity must hold with equality.
    if constraint.sum_to_one:
        pool = grad[interior] if interior.any() else grad
        nu = -float(np.mean(pool))
    else:
        nu = 0.0
    reduced = grad + nu

    viol = [
        float(np.abs(reduced[interior]).max(initial=0.0)),
        float(np.maximum(0.0, -reduced[at_lower]).max(initial=0.0)),
        float(np.maximum(0.0, reduced[at_upper]).max(initial=0.0)),
    ]
    if constraint.intercept:
        viol.append(float(abs(2.0 * resid.sum())))
    stationarity = max(viol) / scale

    primal = [float(np.maximum(0.0, -w).max(initial=0.0)) if constraint.nonneg else 0.0]
    if constraint.sum_to_one:
        primal.append(abs(float(w.sum()) - 1.0))
    if constraint.upper is not None:
        primal.append(float(np.maximum(0.0, w - constraint.upper).max(initial=0.0)))

    return max(stationarity, max(primal))


def _is_unique(B, w, constraint, objective) -> bool:
    """Whether the minimiser is the only one.

    On the face the solution sits on, another optimum exists exactly when some
    direction both preserves feasibility to first order and leaves the fit
    unchanged. Stacking the design's free columns over the active equality row
    makes that set the matrix's null space, so uniqueness is a rank test. A
    positive ridge makes the objective strictly convex and settles it outright.

    ``rank(B) < J`` is necessary for a continuum but not sufficient: the
    constraint can cut out precisely the flat directions. Proposition 99 has 38
    donors over 19 pre-periods, a rank-19 design, and a unique simplex optimum.
    """
    if objective.ridge > 0.0:
        return True
    free = np.ones(w.size, dtype=bool)
    if constraint.nonneg:
        free &= w > SUPPORT_TOL
    if constraint.upper is not None:
        free &= w < constraint.upper - SUPPORT_TOL
    k = int(free.sum())
    if k == 0:
        return True

    blocks = [B[:, free]]
    if constraint.intercept:
        blocks[0] = np.column_stack([blocks[0], np.ones(B.shape[0])])
    M = blocks[0]
    if constraint.sum_to_one:
        row = np.zeros(M.shape[1])
        row[:k] = 1.0
        M = np.vstack([M, row])
    scale = float(np.abs(M).max(initial=0.0))
    if scale == 0.0:
        return False
    return int(np.linalg.matrix_rank(M / scale)) == M.shape[1]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def solve_weights(
    B: np.ndarray,
    A: np.ndarray,
    constraint: Optional[WeightConstraint] = None,
    objective: Optional[WeightObjective] = None,
    *,
    warm_start: Optional[np.ndarray] = None,
) -> WeightSolution:
    """Minimise ``||A - B w - a||^2 + ridge * ||w - toward||^2`` over a polyhedron.

    Parameters
    ----------
    B : np.ndarray, shape (m, J)
        Donor design, usually pre-period donor outcomes.
    A : np.ndarray, shape (m,)
        Target, usually the treated unit's pre-period outcomes.
    constraint : WeightConstraint, optional
        The polyhedron. Defaults to the simplex.
    objective : WeightObjective, optional
        The discrepancy. Defaults to plain least squares.
    warm_start : np.ndarray, optional
        A feasible starting point, passed through to backends that can use one.
        It is an argument and not solver state, so the same call with the same
        inputs gives the same answer whatever ran before it.

    Returns
    -------
    WeightSolution
        The minimiser with its optimality certificate and uniqueness verdict.
    """
    constraint = constraint if constraint is not None else WeightConstraint()
    objective = objective if objective is not None else WeightObjective()

    B = np.asarray(B, dtype=float)
    A = np.asarray(A, dtype=float).ravel()
    if B.ndim != 2:
        raise MlsynthEstimationError(f"Donor design must be 2D; got shape {B.shape}.")
    if B.shape[0] != A.size:
        raise MlsynthEstimationError(
            f"Row count mismatch: design has {B.shape[0]} rows, target has {A.size}."
        )
    if B.shape[1] == 0:
        raise MlsynthEstimationError("Donor design has no columns; no weights to solve for.")
    if not np.all(np.isfinite(B)) or not np.all(np.isfinite(A)):
        raise MlsynthEstimationError("Donor design and target must be finite.")

    backend = _BACKENDS.get(constraint.shape)
    if backend is None:
        raise MlsynthConfigError(
            f"The {constraint.describe()} polyhedron "
            f"(nonneg={constraint.nonneg}, sum_to_one={constraint.sum_to_one}, "
            f"upper={constraint.upper}) is not covered by an exact backend. "
            f"Drop the upper cap, or build the program with cvxpy."
        )

    n = B.shape[1]
    # A free intercept is profiled out: for any w the optimal shift is the
    # residual mean, so centring both sides removes it from the program without
    # touching the constraint set.
    if constraint.intercept:
        Bs, As = B - B.mean(axis=0, keepdims=True), A - A.mean()
    else:
        Bs, As = B, A

    # The ridge penalty is absorbed into the least-squares data.
    if objective.ridge > 0.0:
        root = float(np.sqrt(objective.ridge))
        Bs = np.vstack([Bs, root * np.eye(n)])
        As = np.concatenate([As, root * objective.target(n)])

    w, method = backend(Bs, As, constraint, warm_start)
    w = np.asarray(w, dtype=float).ravel()
    if constraint.nonneg:
        w[w <= 0.0] = 0.0
    a = float(np.mean(A - B @ w)) if constraint.intercept else 0.0

    resid = A - B @ w - a
    value = float(resid @ resid)
    if objective.ridge > 0.0:
        gap = w - objective.target(n)
        value += objective.ridge * float(gap @ gap)

    residual = kkt_residual(B, A, w, a, constraint, objective)
    return WeightSolution(
        weights=w,
        intercept=a,
        objective=value,
        kkt_residual=residual,
        unique=_is_unique(B, w, constraint, objective),
        solver=f"{constraint.describe()}:{method}",
        status="optimal" if residual < KKT_TOL else "inaccurate",
        n_donors=n,
        support=np.flatnonzero(np.abs(w) > SUPPORT_TOL),
    )
