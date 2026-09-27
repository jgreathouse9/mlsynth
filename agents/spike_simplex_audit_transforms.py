"""Check each transform the simplex audit names, against the cvxpy solve it replaces.

``tools/simplex_qp_audit.py`` reports ten sites as the simplex least-squares
program written in another algebra, each with the reshaping a caller applies
before handing the data to ``solve_simplex_qp``. This runs every one of them:
it builds the site's cvxpy program on random panels, applies the named
transform, solves through the active set, and reports the largest weight
disagreement together with the objective each answer reached. Where a program
has a flat direction the two minimisers differ in ``w`` and agree in what ``w``
achieves, so the objective column is what settles a wide weight gap.

    python agents/spike_simplex_audit_transforms.py

The numbers are in ``agents/agents_simplex_audit.md``. Reshapings that cannot
be checked this way do not belong in the audit's transform strings.
"""
from __future__ import annotations

import zlib
from typing import Callable, Dict, List, Tuple

import cvxpy as cp
import numpy as np

from mlsynth.utils.solvers.active_set import solve_simplex_qp
from mlsynth.utils.hsc_helpers.formulation import smoother_and_metric

# A case returns the active-set weights, the cvxpy weights, and the site's
# objective in numpy, with any intercept profiled out. The weights are the
# headline, and the objective is what settles a disagreement: two minimisers
# of a program with a flat direction differ in w and not in what w achieves.
Case = Callable[[np.random.Generator],
                Tuple[np.ndarray, np.ndarray, Callable[[np.ndarray], float]]]
CASES: Dict[str, Tuple[str, Case]] = {}


def case(site: str, transform: str) -> Callable[[Case], Case]:
    def register(fn: Case) -> Case:
        CASES[site] = (transform, fn)
        return fn
    return register


def _panel(rng: np.random.Generator, rows=(8, 25), cols=(3, 12)):
    T = int(rng.integers(*rows))
    J = int(rng.integers(*cols))
    X = rng.normal(size=(T, J))
    return X, X @ rng.dirichlet(np.ones(J)) + 0.05 * rng.normal(size=T)


def _solve(objective, constraints, var):
    cp.Problem(objective, constraints).solve(solver=cp.CLARABEL)
    return np.asarray(var.value, dtype=float).ravel()


@case("clustersc_helpers/pcr/convex.py:60",
      "square the objective, which has the same minimiser")
def _pcr(rng):
    X, y = _panel(rng)
    w = cp.Variable(X.shape[1], nonneg=True)
    ref = _solve(cp.Minimize(cp.norm(y - X @ w, 2)), [cp.sum(w) == 1], w)
    return solve_simplex_qp(X, y), ref, lambda v: float(np.linalg.norm(y - X @ v))


@case("cscm_helpers/engine.py:124",
      "scale the rows by the square root of the metric")
def _cscm(rng):
    X, y = _panel(rng)
    V = rng.uniform(0.2, 3.0, size=X.shape[0])
    W = cp.Variable(X.shape[1])
    ref = _solve(cp.Minimize(cp.quad_form(y - X @ W, cp.psd_wrap(np.diag(V)))),
                 [W >= 0, cp.sum(W) == 1], W)
    s = np.sqrt(V)
    return (solve_simplex_qp(s[:, None] * X, s * y), ref,
            lambda v: float(((y - X @ v) * V) @ (y - X @ v)))


@case("dscar_helpers/weights.py:72",
      "scale the rows by the square root of the metric; "
      "the sum-to-one penalty is zero on the feasible set")
def _dscar(rng):
    X, y = _panel(rng)
    V = rng.uniform(0.2, 3.0, size=X.shape[0])
    w = cp.Variable(X.shape[1], nonneg=True)
    loss = (cp.sum_squares(cp.multiply(np.sqrt(V), y - X @ w))
            + cp.square(cp.sum(w) - 1.0))
    ref = _solve(cp.Minimize(loss), [cp.sum(w) == 1], w)
    s = np.sqrt(V)
    return (solve_simplex_qp(s[:, None] * X, s * y), ref,
            lambda v: float(((y - X @ v) * V) @ (y - X @ v) + (v.sum() - 1.0) ** 2))


@case("fast_scm_helpers/fast_scm_bb_helpers.py:192",
      "factor the Gram as R'R and take B = R")
def _fast_scm(rng):
    X, y = _panel(rng)
    R = np.column_stack([X, -y])                  # the residual system's factor
    Q = R.T @ R + 1e-9 * np.eye(R.shape[1])
    w = cp.Variable(Q.shape[0], nonneg=True)
    ref = _solve(cp.Minimize(cp.quad_form(w, cp.psd_wrap(Q))), [cp.sum(w) == 1], w)
    factor = np.linalg.cholesky(Q).T
    return (solve_simplex_qp(factor, np.zeros(factor.shape[0])), ref,
            lambda v: float(v @ Q @ v))


@case("hsc_helpers/formulation.py:157",
      "factor the Gram as R'R and take B = R, "
      "recovering the target from the linear term")
def _hsc(rng):
    X, y = _panel(rng, rows=(12, 30))
    _, W = smoother_and_metric(X.shape[0], int(rng.integers(1, 3)),
                               float(rng.uniform(0.1, 0.9)))
    H0 = X.T @ W @ X
    H = 0.5 * (H0 + H0.T) + float(rng.uniform(1e-4, 1e-1)) * np.eye(X.shape[1])
    f = X.T @ W @ y
    om = cp.Variable(X.shape[1])
    ref = _solve(cp.Minimize(cp.quad_form(om, cp.psd_wrap(H)) - 2.0 * f @ om),
                 [om >= 0, cp.sum(om) == 1], om)
    R = np.linalg.cholesky(H).T
    return (solve_simplex_qp(R, np.linalg.solve(R.T, f)), ref,
            lambda v: float(v @ H @ v - 2.0 * f @ v))


@case("mlsc_helpers/crossval.py:131",
      "augment the design with a multiple of the identity")
def _mlsc_floor(rng):
    X, y = _panel(rng)
    c = 1e-8
    w = cp.Variable(X.shape[1])
    ref = _solve(cp.Minimize(cp.sum_squares(y - X @ w) + c * cp.sum_squares(w)),
                 [cp.sum(w) == 1, w >= 0], w)
    J = X.shape[1]
    return (solve_simplex_qp(np.vstack([X, np.sqrt(c) * np.eye(J)]),
                             np.concatenate([y, np.zeros(J)])), ref,
            lambda v: float((y - X @ v) @ (y - X @ v) + c * v @ v))


def _generalised_ridge(rng):
    """``||y - X w||^2 + p w'Q w``: mlSC's penalty, in crossval and optimization."""
    X, y = _panel(rng, rows=(10, 25), cols=(4, 12))
    M = X.shape[1]
    L = rng.normal(size=(M, M))
    Q = L.T @ L + 1e-6 * np.eye(M)
    p = float(rng.uniform(0.05, 3.0))
    om = cp.Variable(M)
    ref = _solve(cp.Minimize(cp.sum_squares(y - X @ om)
                             + p * cp.quad_form(om, cp.psd_wrap(Q))),
                 [cp.sum(om) == 1, om >= 0], om)
    R = np.linalg.cholesky(Q).T                   # R'R == Q
    return (solve_simplex_qp(np.vstack([X, np.sqrt(p) * R]),
                             np.concatenate([y, np.zeros(M)])), ref,
            lambda v: float((y - X @ v) @ (y - X @ v) + p * v @ Q @ v))


case("mlsc_helpers/crossval.py:154",
     "augment the design with the penalty's square-root factor")(_generalised_ridge)
case("mlsc_helpers/optimization.py:129",
     "augment the design with the penalty's square-root factor")(_generalised_ridge)


@case("spsydid_helpers/weights.py:54",
      "centre the design and the target to profile out the intercept")
def _spsydid_time(rng):
    X, y = _panel(rng)
    y = y + 1.7
    a, w = cp.Variable(), cp.Variable(X.shape[1], nonneg=True)
    ref = _solve(cp.Minimize(cp.sum_squares(a + X @ w - y)), [cp.sum(w) == 1], w)
    r = lambda v: (y - y.mean()) - (X - X.mean(0)) @ v      # noqa: E731
    return (solve_simplex_qp(X - X.mean(0), y - y.mean()), ref,
            lambda v: float(r(v) @ r(v)))


@case("spsydid_helpers/weights.py:136",
      "centre the design and the target to profile out the intercept; "
      "augment the design with a multiple of the identity")
def _spsydid_unit(rng):
    X, y = _panel(rng)
    y = y + 1.7
    c = float(rng.uniform(0.05, 2.0))             # T0 * zeta**2
    a, w = cp.Variable(), cp.Variable(X.shape[1], nonneg=True)
    ref = _solve(cp.Minimize(cp.sum_squares(a + X @ w - y) + c * cp.sum_squares(w)),
                 [cp.sum(w) == 1], w)
    J = X.shape[1]
    r = lambda v: (y - y.mean()) - (X - X.mean(0)) @ v      # noqa: E731
    return (solve_simplex_qp(
        np.vstack([X - X.mean(0), np.sqrt(c) * np.eye(J)]),
        np.concatenate([y - y.mean(), np.zeros(J)])), ref,
        lambda v: float(r(v) @ r(v) + c * v @ v))


def main(trials: int = 8) -> List[Tuple[str, str, float, float]]:
    rows = []
    for site, (transform, build) in CASES.items():
        # crc32, not hash(): string hashing is salted per process.
        rng = np.random.default_rng(zlib.crc32(site.encode()))
        dw, gap = 0.0, -np.inf
        for _ in range(trials):
            got, ref, objective = build(rng)
            dw = max(dw, float(np.max(np.abs(got - ref))))
            here, there = objective(got), objective(ref)
            gap = max(gap, (here - there) / max(abs(there), 1e-12))
        rows.append((site, transform, dw, gap))
        print(f"{dw:9.2e}  {gap:+10.2e}  {site}\n                      {transform}")
    print(f"\n{len(rows)} transforms over {trials} panels each. "
          f"Worst weight gap {max(r[2] for r in rows):.2e}; worst objective "
          f"excess {max(r[3] for r in rows):+.2e} (negative means the active "
          f"set reached a lower objective than cvxpy did).")
    return rows


if __name__ == "__main__":
    main()
