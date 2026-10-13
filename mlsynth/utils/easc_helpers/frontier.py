"""Synthetic-control weights and the exposure-fit frontier (Spruk 2026, Sections 3.2 and 4.2).

The synthetic-control weights come from mlsynth's active-set simplex solver.
The frontier adds the constraint ``|EI(w)| <= d`` to the same programme, which
the active-set solver does not take, so it is solved with cvxpy and CLARABEL,
compiled once with ``d`` as a parameter. With more donors than pre-periods the
frontier's weights are not unique; an interior-point solver returns the same
member of the solution set as the paper's reference code, and an active-set
vertex does not.
"""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from ...exceptions import MlsynthEstimationError
from ..solvers.active_set import solve_simplex_qp
from .diagnostics import exposure_imbalance, min_imbalance


def sc_weights(a: np.ndarray, A: np.ndarray) -> np.ndarray:
    """``argmin ||a - A w||^2`` over the simplex: the synthetic-control weights."""
    return np.asarray(solve_simplex_qp(np.asarray(A, dtype=float), np.asarray(a, dtype=float)), dtype=float)


class BalancedWeights:
    """``min ||a - A w||^2`` over the simplex subject to ``|EI(w)| <= d`` (equation 11)."""

    def __init__(self, a: np.ndarray, A: np.ndarray, s1: float, s0: np.ndarray):
        self.w = cp.Variable(A.shape[1])
        self.d = cp.Parameter(nonneg=True)
        constraints = [self.w >= 0, cp.sum(self.w) == 1, cp.abs(s1 - s0 @ self.w) <= self.d]
        self.problem = cp.Problem(cp.Minimize(cp.sum_squares(a - A @ self.w)), constraints)

    def _solve(self, d: float) -> None:
        self.d.value = float(d)
        self.problem.solve(solver=cp.CLARABEL)

    def solve(self, d: float) -> np.ndarray:
        """The weights at tolerance ``d``; raises when the solver returns none."""
        try:
            self._solve(d)
        except cp.error.SolverError as exc:
            raise MlsynthEstimationError(
                f"EASC: the exposure-fit frontier could not be solved at tolerance {d:.6g}: {exc}") from exc
        if self.w.value is None:  # pragma: no cover - every tolerance is at least d_min, which is feasible
            raise MlsynthEstimationError(
                f"EASC: the exposure-fit frontier returned no solution at tolerance {d:.6g}.")
        v = np.clip(self.w.value, 0.0, None)
        return v / v.sum()


@dataclass(frozen=True)
class Frontier:
    """The exposure-fit frontier on a grid of tolerances (Proposition 3(b)).

    Row ``i`` of ``weights`` is the best-fitting simplex weights whose exposure
    imbalance is at most ``tolerance[i]``. The grid runs from ``d_min`` (EBSC-min,
    row 0) to the synthetic-control imbalance (row -1, the synthetic-control
    weights).
    """

    tolerance: np.ndarray
    exposure_imbalance: np.ndarray
    rmspe: np.ndarray
    weights: np.ndarray


def exposure_fit_frontier(a: np.ndarray, A: np.ndarray, s1: float, s0: np.ndarray, n_points: int,
                          w_sc: np.ndarray) -> Frontier:
    """Trace ``R(d)`` on ``n_points`` tolerances from ``d_min`` to ``|EI(w_SC)|``.

    The last point is the synthetic-control weights themselves, as in the
    reference's Monte Carlo (``frontier_path``). Each tolerance carries a slack of
    ``1e-9`` so that ``d_min`` itself is feasible to the solver.
    """
    a = np.asarray(a, dtype=float)
    A = np.asarray(A, dtype=float)
    s0 = np.asarray(s0, dtype=float)
    dmin = min_imbalance(s1, s0)
    d_sc = abs(exposure_imbalance(s1, s0, w_sc))
    qp = BalancedWeights(a, A, s1, s0)
    grid = np.linspace(0.0, 1.0, n_points)
    tol, weights = [], []
    for g in grid:
        d = dmin + g * max(d_sc - dmin, 0.0) + 1e-9
        tol.append(d)
        weights.append(np.asarray(w_sc, dtype=float) if g == 1.0 else qp.solve(d))
    W = np.vstack(weights)
    resid = a[:, None] - A @ W.T
    return Frontier(
        tolerance=np.asarray(tol),
        exposure_imbalance=np.array([exposure_imbalance(s1, s0, w) for w in W]),
        rmspe=np.sqrt(np.mean(resid ** 2, axis=0)),
        weights=W,
    )


def selection_criterion(rmspe: np.ndarray, ei: np.ndarray, kappa_latent: np.ndarray, se_phi: np.ndarray,
                        post: np.ndarray) -> np.ndarray:
    """The heuristic post-period MSE of equation (12), at every frontier point.

    ``mean_t[(kappa^lambda_t)^2 (R(d)^2 - R(dbar)^2) + se(phi_t)^2 EI(w(d))^2]``,
    where ``dbar`` is the synthetic-control end (the last point). The extra
    misfit that balancing induces is treated as systematic and priced by the
    latent leverage; the second term is the variance of the correction.
    """
    rmspe = np.asarray(rmspe, dtype=float)
    extra = np.maximum(0.0, rmspe ** 2 - rmspe[-1] ** 2)
    k2 = np.asarray(kappa_latent, dtype=float)[post] ** 2
    se2 = np.asarray(se_phi, dtype=float)[post] ** 2
    return np.array([np.mean(k2 * e + se2 * x ** 2) for e, x in zip(extra, np.asarray(ei, dtype=float))])


def first_minimiser(criterion: np.ndarray) -> int:
    """Index of the first minimum, so ties keep the more balanced point, as the reference does."""
    return int(np.argmin(np.asarray(criterion, dtype=float)))
