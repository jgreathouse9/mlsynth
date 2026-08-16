"""Correlated Synthetic Controls (Moev 2025) -- readable oracle port.

Source: Tzvetan Moev, "Correlated Synthetic Controls" (arXiv:2507.08918),
Section 3.e (the correlated-random-coefficients weight model), Section 3.f
eq. (23) (scalar program) and Appendix A.b Proposition 4 eq. (24) (matrix
program). Reference implementation: ``run_rwscm`` in
``tzvetanmoev/Correlated-Synthetic-Controls``,
``Simulation/2_Code_CSC_IDiD_FDiD.R`` (CVXR).

Model, for treated unit i and pre-period t:

    y_it(0) = eta_i + sum_j w_ij y_jt(0) + e_it
    w_ij    = omega_j + x_i alpha_j                    (Random Coef.)
    sum_j w_ij = 1  for every i,   w_ij >= 0           (SC Constraints)

so the donor weights are a correlated-random-coefficients function of the
treated unit's time-invariant covariates: an individual-invariant part
``omega_j`` shared by every treated unit, plus an individual-specific part
``x_i alpha_j``. Two treated units with the same covariates get the same
synthetic control; similar covariates give correlated synthetic controls.

Matrix form solved here (eq. 24), with W = 1 omega' + X alpha':

    min_{omega, alpha, eta}  || Y1 - eta 1' - W Y0 ||_F^2
    s.t.  W 1 = 1,  W >= 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import cvxpy as cp
import numpy as np


@dataclass
class CSCFit:
    """Solved CSC weights and the objects read off them."""

    W: np.ndarray  # (n1, n0) per-treated-unit donor weights
    omega: np.ndarray  # (n0,) individual-invariant part
    alpha: np.ndarray  # (n0, K) covariate loadings on the weights
    eta: np.ndarray  # (n1,) treated-unit intercepts
    pre_fit: np.ndarray  # (n1, T0) fitted pre-treatment outcomes
    solver: str
    objective: float


def csc_weights(
    Y1_pre: np.ndarray,
    Y0_pre: np.ndarray,
    X1: np.ndarray,
    *,
    intercept: Literal["free", "pin_first", "none"] = "free",
    ridge: float = 0.0,
    solver: Optional[str] = None,
    verbose: bool = False,
) -> CSCFit:
    """Solve the CSC program (eq. 24).

    Parameters
    ----------
    Y1_pre : (n1, T0)
        Pre-treatment outcomes of the treated units.
    Y0_pre : (n0, T0)
        Pre-treatment outcomes of the donors.
    X1 : (n1, K)
        Time-invariant covariates of the treated units, dummy-coded. The
        reference passes every category of a categorical variable (no
        reference level dropped), so each row sums to one; see
        ``adding_up_feasible``.
    intercept : {"free", "pin_first", "none"}
        ``free`` estimates one eta_i per treated unit. ``pin_first``
        reproduces the reference implementation, which constrains
        ``eta_1 = 0`` and drops unit 1's intercept from the design.
        ``none`` drops the intercept entirely.
    ridge : float
        Optional ``ridge * (||omega||^2 + ||alpha||_F^2)`` tie-break. The
        paper's program carries far more parameters than pre-treatment
        residuals, so its minimiser is a face and not a point (measured in
        ``identified_set.py``); any positive ridge makes the objective
        strictly convex and selects one point of that face. Not in the paper.
    """
    Y1_pre = np.asarray(Y1_pre, dtype=float)
    Y0_pre = np.asarray(Y0_pre, dtype=float)
    X1 = np.asarray(X1, dtype=float)

    n1, T0 = Y1_pre.shape
    n0, T0b = Y0_pre.shape
    if T0 != T0b:
        raise ValueError(f"pre-period length disagrees: treated {T0}, donors {T0b}")
    if X1.shape[0] != n1:
        raise ValueError(f"X1 has {X1.shape[0]} rows, expected {n1}")
    K = X1.shape[1]

    omega = cp.Variable(n0, name="omega")
    alpha = cp.Variable((n0, K), name="alpha")
    eta = cp.Variable(n1, name="eta")

    # W = 1_{n1} omega' + X alpha'   ->  (n1, n0)
    W = np.ones((n1, 1)) @ cp.reshape(omega, (1, n0), order="C") + X1 @ alpha.T

    resid = Y1_pre - W @ Y0_pre
    if intercept != "none":
        resid = resid - cp.reshape(eta, (n1, 1), order="C") @ np.ones((1, T0))

    constraints = [W @ np.ones(n0) == np.ones(n1), W >= 0]
    if intercept == "pin_first":
        constraints.append(eta[0] == 0)
    elif intercept == "none":
        constraints.append(eta == 0)

    objective = cp.sum_squares(resid)
    if ridge > 0:
        objective = objective + ridge * (cp.sum_squares(omega) + cp.sum_squares(alpha))

    problem = cp.Problem(cp.Minimize(objective), constraints)
    chosen = solver or cp.CLARABEL
    problem.solve(solver=chosen, verbose=verbose)

    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"CSC solve returned status {problem.status!r}")

    W_hat = np.asarray(W.value, dtype=float)
    eta_hat = np.asarray(eta.value, dtype=float) if eta.value is not None else np.zeros(n1)
    return CSCFit(
        W=W_hat,
        omega=np.asarray(omega.value, dtype=float),
        alpha=np.asarray(alpha.value, dtype=float),
        eta=eta_hat,
        pre_fit=W_hat @ Y0_pre + eta_hat[:, None],
        solver=str(chosen),
        objective=float(cp.sum_squares(resid).value),  # fit term only, ridge excluded
    )


def csc_counterfactual(fit: CSCFit, Y0_post: np.ndarray) -> np.ndarray:
    """Post-treatment untreated potential outcomes for the treated units.

    ``yhat_it(0) = eta_i + sum_j w_ij y_jt``, matching the reference's
    ``t(t(X_post) %*% random_weights) + intercepts``.
    """
    Y0_post = np.asarray(Y0_post, dtype=float)
    return fit.W @ Y0_post + fit.eta[:, None]


def adding_up_feasible(X1: np.ndarray, tol: float = 1e-9) -> bool:
    """Whether the per-unit adding-up constraint can hold at all.

    ``sum_j w_ij = 1`` for every i means
    ``sum_k x_ik (sum_j alpha_jk) = 1 - sum_j omega_j`` (paper eq. 9), whose
    right-hand side does not depend on i. So the program is feasible only if
    some vector ``a`` and scalar ``c`` satisfy ``X1 a = c 1`` -- i.e. the
    all-ones vector lies in the column space of ``X1`` (the dummy-block case,
    where every row sums to one), or ``X1`` is degenerate. A continuous
    covariate breaks it and the solve returns ``infeasible``.
    """
    X1 = np.asarray(X1, dtype=float)
    ones = np.ones(X1.shape[0])
    coef, *_ = np.linalg.lstsq(X1, ones, rcond=None)
    return bool(np.allclose(X1 @ coef, ones, atol=tol))
