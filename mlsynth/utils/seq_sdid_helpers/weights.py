"""Closed-form QP solvers for the Sequential SDiD weights.

Both weight problems in Algorithm 1 are equality-constrained convex
quadratic programs with the simplex sum constraint and *no* non-negativity
constraint. We solve them via the KKT linear system rather than handing
them to cvxpy — the bootstrap calls these tens of thousands of times, and
the linear system is small (~ N_cohort + 1 unknowns).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _solve_kkt(H: np.ndarray, g: np.ndarray, A: np.ndarray, b: np.ndarray,
               ridge: float = 1e-12) -> np.ndarray:
    """Solve ``min z' H z + 2 g' z  s.t.  A z = b`` via KKT.

    Parameters
    ----------
    H : np.ndarray
        Symmetric PSD Hessian, shape ``(n, n)``.
    g : np.ndarray
        Linear coefficient, shape ``(n,)``.
    A : np.ndarray
        Equality constraint matrix, shape ``(m, n)``.
    b : np.ndarray
        Equality constraint RHS, shape ``(m,)``.
    ridge : float
        Tiny ridge added to ``H`` if the KKT system is singular.

    Returns
    -------
    np.ndarray
        Optimal ``z``, shape ``(n,)``.
    """

    n = H.shape[0]
    m = A.shape[0]
    KKT = np.zeros((n + m, n + m))
    KKT[:n, :n] = H
    KKT[:n, n:] = A.T
    KKT[n:, :n] = A
    rhs = np.concatenate([-g, b])
    try:
        sol = np.linalg.solve(KKT, rhs)
    except np.linalg.LinAlgError:
        KKT[:n, :n] += ridge * np.eye(n)
        sol = np.linalg.solve(KKT, rhs)
    return sol[:n]


def solve_unit_qp(
    Y_pre_donors: np.ndarray,
    y_pre_treated: np.ndarray,
    pi_donors: np.ndarray,
    eta: float,
) -> Tuple[np.ndarray, float]:
    """Solve the Sequential SDiD unit-weight QP for one (a, k) step.

    Optimization (paper Algorithm 1, line 5)::

        min over (omega, omega_0)
            sum_{l < a + k} (omega_0 + sum_j omega_j Y_{j, l} - Y_{a, l})^2
            + eta^2 * sum_j omega_j^2 / pi_j
        s.t.  sum_j omega_j = 1.

    Parameters
    ----------
    Y_pre_donors : np.ndarray
        Pre-event outcomes of later-adopting cohorts, shape
        ``(T_pre, J)`` where ``T_pre = a + k - 1`` and ``J`` is the number
        of donor cohorts (``j > a``).
    y_pre_treated : np.ndarray
        Pre-event outcomes of the treated cohort, shape ``(T_pre,)``.
    pi_donors : np.ndarray
        Cohort shares ``pi_j`` for the donor cohorts, shape ``(J,)``.
    eta : float
        Non-negative regularization parameter.

    Returns
    -------
    omega : np.ndarray
        Optimal unit weights, shape ``(J,)``, summing to 1.
    omega_0 : float
        Optimal intercept.
    """

    T_pre, J = Y_pre_donors.shape
    X_aug = np.column_stack([Y_pre_donors, np.ones(T_pre)])  # (T_pre, J + 1)
    H = X_aug.T @ X_aug
    if eta > 0:
        # Penalty acts on omega only, not on omega_0.
        H[:J, :J] += (eta ** 2) * np.diag(1.0 / np.clip(pi_donors, 1e-12, None))
    g = -X_aug.T @ y_pre_treated  # (J + 1,)
    A = np.zeros((1, J + 1))
    A[0, :J] = 1.0
    b = np.array([1.0])
    z = _solve_kkt(H, g, A, b)
    return z[:J], float(z[J])


def solve_time_qp(
    Y_pre_donors: np.ndarray,
    y_event_donors: np.ndarray,
    eta: float,
) -> Tuple[np.ndarray, float]:
    """Solve the Sequential SDiD time-weight QP for one (a, k) step.

    Optimization (paper Algorithm 1, line 5)::

        min over (lambda, lambda_0)
            sum_{j > a} (lambda_0 + sum_l lambda_l Y_{j, l} - Y_{j, a + k})^2
            + eta^2 * sum_l lambda_l^2
        s.t.  sum_l lambda_l = 1.

    Parameters
    ----------
    Y_pre_donors : np.ndarray
        Donor outcomes in the pre-event window, shape ``(T_pre, J)``.
    y_event_donors : np.ndarray
        Donor outcomes at event time ``a + k``, shape ``(J,)``.
    eta : float
        Non-negative regularization parameter.

    Returns
    -------
    lambda_w : np.ndarray
        Optimal time weights, shape ``(T_pre,)``, summing to 1.
    lambda_0 : float
        Optimal intercept.
    """

    T_pre, J = Y_pre_donors.shape
    # Stack each donor as a row: the QP fits one observation per donor with
    # features = pre-event outcomes of that donor.
    X = Y_pre_donors.T  # (J, T_pre)
    X_aug = np.column_stack([X, np.ones(J)])  # (J, T_pre + 1)
    H = X_aug.T @ X_aug
    if eta > 0:
        H[:T_pre, :T_pre] += (eta ** 2) * np.eye(T_pre)
    g = -X_aug.T @ y_event_donors  # (T_pre + 1,)
    A = np.zeros((1, T_pre + 1))
    A[0, :T_pre] = 1.0
    b = np.array([1.0])
    z = _solve_kkt(H, g, A, b)
    return z[:T_pre], float(z[T_pre])
