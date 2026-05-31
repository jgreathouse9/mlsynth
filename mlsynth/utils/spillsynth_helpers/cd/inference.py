"""Cao-Dowd 2023 Section 4 inference.

Implements the Andrews-type end-of-sample P-test adapted to the
spillover-aware SCM estimator. The pre-treatment squared (or
quadratic-form) residuals furnish the reference distribution against
which the post-period treatment-and-spillover effect estimates are
compared.

Two granularities are exposed:

* :func:`p_test` -- the general procedure for the hypothesis
  ``H_0: C alpha = d`` at a single post-period. Returns the test
  statistic, the empirical reference CDF, and a p-value.
* :func:`run_per_period_tests` -- runs the special cases of interest
  for SPILLSYNTH:

  - per-period **treatment-effect test** ``H_0: alpha_1(t) = 0`` for
    each post-period :math:`t`, with selector
    :math:`C = e_1^\\prime`;
  - per-period **spillover test** for each declared affected unit
    :math:`k`, ``H_0: alpha_k(t) = 0``, with selector
    :math:`C = e_k^\\prime`.

The default weighting matrix :math:`W_T = I` satisfies Lemma 3 of the
paper under both Condition ST and Condition CO.

See Section 4 of Cao & Dowd (2023) and Andrews (2003).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


def compute_pre_residuals(Y_pre: np.ndarray, a: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pre-period residuals :math:`\\widehat u_t = (I - B) Y_t - a`.

    Parameters
    ----------
    Y_pre : np.ndarray
        Shape ``(N, T0)`` pre-treatment outcome panel.
    a : np.ndarray
        Length-``N`` leave-one-out intercepts.
    B : np.ndarray
        Shape ``(N, N)`` leave-one-out SCM weight matrix.

    Returns
    -------
    np.ndarray
        Shape ``(N, T0)`` matrix where column ``t`` is :math:`\\widehat u_t`.
    """
    N = Y_pre.shape[0]
    return (np.eye(N) - B) @ Y_pre - a[:, None]


def G_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Sandwich matrix :math:`\\widehat G = A (A' \\widehat M A)^{-1} A' (I - \\widehat B)'`.

    This is the operator that maps the post-period residual
    :math:`\\widehat u_{T+1}` to the leading-order term of
    :math:`\\widehat \\alpha - \\alpha` in Theorem 1 of Cao-Dowd 2023.
    """
    N = B.shape[0]
    I_B = np.eye(N) - B
    M = I_B.T @ I_B
    AMA_inv = np.linalg.inv(A.T @ M @ A)
    return A @ AMA_inv @ A.T @ I_B.T


@dataclass(frozen=True)
class PTestResult:
    """Per-post-period Cao-Dowd Andrews-style P-test outcome.

    Parameters
    ----------
    P_post : np.ndarray
        Shape ``(T1,)``. Per-post-period test statistic
        :math:`P_t = (C \\widehat \\alpha_t - d)' W_T (C \\widehat \\alpha_t - d)`.
    P_pre : np.ndarray
        Shape ``(T0,)``. Pre-period reference values
        :math:`\\widehat P_t = \\widehat u_t' \\widehat G' C' W_T C \\widehat G \\widehat u_t`.
    p_value : np.ndarray
        Shape ``(T1,)``. One-sided p-value
        :math:`\\Pr(P_{\\text{pre}} \\geq P_t)` computed from the
        empirical CDF of ``P_pre`` (no continuity correction; ties
        counted as :math:`\\geq`).
    cutoff_05 : float
        The 95th percentile of ``P_pre``; reject ``H_0`` at the 5%
        level when ``P_post > cutoff_05``.
    reject_05 : np.ndarray
        Shape ``(T1,)`` of bool. Convenience flag for the 5% test.
    """

    P_post: np.ndarray
    P_pre: np.ndarray
    p_value: np.ndarray
    cutoff_05: float
    reject_05: np.ndarray


def p_test(
    *,
    alpha_hat: np.ndarray,
    U_pre: np.ndarray,
    G_hat: np.ndarray,
    C: np.ndarray,
    d: Optional[np.ndarray] = None,
    W_T: Optional[np.ndarray] = None,
) -> PTestResult:
    """Cao-Dowd Section 4.2 P-test for ``H_0: C alpha = d`` at each post-period.

    Parameters
    ----------
    alpha_hat : np.ndarray
        Shape ``(N, T1)`` SP effect estimates (rows ordered as in
        :class:`SpillSynthInputs`: row 0 treated, rows 1..p affected,
        rows p+1..N-1 zero).
    U_pre : np.ndarray
        Shape ``(N, T0)`` pre-period residuals (see
        :func:`compute_pre_residuals`).
    G_hat : np.ndarray
        Shape ``(N, N)`` operator (see :func:`G_matrix`).
    C : np.ndarray
        Shape ``(q, N)`` linear-hypothesis selector.
    d : np.ndarray, optional
        Length-``q`` null value. Defaults to zero.
    W_T : np.ndarray, optional
        Shape ``(q, q)`` weighting matrix. Defaults to the identity,
        which satisfies Lemma 3 of the paper under Condition ST or
        Condition CO.

    Returns
    -------
    PTestResult
    """
    q, N = C.shape
    T1 = alpha_hat.shape[1]
    if d is None:
        d = np.zeros(q)
    if W_T is None:
        W_T = np.eye(q)

    diff_post = C @ alpha_hat - d[:, None]                # (q, T1)
    # P_post[t] = diff_post[:, t]' W_T diff_post[:, t]
    P_post = np.einsum("qt,qr,rt->t", diff_post, W_T, diff_post)

    Q = G_hat.T @ C.T @ W_T @ C @ G_hat                   # (N, N)
    # P_pre[t] = U_pre[:, t]' Q U_pre[:, t]
    P_pre = np.einsum("nt,nm,mt->t", U_pre, Q, U_pre)

    cutoff_05 = float(np.quantile(P_pre, 0.95))
    reject_05 = P_post > cutoff_05

    p_value = np.empty(T1)
    T0 = P_pre.shape[0]
    for t in range(T1):
        p_value[t] = float(np.mean(P_pre >= P_post[t]))

    return PTestResult(
        P_post=P_post, P_pre=P_pre,
        p_value=p_value, cutoff_05=cutoff_05, reject_05=reject_05,
    )


def run_per_period_tests(
    *,
    alpha_hat: np.ndarray,
    Y_pre: np.ndarray,
    a: np.ndarray,
    B: np.ndarray,
    A: np.ndarray,
    affected_labels,
) -> "tuple[PTestResult, Dict[Any, PTestResult]]":
    """Run the two SPILLSYNTH inferences of practical interest.

    Returns
    -------
    treatment_test : PTestResult
        Per-post-period test of ``H_0: alpha_1(t) = 0``.
    spillover_tests : Dict[label, PTestResult]
        Per-affected-unit, per-post-period test of
        ``H_0: alpha_k(t) = 0``.
    """
    N = Y_pre.shape[0]
    U_pre = compute_pre_residuals(Y_pre, a, B)
    G_hat = G_matrix(A, B)

    e_treat = np.zeros((1, N)); e_treat[0, 0] = 1.0
    treatment_test = p_test(
        alpha_hat=alpha_hat, U_pre=U_pre, G_hat=G_hat, C=e_treat,
    )

    spillover_tests: Dict[Any, PTestResult] = {}
    for k, label in enumerate(affected_labels):
        row = 1 + k                                     # affected units live at rows 1..p
        e_k = np.zeros((1, N)); e_k[0, row] = 1.0
        spillover_tests[label] = p_test(
            alpha_hat=alpha_hat, U_pre=U_pre, G_hat=G_hat, C=e_k,
        )

    return treatment_test, spillover_tests
