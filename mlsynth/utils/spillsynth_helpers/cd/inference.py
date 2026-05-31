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


def signed_ci(
    *,
    alpha_hat: np.ndarray,
    U_pre: np.ndarray,
    G_hat: np.ndarray,
    C: np.ndarray,
    alpha_level: float = 0.05,
) -> np.ndarray:
    """Confidence interval for ``C alpha`` obtained by inverting the P-test.

    The R reference implementation (``scmSpillover::sp_andrews_te``) and
    Section 6.2 of the paper construct the CI by exploiting the fact
    that under :math:`H_0: C \\alpha = c_0`, the test statistic is
    asymptotically equivalent to :math:`(C G u_{T+1} - (c_0 - C
    \\alpha))^2`. Inverting the level-:math:`\\tau` test gives

    .. math::

       \\text{CI} = \\Big[\\, C \\widehat \\alpha + q_{\\tau/2}(\\{C
       \\widehat G \\widehat u_t\\}), \\quad C \\widehat \\alpha +
       q_{1-\\tau/2}(\\{C \\widehat G \\widehat u_t\\})\\, \\Big].

    Only well-defined for a *single-row* selector ``C``; this function
    raises if ``C.shape[0] != 1``.

    Parameters
    ----------
    alpha_hat : np.ndarray
        Shape ``(N, T1)``.
    U_pre : np.ndarray
        Shape ``(N, T0)`` pre-period residuals.
    G_hat : np.ndarray
        Shape ``(N, N)`` operator (see :func:`G_matrix`).
    C : np.ndarray
        Shape ``(1, N)`` linear selector.
    alpha_level : float
        Significance level (default 0.05 → 95% CI).

    Returns
    -------
    np.ndarray
        Shape ``(T1, 2)`` with columns ``[lower, upper]``.
    """
    if C.ndim != 2 or C.shape[0] != 1:
        raise ValueError(
            "signed_ci requires C with shape (1, N); for multi-row C "
            "the inversion is not a scalar interval."
        )
    point = (C @ alpha_hat).ravel()                       # (T1,)
    series = (C @ G_hat @ U_pre).ravel()                  # (T0,)
    q_lo = float(np.quantile(series, alpha_level / 2.0))
    q_hi = float(np.quantile(series, 1.0 - alpha_level / 2.0))
    out = np.empty((point.size, 2))
    out[:, 0] = point + q_lo
    out[:, 1] = point + q_hi
    return out


def run_per_period_tests(
    *,
    alpha_hat: np.ndarray,
    Y_pre: np.ndarray,
    a: np.ndarray,
    B: np.ndarray,
    A: np.ndarray,
    treated_labels,
    affected_labels,
) -> "tuple[Dict[Any, PTestResult], Dict[Any, PTestResult], Dict[Any, np.ndarray], Dict[Any, np.ndarray], Optional[PTestResult]]":
    """Run the SPILLSYNTH inferences of practical interest, plus CIs.

    Returns
    -------
    treatment_tests : Dict[treated_label, PTestResult]
        Per-treated-unit, per-post-period test of
        ``H_0: alpha_treated(t) = 0``.
    spillover_tests : Dict[affected_label, PTestResult]
        Per-affected-unit, per-post-period test of
        ``H_0: alpha_affected(t) = 0`` (one rejection per declared
        affected unit per period).
    treatment_cis_95 : Dict[treated_label, np.ndarray]
        Per-treated-unit 95% confidence interval on the treatment
        effect in each post-period (shape ``(T1, 2)``).
    spillover_ci_95 : Dict[affected_label, np.ndarray]
        Per-affected-unit 95% CI on the spillover effect in each
        post-period.
    joint_spillover_test : Optional[PTestResult]
        Cao-Dowd MATLAB-reference *joint* spillover hypothesis with a
        ``(p, N)`` selector that picks out every declared
        affected-unit row of alpha. ``None`` when ``p == 0``.
    """
    N = Y_pre.shape[0]
    n_treated = len(treated_labels)
    p = len(affected_labels)

    U_pre = compute_pre_residuals(Y_pre, a, B)
    G_hat = G_matrix(A, B)

    # One treatment-effect test + CI per treated unit.
    treatment_tests: Dict[Any, PTestResult] = {}
    treatment_cis_95: Dict[Any, np.ndarray] = {}
    for i, label in enumerate(treated_labels):
        e_i = np.zeros((1, N)); e_i[0, i] = 1.0
        treatment_tests[label] = p_test(
            alpha_hat=alpha_hat, U_pre=U_pre, G_hat=G_hat, C=e_i,
        )
        treatment_cis_95[label] = signed_ci(
            alpha_hat=alpha_hat, U_pre=U_pre, G_hat=G_hat, C=e_i,
        )

    # Per-affected-unit spillover tests + CIs. Affected units live at
    # rows ``n_treated .. n_treated + p - 1`` of alpha.
    spillover_tests: Dict[Any, PTestResult] = {}
    spillover_ci_95: Dict[Any, np.ndarray] = {}
    for k, label in enumerate(affected_labels):
        row = n_treated + k
        e_k = np.zeros((1, N)); e_k[0, row] = 1.0
        spillover_tests[label] = p_test(
            alpha_hat=alpha_hat, U_pre=U_pre, G_hat=G_hat, C=e_k,
        )
        spillover_ci_95[label] = signed_ci(
            alpha_hat=alpha_hat, U_pre=U_pre, G_hat=G_hat, C=e_k,
        )

    # Joint spillover hypothesis: C selects every affected-unit row of
    # alpha simultaneously.
    joint_spillover_test: Optional[PTestResult] = None
    if p > 0:
        C_joint = np.zeros((p, N))
        for k in range(p):
            C_joint[k, n_treated + k] = 1.0
        joint_spillover_test = p_test(
            alpha_hat=alpha_hat, U_pre=U_pre, G_hat=G_hat, C=C_joint,
        )

    return (treatment_tests, spillover_tests, treatment_cis_95,
            spillover_ci_95, joint_spillover_test)


# ---------------------------------------------------------------------------
# Cao-Dowd v3 Section 5.1.2: kappa_A specification test
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KappaATestResult:
    """Outcome of the Cao-Dowd v3 :math:`\\kappa_A` specification test.

    Tests :math:`H_0`: the chosen spillover-structure matrix :math:`A`
    correctly specifies the spillover effects.

    Parameters
    ----------
    kappa_A : np.ndarray
        Shape ``(T1,)``. Per-post-period
        :math:`\\kappa_A = \\|(I - \\widehat B)(Y_{T+s} - \\widehat
        \\alpha_{T+s}) - \\widehat a\\|`.
    kappa_pre : np.ndarray
        Shape ``(T0,)``. Pre-period reference values
        :math:`\\|(I - \\widehat \\Gamma_A) \\widehat u_t\\|`,
        where :math:`\\widehat \\Gamma_A = (I - \\widehat B) A
        (A^\\prime (I - \\widehat B)^\\prime (I - \\widehat B) A)^{-1}
        A^\\prime (I - \\widehat B)^\\prime` is the projection onto the
        span of columns of :math:`(I - \\widehat B) A`.
    p_value : np.ndarray
        Shape ``(T1,)``. Per-period p-value
        :math:`\\Pr(\\kappa_{\\text{pre}} \\geq \\kappa_A)` from the
        pre-period empirical CDF.
    cutoff_05 : float
        95th percentile of ``kappa_pre`` — reject correct specification
        at the 5% level when ``kappa_A > cutoff_05``.
    reject_05 : np.ndarray
        Shape ``(T1,)`` of bool.
    """

    kappa_A: np.ndarray
    kappa_pre: np.ndarray
    p_value: np.ndarray
    cutoff_05: float
    reject_05: np.ndarray


def kappa_A_test(
    *,
    Y_post: np.ndarray,
    alpha_hat: np.ndarray,
    Y_pre: np.ndarray,
    a: np.ndarray,
    B: np.ndarray,
    A: np.ndarray,
) -> KappaATestResult:
    """Per-post-period :math:`\\kappa_A` specification test (Section 5.1.2).

    ``kappa_A`` measures how well the chosen A-matrix explains the
    spillover-residualised post-period outcome. Small values indicate a
    correctly-specified A; large values indicate missing spillover
    structure.

    Reference distribution: project the pre-period residuals
    :math:`\\widehat u_t` orthogonally to the span of
    :math:`(I - \\widehat B) A` and take the Euclidean norm of the
    remainder. Under correct specification, the post-period
    :math:`\\kappa_A` is asymptotically distributed like a draw from
    this reference (Proposition 2 of Cao-Dowd v3).

    For multi-period selection, the user may take the column-mean of
    ``kappa_A`` to obtain a single statistic per A-matrix candidate
    (Section S.1.3 of the paper).
    """
    N = Y_pre.shape[0]
    I_B = np.eye(N) - B

    # Sample projection Γ̂_A onto colspan((I - B̂) A).
    IB_A = I_B @ A                                          # (N, k)
    AMA = IB_A.T @ IB_A                                     # (k, k)
    AMA_inv = np.linalg.inv(AMA)
    Gamma_A = IB_A @ AMA_inv @ IB_A.T                       # (N, N)
    I_minus_Gamma = np.eye(N) - Gamma_A

    # Post-period: kappa_A_s = ||(I - B̂)(Y_post - α̂_s) - â||.
    post_residual = I_B @ (Y_post - alpha_hat) - a[:, None]   # (N, T1)
    kappa_A = np.linalg.norm(post_residual, axis=0)           # (T1,)

    # Pre-period reference: ||(I - Γ̂_A) û_t||
    U_pre = I_B @ Y_pre - a[:, None]                          # (N, T0)
    kappa_pre = np.linalg.norm(I_minus_Gamma @ U_pre, axis=0) # (T0,)

    cutoff_05 = float(np.quantile(kappa_pre, 0.95))
    reject_05 = kappa_A > cutoff_05
    p_value = np.array([float(np.mean(kappa_pre >= k)) for k in kappa_A])

    return KappaATestResult(
        kappa_A=kappa_A, kappa_pre=kappa_pre,
        p_value=p_value, cutoff_05=cutoff_05, reject_05=reject_05,
    )


def select_A_by_kappa(
    *,
    Y_post: np.ndarray,
    Y_pre: np.ndarray,
    a: np.ndarray,
    B: np.ndarray,
    candidates,
) -> "tuple[int, np.ndarray]":
    """Heuristic Cao-Dowd v3 A-selection: ``argmin_A kappa_A``.

    For each candidate A-matrix, refits the SP estimator under that A,
    computes the multi-period mean of :math:`\\kappa_A` (Section S.1.3,
    averaged across post-periods), and returns the index of the
    smallest. Useful for choosing between, say, "per_unit" and
    "homogeneous" structures given the same affected-unit set.

    Parameters
    ----------
    candidates : Sequence[np.ndarray]
        List of ``A`` matrices, all of shape ``(N, k_i)`` (``k_i`` may
        differ across candidates).

    Returns
    -------
    best_index : int
        Index of the candidate with the smallest mean-kappa_A.
    kappa_means : np.ndarray
        Length ``len(candidates)`` vector of mean-kappa_A values.

    Notes
    -----
    Section S.1.3 of the paper notes that consistent selection requires
    multiple post-periods (so ``T1 >= 2``); with a single post-period
    this is a heuristic that may misselect under uninformative noise.
    """
    from .estimation import build_M, sp_estimate                  # local import to avoid cycle

    kappa_means = np.empty(len(candidates))
    M = build_M(B)
    for i, A in enumerate(candidates):
        _gamma, alpha, _cond = sp_estimate(Y_post, a=a, B=B, M=M, A=A)
        res = kappa_A_test(
            Y_post=Y_post, alpha_hat=alpha, Y_pre=Y_pre, a=a, B=B, A=A,
        )
        kappa_means[i] = float(res.kappa_A.mean())
    return int(np.argmin(kappa_means)), kappa_means
