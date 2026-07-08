"""Asymptotic inference for the CFM systematic causal effect.

The sampling variance of ``tau*_t`` has two asymptotically-uncorrelated
pieces (Bai & Wang appendix A.2):

* ``V_reg`` -- uncertainty from the treated-unit pre/post regressions,
  estimated by a *block-additive* heteroskedasticity-robust (HC1) sandwich:
  ``Var(tau*_t) = c_t' Var(theta_1) c_t + c_t' Var(theta_0) c_t`` with
  ``c_t = [1, f_t]`` and ``Var(theta_d)`` the HC sandwich of block ``d``
  (appendix A.14-A.19). The blocks are independent, so their variances add.
* ``V_f`` -- uncertainty from estimating the common factors off the control
  units (appendix A.20): ``Var(f_hat_t) = (1/n1) Q^{-1} S_t Q^{-1}`` with
  ``Q = (1/n1) sum_k lam_k lam_k'`` and
  ``S_t = (1/n1) sum_k lam_k lam_k' e_kt^2``. It enters ``tau*_t`` through
  ``(lambda1 - lambda0)' Var(f_hat_t) (lambda1 - lambda0)``.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.stats import norm

from .pipeline import _design, fit_systematic_effect


def block_regression_variance(
    treated_outcome: np.ndarray, factors: np.ndarray, T0: int, hc: str = "HC1"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Heteroskedasticity-robust sandwich variance for each treated block.

    Returns
    -------
    V0, V1 : np.ndarray
        ``(r+1, r+1)`` HC sandwich covariance of ``theta_0`` (pre) and
        ``theta_1`` (post).
    theta0, theta1 : np.ndarray
        OLS coefficient vectors ``[a_d, lambda_d]``.
    """
    y = np.asarray(treated_outcome, dtype=float).ravel()
    Z = _design(factors)

    def _block(Zb: np.ndarray, yb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n, k = Zb.shape
        beta, *_ = np.linalg.lstsq(Zb, yb, rcond=None)
        e = yb - Zb @ beta
        XtXi = np.linalg.pinv(Zb.T @ Zb)
        scale = {"HC0": 1.0, "HC1": n / max(n - k, 1)}[hc]
        meat = Zb.T @ (Zb * (e ** 2)[:, None])
        return XtXi @ meat @ XtXi * scale, beta

    V0, theta0 = _block(Z[:T0], y[:T0])
    V1, theta1 = _block(Z[T0:], y[T0:])
    return V0, V1, theta0, theta1


def factor_estimation_variance(
    control_panel: np.ndarray, factors: np.ndarray
) -> np.ndarray:
    """Per-period variance of the estimated factors (appendix A.20).

    Parameters
    ----------
    control_panel : np.ndarray
        ``(T, N_co)`` control outcomes.
    factors : np.ndarray
        ``(T, r)`` factors in the Bai normalization (``F'F / T = I``).

    Returns
    -------
    np.ndarray
        ``(T, r, r)`` array of ``Var(f_hat_t)``.
    """
    Xc = control_panel - control_panel.mean(axis=0, keepdims=True)
    T, n1 = Xc.shape
    F = np.asarray(factors, dtype=float)
    r = F.shape[1]
    # Control loadings: lam_k = (F'F)^-1 F' X_k = (1/T) F' X_k (Bai norm).
    Lam = (F.T @ Xc) / T                        # (r, N_co)
    resid = Xc - F @ Lam                        # (T, N_co)
    Q = (Lam @ Lam.T) / n1                       # (r, r)
    Qi = np.linalg.pinv(Q)
    Vf = np.empty((T, r, r))
    for t in range(T):
        w = Lam * (resid[t] ** 2)[None, :]       # (r, N_co)
        S_t = (w @ Lam.T) / n1                    # (r, r)
        Vf[t] = (Qi @ S_t @ Qi) / n1
    return Vf


def cfm_inference(
    treated_outcome: np.ndarray,
    factors: np.ndarray,
    control_panel: np.ndarray,
    T0: int,
    alpha: float = 0.05,
    factor_variance: bool = True,
    hc: str = "HC1",
) -> Dict[str, object]:
    """Full asymptotic inference for the systematic causal effect.

    Returns a dict with per-period (``tau``, ``se_t``, ``ci_lower_t``,
    ``ci_upper_t``), ATT-level (``att``, ``att_se``, ``att_lower``,
    ``att_upper``, ``att_p_value``), and intercept-shift (``kappa``,
    ``kappa_se``, ``kappa_t``) fields. The ``kappa`` t-statistic uses the
    treated-regression (``V_reg``) component only -- the paper's
    post-treatment intercept-shift test.
    """
    y = np.asarray(treated_outcome, dtype=float).ravel()
    fit = fit_systematic_effect(y, factors, T0)
    V0, V1, _, _ = block_regression_variance(y, factors, T0, hc=hc)

    Z = _design(factors)
    c_post = Z[T0:]                              # (n_post, r+1)
    n_post = c_post.shape[0]
    V_reg = V0 + V1
    v_reg_t = np.einsum("ti,ij,tj->t", c_post, V_reg, c_post)

    dlam = fit.lambda1 - fit.lambda0             # (r,)
    if factor_variance:
        Vf = factor_estimation_variance(control_panel, factors)   # (T,r,r)
        v_f_t = np.einsum("i,tij,j->t", dlam, Vf[T0:], dlam)
        v_f_t = np.clip(v_f_t, 0.0, None)
    else:
        Vf = None
        v_f_t = np.zeros(n_post)

    se_t = np.sqrt(np.clip(v_reg_t + v_f_t, 0.0, None))
    z = float(norm.ppf(1.0 - alpha / 2.0))
    tau = fit.tau
    ci_lower_t = tau - z * se_t
    ci_upper_t = tau + z * se_t

    # ATT = c_bar'(theta1 - theta0); Var adds the two block sandwiches.
    c_bar = c_post.mean(axis=0)
    v_reg_att = float(c_bar @ V_reg @ c_bar)
    if factor_variance and n_post > 0:
        # Var of the post-period mean factor, treating periods as
        # approximately independent: (1/n_post^2) sum_t Var(f_hat_t).
        Vf_bar = Vf[T0:].mean(axis=0) / n_post
        v_f_att = float(dlam @ Vf_bar @ dlam)
    else:
        v_f_att = 0.0
    att_se = float(np.sqrt(max(v_reg_att + v_f_att, 0.0)))
    att = fit.att
    if att_se > 0:
        att_lower = att - z * att_se
        att_upper = att + z * att_se
        att_p = float(2.0 * (1.0 - norm.cdf(abs(att) / att_se)))
    else:  # pragma: no cover - att_se is > 0 whenever residual variance is nonzero
        att_lower = att_upper = att
        att_p = float("nan")

    # Intercept-shift (kappa) test: V_reg only, from the [0,0] blocks.
    kappa_var = float(V0[0, 0] + V1[0, 0])
    kappa_se = float(np.sqrt(kappa_var)) if kappa_var > 0 else float("nan")
    kappa_t = float(fit.kappa / kappa_se) if kappa_se and np.isfinite(kappa_se) else float("nan")

    return {
        "tau": tau,
        "se_t": se_t,
        "ci_lower_t": ci_lower_t,
        "ci_upper_t": ci_upper_t,
        "att": att,
        "att_se": att_se,
        "att_lower": att_lower,
        "att_upper": att_upper,
        "att_p_value": att_p,
        "kappa": fit.kappa,
        "kappa_se": kappa_se,
        "kappa_t": kappa_t,
    }
