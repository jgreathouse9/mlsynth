"""Multiple-treated-units L2-relaxation PDA (Shi & Wang 2024, Sec. 3 / their
``PDA.multitreat``).

When *several* units are treated by the same intervention (e.g. all UK firms at
the Brexit referendum), the L2-relaxation counterfactual is fit **per treated
unit** against the shared control pool, and the effects are aggregated
**cross-sectionally** into a per-period average treatment effect with a
covariance-based standard error:

    ATE_t = mean_j ( y_{j,t} - yhat_{j,t} ),     t in the post-period,
    se    = sqrt( 1' Sigma_e 1 ) / J,            Sigma_e = E1' E1 / T1,

where ``E1`` are the pre-period prediction residuals stacked across the ``J``
treated units (``Sigma_e`` their cross-sectional covariance). ``se`` is constant
across post-periods (it depends only on the pre-period residual covariance), and
the test statistic is ``ATE_t / se -> N(0,1)``.

All ``J`` per-unit fits share the same ``Sigma = X'X/T1``, so the whole thing
runs through one OSQP factorisation (:func:`l2_relax_batch`). ``tau`` is tuned
per unit by **time-respecting** sequential validation (fit on the earlier
window, validate on the recent tail) -- never a future-leaking K-fold.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .l2.batch import l2_relax_batch


@dataclass(frozen=True)
class MultiTreatResult:
    """Per-period cross-sectional ATE for the multiple-treated-units PDA.

    Attributes
    ----------
    ate : np.ndarray
        Per-post-period average treatment effect, shape ``(T2,)``.
    se : float
        Cross-sectional standard error (constant across post-periods).
    tstat, pvalue : np.ndarray
        Per-period ``ate / se`` and two-sided normal p-values, shape ``(T2,)``.
    ate_mean : float
        Average of ``ate`` over the post-period (the scalar ATE).
    tau : np.ndarray
        Per-unit selected penalty, shape ``(J,)``.
    """

    ate: np.ndarray
    se: float
    tstat: np.ndarray
    pvalue: np.ndarray
    ate_mean: float
    tau: np.ndarray


def _moments(X: np.ndarray, Y: np.ndarray, standardize: bool):
    """Shared ``Sigma`` and per-unit ``Eta`` on the (standardised) pre-period."""
    T = X.shape[0]
    Mu = X.mean(0)
    Sd = X.std(0, ddof=1) if standardize else np.ones(X.shape[1])
    Sd = np.where(Sd > 0, Sd, 1.0)
    Xt = (X - Mu) / Sd
    Sigma = Xt.T @ Xt / T
    muY = Y.mean(0)
    sdY = Y.std(0, ddof=1) if standardize else np.ones(Y.shape[1])
    sdY = np.where(sdY > 0, sdY, 1.0)
    Eta = Xt.T @ ((Y - muY) / sdY) / T
    return Sigma, Eta, Mu, Sd, muY, sdY


def run_pda_multitreat(
    Y_treated: np.ndarray, X_controls: np.ndarray, T0: int,
    tau_grid: np.ndarray, standardize: bool = True, val_frac: float = 0.2,
    eps: float = 1e-4, max_iter: int = 2000,
) -> MultiTreatResult:
    """Cross-sectional L2-relaxation PDA over ``J`` treated units.

    Parameters
    ----------
    Y_treated : np.ndarray
        ``(T, J)`` treated-unit outcomes (pre + post stacked).
    X_controls : np.ndarray
        ``(T, N)`` control-unit outcomes.
    T0 : int
        Number of pre-treatment periods.
    tau_grid : np.ndarray
        Penalty grid searched per unit.
    standardize : bool
        Standardise series before solving (matches the L2relax default).
    val_frac : float
        Tail fraction used for time-respecting validation.
    eps, max_iter : float, int
        Batched-solver tolerance / iteration cap.
    """
    Y_treated = np.asarray(Y_treated, dtype=float)
    X_controls = np.asarray(X_controls, dtype=float)
    tau_grid = np.asarray(tau_grid, dtype=float)
    T, J = Y_treated.shape
    Xpre, Ypre = X_controls[:T0], Y_treated[:T0]

    # --- time-respecting per-unit tau selection -------------------------------
    n_val = max(2, int(round(val_frac * T0)))
    t_cv = T0 - n_val
    Sig_tr, Eta_tr, Mu_tr, Sd_tr, muY_tr, sdY_tr = _moments(
        Xpre[:t_cv], Ypre[:t_cv], standardize)
    beta_tr = l2_relax_batch(Sig_tr, Eta_tr, tau_grid, eps=eps, max_iter=max_iter)
    # unstandardise to original scale, evaluate on the validation tail
    Xval = Xpre[t_cv:T0]
    tau_star = np.empty(J, dtype=float)
    for j in range(J):
        best, best_mse = tau_grid[0], np.inf
        for k, tau in enumerate(tau_grid):
            b = sdY_tr[j] * (beta_tr[j, k] / Sd_tr)
            a = float(Ypre[:t_cv, j].mean() - Mu_tr @ b)
            mse = float(np.mean((Ypre[t_cv:T0, j] - (Xval @ b + a)) ** 2))
            if mse < best_mse:
                best_mse, best = mse, tau
        tau_star[j] = best

    # --- final fit on the full pre-period (one shared factorisation) ----------
    Sig, Eta, Mu, Sd, muY, sdY = _moments(Xpre, Ypre, standardize)
    beta_full = l2_relax_batch(Sig, Eta, tau_grid, eps=1e-6, max_iter=8000)
    tau_idx = {float(t): k for k, t in enumerate(tau_grid)}
    E1 = np.zeros((T0, J))
    Gap2 = np.zeros((T - T0, J))
    for j in range(J):
        b = sdY[j] * (beta_full[j, tau_idx[float(tau_star[j])]] / Sd)
        a = float(Ypre[:, j].mean() - Mu @ b)
        cf = X_controls @ b + a
        gap = Y_treated[:, j] - cf
        E1[:, j] = gap[:T0]
        Gap2[:, j] = gap[T0:]

    Sigma_e = E1.T @ E1 / T0
    se = float(np.sqrt(max(Sigma_e.sum(), 0.0)) / J)
    ate = Gap2.mean(axis=1)
    from scipy.stats import norm
    tstat = ate / se if se > 0 else np.full_like(ate, np.nan)
    pvalue = 2.0 * (1.0 - norm.cdf(np.abs(tstat)))
    return MultiTreatResult(ate=ate, se=se, tstat=tstat, pvalue=pvalue,
                            ate_mean=float(ate.mean()), tau=tau_star)
