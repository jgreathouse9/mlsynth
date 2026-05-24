"""Doubly robust proximal synthetic control (DR).

Implements the doubly-robust ATT of Qiu, Shi, Miao, Dobriban and Tchetgen
Tchetgen (2024, Biometrics):

    phi* = E_post[Y - h(W)] - E_pre[q(Z){Y - h(W)}],

which is consistent if *either* the outcome bridge ``h`` *or* the
treatment bridge ``q`` is correctly specified. Both nuisances are fit on
the pre-period; the estimand and all nuisance/auxiliary parameters are
stacked into one just-identified GMM, and the ATT standard error is the
GMM sandwich with a Bartlett-HAC middle. Validated against the authors'
reference code (``QIU-Hongxiang-David/DR_Proximal_SC``): the just-identified
point estimate matches by construction, and the sandwich SE is calibrated
(~95% Wald coverage on their simulation DGP).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..bridges import (
    augment,
    fit_outcome_bridge,
    fit_treatment_bridge,
    gmm_sandwich_se,
)


def estimate_dr(
    outcome_vector: np.ndarray,
    donor_outcomes: np.ndarray,
    donor_proxies: np.ndarray,
    num_pre_treatment_periods: int,
    hac_bandwidth: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Doubly-robust proximal ATT.

    Parameters
    ----------
    outcome_vector : np.ndarray
        Treated outcome ``Y``, shape ``(T,)``.
    donor_outcomes : np.ndarray
        Donor outcomes ``W`` (the outcome-bridge regressors), shape
        ``(T, n_donors)``.
    donor_proxies : np.ndarray
        Supplemental proxies ``Z`` (instruments for ``h`` and regressors
        for ``q``), shape ``(T, n_proxies)``.
    num_pre_treatment_periods : int
        ``T0``.
    hac_bandwidth : int
        Bartlett-HAC bandwidth for the sandwich SE.

    Returns
    -------
    counterfactual : np.ndarray
        Outcome-bridge synthetic control ``h(W) = (1, W) alpha`` over all
        periods, shape ``(T,)``.
    alpha : np.ndarray
        Outcome-bridge coefficients (intercept first).
    beta : np.ndarray
        Treatment-bridge coefficients (intercept first).
    att : float
        Doubly-robust ATT estimate ``phi``.
    se : float
        GMM/HAC standard error of ``phi``.
    """

    Y = np.asarray(outcome_vector, dtype=float).ravel()
    W = np.asarray(donor_outcomes, dtype=float)
    Z = np.asarray(donor_proxies, dtype=float)
    T = len(Y)
    T0 = int(num_pre_treatment_periods)
    nU = W.shape[1]
    pre = np.arange(T) < T0
    post = ~pre
    Wc, Zc = augment(W), augment(Z)

    alpha = fit_outcome_bridge(Y[pre], Wc[pre], Zc[pre])
    psi = Wc[post].mean(0)
    beta = fit_treatment_bridge(Zc[pre], Wc[pre], psi)
    h = Wc @ alpha
    q = np.exp(Zc @ beta)
    psi_minus = float((q[pre] * (Y[pre] - h[pre])).mean())
    phi = float((Y[post] - h[post]).mean() - psi_minus)

    # Stacked just-identified moments, parameter order [alpha, beta, psi, phi, psi-].
    theta = np.concatenate([alpha, beta, psi, [phi], [psi_minus]])
    phi_index = 3 * nU + 3

    def moments(th: np.ndarray) -> np.ndarray:
        a = th[: nU + 1]
        b = th[nU + 1 : 2 * nU + 2]
        ps = th[2 * nU + 2 : 3 * nU + 3]
        ph = th[3 * nU + 3]
        pm = th[3 * nU + 4]
        hh = Wc @ a
        qq = np.exp(Zc @ b)
        g1 = pre[:, None] * ((Y - hh)[:, None] * Zc)          # outcome bridge
        g2 = post[:, None] * (ps[None, :] - Wc)                # psi = E_post[(1,W)]
        g3 = pre[:, None] * (qq[:, None] * Wc - ps[None, :])   # treatment bridge
        g4 = post.astype(float) * (ph - (Y - hh) + pm)         # phi (ATT)
        g5 = pre.astype(float) * (pm - qq * (Y - hh))          # psi- = E_pre[q(Y-h)]
        return np.column_stack([g1, g2, g3, g4, g5])

    se = gmm_sandwich_se(theta, moments, phi_index, T, hac_bandwidth)
    counterfactual = h
    return counterfactual, alpha, beta, phi, se
