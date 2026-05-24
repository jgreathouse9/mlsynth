"""Treatment-bridge (proximal inverse-probability weighting) estimator (PIPW).

Implements the weighting-only ATT of Qiu, Shi, Miao, Dobriban and Tchetgen
Tchetgen (2024, Biometrics):

    phi* = E_post[Y] - E_pre[q(Z) Y],

where ``q_beta(Z) = exp((1, Z) beta)`` is the treatment confounding bridge
(a covariate-shift / likelihood-ratio weight) solving the pre-period moment
``E_pre[q(Z)(1, W)] = E_post[(1, W)]``. Unlike the outcome-bridge methods,
this relies on **no** model for the treated unit's counterfactual outcome
trajectory -- only on correctly modelling the weights. The estimand and
auxiliary means are stacked into one just-identified GMM with a Bartlett-HAC
sandwich SE.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..bridges import augment, fit_treatment_bridge, gmm_sandwich_se


def estimate_pipw(
    outcome_vector: np.ndarray,
    donor_outcomes: np.ndarray,
    donor_proxies: np.ndarray,
    num_pre_treatment_periods: int,
    hac_bandwidth: int,
) -> Tuple[np.ndarray, float, float]:
    """Treatment-bridge weighting ATT.

    Parameters
    ----------
    outcome_vector : np.ndarray
        Treated outcome ``Y``, shape ``(T,)``.
    donor_outcomes : np.ndarray
        Donor outcomes ``W`` used in the weighting moment, shape
        ``(T, n_donors)``.
    donor_proxies : np.ndarray
        Supplemental proxies ``Z`` (treatment-bridge regressors), shape
        ``(T, n_proxies)``.
    num_pre_treatment_periods : int
        ``T0``.
    hac_bandwidth : int
        Bartlett-HAC bandwidth for the sandwich SE.

    Returns
    -------
    beta : np.ndarray
        Treatment-bridge coefficients (intercept first).
    att : float
        Weighting ATT estimate ``phi``.
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

    psi = Wc[post].mean(0)
    beta = fit_treatment_bridge(Zc[pre], Wc[pre], psi)
    q = np.exp(Zc @ beta)
    psi_minus = float((q[pre] * Y[pre]).mean())
    phi = float(Y[post].mean() - psi_minus)

    # Parameter order [beta, psi, phi, psi-].
    theta = np.concatenate([beta, psi, [phi], [psi_minus]])
    phi_index = 2 * nU + 3

    def moments(th: np.ndarray) -> np.ndarray:
        b = th[: nU + 1]
        ps = th[nU + 1 : 2 * nU + 2]
        ph = th[2 * nU + 2]
        pm = th[2 * nU + 3]
        qq = np.exp(Zc @ b)
        g1 = post[:, None] * (ps[None, :] - Wc)
        g2 = pre[:, None] * (qq[:, None] * Wc - ps[None, :])
        g3 = post.astype(float) * (ph - Y + pm)
        g4 = pre.astype(float) * (pm - qq * Y)
        return np.column_stack([g1, g2, g3, g4])

    se = gmm_sandwich_se(theta, moments, phi_index, T, hac_bandwidth)
    return beta, phi, se
