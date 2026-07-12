"""Over-identified Proximal Inference (PIOID) -- unit-instrument outcome bridge.

Implements the proximal outcome-bridge estimator of Shi, Li, Yu, Miao,
Kuchibhotla, Hu & Tchetgen Tchetgen (2026), *"Theory for Identification and
Inference with Synthetic Controls: A Proximal Causal Inference Framework"*
(JASA), as coded in the authors' manuscript replication (``KenLi93/
proximal_sc_manuscript``, function ``NC_nocov``).

Unlike the just-identified :func:`..pi.estimation.estimate_pi` -- which uses a
proxy *variable* measured on the same donor units, giving one instrument per
donor -- here the donor pool ``W`` (the ``donors``) is instrumented by a
*distinct set of donor units* ``Z`` (the ``outcome_instruments``), the same
outcome variable throughout. With more instruments than donors the outcome
bridge is over-identified and solved by one-step GMM under the identity weight,

    omega = (W'Z Z'W)^{-1} W'Z Z'Y,   fit on the pre-period,

which is 2SLS of the treated series on ``W`` using ``Z`` as instruments. The
fitted ``W omega`` is the counterfactual and its post-period gap the ATT. This
is the configuration the manuscript's German-reunification application uses; on
``scpi_germany`` it reproduces the paper's PI headline of -1709 USD.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..inference import hac


def estimate_pi_overid(
    outcome_vector: np.ndarray,
    design_matrix: np.ndarray,
    instrument_matrix: np.ndarray,
    num_pre_treatment_periods: int,
    num_post_periods_for_effect_eval: int,
    total_periods: int,
    hac_truncation_lag: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Over-identified PI counterfactual, donor weights, and ATT SE.

    Parameters
    ----------
    outcome_vector : np.ndarray
        Treated outcome, shape ``(total_periods,)``.
    design_matrix : np.ndarray
        Donor outcomes ``W``, shape ``(total_periods, n_donors)``.
    instrument_matrix : np.ndarray
        Instrument-unit outcomes ``Z``, shape ``(total_periods, n_instruments)``
        with ``n_instruments >= n_donors`` (over-identified).
    num_pre_treatment_periods : int
        Number of pre-treatment periods ``T0``.
    num_post_periods_for_effect_eval : int
        Number of post-treatment periods used to average the ATT.
    total_periods : int
        Total number of periods ``T``.
    hac_truncation_lag : int
        Bartlett bandwidth for the HAC variance.

    Returns
    -------
    counterfactual : np.ndarray
        Predicted counterfactual ``W omega``, shape ``(total_periods,)``.
    alpha : np.ndarray
        Donor coefficients ``omega``.
    se_tau : float
        Standard error of the ATT (``np.nan`` if GMM inference fails).
    """

    W = np.asarray(design_matrix, dtype=float)
    Z = np.asarray(instrument_matrix, dtype=float)
    Y = np.asarray(outcome_vector, dtype=float).ravel()
    T0 = int(num_pre_treatment_periods)

    Wp, Zp, Yp = W[:T0], Z[:T0], Y[:T0]
    # One-step GMM under the identity weight (2SLS): omega = (W'Z Z'W)^{-1} W'Z Z'Y.
    WZ = Wp.T @ Zp                       # (n_donors, n_instruments)
    A = WZ @ WZ.T                        # (n_donors, n_donors)
    b = WZ @ (Zp.T @ Yp)
    alpha = np.linalg.solve(A, b)

    counterfactual = W @ alpha
    gap = Y - counterfactual
    tau = float(np.mean(gap[T0 : T0 + num_post_periods_for_effect_eval]))

    se_tau = _overid_att_se(
        Y, W, Z, alpha, tau, T0, total_periods, hac_truncation_lag,
    )
    return counterfactual, alpha, se_tau


def _overid_att_se(
    Y: np.ndarray,
    W: np.ndarray,
    Z: np.ndarray,
    alpha: np.ndarray,
    tau: float,
    T0: int,
    total_periods: int,
    hac_truncation_lag: int,
) -> float:
    """GMM sandwich SE of the ATT for the one-step (identity-weight) over-id bridge.

    Stacks the pre-period bridge moment ``g_t = (Z'W)(Z_t)(Y_t - W_t' alpha)``
    (the identity-weighted projection that defines ``alpha``) with the
    post-period ATT moment, and returns the HAC-cored sandwich standard error of
    the ATT. Returns ``np.nan`` if the Jacobian is singular.
    """

    n_donors = W.shape[1]
    WZ = W[:T0].T @ Z[:T0]                        # (n_donors, n_instruments)
    resid = Y - W @ alpha

    # Effective bridge instrument for the just-identified projection: WZ @ Z_t.
    # Pre-period moments (n_donors), post-period ATT moment (1).
    U0 = (WZ @ Z.T) * resid.reshape(1, -1)        # (n_donors, T)
    U0[:, T0:] = 0.0
    U1 = resid - tau
    U1[:T0] = 0.0
    U = np.column_stack((U0.T, U1))               # (T, n_donors + 1)

    G = np.zeros((n_donors + 1, n_donors + 1))
    # d(pre moments)/d(alpha) = -(WZ Z') W summed, scaled by T.
    G[:n_donors, :n_donors] = -(WZ @ Z[:T0].T @ W[:T0]) / total_periods
    G[-1, :n_donors] = np.sum(W[T0:], axis=0) / total_periods
    G[-1, -1] = (total_periods - T0) / total_periods

    omega = hac(U, hac_truncation_lag)
    try:
        G_inv = np.linalg.inv(G)
        cov = G_inv @ omega @ G_inv.T
        var_tau = cov[-1, -1] / total_periods
        return float(np.sqrt(var_tau)) if var_tau >= 0 else np.nan
    except np.linalg.LinAlgError:
        return np.nan
