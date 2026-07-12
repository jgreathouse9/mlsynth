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


def _solve_simplex_bridge(Wp: np.ndarray, Zp: np.ndarray, Yp: np.ndarray) -> np.ndarray:
    """Simplex-constrained proximal outcome bridge (the authors' cPI / NC_constrained_nocov).

    Solves ``min_omega (Y - W omega)' Z Z' (Y - W omega)`` subject to
    ``omega >= 0, sum(omega) = 1`` on the pre-period. Writing ``Z'(Y - W omega)``
    as a residual makes this the simplex least-squares
    ``min ||(Z'Y) - (Z'W) omega||^2``, solved exactly by the pure-NumPy,
    PSD-safe active-set method :func:`mlsynth.utils.bilevel.active_set.solve_simplex_qp`
    -- the ``lstsq`` free-set solve handles the rank-deficient/ill-conditioned
    ``Z'W`` Gram without an epsilon-I fudge, reaching the same optimum the
    convex program ``scpi::scest(w.constr="simplex", V.mat = Z'Z / T0)`` does.
    """
    from ...bilevel.active_set import solve_simplex_qp

    return solve_simplex_qp(Zp.T @ Wp, Zp.T @ Yp)


def estimate_pi_overid(
    outcome_vector: np.ndarray,
    design_matrix: np.ndarray,
    instrument_matrix: np.ndarray,
    num_pre_treatment_periods: int,
    num_post_periods_for_effect_eval: int,
    total_periods: int,
    hac_truncation_lag: int,
    simplex: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Over-identified PI counterfactual, donor weights, and ATT SE.

    The point estimate is the one-step GMM (identity-weight) outcome bridge
    ``omega = (W'Z Z'W)^{-1} W'Z Z'Y`` on the pre-period, i.e. 2SLS of the
    treated series on ``W`` using ``Z`` as instruments; ``W omega`` is the
    counterfactual and its post-period gap the ATT. With ``simplex=True`` the
    coefficients are instead constrained to the simplex (``omega >= 0``,
    ``sum(omega) = 1``) under the same ``Z'Z`` metric -- the authors' cPI
    (constrained proximal inference) -- and no GMM standard error is reported
    (the paper's constrained inference is by permutation).

    The ATT standard error follows the authors' ``NC_nocov_gmm`` exactly: a
    single one-step GMM over all periods with parameters ``theta = (tau, omega)``,
    moment ``(Y - S1 theta) S2`` for ``S1 = [X, W]`` and ``S2 = [X, Z(1-X)]`` (the
    treatment indicator identifies ``tau``, the pre-period instruments identify
    ``omega``), and the over-identified sandwich ``(G'G)^{-1} G' Omega G (G'G)^{-1}``
    with a Bartlett-HAC ``Omega`` at ``hac_truncation_lag``. At the manuscript's
    Newey-West lag (10) this reproduces the paper's GMM confidence interval.

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
        Bartlett/Newey-West bandwidth for the HAC variance.

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
    if simplex:
        # Constrained proximal inference (cPI): simplex-constrained bridge, no
        # GMM SE (the paper's constrained inference is by permutation).
        alpha = _solve_simplex_bridge(Wp, Zp, Yp)
        counterfactual = W @ alpha
        return counterfactual, alpha, np.nan

    # One-step GMM under the identity weight (2SLS): omega = (W'Z Z'W)^{-1} W'Z Z'Y.
    WZ = Wp.T @ Zp                       # (n_donors, n_instruments)
    A = WZ @ WZ.T                        # (n_donors, n_donors)
    b = WZ @ (Zp.T @ Yp)
    alpha = np.linalg.solve(A, b)

    counterfactual = W @ alpha
    gap = Y - counterfactual
    tau = float(np.mean(gap[T0 : T0 + num_post_periods_for_effect_eval]))

    se_tau = _overid_att_se(Y, W, Z, alpha, tau, T0, total_periods, hac_truncation_lag)
    return counterfactual, alpha, se_tau


def _overid_joint_cov(
    Y: np.ndarray,
    W: np.ndarray,
    Z: np.ndarray,
    alpha: np.ndarray,
    tau: float,
    T0: int,
    total_periods: int,
    hac_truncation_lag: int,
):
    """Joint one-step-GMM sandwich covariance of ``theta = (tau, omega)``.

    Builds the moment matrix ``bg = (Y - S1 theta) * S2`` with ``S1 = [X, W]``,
    ``S2 = [X, Z(1-X)]``; the over-identified sandwich is
    ``(G'G)^{-1} G' Omega G (G'G)^{-1}`` with a Bartlett-HAC ``Omega``. Returns
    the full ``(1 + n_donors, 1 + n_donors)`` matrix (still to be divided by
    ``T`` for the variance), or ``None`` if the projection is singular. The ATT
    SE and the per-period counterfactual band both read off this one object.
    """
    T = total_periods
    X = np.concatenate([np.zeros(T0), np.ones(T - T0)])
    theta = np.concatenate([[tau], alpha])
    S1 = np.column_stack([X, W])                 # (T, 1 + n_donors)
    S2 = np.column_stack([X, Z * (1.0 - X)[:, None]])  # (T, 1 + n_instruments)

    bg = (Y - S1 @ theta)[:, None] * S2          # (T, 1 + n_instruments) moments
    G = S2.T @ S1 / T                            # (1 + n_instr, 1 + n_donors) Jacobian
    omega = hac(bg, hac_truncation_lag)
    try:
        proj = np.linalg.inv(G.T @ G) @ G.T      # (1 + n_donors, 1 + n_instr)
        return proj @ omega @ proj.T             # (1 + n_donors, 1 + n_donors)
    except np.linalg.LinAlgError:
        return None


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
    """Over-identified one-step-GMM sandwich SE of the ATT (authors' NC_nocov_gmm).

    The ATT variance is the top-left entry of the joint sandwich
    (:func:`_overid_joint_cov`) divided by ``T``. Returns ``np.nan`` if the
    projection is singular.
    """
    cov = _overid_joint_cov(
        Y, W, Z, alpha, tau, T0, total_periods, hac_truncation_lag)
    if cov is None:
        return np.nan
    var_tau = cov[0, 0] / total_periods
    return float(np.sqrt(var_tau)) if var_tau >= 0 else np.nan


def overid_counterfactual_band(
    outcome_vector: np.ndarray,
    design_matrix: np.ndarray,
    instrument_matrix: np.ndarray,
    alpha: np.ndarray,
    num_pre_treatment_periods: int,
    num_post_periods_for_effect_eval: int,
    total_periods: int,
    hac_truncation_lag: int,
    level: float = 0.90,
    method: str = "gmm",
):
    """Per-period confidence band on the PIOID counterfactual (Shi et al. 2026).

    Two routes, both on the outcome bridge ``h(W_t) = W_t' omega``:

    ``method="gmm"`` (Section 3.2.3, default) -- the delta-method band
    ``W omega +/- z_{level} * sqrt(W_t' Var(omega_hat) W_t)`` with
    ``Var(omega_hat)`` the ``omega`` block of the joint ``(tau, omega)`` sandwich
    (:func:`_overid_joint_cov`), the per-period companion to the ATT SE the
    estimator already reports.

    ``method="conformal"`` (Section 3.2.1; Chernozhukov, Wuthrich & Zhu 2021) --
    the split-conformal prediction band ``W omega +/- q`` with ``q`` the
    ``level`` quantile of the absolute pre-period residuals. The bridge is fit on
    the pre-period only, so the counterfactual never sees the post outcomes and
    the interval needs no refitting; its half-width is a single residual quantile
    shared across post periods.

    Returns full-length ``(lower, upper)`` arrays, or ``(None, None)`` if the GMM
    sandwich is singular.
    """
    from ....exceptions import MlsynthConfigError

    W = np.asarray(design_matrix, dtype=float)
    Z = np.asarray(instrument_matrix, dtype=float)
    Y = np.asarray(outcome_vector, dtype=float).ravel()
    alpha = np.asarray(alpha, dtype=float).ravel()
    T = int(total_periods)
    T0 = int(num_pre_treatment_periods)
    cf = W @ alpha

    if method == "conformal":
        resid_pre = np.abs((Y - cf)[:T0])
        q = float(np.quantile(resid_pre, float(level)))
        return cf - q, cf + q
    if method != "gmm":
        raise MlsynthConfigError(
            f"Unknown PIOID band method {method!r}; expected 'gmm' or 'conformal'.")

    from scipy.stats import norm
    tau = float(np.mean((Y - cf)[T0 : T0 + num_post_periods_for_effect_eval]))
    cov = _overid_joint_cov(Y, W, Z, alpha, tau, T0, T, hac_truncation_lag)
    if cov is None:
        return None, None
    var_alpha = cov[1:, 1:] / T                  # Var(omega_hat)
    var_t = np.einsum("tj,jk,tk->t", W, var_alpha, W)
    se_t = np.sqrt(np.clip(var_t, 0.0, None))
    z = float(norm.ppf(0.5 + float(level) / 2.0))
    return cf - z * se_t, cf + z * se_t
