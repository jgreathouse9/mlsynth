"""Top-level SI solve: fit every intervention arm and assemble results."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.stats import norm

from ...exceptions import MlsynthEstimationError
from ..clustersc_helpers.pcr.hsvt import hsvt
from .estimation import (
    resolve_rank,
    select_omega,
    si_pcr_weights,
    variance_estimation,
)
from .structures import SIArm, SIInputs, SIResults


def _interval(theta, w, sigma, T1, alpha, kind):
    """Asymptotic interval half-width around ``theta`` (Agarwal-Shah-Shen).

    ``kind="confidence"`` gives the eq.-13 CI (half-width
    ``z*sigma*||w||/sqrt(T1)``); ``kind="prediction"`` adds the extra
    prediction term (``z*sigma*sqrt(1+||w||^2)/sqrt(T1)``).
    """
    z = float(norm.ppf(1.0 - alpha / 2.0))
    wn = float(np.linalg.norm(w))
    if kind == "prediction":
        half = z * sigma * np.sqrt(1.0 + wn ** 2) / np.sqrt(T1)
    else:
        half = z * sigma * wn / np.sqrt(T1)
    return (theta - half, theta + half), half


def _fit_arm(
    inputs: SIInputs,
    intervention: str,
    rank_method: str,
    rank: Optional[int],
    cumvar_threshold: float,
    bias_correct: bool,
    alpha: float,
    variance: str,
    interval: str,
) -> SIArm:
    """Fit one intervention arm (the focal unit's counterfactual under it)."""
    pool = inputs.pools[intervention]
    T0 = inputs.T0
    donor_pre = pool.matrix[:T0]            # (T0, Nd) under control
    donor_post = pool.matrix[T0:]           # (T1, Nd) under intervention d
    target_pre = inputs.Y_pre
    names = pool.names

    k = resolve_rank(donor_pre, rank_method, rank, cumvar_threshold)
    k = max(1, min(k, donor_pre.shape[1]))

    # rank-k HSVT of the donor pre-matrix (shared ClusterSC primitive)
    M_hat, U_k, _, Vt_k = hsvt(donor_pre, k)
    V_k = Vt_k.T

    sigma_hat: Optional[float] = None
    weight_norm: Optional[float] = None
    cf_mean_ci: Optional[Tuple[float, float]] = None
    att_ci: Optional[Tuple[float, float]] = None

    if bias_correct:
        omega = select_omega(donor_pre, k)
        omega_names = [names[j] for j in omega]
        w_omega = np.linalg.pinv(M_hat[:, omega]) @ target_pre
        weights = {names[j]: float(w) for j, w in zip(omega, w_omega)}
        cf_pre = M_hat[:, omega] @ w_omega
        cf_post = donor_post[:, omega] @ w_omega
        weight_norm = float(np.linalg.norm(w_omega))
        sig_double, sig_units, sig_time = variance_estimation(
            U_k, V_k, target_pre, donor_post
        )
        sigma_hat = {"double": sig_double, "units": sig_units,
                     "time_iv": sig_time}[variance]
    else:
        w = si_pcr_weights(donor_pre, target_pre, k)
        omega_names = list(names)
        weights = {names[j]: float(w[j]) for j in range(len(names)) if abs(w[j]) > 1e-10}
        cf_pre = donor_pre @ w
        cf_post = donor_post @ w

    counterfactual = np.concatenate([cf_pre, cf_post])
    gap = inputs.y_target - counterfactual
    post_gap = gap[T0:]
    att = float(np.mean(post_gap)) if post_gap.size else float("nan")
    cf_mean = float(np.mean(cf_post)) if cf_post.size else float("nan")
    pre_resid = target_pre - cf_pre
    pre_rmse = float(np.sqrt(np.mean(pre_resid ** 2))) if pre_resid.size else float("nan")

    if bias_correct and sigma_hat is not None and inputs.n_post > 0:
        cf_mean_ci, half = _interval(
            cf_mean, w_omega, sigma_hat, inputs.n_post, alpha, interval
        )
        # ATT = observed_post_mean - cf_mean; observed is fixed, so the
        # half-width is shared between the counterfactual and the effect.
        att_ci = (att - half, att + half)

    return SIArm(
        name=intervention,
        donor_names=list(names),
        weights=weights,
        selected_rank=int(k),
        omega_names=omega_names,
        counterfactual=counterfactual,
        gap=gap,
        att=att,
        cf_mean=cf_mean,
        pre_rmse=pre_rmse,
        bias_corrected=bias_correct,
        sigma_hat=sigma_hat,
        weight_norm=weight_norm,
        cf_mean_ci=cf_mean_ci,
        att_ci=att_ci,
    )


def solve_si(
    inputs: SIInputs,
    rank_method: str = "donoho",
    rank: Optional[int] = None,
    cumvar_threshold: float = 0.95,
    bias_correct: bool = True,
    alpha: float = 0.05,
    variance: str = "double",
    interval: str = "confidence",
) -> SIResults:
    """Fit the SI estimator for every alternative intervention.

    Parameters
    ----------
    inputs : SIInputs
        Prepared focal-unit data and donor pools.
    rank_method : {"donoho", "usvt", "cumvar", "fixed"}
        Spectral-rank rule. ``"donoho"`` (default) reproduces the paper's exact
        Gavish-Donoho rank (``ratio = T0 / Nd``).
    rank : int, optional
        Explicit rank for ``rank_method="fixed"``.
    cumvar_threshold : float
        Cumulative-energy target for ``rank_method="cumvar"``.
    bias_correct : bool
        Use the bias-corrected SI-PCR estimator (enables intervals).
    alpha : float
        Two-sided significance level for the intervals.
    variance : {"double", "units", "time_iv"}
        Noise-variance estimator behind the interval. ``"double"`` (default)
        matches the paper's code; ``"units"`` is the main-text eq. 14.
    interval : {"confidence", "prediction"}
        Interval type. ``"confidence"`` is the eq.-13 CI for the counterfactual
        mean; ``"prediction"`` is the wider prediction interval the case study
        uses for coverage validation.

    Returns
    -------
    SIResults
    """
    if not inputs.pools:
        raise MlsynthEstimationError("SI requires at least one intervention pool.")
    arms = {
        name: _fit_arm(
            inputs, name, rank_method, rank, cumvar_threshold,
            bias_correct, alpha, variance, interval,
        )
        for name in inputs.pools
    }
    return SIResults(
        inputs=inputs, arms=arms, alpha=alpha, bias_corrected=bias_correct
    )
