"""Missing donor cells on the PCR path: the two-block PCR of Agarwal, Shah and
Shen (2025).

Source: A. Agarwal, D. Shah and D. Shen, "On Model Identification and
Out-of-Sample Prediction of Principal Component Regression: Applications to
Synthetic Controls", JMLR 26 (2025), Section 2.2, mapped to synthetic control in
Section 7.1: the in-sample covariates :math:`\\mathbf{Z}` are the donors'
pre-period outcomes (``T0 x J`` here), the response :math:`\\mathbf{y}` is the
treated unit's pre-period series, and the out-of-sample covariates
:math:`\\mathbf{Z}'` are the donors' post-period outcomes.

I. Parameter estimation. :math:`\\hat\\rho` is the fraction of observed entries
   of :math:`\\mathbf{Z}`. Missing entries are set to 0 and
   :math:`\\widetilde{\\mathbf{Z}} = \\mathbf{Z} / \\hat\\rho`; with
   :math:`\\widetilde{\\mathbf{Z}}^k` its rank-``k`` truncation,
   :math:`\\hat\\beta = (\\widetilde{\\mathbf{Z}}^k)^\\dagger \\mathbf{y}`
   (their eq. 3).
II. Out-of-sample prediction. :math:`\\hat\\rho'` is the fraction of observed
   entries of :math:`\\mathbf{Z}'`, rescaled the same way into
   :math:`\\widetilde{\\mathbf{Z}}'`, truncated at rank :math:`\\ell`, and
   :math:`\\hat{\\mathbf{y}}' = \\widetilde{\\mathbf{Z}}'^{\\ell}\\hat\\beta`.

Their Section 2.4.1 gives the reason for the zero-fill: with entries observed
independently with probability :math:`\\rho`,
:math:`\\mathbb{E}[Z_{ij}] = \\rho X_{ij}`, so
:math:`\\mathbb{E}[\\widetilde Z_{ij}] = X_{ij}`. Each block is rescaled by its
own observed fraction, so a post block missing at a different rate from the
pre block stays on the scale of the weights.

The pre-period design (the truncation reads the pre-period donor block only,
and the counterfactual applies the weights to the post-period donor rows) is
also that of Agarwal, Amjad, Shah and Shen, "Model Agnostic Time Series
Analysis via Matrix Estimation", Proc. ACM Meas. Anal. Comput. Syst. 2(3),
Article 40 (2018), Section 3.3 (the paper behind the authors' ``tslib``, whose
Property 2.1 is Step I's de-noising), and of Shen, Ding, Sekhon and Yu, "Same
Root Different Leaves" (arXiv:2207.14481), Corollary on PCR.

CLUSTERSC uses :math:`\\ell = \\min(m, p)`, full rank. The rank-full truncation
of a matrix is the matrix, so Step II is the closed form
:math:`\\hat{\\mathbf{y}}' = \\widetilde{\\mathbf{Z}}'\\hat\\beta` and no SVD of
the post block is taken (invariant 8). With every cell observed,
:math:`\\hat\\rho = \\hat\\rho' = 1` and the procedure is CLUSTERSC's
complete-data path (HSVT of the pre-period donor block, minimum-norm least
squares at rank ``k``, raw post-period projection) bit for bit. The in-sample
fit is :math:`\\widetilde{\\mathbf{Z}}\\hat\\beta`: it equals
:math:`\\widetilde{\\mathbf{Z}}^k\\hat\\beta` because :math:`\\hat\\beta` lies in
the row space of :math:`\\widetilde{\\mathbf{Z}}^k` (their Property 2.1) and
the discarded components are orthogonal to that space.

:math:`\\hat\\rho` is floored at ``1/(mn)``, the convention of footnote 6 of the
2018 paper, so an all-missing block divides by a positive number; such a
block, and any period without one observed donor, is refused before it is
used.

The truncation goes through :func:`mlsynth.utils.pcr.core.hsvt`, an SVD, and
the weights through :func:`mlsynth.utils.pcr.core.pcr_weights`, never through
normal equations (invariant 9).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from ....exceptions import MlsynthDataError, MlsynthEstimationError
from ...pcr.core import hsvt


@dataclass(frozen=True)
class ObservedDenoise:
    """Step I's de-noised pre-period block.

    Attributes
    ----------
    Z_tilde : np.ndarray
        :math:`\\widetilde{\\mathbf{Z}}`: zero-filled block over
        :math:`\\hat\\rho`, shape ``(T0, J)``.
    M_hat : np.ndarray
        :math:`\\widetilde{\\mathbf{Z}}^k`, its rank-``k`` truncation.
    p_hat : float
        :math:`\\hat\\rho`, the observed fraction of the block.
    rank : int
        Truncation rank ``k``.
    """

    Z_tilde: np.ndarray
    M_hat: np.ndarray
    p_hat: float
    rank: int


def observed_fraction(mask: np.ndarray) -> float:
    """:math:`\\hat\\rho`: the observed fraction, floored at ``1/(mn)``."""
    mask = np.asarray(mask, dtype=bool)
    return max(float(mask.mean()), 1.0 / mask.size)


def rescale_observed(values: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, float]:
    """:math:`\\widetilde{\\mathbf{Z}} = \\mathbf{Z}/\\hat\\rho` with missing entries set to 0.

    Entries where ``mask`` is False are never read: they may hold NaN or any
    other number. Returns the rescaled block and :math:`\\hat\\rho`.
    """
    values = np.asarray(values, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if values.shape != mask.shape:
        raise MlsynthEstimationError(
            f"The observed mask has shape {mask.shape} for values of shape "
            f"{values.shape}.")
    p_hat = observed_fraction(mask)
    return np.where(mask, values, 0.0) / p_hat, p_hat


def denoise_observed(values: np.ndarray, mask: np.ndarray,
                     rank: int) -> ObservedDenoise:
    """Step I's de-noising of the pre-period donor block at rank ``rank``."""
    Z_tilde, p_hat = rescale_observed(values, mask)
    M_hat, _, _, _ = hsvt(Z_tilde, rank)
    return ObservedDenoise(Z_tilde=Z_tilde, M_hat=M_hat, p_hat=p_hat,
                           rank=int(rank))


def refuse_empty_periods(mask: np.ndarray, labels: Optional[Sequence] = None) -> None:
    """Refuse any period with no observed donor.

    Its row of :math:`\\widetilde{\\mathbf{Z}}` or
    :math:`\\widetilde{\\mathbf{Z}}'` is all zeros, so its estimate would be 0
    whatever the data.
    """
    empty = np.flatnonzero(~np.asarray(mask, dtype=bool).any(axis=1))
    if empty.size:
        names = ", ".join(
            str(t if labels is None else labels[t]) for t in empty)
        raise MlsynthDataError(
            f"CLUSTERSC: period(s) {names} have no observed donor outcome, so "
            "the zero-filled donor row carries no information and the "
            "counterfactual there would be 0. Drop those periods.")


def two_block_counterfactual(
    values: np.ndarray,
    mask: np.ndarray,
    T0: int,
    beta: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """The counterfactual over every period, and :math:`\\hat\\rho'`.

    Pre-periods get the in-sample fit :math:`\\widetilde{\\mathbf{Z}}\\hat\\beta`;
    post-periods get Step II at full rank,
    :math:`\\widetilde{\\mathbf{Z}}'\\hat\\beta`, each block rescaled by its own
    observed fraction. With every cell observed this is ``values @ beta``.
    """
    values = np.asarray(values, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    beta = np.asarray(beta, dtype=float).ravel()
    Z_tilde, _ = rescale_observed(values[:T0], mask[:T0])
    Z_prime, p_prime = rescale_observed(values[T0:], mask[T0:])
    return np.concatenate([Z_tilde @ beta, Z_prime @ beta]), p_prime


def check_missing_supported(
    *,
    method: str,
    clustering: bool,
    estimator: str,
    objective: str,
    lambda_penalty: Optional[float],
    project_denoised: bool,
    compute_shen_ci: bool,
    compute_scpi_pi: bool,
) -> None:
    """Refuse a panel with missing donor cells on a branch the paper does not cover.

    The two-block algorithm covers the PCR path with minimum-norm
    least-squares weights. Every other option rests on a method that assumes
    a complete donor matrix, or on weights the 1/rho-hat rescaling does not
    leave invariant.

    Raises
    ------
    MlsynthDataError
        Listing every unsupported option set.
    """
    reasons = []
    if method != "pcr":
        reasons.append(
            f"method={method!r}: the rpca family (PCP, HQF, HSVT, FGRC "
            "de-noising and simplex weights) assumes a complete donor matrix; "
            "use method='pcr'")
    if clustering:
        reasons.append(
            "clustering=True: the donor clustering of Rho et al. (2025) "
            "assumes a complete donor matrix; set clustering=False")
    if estimator != "frequentist":
        reasons.append(
            f"estimator={estimator!r}: the bayesian posterior (Amjad, Shah and "
            "Shen 2018, Algorithm 2) projects through a full-period truncation; "
            "use estimator='frequentist'")
    if objective != "OLS":
        reasons.append(
            f"pcr_objective={objective!r}: Step I is the minimum-norm least "
            "squares fit, and simplex weights are not invariant to the "
            "1/rho-hat rescaling; use 'OLS'")
    if lambda_penalty not in (None, 0, 0.0):
        reasons.append(
            "lambda_penalty>0: Step I is the minimum-norm least squares fit, "
            "and a penalised fit is not invariant to the 1/rho-hat rescaling; "
            "leave lambda_penalty unset")
    if project_denoised:
        reasons.append(
            "project_denoised=True: projects through a rank-r truncation of "
            "the full-period donor matrix, which is not Step II of Agarwal, "
            "Shah and Shen (2025); leave it False")
    if compute_shen_ci:
        reasons.append(
            "compute_shen_ci=True: the Shen et al. (2023) intervals assume a "
            "complete donor matrix; set compute_shen_ci=False")
    if compute_scpi_pi:
        reasons.append(
            "compute_scpi_pi=True: the scpi prediction intervals assume a "
            "complete donor matrix; set compute_scpi_pi=False")
    if reasons:
        raise MlsynthDataError(
            "CLUSTERSC: the panel has missing donor cells, which only the PCR "
            "path handles (Agarwal, Shah and Shen 2025, Section 2.2). "
            "Unsupported here: " + "; ".join(reasons) + ".")
