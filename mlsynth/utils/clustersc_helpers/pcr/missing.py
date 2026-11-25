"""Missing donor cells on the PCR path: the two-step PCR of Agarwal, Shah and
Shen (2025), with a donor-mean fill.

Source: A. Agarwal, D. Shah and D. Shen, "On Model Identification and
Out-of-Sample Prediction of Principal Component Regression: Applications to
Synthetic Controls", JMLR 26 (2025), Section 2.2, mapped to synthetic control in
Section 7.1: the in-sample covariates :math:`\\mathbf{Z}` are the donors'
pre-period outcomes (``T0 x J`` here), the response :math:`\\mathbf{y}` is the
treated unit's pre-period series, and the out-of-sample covariates
:math:`\\mathbf{Z}'` are the donors' post-period outcomes.

I. Parameter estimation. :math:`\\hat\\rho` is the fraction of observed entries
   of :math:`\\mathbf{Z}`; the block is filled and rescaled into
   :math:`\\widetilde{\\mathbf{Z}}`, and with :math:`\\widetilde{\\mathbf{Z}}^k`
   its rank-``k`` truncation,
   :math:`\\hat\\beta = (\\widetilde{\\mathbf{Z}}^k)^\\dagger \\mathbf{y}`
   (their eq. 3).
II. Out-of-sample prediction. :math:`\\hat\\rho'` is the fraction of observed
   entries of :math:`\\mathbf{Z}'`, filled and rescaled the same way into
   :math:`\\widetilde{\\mathbf{Z}}'`, truncated at rank :math:`\\ell`, and
   :math:`\\hat{\\mathbf{y}}' = \\widetilde{\\mathbf{Z}}'^{\\ell}\\hat\\beta`.

The fill departs from the paper. Their Section 2.4.1 sets missing entries to 0
and divides by :math:`\\hat\\rho`, so that with entries observed independently
with probability :math:`\\rho`, :math:`\\mathbb{E}[\\widetilde Z_{ij}] = X_{ij}`.
The variance that fill adds to a cell is :math:`X_{ij}^2 (1-\\rho)/\\rho`, in
proportion to the squared level of the outcome, and Step I is an
errors-in-variables regression, so on outcomes far from zero the weights are
biased. On the Amjad, Shah and Shen (2018, Section 5.3) model shifted by +100,
with 20% of pre-period donor cells missing, the post-period error is -0.31 at
rank 2; on California Proposition 99 at the complete panel's rank, 10% missing
in both blocks moves the ATT from -19.4 to -27.9. CLUSTERSC centres each donor
at its observed mean :math:`m_j` before the rescaling,

.. math::

   \\widetilde Z_{ij} = m_j + (Z_{ij} - m_j)\\,
       \\mathbb{1}[Z_{ij}\\ \\text{observed}] / \\hat\\rho,

which keeps :math:`\\mathbb{E}[\\widetilde Z_{ij} \\mid m_j] = X_{ij}` and
replaces the level in the added variance by the deviation :math:`X_{ij} - m_j`.
On the same model the error is -0.03, and on Proposition 99 the ATT is -21.0.
The pre block uses pre-period means. The post block uses post-period means, and
a donor with no observed post-period cell takes its pre-period mean. A donor
with no observed pre-period cell has no mean and is refused. Each block is
rescaled by its own observed fraction, so a post block missing at a different
rate from the pre block stays on the scale of the weights.

The pre-period design (the truncation reads the pre-period donor block only,
and the counterfactual applies the weights to the post-period donor rows) is
also that of Agarwal, Amjad, Shah and Shen, "Model Agnostic Time Series
Analysis via Matrix Estimation", Proc. ACM Meas. Anal. Comput. Syst. 2(3),
Article 40 (2018), Section 3.3 (the paper behind the authors' ``tslib``), and
of Shen, Ding, Sekhon and Yu, "Same Root Different Leaves" (arXiv:2207.14481),
Corollary on PCR.

Closed form (invariant 8). A block with every cell observed is returned as it
is: :math:`m + (Z - m)` is :math:`Z` in exact arithmetic, but not in floating
point (``[0.1, 0.7, 0.3]`` loses the 0.1), so the identity is taken as a
branch. CLUSTERSC uses :math:`\\ell = \\min(m, p)`, full rank, and the
rank-full truncation of a matrix is the matrix, so Step II is
:math:`\\widetilde{\\mathbf{Z}}'\\hat\\beta` with no SVD of the post block.
With every cell observed the procedure is CLUSTERSC's complete-data path
(HSVT of the pre-period donor block, minimum-norm least squares at rank ``k``,
raw post-period projection) bit for bit. The in-sample fit is
:math:`\\widetilde{\\mathbf{Z}}\\hat\\beta`, which equals
:math:`\\widetilde{\\mathbf{Z}}^k\\hat\\beta` because :math:`\\hat\\beta` lies in
the row space of :math:`\\widetilde{\\mathbf{Z}}^k` (their Property 2.1).

:math:`\\hat\\rho` is floored at ``1/(mn)``, the convention of footnote 6 of the
2018 paper. A period without one observed donor is refused before it is used.

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
        :math:`\\widetilde{\\mathbf{Z}}`: the filled, rescaled block,
        shape ``(T0, J)``.
    M_hat : np.ndarray
        :math:`\\widetilde{\\mathbf{Z}}^k`, its rank-``k`` truncation.
    p_hat : float
        :math:`\\hat\\rho`, the observed fraction of the block.
    rank : int
        Truncation rank ``k``.
    means : np.ndarray
        Each donor's mean over its observed pre-period cells, shape ``(J,)``.
    """

    Z_tilde: np.ndarray
    M_hat: np.ndarray
    p_hat: float
    rank: int
    means: np.ndarray


def observed_fraction(mask: np.ndarray) -> float:
    """:math:`\\hat\\rho`: the observed fraction, floored at ``1/(mn)``."""
    mask = np.asarray(mask, dtype=bool)
    return max(float(mask.mean()), 1.0 / mask.size)


def _check_shape(values: np.ndarray, mask: np.ndarray) -> None:
    if values.shape != mask.shape:
        raise MlsynthEstimationError(
            f"The observed mask has shape {mask.shape} for values of shape "
            f"{values.shape}.")


def donor_means(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Each column's mean over its observed cells; NaN for a column with none.

    Entries where ``mask`` is False are never read.
    """
    values = np.asarray(values, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    _check_shape(values, mask)
    total = np.where(mask, values, 0.0).sum(axis=0)
    count = mask.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return total / count


def rescale_observed(
    values: np.ndarray,
    mask: np.ndarray,
    means: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """:math:`\\widetilde{\\mathbf{Z}} = m + (\\mathbf{Z} - m)\\,\\mathbb{1}[\\text{observed}]/\\hat\\rho`.

    Parameters
    ----------
    values, mask : np.ndarray
        The block and its observed mask. Entries where ``mask`` is False are
        never read: they may hold NaN or any other number.
    means : np.ndarray, optional
        The column centres ``m``; defaults to :func:`donor_means` of the block.

    Returns
    -------
    (np.ndarray, float)
        The filled block and :math:`\\hat\\rho`. A block with every cell
        observed is returned unchanged, with :math:`\\hat\\rho = 1`.
    """
    values = np.asarray(values, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    _check_shape(values, mask)
    if mask.all():
        return values.copy(), 1.0
    p_hat = observed_fraction(mask)
    m = donor_means(values, mask) if means is None else means
    return m + np.where(mask, values - m, 0.0) / p_hat, p_hat


def pre_period_means(values: np.ndarray, mask: np.ndarray,
                     names: Optional[Sequence] = None) -> np.ndarray:
    """:func:`donor_means` of the pre-period block, refusing a donor with none.

    Raises
    ------
    MlsynthDataError
        If a donor has no observed pre-period cell, so no mean to centre at.
    """
    means = donor_means(values, mask)
    empty = np.flatnonzero(np.isnan(means))
    if empty.size:
        who = ", ".join(str(j if names is None else names[j]) for j in empty)
        raise MlsynthDataError(
            f"CLUSTERSC: donor(s) {who} have no observed pre-period outcome, so "
            "there is no mean to fill their missing cells with. Drop them.")
    return means


def denoise_observed(values: np.ndarray, mask: np.ndarray, rank: int,
                     names: Optional[Sequence] = None) -> ObservedDenoise:
    """Step I's fill and rank-``rank`` truncation of the pre-period donor block.

    Raises
    ------
    MlsynthDataError
        If a donor has no observed pre-period cell, so no mean to centre at.
    """
    means = pre_period_means(values, mask, names)
    Z_tilde, p_hat = rescale_observed(values, mask, means)
    M_hat, _, _, _ = hsvt(Z_tilde, rank)
    return ObservedDenoise(Z_tilde=Z_tilde, M_hat=M_hat, p_hat=p_hat,
                           rank=int(rank), means=means)


def refuse_empty_periods(mask: np.ndarray, labels: Optional[Sequence] = None) -> None:
    """Refuse any period with no observed donor.

    Its filled row carries only the donor means, the same row for every such
    period whatever happened in it.
    """
    empty = np.flatnonzero(~np.asarray(mask, dtype=bool).any(axis=1))
    if empty.size:
        names = ", ".join(
            str(t if labels is None else labels[t]) for t in empty)
        raise MlsynthDataError(
            f"CLUSTERSC: period(s) {names} have no observed donor outcome, so "
            "the filled donor row carries no information about the period. "
            "Drop those periods.")


def two_block_counterfactual(
    values: np.ndarray,
    mask: np.ndarray,
    T0: int,
    beta: np.ndarray,
    pre_means: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, int]:
    """The counterfactual over every period, :math:`\\hat\\rho'`, and the fallback count.

    Pre-periods get the in-sample fit :math:`\\widetilde{\\mathbf{Z}}\\hat\\beta`;
    post-periods get Step II at full rank,
    :math:`\\widetilde{\\mathbf{Z}}'\\hat\\beta`, each block filled at its own
    donor means and rescaled by its own observed fraction. A donor with no
    observed post-period cell is centred at its pre-period mean; the third
    return value counts them. With every cell observed this is
    ``values @ beta``.
    """
    values = np.asarray(values, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    beta = np.asarray(beta, dtype=float).ravel()
    if pre_means is None:
        pre_means = donor_means(values[:T0], mask[:T0])
    Z_tilde, _ = rescale_observed(values[:T0], mask[:T0], pre_means)
    post_means = donor_means(values[T0:], mask[T0:])
    fallback = np.isnan(post_means)
    post_means = np.where(fallback, pre_means, post_means)
    Z_prime, p_prime = rescale_observed(values[T0:], mask[T0:], post_means)
    return (np.concatenate([Z_tilde @ beta, Z_prime @ beta]), p_prime,
            int(fallback.sum()))


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

    The two-step algorithm covers the PCR path with minimum-norm
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
