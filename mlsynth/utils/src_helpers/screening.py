"""SIRS donor screening for SRC (Zhu 2023, Algorithm 2).

When the donor pool is not small relative to the pre-period length
(:math:`J \\ge T_0`, recommended in the paper at :math:`J \\ge 4T_0/5`), the
unit-regression fit is ill-posed: the pre-period system has more donor
coefficients than periods, so the noise estimate :math:`\\widehat{\\sigma}^2`
collapses and the Cp penalty switches off. Algorithm 2 screens the pool down to
a subset of active donors *before* running Algorithm 1, restoring a well-posed
fit.

Screening ranks donors by the sure-independent-ranking-and-screening (SIRS)
marginal utility of Zhu, Fan, Li & Zhu (2011), which the paper adopts. For a
standardized donor path :math:`\\mathbf{z}_j` and the treated path
:math:`\\mathbf{y}_1` over the pre-period,

.. math::

   \\widehat{\\omega}_j = \\frac{1}{T_0}\\sum_{t=1}^{T_0}
     \\Bigl[\\frac{1}{T_0}\\sum_{\\ell=1}^{T_0}
       z_{j\\ell}\\,\\mathbf{1}\\{y_{1\\ell} \\le y_{1t}\\}\\Bigr]^2 .

It is a model-free measure of dependence between donor :math:`j` and the treated
outcome: a donor whose (standardized) path co-moves -- or anti-moves, since the
inner term is squared and SRC allows negative :math:`\\theta_j` -- with the
treated's ordering scores high; a flat / noise donor scores near zero.

Note on the paper's Eq. 22. As printed, Eq. 22 carries the inner term as
:math:`Y_{jt}` (independent of the summation index :math:`\\ell`), which would
factor out of the inner sum; we read it as the cited-form :math:`z_{j\\ell}`
(Zhu et al. 2011) and standardize the donor paths, as SIRS requires for a
scale-free ranking.
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Optional

import numpy as np


def sirs_scores(donors: np.ndarray, treated: np.ndarray) -> np.ndarray:
    """SIRS marginal-utility score for each donor over the pre-period.

    Parameters
    ----------
    donors : np.ndarray, shape (T0, J)
        Donor pre-period outcome paths (one column per donor).
    treated : np.ndarray, shape (T0,)
        Treated unit's pre-period outcome path.

    Returns
    -------
    np.ndarray, shape (J,)
        Non-negative SIRS scores; higher means stronger dependence on the
        treated outcome. A zero-variance (constant) donor scores exactly zero.
    """
    D = np.asarray(donors, dtype=float)
    y = np.asarray(treated, dtype=float).ravel()
    T0 = y.shape[0]
    sd = D.std(axis=0, ddof=0)
    Z = (D - D.mean(axis=0)) / np.where(sd > 0, sd, 1.0)
    Z[:, sd == 0] = 0.0                                   # constant donor -> score 0
    ind = (y[None, :] <= y[:, None]).astype(float)        # ind[t, l] = 1{y_l <= y_t}
    cond = (ind @ Z) / T0                                 # (T0, J) inner conditional means
    return (cond ** 2).mean(axis=0)


def screen_count(T0: int) -> int:
    """Number of active donors to keep: ``min(floor(T0 / log(T0/2)), T0 - 1)``.

    The floor is the paper's rule (Zhu et al. 2011, adopted by Zhu 2023). The
    ``T0 - 1`` cap is a well-posedness guard beyond the paper: it guarantees the
    screened pool is strictly smaller than the pre-period length, so the Cp fit
    is never degenerate. In the paper's regime (large ``T0``) the cap never
    binds; it only tames pathologically small ``T0``.
    """
    T0 = int(T0)
    if T0 <= 2:
        return max(1, T0 - 1)
    denom = math.log(T0 / 2.0)
    k = int(np.floor(T0 / denom)) if denom > 0 else T0 - 1
    return int(max(1, min(k, T0 - 1)))


def screen_donors(
    donors: np.ndarray,
    treated: np.ndarray,
    *,
    n_screen: Optional[int] = None,
) -> np.ndarray:
    """Return the indices of the donors to keep, ranked by SIRS.

    Keeps the top ``k`` donors, where ``k = n_screen`` if given else
    :func:`screen_count`. If ``k >= J`` the whole pool is kept (a no-op, so
    screening a pool already smaller than ``k`` changes nothing). Returned
    indices are sorted ascending for a stable downstream donor ordering.
    """
    D = np.asarray(donors, dtype=float)
    T0, J = D.shape
    k = int(n_screen) if n_screen is not None else screen_count(T0)
    k = max(1, min(k, J))
    if k >= J:
        return np.arange(J)
    scores = sirs_scores(D, treated)
    keep = np.argsort(scores, kind="stable")[::-1][:k]
    return np.sort(keep)


def screen_inputs(inputs, keep: np.ndarray):
    """Return a copy of ``SRCInputs`` restricted to the kept donor columns."""
    keep = np.asarray(keep, dtype=int)
    cov_donors = (inputs.cov_donors[:, keep]
                  if inputs.cov_donors is not None else None)
    return replace(
        inputs,
        Y_donors=inputs.Y_donors[:, keep],
        donor_labels=tuple(np.asarray(inputs.donor_labels, dtype=object)[keep]),
        cov_donors=cov_donors,
        J=int(keep.size),
    )
