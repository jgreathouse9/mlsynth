"""Fit metric and weighted-least-squares treatment effect for ISCM.

Given the all-units synthetic-control weights :math:`W`, ISCM forms

* the SC **residuals** :math:`R_{it} = Y_{it} - \\sum_j w_{ij} Y_{jt}`,
* the treatment **exposure** :math:`E_{it} = D_{it} - \\sum_j w_{ij}
  D_{jt}` (the regressor: a control unit that borrows the treated unit
  as a donor has non-zero post-period exposure :math:`-w_{i,\\text{tr}}`),
* a per-unit **fit metric** :math:`a_i` (paper eq. 14) that weights
  units by how well their synthetic control satisfies the SCM moment
  conditions in the pre-period -- asymptotically excluding units without
  a valid synthetic control,

and estimates the ATT by weighted least squares pooled over all units
(paper eq. 8 / 15):

.. math::

   \\widehat\\alpha =
     \\frac{\\sum_i a_i \\sum_{t > T_0} E_{it} R_{it}}
          {\\sum_i a_i \\sum_{t > T_0} E_{it}^2}.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

# Floor to keep the fit metric finite when a unit fits (near-)perfectly.
_EPS = 1e-12


def residuals_and_exposure(
    Y: np.ndarray, D: np.ndarray, W: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (residuals R, exposure E), each shape ``(N, T)``."""
    R = Y - W @ Y
    E = D - W @ D
    return R, E


def fit_metric(R: np.ndarray, Y: np.ndarray, T0: int) -> np.ndarray:
    """Per-unit fit weights :math:`a_i \\in (0, 1]` (paper eq. 14).

    The pre-period moment vector for unit ``i`` collects the empirical
    covariances of its SC residual with every other unit's outcomes,
    :math:`M_i^k = \\tfrac{1}{T_0}\\sum_{t \\le T_0} R_{it} Y_{kt}`. A good
    synthetic control makes these (near) zero. The metric normalises so
    the best-fitting unit gets ``1`` and poorer fits get progressively
    smaller weights:

    .. math::

       a_i = \\frac{\\min_\\ell M_\\ell' M_\\ell}{M_i' M_i}.
    """
    N = R.shape[0]
    R_pre = R[:, :T0]
    Y_pre = Y[:, :T0]
    # M[i] = (1/T0) * R_pre[i] @ Y_pre[k]^T for all k (k == i contributes 0-ish).
    M = (R_pre @ Y_pre.T) / T0            # (N, N)
    np.fill_diagonal(M, 0.0)              # drop the self moment
    mtm = np.sum(M ** 2, axis=1)          # (N,) -- M_i' M_i
    min_mtm = float(np.min(mtm))
    a = min_mtm / (mtm + _EPS)
    return np.clip(a, 0.0, 1.0)


def weighted_att(
    R: np.ndarray, E: np.ndarray, a: np.ndarray, T0: int
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Aggregate ATT and per-unit decomposition (paper eq. 15).

    Returns
    -------
    att : float
        WLS aggregate effect.
    unit_att : np.ndarray, shape ``(N,)``
        Per-unit estimate :math:`\\widehat\\alpha_i = \\sum_{t>T_0} E_{it}
        R_{it} / \\sum_{t>T_0} E_{it}^2`; ``NaN`` outside the contributing
        set (units with zero post-period exposure).
    contribution : np.ndarray, shape ``(N,)``
        Per-unit share :math:`v_i` of the aggregate effect; sums to one
        over the contributing set.
    """
    N = R.shape[0]
    R_post = R[:, T0:]
    E_post = E[:, T0:]
    se_sq = np.sum(E_post ** 2, axis=1)          # (N,) sum_t E_it^2
    ser = np.sum(E_post * R_post, axis=1)        # (N,) sum_t E_it R_it

    # Contributing set C: units with meaningful treatment exposure. A
    # relative floor excludes units whose exposure is only numerically
    # non-zero (tiny donor weight on a treated unit) -- their per-unit
    # ratio R/E is ill-conditioned and would pollute inference, though
    # their v_i = a_i * E^2 weight in the aggregate is already negligible.
    max_se_sq = float(se_sq.max()) if se_sq.size else 0.0
    contributing = se_sq > max(_EPS, 1e-6 * max_se_sq)
    unit_att = np.full(N, np.nan)
    unit_att[contributing] = ser[contributing] / se_sq[contributing]

    num = float(np.sum(a * ser))
    den = float(np.sum(a * se_sq))
    att = num / den if abs(den) > _EPS else np.nan

    weights_v = a * se_sq
    total_v = float(np.sum(weights_v))
    contribution = np.zeros(N)
    if total_v > _EPS:
        contribution = weights_v / total_v
    return att, unit_att, contribution
