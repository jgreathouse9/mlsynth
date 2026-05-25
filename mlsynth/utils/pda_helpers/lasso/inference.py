"""ATE inference for LASSO PDA (Li & Bell 2017, Theorem 3.2).

The test statistic is ``sqrt(T2) * Delta_bar / sqrt(Sigma_hat)`` with
``Sigma_hat = Sigma_hat_1 + Sigma_hat_2``:

* ``Sigma_hat_2`` -- the HAC (Newey-West) long-run variance of the post-period
  effects, the always-present post-period averaging variance;
* ``Sigma_hat_1`` -- the first-stage (pre-period estimation) variance, here the
  OLS prediction variance of the mean post-period counterfactual on the
  LASSO-selected support. Li & Bell note it is negligible when ``T1 >> T2``.

Equivalently the ATE standard error is
``SE = sqrt(var_firststage + Sigma_hat_2 / T2)``.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..inference import hac_lrv, normal_test


def lasso_ate_inference(
    y: np.ndarray, X: np.ndarray, counterfactual: np.ndarray, support: np.ndarray,
    T0: int, alpha: float = 0.05,
) -> Tuple[float, float, Tuple[float, float], float]:
    """Return ``(att, se, ci, p_value)`` for the LASSO PDA ATE."""
    gap = np.asarray(y, dtype=float) - np.asarray(counterfactual, dtype=float)
    post_effect = gap[T0:]
    T2 = post_effect.shape[0]
    att = float(np.mean(post_effect))

    # post-period averaging variance (Sigma_hat_2 / T2)
    var_post = hac_lrv(post_effect - att) / T2

    # first-stage variance: OLS prediction variance of the post-mean on the
    # LASSO-selected support (negligible when T1 >> T2)
    var_first = 0.0
    S = np.where(support)[0]
    if S.size > 0 and T0 > S.size + 1:
        Z = np.column_stack([np.ones(T0), X[:T0, S]])
        ZtZ_inv = np.linalg.pinv(Z.T @ Z)
        resid = y[:T0] - Z @ (ZtZ_inv @ (Z.T @ y[:T0]))
        sigma2 = float(resid @ resid) / (T0 - S.size - 1)
        z_post = np.concatenate([[1.0], X[T0:, S].mean(axis=0)])
        var_first = sigma2 * float(z_post @ ZtZ_inv @ z_post)

    se = float(np.sqrt(var_first + var_post))
    p_value, ci = normal_test(att, se, alpha)
    return att, se, ci, p_value
