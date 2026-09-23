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

Supplying ``lrvar_lag`` asks for a different paper's test. Shi & Huang's
``fsPDA`` studentizes by the post-period long-run variance alone, at a
truncation lag the caller fixes -- ``Z = ATE / sqrt(lrvar(d, h) / T2)`` in their
``lasso.BIC`` -- with no first-stage term. The two conventions are far apart when
``T1`` is comparable to ``T2``: on their Table 1 design the first-stage term is
44-46% of the total variance, so Li & Bell's standard error runs about a third
larger and the test under-rejects against their cells.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..inference import fspda_lrvar_lag, hac_lrv, normal_test


def lasso_ate_inference(
    y: np.ndarray, X: np.ndarray, counterfactual: np.ndarray, support: np.ndarray,
    T0: int, alpha: float = 0.05, lrvar_lag: Optional[int] = None,
) -> Tuple[float, float, Tuple[float, float], float]:
    """Return ``(att, se, ci, p_value)`` for the LASSO PDA ATE.

    Parameters
    ----------
    lrvar_lag : int, optional
        ``None`` (default) gives Li & Bell's two-component variance. An integer
        gives the released ``fsPDA`` package's t-test: a Bartlett long-run
        variance of the post-period effects at that truncation lag, and nothing
        else. It must be a non-negative integer no larger than
        ``floor(sqrt(T2))``.

    Raises
    ------
    ValueError
        If ``lrvar_lag`` is negative or above that cap.
    """
    gap = np.asarray(y, dtype=float) - np.asarray(counterfactual, dtype=float)
    post_effect = gap[T0:]
    T2 = post_effect.shape[0]
    att = float(np.mean(post_effect))

    if lrvar_lag is not None:
        lag = fspda_lrvar_lag(T2, lrvar_lag)
        se = float(np.sqrt(hac_lrv(post_effect - att, lag=lag) / T2))
        p_value, ci = normal_test(att, se, alpha)
        return att, se, ci, p_value

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
