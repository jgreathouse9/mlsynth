"""ATE inference for L2-relaxation PDA (Shi & Wang 2024, Theorem 3).

Single treated unit. With the post-period estimated effects
``Delta_hat_t = y_t - y_hat_t`` (``t in T2``) and the pre-period prediction
residuals ``e_t = y_t - y_hat_t`` (``t in T1``), the ATE ``Delta_bar`` is
asymptotically normal,

    Z_hat = Delta_bar / sqrt( rho_hat_(1)^2 / T1 + rho_hat_(2)^2 / T2 ) -> N(0, 1),

where ``rho_hat_(1)^2`` is the HAC long-run variance of the pre-period
residuals and ``rho_hat_(2)^2`` is the HAC long-run variance of the de-meaned
post-period effects. Both estimation uncertainty (pre) and post-period noise
contribute.

The kernel is Bartlett; Theorem 3 states the estimator with the uniform kernel,
an unweighted two-sided sum ``sum_{l=-h}^{h}``, which the authors adopt "for
simplicity" while noting the result "is compatible with the Bartlett kernel
(Newey & West 1987)". The substitution is theirs to sanction and it is not free:
on Hong Kong it moves the t-statistic from 7.69 to 7.83. Bartlett is what makes
the estimate non-negative without a clamp, which is why it is the default here.

The autocovariances divide by the length of the index set, matching the paper's
``E_S(x_t) = |S|^{-1} sum_{t in S} x_t``. The truncation lag is Newey & West
(1994), which is the rule the paper points to.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..inference import hac_lrv, normal_test


def l2_ate_inference(
    y: np.ndarray, counterfactual: np.ndarray, T0: int, alpha: float = 0.05,
) -> Tuple[float, float, Tuple[float, float], float]:
    """Return ``(att, se, ci, p_value)`` for the L2-relaxation ATE."""
    gap = np.asarray(y, dtype=float) - np.asarray(counterfactual, dtype=float)
    pre_resid = gap[:T0]
    post_effect = gap[T0:]
    T1, T2 = T0, gap.shape[0] - T0

    att = float(np.mean(post_effect))
    rho1_sq = hac_lrv(pre_resid)                      # pre-period prediction residuals
    rho2_sq = hac_lrv(post_effect - att)              # de-meaned post-period effects
    se = float(np.sqrt(rho1_sq / T1 + rho2_sq / T2))
    p_value, ci = normal_test(att, se, alpha)
    return att, se, ci, p_value
