"""Weak-dependence ATE inference for the RESCM counterfactual.

Once donor weights are fixed, the post-treatment gap ``d_t = y_t - yhat_t`` has
mean equal to the ATE. Under weak dependence (Li 2020, extended to dense weights
by Wang, Xing & Ye 2025), the ATE is asymptotically normal,

    Z = ATE / sqrt( rho1^2 / T1 + rho2^2 / T2 ) -> N(0, 1),

with ``rho1^2`` the HAC long-run variance of the **pre-period prediction
residuals** (which carries the first-stage weight-estimation uncertainty) and
``rho2^2`` the HAC long-run variance of the de-meaned **post-period effects**.

The pre-period term is essential for dense penalized/relaxed weights: unlike
forward selection (whose sample splitting makes the pre/post periods
asymptotically independent and lets a post-only variance suffice), the dense
estimators reuse the whole pre-window to fit the weights, so ignoring ``rho1``
severely understates the standard error when ``N`` is large relative to ``T1``.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..pda_helpers.inference import hac_lrv, normal_test


def ate_inference(
    gap: np.ndarray, T0: int, alpha: float = 0.05,
) -> Tuple[float, float, Tuple[float, float], float]:
    """Return ``(att, se, ci, p_value)`` for the post-period gap mean.

    Uses the Li (2020) two-term long-run variance: pre-period residual LRV
    (estimation uncertainty) plus de-meaned post-period effect LRV.
    """
    gap = np.asarray(gap, dtype=float)
    pre_resid = gap[:T0]
    post = gap[T0:]
    T1, T2 = T0, post.shape[0]
    att = float(np.mean(post))
    if T2 < 2 or T1 < 2:
        return att, float("nan"), (float("nan"), float("nan")), float("nan")
    lag2 = int(np.floor(T2 ** 0.25))
    rho1_sq = hac_lrv(pre_resid)                       # first-stage estimation uncertainty
    rho2_sq = hac_lrv(post - att, lag=lag2)            # post-period effect noise
    se = float(np.sqrt(rho1_sq / T1 + rho2_sq / T2))
    p_value, ci = normal_test(att, se, alpha)
    return att, se, ci, p_value
