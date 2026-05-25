"""Post-selection ATE inference for forward-selected PDA (Shi & Huang 2023, Eq. 4).

Because forward selection uses pre-treatment data only and the pre/post periods
become asymptotically independent under weak dependence, the naive conditional
t-statistic is valid:

    Z_U = sqrt(T2) * Delta_bar / rho_hat_tau -> N(0, 1),

where ``rho_hat_tau^2`` is the HAC long-run variance of the (de-meaned)
post-period treatment effects. No first-stage variance term is required.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..inference import hac_lrv, normal_test


def fs_ate_inference(
    y: np.ndarray, counterfactual: np.ndarray, T0: int, alpha: float = 0.05,
) -> Tuple[float, float, Tuple[float, float], float]:
    """Return ``(att, se, ci, p_value)`` for the forward-selected PDA ATE."""
    gap = np.asarray(y, dtype=float) - np.asarray(counterfactual, dtype=float)
    post_effect = gap[T0:]
    T2 = post_effect.shape[0]
    att = float(np.mean(post_effect))
    se = float(np.sqrt(hac_lrv(post_effect - att) / T2))   # post-period HAC only
    p_value, ci = normal_test(att, se, alpha)
    return att, se, ci, p_value
