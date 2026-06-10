"""Post-selection ATE inference for forward-selected PDA (Shi & Huang 2023, Eq. 4).

Because forward selection uses pre-treatment data only and the pre/post periods
become asymptotically independent under weak dependence, the naive conditional
t-statistic is valid:

    Z_U = Delta_bar / sqrt(lrvar_hat) -> N(0, 1),

where ``lrvar_hat`` is the long-run variance of the sample mean of the
(de-meaned) post-period treatment effects. By default mlsynth uses the
**prewhitened** Newey-West estimator (``sandwich::lrvar(..., prewhite = TRUE,
adjust = TRUE)``) that Shi & Huang use in their applications
(``app1_luxury_watch/fsPDA.R``); on serially-dependent effects this is far less
conservative than a plain Bartlett kernel. Supplying ``lrvar_lag`` instead
switches to the released ``est.fsPDA`` package's fixed-lag Bartlett estimator
(``floor(T2 ** (1/4))`` if the cap, see :func:`fspda_lrvar_lag`).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..inference import fspda_lrvar_lag, hac_lrv, lrvar_prewhite_nw, normal_test


def fs_ate_inference(
    y: np.ndarray, counterfactual: np.ndarray, T0: int, alpha: float = 0.05,
    lrvar_lag: Optional[int] = None,
) -> Tuple[float, float, Tuple[float, float], float]:
    """Return ``(att, se, ci, p_value)`` for the forward-selected PDA ATE."""
    gap = np.asarray(y, dtype=float) - np.asarray(counterfactual, dtype=float)
    post_effect = gap[T0:]
    T2 = post_effect.shape[0]
    att = float(np.mean(post_effect))
    if lrvar_lag is None:
        # Prewhitened Newey-West variance of the mean (the application default).
        se = float(np.sqrt(lrvar_prewhite_nw(post_effect)))
    else:
        # Released-package fixed-lag Bartlett: se = sqrt(lrvar_series / T2).
        lag = fspda_lrvar_lag(T2, lrvar_lag)
        se = float(np.sqrt(hac_lrv(post_effect - att, lag=lag) / T2))
    p_value, ci = normal_test(att, se, alpha)
    return att, se, ci, p_value
