"""ATE inference for the HCW best-subset panel data approach.

HCW (2012) construct the counterfactual from pre-treatment data only and test
the average post-intervention effect with a heteroscedasticity-and-autocorre-
lation-consistent (HAC / Newey-West) long-run variance (their Lemma 4). Because
the donor set is chosen on the pre-period and the pre/post windows are
asymptotically independent under weak dependence, the conditional t-statistic on
the post-period effect mean is valid -- the same argument the forward-selected
PDA uses -- so this reuses mlsynth's shared PDA long-run-variance machinery.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..inference import fspda_lrvar_lag, hac_lrv, lrvar_prewhite_nw, normal_test


def hcw_ate_inference(
    y: np.ndarray, counterfactual: np.ndarray, T0: int, alpha: float = 0.05,
    lrvar_lag: Optional[int] = None,
) -> Tuple[float, float, Tuple[float, float], float]:
    """Return ``(att, se, ci, p_value)`` for the HCW best-subset PDA ATE.

    The ATT is the mean post-period gap; its standard error is the square root of
    the HAC long-run variance of that mean -- the prewhitened Newey-West
    estimator by default, or a fixed-lag Bartlett kernel when ``lrvar_lag`` is
    supplied (HCW's Newey-West, Lemma 4).
    """
    gap = np.asarray(y, dtype=float) - np.asarray(counterfactual, dtype=float)
    post_effect = gap[T0:]
    T2 = post_effect.shape[0]
    att = float(np.mean(post_effect))
    if lrvar_lag is None:
        se = float(np.sqrt(lrvar_prewhite_nw(post_effect)))
    else:
        lag = fspda_lrvar_lag(T2, lrvar_lag)
        se = float(np.sqrt(hac_lrv(post_effect - att, lag=lag) / T2))
    p_value, ci = normal_test(att, se, alpha)
    return att, se, ci, p_value
