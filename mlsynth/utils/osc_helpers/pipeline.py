"""OSC orchestrator: regularized nuisance -> orthogonalized ATT -> Series-HAC
fixed-smoothing inference. Mirrors the reference ``OrthoganilzedSCE`` end to end.
"""
from __future__ import annotations

import numpy as np

from .regularized import estimate_delta, estimate_eta
from .orthogonal import orthogonalized_att
from .serieshac import cpe_optimal_h, series_hac_variance, ttest_pvalue, ttest_ci


def orthogonalized_sce(pre_y0, pre_yj, Z, post_y0, post_yj, *,
                       alpha: float = 0.05, beta0: float = 0.0,
                       include_constant: bool = True):
    """Run the full Orthogonalized Synthetic Control estimate + inference.

    Returns ``dict`` with ``beta``, ``pvalue``, ``ci`` (lo, hi), ``df`` (smoothing
    K), ``control_weights`` (delta), ``instrument_weights`` (eta).
    """
    T0 = np.atleast_2d(np.asarray(pre_yj, float)).shape[1]
    T1 = np.atleast_2d(np.asarray(post_yj, float)).shape[1]

    delta = estimate_delta(pre_y0, pre_yj, Z, scaled=True,
                           include_constant=include_constant, T1=T1)["delta"]
    eta = estimate_eta(pre_y0, pre_yj, post_y0, post_yj, Z, scaled=True,
                       include_constant=include_constant)["eta"]
    o = orthogonalized_att(pre_y0, pre_yj, Z, post_y0, post_yj, delta, eta,
                           include_constant=include_constant)
    beta, preg, postg = o["beta"], o["preg"], o["postg"]

    h = cpe_optimal_h(preg, p=1, sig=0.05)
    V = series_hac_variance(preg, postg, eta, h)
    n = min(T0, T1)
    return {
        "beta": beta,
        "pvalue": ttest_pvalue(beta, V, h, n, beta0=beta0),
        "ci": ttest_ci(beta, V, h, alpha),
        "df": h,
        "control_weights": delta,
        "instrument_weights": eta,
    }
