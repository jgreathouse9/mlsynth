"""Canonical two-period difference in differences, the paper's DID column.

Table 1's DID row depends only on the data-generating process, not on any of the
paper's machinery, so it is the cheapest available check that the DGP port is
right before the estimator is trusted.
"""

from __future__ import annotations

import numpy as np

Z95 = 1.959963984540054


def did(Y: np.ndarray, D: np.ndarray, T0: int, horizon: int = 0) -> tuple[float, float]:
    """2x2 DiD for the ATT at post period ``T0 + horizon``, baselined at ``T0 - 1``.

    Returns the point estimate and the two-sample standard error.
    """
    delta = Y[:, T0 + horizon] - Y[:, T0 - 1]
    treated, control = delta[D == 1], delta[D == 0]
    n1, n0 = treated.size, control.size
    estimate = treated.mean() - control.mean()
    se = np.sqrt(treated.var(ddof=1) / n1 + control.var(ddof=1) / n0)
    return float(estimate), float(se)
