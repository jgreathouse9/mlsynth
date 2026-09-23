"""The split-conformal order statistic -- the one calibration primitive.

Every conformal band in mlsynth ultimately asks the same question of a set of
conformity scores: how far out must the band reach to cover ``1 - alpha`` of them
in finite samples? That is one order statistic, and it lives here so the answer
cannot drift between estimators.
"""

from __future__ import annotations

import numpy as np


def split_conformal_quantile(residuals, alpha: float = 0.05) -> float:
    r"""Split-conformal prediction-band half-width (Chernozhukov, Wuthrich & Zhu 2021).

    Returns ``q``, the constant half-width of the symmetric prediction band
    ``counterfactual +/- q``: the ``ceil((n+1)(1-alpha))``-th order statistic of
    the absolute pre-period residuals (gaps). Under exchangeability of the
    residuals this band has finite-sample :math:`(1-\alpha)` coverage. When
    ``n < ceil(1/alpha) - 1`` the required order statistic does not exist and
    ``q`` is ``+inf`` (an uninformative band).

    This is the constant-width "split" construction used by R ``Synth``'s
    ``synth_inference(method = "conformal")`` (Hainmueller's j-hai/Synth), as
    distinct from the test-inversion conformal band (which widens over the
    post-period).

    Parameters
    ----------
    residuals : array-like
        Conformity scores -- classically the pre-treatment gaps (treated minus
        synthetic), one per pre-period; for a cumulative band, the centred
        out-of-sample block sums.
    alpha : float
        Miscoverage level in ``(0, 1)``; the band targets ``1 - alpha`` coverage.
    """
    r = np.sort(np.abs(np.asarray(residuals, dtype=float)))
    n = r.size
    if n == 0:
        return float("inf")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    return float(r[k - 1]) if k <= n else float("inf")
