"""ATE inference for rfPDA (Liu, Long & Luo 2025, Eq. 8-10).

The test statistic is the one Hsiao, Ching & Wan's framework gives,

    Z_U = sqrt(T2) * Delta_bar / sqrt(gamma_1 + gamma_2)  ->  N(0, 1),

with the long-run variance estimated by West (1997) off an MA model of the
prediction error. The paper's reason for an MA and not an autoregression is in
Section 2.3: the error sequence is short-memory and close to stationary, so a
moving average is the more suitable model at the sample sizes PDA runs at.

The two components are estimated on the two windows separately. Writing
``u_hat(s)`` for the residuals of an MA(q_s) fit to the error sequence on window
``s`` and ``theta(s)`` for its coefficients,

    gamma_1 = T2 / (T1 (T1 - q1)) * sum_t [(1 + sum theta(1)) u_hat(1)_t]^2,
    gamma_2 = 1 / (T2 - q2)       * sum_t [(1 + sum theta(2)) u_hat(2)_t]^2,

the first summed over the pre-treatment window and the second over the
post-treatment window, each dropping the first ``q_s`` residuals the MA fit
cannot form. The T2 in gamma_1 carries the pre-period term onto the scale of the
post-period mean, which is what ``sqrt(T2) * Delta_bar`` is standardised by.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..inference import normal_test

_MIN_RESID = 4  # below this an MA fit has nothing to estimate from


def _ma_fit(e: np.ndarray, q: int) -> Tuple[float, np.ndarray]:
    """``(1 + sum(theta), residuals)`` from a zero-mean MA(q) fit.

    Falls back to the raw series with unit scaling when the fit cannot be formed
    -- too few observations, or a degenerate (constant, near-zero) error -- which
    leaves the long-run variance at the series' own second moment.
    """
    e = np.asarray(e, dtype=float).ravel()
    if q <= 0 or e.size < max(_MIN_RESID, 2 * q + 2) or not np.isfinite(e).all() \
            or np.allclose(e, e[0]):
        return 1.0, e
    try:
        from statsmodels.tsa.arima.model import ARIMA
        fit = ARIMA(e, order=(0, 0, q), trend="n").fit()
        theta = np.asarray(fit.maparams, dtype=float)
        resid = np.asarray(fit.resid, dtype=float)
        # A converged fit carrying a non-finite parameter: statsmodels raises
        # instead of returning one, so the except below is what fires in
        # practice and this is the belt to its braces.
        if not (np.isfinite(theta).all() and np.isfinite(resid).all()):  # pragma: no cover
            return 1.0, e
        return 1.0 + float(np.sum(theta)), resid
    except Exception:  # pragma: no cover - statsmodels convergence failure
        return 1.0, e


def west_lrvar(error_pre: np.ndarray, error_post: np.ndarray,
               T1: Optional[int] = None, T2: Optional[int] = None,
               q1: int = 1, q2: int = 1) -> dict:
    """The two long-run-variance components of Eq. (9) and (10).

    ``T1`` and ``T2`` are the scaling window lengths; they default to the two
    arrays' own lengths and are taken separately so a caller can reproduce the
    reference's scaling exactly.
    """
    error_pre = np.asarray(error_pre, dtype=float).ravel()
    error_post = np.asarray(error_post, dtype=float).ravel()
    T1 = int(error_pre.size if T1 is None else T1)
    T2 = int(error_post.size if T2 is None else T2)

    scale1, resid1 = _ma_fit(error_pre, q1)
    scale2, resid2 = _ma_fit(error_post, q2)
    drop1 = q1 if resid1.size > q1 else 0
    drop2 = q2 if resid2.size > q2 else 0

    denom1 = T1 * max(T1 - q1, 1)
    before = (T2 / denom1) * float(np.sum((scale1 * resid1[drop1:]) ** 2))
    after = float(np.sum((scale2 * resid2[drop2:]) ** 2)) / max(T2 - q2, 1)
    return {"before": before, "after": after, "q1": int(q1), "q2": int(q2)}


def rf_ate_inference(
    y: np.ndarray, counterfactual: np.ndarray, T0: int, alpha: float = 0.05,
    q1: int = 1, q2: int = 1,
) -> Tuple[float, float, Tuple[float, float], float]:
    """Return ``(att, se, ci, p_value)`` for the rfPDA ATE."""
    gap = np.asarray(y, dtype=float) - np.asarray(counterfactual, dtype=float)
    post = gap[T0:]
    T2 = int(post.size)
    att = float(np.mean(post))
    lrv = west_lrvar(gap[:T0], post, T1=int(T0), T2=T2, q1=q1, q2=q2)
    total = lrv["before"] + lrv["after"]
    # Z = sqrt(T2) * att / sqrt(total), so the standard error of the mean is
    # sqrt(total / T2) and the shared normal_test reads it unchanged.
    se = float(np.sqrt(total / T2)) if total > 0 else 0.0
    if not np.isfinite(se) or se <= 0.0:
        return att, 0.0, (att, att), 1.0
    p_value, ci = normal_test(att, se, alpha)
    return att, se, ci, p_value
