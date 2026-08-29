r"""Inference for Forward Difference-in-Differences (Li 2023).

Li (2023) derives a closed-form variance for the difference-in-differences
ATT estimator. Writing the pre-treatment residuals of the treated unit
against its difference-in-differences fit as ``e_t``, the post-period
average treatment effect has asymptotic variance

    Var(ATT) = (omega_1 + omega_2) / T1,

where ``omega_2 = mean(e_t^2)`` is the pre-period residual variance and
``omega_1 = (T1 / T0) * omega_2`` inflates it for the post-period sample
size ``T1``. The standard error is the square root of this quantity.

Serial correlation
------------------

That formula studentises by the residual's *marginal* variance. The
estimator's sampling error is a difference of two block means,
:math:`\bar e_{\text{post}} - \bar e_{\text{pre}}`, and a block mean's
variance is governed by the residual's autocovariances. The two agree
exactly when the residual is serially uncorrelated, which Online Appendix
A's Assumptions 2(ii) and 3(i) impose. Assumption 2.1 in the main text asks
only for weak dependence, and the appendix remarks that the iid assumptions
"can be easily relaxed" to it. The estimator survives that relaxation; the
standard error does not, since nothing in ``omega_1 + omega_2`` estimates an
autocovariance. ``benchmarks/cases/fdid_serial_correlation_mc`` measures the
cost: coverage of the nominal 95% interval falls from 0.94 to 0.53 as an
AR(1) residual's coefficient goes from 0 to 0.9, with the point estimate
consistent throughout.

``method="hac"`` prices the autocovariances in. It estimates them on the
pre-period residuals -- the only stretch long enough to estimate them, and
the stretch Li already uses for ``omega_2`` -- and puts them through the
exact finite-sample variance of a block mean,

.. math::

   \operatorname{Var}(\bar e_T)
     = \frac{1}{T}\Bigl[\gamma_0
         + 2\sum_{k=1}^{\min(L,\,T-1)}\bigl(1 - \tfrac{k}{T}\bigr)\gamma_k
       \Bigr],

summed over the pre and post blocks. The Bartlett weight :math:`1 - k/T`
here is not a kernel choice: it is the exact coefficient lag :math:`k`
carries in the variance of a length-``T`` mean, so at :math:`L = T - 1`
the expression is the variance itself and not an approximation to it.
Truncation at ``L`` is the only approximation, and the sum is floored at
:math:`\gamma_0`, since truncating an alternating autocovariance sequence
can drive the sum below the iid value or negative.

The default truncation is ``min(T2 - 1, T1 // 10)``. The first term is
exhaustive, not conservative: lag ``k`` enters the post block with weight
``1 - k/T2``, which is zero at ``k = T2``. The second is the usual
one-tenth-of-sample HAC cap, on the pre-period sample that supplies the
estimates. Both bind in practice, and dropping either loses coverage: at
:math:`T_1 = 100, T_2 = 100` the uncapped ``T2 - 1`` gives 0.53 against
0.75 for the capped rule, and at :math:`T_1 = 400, T_2 = 40` a
Newey-West-style ``T1``-only rule gives 0.81 against 0.88.

References
----------
Li, K. T. (2023). Frontiers: A Simple Forward Difference-in-Differences
Method. Marketing Science, 43(2), 267-279.
https://doi.org/10.1287/mksc.2022.0212
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.stats import norm


#: Inference methods accepted by :func:`did_inference`.
INFERENCE_METHODS = ("analytic", "hac")


def residual_autocovariances(residuals: np.ndarray, max_lag: int) -> np.ndarray:
    """Sample autocovariances of the pre-period residuals, lags ``0..max_lag``.

    Parameters
    ----------
    residuals : np.ndarray
        Pre-treatment residual series, shape ``(n,)``.
    max_lag : int
        Highest lag to return. Clamped to ``n - 1``, past which no lagged
        product exists.

    Returns
    -------
    np.ndarray
        Length ``min(max_lag, n - 1) + 1``. Entry ``k`` is
        ``sum_t (e_t - ebar)(e_{t-k} - ebar) / n``, the standard divisor-``n``
        estimator, which keeps the implied spectral density non-negative.

    Raises
    ------
    ValueError
        If ``max_lag`` is negative or ``residuals`` is empty.
    """
    e = np.asarray(residuals, dtype=float).ravel()
    if e.size < 1:
        raise ValueError("residuals must hold at least one observation.")
    if max_lag < 0:
        raise ValueError(f"max_lag must be non-negative; got {max_lag}.")

    centred = e - e.mean()
    n = centred.size
    top = min(int(max_lag), n - 1)
    return np.array(
        [float(centred[k:] @ centred[: n - k]) / n for k in range(top + 1)]
    )


def block_mean_variance(autocovariances: np.ndarray, block_length: int) -> float:
    r"""Variance of the mean of a length-``block_length`` stretch.

    Parameters
    ----------
    autocovariances : np.ndarray
        ``gamma_0, ..., gamma_L``. Lags beyond ``L`` are treated as zero.
    block_length : int
        Number of periods averaged, ``T``.

    Returns
    -------
    float
        :math:`\bigl[\gamma_0 + 2\sum_{k\ge1}(1 - k/T)\gamma_k\bigr] / T`,
        floored at :math:`\gamma_0 / T`.

    Raises
    ------
    ValueError
        If ``block_length`` is not positive or the array is empty.
    """
    gamma = np.asarray(autocovariances, dtype=float).ravel()
    if gamma.size < 1:
        raise ValueError("autocovariances must hold at least gamma_0.")
    if block_length < 1:
        raise ValueError(f"block_length must be positive; got {block_length}.")

    T = int(block_length)
    top = min(gamma.size - 1, T - 1)
    total = gamma[0]
    for k in range(1, top + 1):
        total += 2.0 * (1.0 - k / T) * gamma[k]
    # Truncating an alternating sequence can drive the sum below the iid
    # value, or negative; neither is a usable variance.
    return float(max(total, gamma[0]) / T)


def hac_lag(pre_periods: int, post_periods: int) -> int:
    """Default truncation lag, ``min(T2 - 1, T1 // 10)``.

    Parameters
    ----------
    pre_periods : int
        Pre-treatment period count ``T0``, the sample the autocovariances are
        estimated on.
    post_periods : int
        Post-treatment period count ``T1``.

    Returns
    -------
    int
        A non-negative lag. See the module docstring for why both terms bind.
    """
    return int(max(0, min(int(post_periods) - 1, int(pre_periods) // 10)))


def did_inference(
    att: float,
    pre_residuals: np.ndarray,
    pre_periods: int,
    post_periods: int,
    method: str = "analytic",
    lrvar_lag: Optional[int] = None,
) -> Tuple[float, Tuple[float, float], float, float]:
    """Compute the FDID standard error, 95% CI, p-value, and SATT.

    Parameters
    ----------
    att : float
        Estimated average treatment effect on the treated.
    pre_residuals : np.ndarray
        Pre-treatment residuals of the treated unit against its
        difference-in-differences fit, shape ``(T0,)``.
    pre_periods : int
        Number of pre-treatment periods ``T0``.
    post_periods : int
        Number of post-treatment periods ``T1``.
    method : {"analytic", "hac"}, default "analytic"
        ``"analytic"`` is Li (2023) Proposition 2.1, exact under a serially
        uncorrelated residual. ``"hac"`` estimates the residual's
        autocovariances on the pre-period and prices them into the variance
        of both block means; see the module docstring.
    lrvar_lag : int, optional
        Truncation lag for ``method="hac"``. Defaults to
        :func:`hac_lag`. Ignored by the analytic path.

    Returns
    -------
    se : float
        Standard error of the ATT (``nan`` if undefined).
    ci : tuple of float
        ``(lower, upper)`` 95% confidence interval.
    p_value : float
        Two-sided p-value for the ATT.
    satt : float
        Standardised ATT (``att / se * sqrt(T1)``).

    Raises
    ------
    ValueError
        If ``method`` is not one of :data:`INFERENCE_METHODS`, or
        ``lrvar_lag`` is negative.
    """
    if method not in INFERENCE_METHODS:
        raise ValueError(
            f"method must be one of {INFERENCE_METHODS}; got {method!r}."
        )
    if lrvar_lag is not None and lrvar_lag < 0:
        raise ValueError(f"lrvar_lag must be non-negative; got {lrvar_lag}.")

    if pre_periods <= 0 or post_periods <= 0:
        return np.nan, (np.nan, np.nan), np.nan, np.nan

    omega2 = float(np.mean(pre_residuals ** 2))
    if method == "analytic":
        omega1 = (post_periods / pre_periods) * omega2
        se = np.sqrt(omega1 + omega2) / np.sqrt(post_periods)
    else:
        lag = hac_lag(pre_periods, post_periods) if lrvar_lag is None else lrvar_lag
        gamma = residual_autocovariances(pre_residuals, lag)
        se = np.sqrt(
            block_mean_variance(gamma, pre_periods)
            + block_mean_variance(gamma, post_periods)
        )

    if not (se > 0):
        return float(se), (np.nan, np.nan), np.nan, np.nan

    z = norm.ppf(0.975)
    ci = (att - z * se, att + z * se)
    p_value = 2.0 * (1.0 - norm.cdf(np.abs(att / se)))
    satt = att / se * np.sqrt(post_periods)
    return float(se), ci, float(p_value), float(satt)
