r"""Inference for Forward Difference-in-Differences (Li 2023).

Li (2023) derives a closed-form variance for the difference-in-differences
ATT estimator. Throughout this module ``T1`` is the pre-period length and
``T2`` the post-period, which is Li's notation; the function parameters are
named ``pre_periods`` and ``post_periods``, so no reader has to hold the
mapping in their head.

Writing the pre-treatment residuals of the treated unit against its
difference-in-differences fit as ``e_t``, and ``omega_2 = mean(e_t^2)`` for
their mean square, the post-period average treatment effect has asymptotic
variance

    omega_1 = (T2 / T1) * omega_2,
    Var(ATT) = (omega_1 + omega_2) / T2,

so the standard error is ``sqrt(omega_1 + omega_2) / sqrt(T2)``. The
``omega_1`` term prices in the error from estimating the level shift on
``T1`` pre-periods, which the post-period average inherits.

The standardised ATT
--------------------

It is the estimate over that standard error, ``att / se``. Proposition 2.1
writes it as ``sqrt(T2) * ATT / sqrt(omega_1 + omega_2)``, which is the same
number, and Li's own replication code computes it that way
(``FDID_Matlab.m`` line 45: ``ATT_std_FDID = sqrt(t2) * ATT_FDID /
std_Omega_hat_FDID``, annotated "it is N(0,1) under H0, ATT=0"). Her
confidence interval on line 49 is ``ATT +/- 1.96 * std_Omega_hat_FDID /
sqrt(t2)``, so her standard error is the one this module returns.
:func:`mlsynth.utils.effectutils.standardized_att` computes the same
statistic for the library at large.

That identity is also what makes the returned p-value and the returned
``satt`` one statement: the p-value is the two-sided normal tail of
``att / se``, so it is the tail of ``satt``.

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

from ..groupfit import GroupSums, prediction_variance, unscaled_cov


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
        Standardised ATT, ``att / se`` -- standard normal under the null
        of no effect (Proposition 2.1).

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
    satt = att / se
    return float(se), ci, float(p_value), float(satt)


def adid_inference(
    att: float,
    pre_residuals: np.ndarray,
    sums: GroupSums,
    pre_regressor: np.ndarray,
    post_regressor_mean: float,
    pre_periods: int,
    post_periods: int,
    method: str = "analytic",
    lrvar_lag: Optional[int] = None,
) -> Tuple[float, Tuple[float, float], float, float]:
    """Li and Van den Bulte (2022) Appendix A.1, the ADID variance.

    Their ``Sigma = Sigma_1 + Sigma_2`` where ``Sigma_1 = (T2/T1) B V B'``
    carries the pre-period estimation error and ``Sigma_2`` the post-period
    idiosyncratic error, with ``B = T2^-1 sum_post x_t' Psi^-1`` and
    ``Psi = T1^-1 sum_pre x_t x_t'``. Proposition 3.1 then gives
    ``sqrt(T2) (ATT - Delta) / sqrt(Sigma) -> N(0, 1)``, so the standard error
    of the ATT is ``sqrt(Sigma / T2)``.

    This is not :func:`did_inference` with an extra term. DID estimates one
    parameter and its ``Sigma_1`` collapses to ``(T2/T1) sigma^2``; ADID
    estimates two, and the design enters through ``B``, so a panel where the
    post-period control average sits far from its pre-period mean is penalised
    for the extrapolation.

    Parameters
    ----------
    att : float
        The ADID ATT.
    pre_residuals : np.ndarray
        Pre-period residuals ``y_1t - x_t' delta``, shape ``(T1,)``.
    sums : GroupSums
        The pre-period regression's sufficient statistics, from
        :mod:`mlsynth.utils.groupfit`. ``Psi^-1`` is read off these in closed
        form, so the design's Gram matrix is never formed or inverted.
    pre_regressor : np.ndarray
        The donor average over the pre-period, shape ``(T1,)``. Only
        ``method="hac"`` reads it, to weight the residual cross-products by the
        design.
    post_regressor_mean : float
        The donor average's post-period mean. With the implied constant it is
        their ``eta = (1, this)``.
    pre_periods, post_periods : int
        ``T1`` and ``T2``.
    method : {"analytic", "hac"}, default "analytic"
        ``"analytic"`` takes ``V = sigma^2 Psi``, which reduces ``Sigma_1`` to
        ``(T2/T1) sigma^2 eta' Psi^-1 eta`` -- the form the authors' own script
        computes at its lines 57 to 66. ``"hac"`` estimates the residual
        autocovariances on the pre-period and prices them into both terms,
        which is their general expression with the truncation at
        ``l = O(T1^(1/4))``.

        Their Appendix A.1 gives a third form for the serially uncorrelated
        case, ``V = T1^-1 sum e_t^2 x_t x_t'``, robust to heteroskedasticity
        where ``sigma^2 Psi`` is not. That form is ``method="hac"`` with
        ``lrvar_lag=0``: the double sum collapses to its diagonal, so the three
        branches are reachable as analytic, hac at lag zero, and hac at a
        positive lag. The two uncorrelated forms coincide only under
        homoskedasticity, and on a heteroskedastic pre-period they do not.
    lrvar_lag : int, optional
        Truncation lag for ``method="hac"``; defaults to :func:`hac_lag`.

    Returns
    -------
    se : float
        Standard error of the ATT (``nan`` if undefined).
    ci : tuple of float
        ``(lower, upper)`` 95% confidence interval.
    p_value : float
        Two-sided p-value.
    satt : float
        ``sqrt(T2) ATT / sqrt(Sigma)``, standard normal under the null.

    Raises
    ------
    ValueError
        If ``method`` is not one of :data:`INFERENCE_METHODS`, or
        ``lrvar_lag`` is negative.
    MlsynthEstimationError
        If ``sums`` does not identify a slope. :func:`adid_from_mean` refuses
        such a design on a stricter, relative threshold before reaching here, so
        the estimator cannot raise this.
    """
    if method not in INFERENCE_METHODS:
        raise ValueError(
            f"method must be one of {INFERENCE_METHODS}; got {method!r}."
        )
    if lrvar_lag is not None and lrvar_lag < 0:
        raise ValueError(f"lrvar_lag must be non-negative; got {lrvar_lag}.")
    if pre_periods <= 0 or post_periods <= 0:
        return np.nan, (np.nan, np.nan), np.nan, np.nan

    resid = np.asarray(pre_residuals, dtype=float).ravel()
    # Psi = X'X / T1, so Psi^-1 = T1 (X'X)^-1 and the shared closed form about
    # S_xx supplies it. The two branches want different things from it: the
    # analytic one needs only the scalar eta' Psi^-1 eta, which is T1 times the
    # prediction variance at the post-period mean, and never forms the matrix;
    # the HAC one needs the vector Psi^-1 eta to contract V, so it takes the
    # covariance itself.
    if method == "analytic":
        sigma2 = float(np.mean(resid ** 2))
        omega1 = sigma2 * pre_periods * prediction_variance(
            sums, post_regressor_mean)
        omega2 = sigma2
    else:
        lag = hac_lag(pre_periods, post_periods) if lrvar_lag is None else lrvar_lag
        x = np.asarray(pre_regressor, dtype=float).ravel()
        X = np.column_stack([np.ones(x.size), x])
        # V = T1^-1 sum_{|s-t| <= l} e_t e_s x_t x_s', Bartlett weighted
        V = np.zeros((X.shape[1], X.shape[1]))
        for j in range(0, lag + 1):
            w = 1.0 if j == 0 else 1.0 - j / (lag + 1)
            cross = (X[j:] * (resid[j:] * resid[:resid.size - j])[:, None]).T @ X[:X.shape[0] - j]
            V += (w * (cross if j == 0 else cross + cross.T)) / pre_periods
        eta = np.array([1.0, float(post_regressor_mean)])
        b = pre_periods * (unscaled_cov(sums) @ eta)
        omega1 = float(b @ V @ b)
        # Sigma_2 = T1^-1 sum_{|s-t| <= l} e_t e_s, the residual's long-run
        # variance, which is gamma_0 alone only at lag zero.
        gamma = residual_autocovariances(resid, lag)
        omega2 = float(gamma[0] + 2.0 * sum(
            (1.0 - k / (lag + 1)) * gamma[k] for k in range(1, gamma.size)))
        omega2 = max(omega2, float(gamma[0]))

    omega = (post_periods / pre_periods) * omega1 + omega2
    if not (omega > 0):                 # pragma: no cover - a degenerate fit
        return np.nan, (np.nan, np.nan), np.nan, np.nan

    se = float(np.sqrt(omega / post_periods))
    satt = float(np.sqrt(post_periods) * att / np.sqrt(omega))
    z = float(norm.ppf(0.975))
    ci = (float(att - z * se), float(att + z * se))
    p_value = float(2.0 * (1.0 - norm.cdf(abs(satt))))
    return se, ci, p_value, satt
