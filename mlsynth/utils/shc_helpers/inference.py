"""Inference for the Synthetic Historical Control estimator.

Primary inference is the conformal permutation test of Chen, Yang & Yang
(2024, footnote 21) -- their application of Chernozhukov, Wuthrich & Zhu
(2021) to SHC -- computed by :func:`mlsynth.utils.inferutils.shc_conformal_test`.
Andrews-Genton conformal prediction bands are computed alongside for the
plot.
"""

from __future__ import annotations

from typing import Sequence

from scipy.optimize import lsq_linear
import numpy as np

from .structures import SHCDesign, SHCInference, SHCInputs


def ag_conformal(
    actual_outcomes_pre_treatment: np.ndarray,
    predicted_outcomes_pre_treatment: np.ndarray,
    predicted_outcomes_post_treatment: np.ndarray,
    miscoverage_rate: float = 0.1,
    pad_value: Any = np.nan,
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct agnostic conformal prediction intervals.

    Generates prediction intervals for post-treatment predictions based on
    pre-treatment residuals and assuming residuals follow a distribution
    for which sub-Gaussian concentration bounds apply. The interval width
    is determined by the variability of pre-treatment residuals and the
    desired coverage level `miscoverage_rate`.

    Parameters
    ----------
    actual_outcomes_pre_treatment : np.ndarray
        Actual pre-treatment outcomes. Shape (T_pre,), where T_pre is the
        number of pre-treatment periods.
    predicted_outcomes_pre_treatment : np.ndarray
        Predicted pre-treatment outcomes, corresponding to `actual_outcomes_pre_treatment`.
        Shape (T_pre,). Must have the same length as `actual_outcomes_pre_treatment`.
    predicted_outcomes_post_treatment : np.ndarray
        Predicted post-treatment outcomes for which intervals are desired.
        Shape (T_post,), where T_post is the number of post-treatment periods.
    miscoverage_rate : float, optional
        Desired miscoverage level (e.g., 0.1 for 90% prediction intervals,
        meaning (1-miscoverage_rate) coverage). Must be between 0 and 1. Default is 0.1.
    pad_value : Any, optional
        Value used to pad the pre-treatment portion of the returned interval
        arrays. This makes the output arrays align with a full time series
        (pre- and post-treatment). Default is `np.nan`.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - lower_bounds_full_series : np.ndarray
            Lower bounds of the prediction intervals. Shape (T_pre + T_post,).
            The first T_pre elements are filled with `pad_value`.
        - upper_bounds_full_series : np.ndarray
            Upper bounds of the prediction intervals. Shape (T_pre + T_post,).
            The first T_pre elements are filled with `pad_value`.

    Raises
    ------
    MlsynthDataError
        If `actual_outcomes_pre_treatment` and `predicted_outcomes_pre_treatment` have different lengths.
        If `actual_outcomes_pre_treatment` is empty.
    MlsynthConfigError
        If `miscoverage_rate` is not between 0 and 1.

    Examples
    --------
    >>> actual_outcomes_pre_treatment_ex = np.array([10, 12, 11, 13, 12])
    >>> predicted_outcomes_pre_treatment_ex = np.array([10.5, 11.5, 10.5, 12.5, 11.5])
    >>> predicted_outcomes_post_treatment_ex = np.array([14, 15, 14.5])
    >>> miscoverage_rate_ex = 0.1 # For 90% prediction intervals
    >>> lower_b, upper_b = ag_conformal(
    ...     actual_outcomes_pre_treatment_ex, predicted_outcomes_pre_treatment_ex,
    ...     predicted_outcomes_post_treatment_ex, miscoverage_rate=miscoverage_rate_ex
    ... )
    >>> print("Lower bounds:", np.round(lower_b, 2))
    Lower bounds: [  nan   nan   nan   nan   nan 12.01 13.01 12.51]
    >>> print("Upper bounds:", np.round(upper_b, 2))
    Upper bounds: [  nan   nan   nan   nan   nan 15.99 16.99 16.49]

    >>> # Example with empty pre-treatment data (raises MlsynthDataError)
    >>> try:
    ...     ag_conformal(np.array([]), np.array([]), predicted_outcomes_post_treatment_ex)
    ... except MlsynthDataError as e:
    ...     print(e)
    Pre-treatment arrays cannot be empty.

    >>> # Example with invalid miscoverage_rate (raises MlsynthConfigError)
    >>> try:
    ...     ag_conformal(actual_outcomes_pre_treatment_ex, predicted_outcomes_pre_treatment_ex,
    ...                  predicted_outcomes_post_treatment_ex, miscoverage_rate=1.1)
    ... except MlsynthConfigError as e:
    ...     print(e)
    miscoverage_rate must be between 0 and 1.
    """
    # --- Input Validation ---
    if len(actual_outcomes_pre_treatment) != len(predicted_outcomes_pre_treatment):
        raise MlsynthDataError("actual_outcomes_pre_treatment and predicted_outcomes_pre_treatment must have the same length.")
    if len(actual_outcomes_pre_treatment) == 0: # Check if pre-treatment data is empty
        raise MlsynthDataError("Pre-treatment arrays cannot be empty.")
    if not (0 < miscoverage_rate < 1): # miscoverage_rate (alpha) must be in (0, 1)
        raise MlsynthConfigError("miscoverage_rate must be between 0 and 1.")

    # --- Conformal Interval Calculation ---
    # 1. Calculate pre-treatment residuals
    residuals = actual_outcomes_pre_treatment - predicted_outcomes_pre_treatment
    
    # 2. Calculate mean and variance of these residuals
    mean_residuals = np.mean(residuals)
    # Use ddof=1 for sample variance (unbiased estimator)
    variance_residuals = np.var(residuals, ddof=1) 

    # 3. Calculate the half-width of the prediction interval.
    # This is based on a sub-Gaussian concentration inequality.
    # The term sqrt(2 * var * log(2/alpha)) is derived from Hoeffding's inequality
    # or similar bounds for sums of bounded random variables, adapted for residuals.
    interval_half_width = np.sqrt(2 * variance_residuals * np.log(2 / miscoverage_rate))

    # 4. Construct prediction intervals for post-treatment predictions.
    # The interval is centered around the prediction adjusted by the mean of pre-treatment residuals.
    # Interval: [prediction + mean_residual - half_width, prediction + mean_residual + half_width]
    lower_bounds_post_treatment = predicted_outcomes_post_treatment + mean_residuals - interval_half_width
    upper_bounds_post_treatment = predicted_outcomes_post_treatment + mean_residuals + interval_half_width

    # --- Prepare Output ---
    # Create an array of pad_value for the pre-treatment period length
    padding_array_pre_treatment = np.full(len(actual_outcomes_pre_treatment), pad_value)

    # Concatenate the padding with the post-treatment bounds to get full series
    lower_bounds_full_series = np.concatenate([padding_array_pre_treatment, lower_bounds_post_treatment])
    upper_bounds_full_series = np.concatenate([padding_array_pre_treatment, upper_bounds_post_treatment])

    # Ensure the output arrays are 1D
    return lower_bounds_full_series.flatten(), upper_bounds_full_series.flatten()


def shc_conformal_test(
    pre_intervention_residuals: np.ndarray,
    post_intervention_residuals: np.ndarray,
    num_resamples: int = 1000,
    levels: Tuple[float, ...] = (0.01, 0.05, 0.10),
    random_state: int = 0,
) -> dict:
    r"""Conformal permutation test for the SHC intervention effect.

    Implements the inference procedure of Chen, Yang & Yang (Synthetic
    Historical Control for Policy Evaluation, 2024), which applies the
    conformal inference of Chernozhukov, Wuthrich & Zhu (2021) to the SHC
    estimator. The procedure tests the sharp null of no intervention
    effect,

    .. math::

       H_0: \delta_t = 0 \quad \text{for } t = T_o + 1, \dots, T_o + n,

    using the test statistic (their footnote 21)

    .. math::

       S = n^{-1/2} \sum_{t=T_o+1}^{T_o+n} \bigl| \hat\varepsilon_t^0 \bigr|,
       \qquad \hat\varepsilon_t^0 = y_t - \hat\ell_t,

    where :math:`\hat\varepsilon_t^0` are the post-intervention residuals
    (the estimated gaps :math:`\hat\delta_t`). The null distribution of
    ``S`` is constructed by **randomly sampling ``n`` observations with
    replacement** from the :math:`T_o` pre-intervention residuals
    :math:`\{\hat\varepsilon_t^0\}_{t=1}^{T_o}`, repeated ``num_resamples``
    (default 1,000) times, exactly as described in the paper.

    Parameters
    ----------
    pre_intervention_residuals : np.ndarray
        The :math:`T_o` pre-intervention residuals
        :math:`\hat\varepsilon_t^0 = y_t - \hat\ell_t`, shape ``(T_o,)``.
        These form the resampling pool for the null distribution.
    post_intervention_residuals : np.ndarray
        The ``n`` post-intervention residuals (estimated gaps), shape
        ``(n,)``. Their absolute sum forms the observed statistic.
    num_resamples : int, optional
        Number of resamples used to build the null distribution. Default
        1000, matching the paper.
    levels : tuple of float, optional
        Significance levels at which to report upper-tail critical values
        and reject/retain decisions. Default ``(0.01, 0.05, 0.10)``.
    random_state : int, optional
        Seed for the resampling RNG. Default 0.

    Returns
    -------
    dict
        Keys: ``test_statistic`` (S), ``p_value``
        (:math:`\Pr(S^* \ge S)`), ``critical_values`` (mapping level ->
        upper-tail quantile of the null), ``reject`` (mapping level ->
        bool), ``null_distribution`` (the resampled ``S^*`` array),
        ``num_resamples``, and ``levels``.

    Raises
    ------
    MlsynthDataError
        If either residual array is empty.
    """
    pre = np.asarray(pre_intervention_residuals, dtype=float).ravel()
    post = np.asarray(post_intervention_residuals, dtype=float).ravel()
    if pre.size == 0:
        raise MlsynthDataError("pre_intervention_residuals cannot be empty.")
    if post.size == 0:
        raise MlsynthDataError("post_intervention_residuals cannot be empty.")

    n = post.size
    scale = 1.0 / np.sqrt(n)
    test_statistic = float(scale * np.sum(np.abs(post)))

    # Null distribution: sample n residuals WITH REPLACEMENT from the
    # pre-intervention pool, num_resamples times (paper, footnote 21).
    rng = np.random.default_rng(random_state)
    resamples = rng.choice(pre, size=(num_resamples, n), replace=True)
    null_distribution = scale * np.sum(np.abs(resamples), axis=1)

    p_value = float(np.mean(null_distribution >= test_statistic))
    critical_values = {
        lvl: float(np.quantile(null_distribution, 1.0 - lvl)) for lvl in levels
    }
    reject = {lvl: bool(test_statistic > critical_values[lvl]) for lvl in levels}

    return {
        "test_statistic": test_statistic,
        "p_value": p_value,
        "critical_values": critical_values,
        "reject": reject,
        "null_distribution": null_distribution,
        "num_resamples": int(num_resamples),
        "levels": tuple(levels),
    }





def run_conformal_inference(
    inputs: SHCInputs,
    design: SHCDesign,
    observed: np.ndarray,
    counterfactual: np.ndarray,
    *,
    miscoverage_rate: float = 0.10,
    num_resamples: int = 1000,
    levels: Sequence[float] = (0.01, 0.05, 0.10),
    random_state: int = 0,
) -> SHCInference:
    """Assemble the SHC conformal permutation test and conformal bands.

    Parameters
    ----------
    inputs : SHCInputs
        Preprocessed series (supplies the pre-period and latent trend pool).
    design : SHCDesign
        Fitted design (supplies ``latent_pre`` for the pre-period residuals).
    observed, counterfactual : np.ndarray
        Observed and SHC series over the ``m + n`` block window.
    miscoverage_rate : float
        ``1 - coverage`` for the Andrews-Genton bands (0.10 -> 90%).
    num_resamples, levels, random_state
        Forwarded to :func:`shc_conformal_test`.
    """
    m = inputs.m
    T0 = inputs.T0

    # Paper's residuals: pre-period eps_t^0 = y_t - ell_hat_t over t = 1..T0
    # (the kernel-smoother residuals); post-period eps_t^0 = the gap.
    pre_residuals = inputs.y[:T0] - np.asarray(design.latent_pre).ravel()
    post_residuals = observed[m:] - counterfactual[m:]

    test = shc_conformal_test(
        pre_intervention_residuals=pre_residuals,
        post_intervention_residuals=post_residuals,
        num_resamples=num_resamples,
        levels=tuple(levels),
        random_state=random_state,
    )

    lower, upper = ag_conformal(
        actual_outcomes_pre_treatment=observed[:m],
        predicted_outcomes_pre_treatment=counterfactual[:m],
        predicted_outcomes_post_treatment=counterfactual[m:],
        miscoverage_rate=miscoverage_rate,
        pad_value=np.nan,
    )

    return SHCInference(
        method="conformal_permutation",
        test_statistic=test["test_statistic"],
        p_value=test["p_value"],
        critical_values=test["critical_values"],
        reject=test["reject"],
        num_resamples=test["num_resamples"],
        null_distribution=test["null_distribution"],
        conformal_lower=lower[m:],
        conformal_upper=upper[m:],
        confidence_level=1.0 - miscoverage_rate,
    )
