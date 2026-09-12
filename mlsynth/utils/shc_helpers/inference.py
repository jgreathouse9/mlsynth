"""Inference for the Synthetic Historical Control estimator.

Primary inference is the conformal permutation test of Chen, Yang & Yang
(2024, footnote 21) -- their application of Chernozhukov, Wuthrich & Zhu
(2021) to SHC -- computed by :func:`mlsynth.utils.inferutils.shc_conformal_test`.
Andrews-Genton conformal prediction bands are computed alongside for the
plot.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional, Sequence, Tuple

from scipy.optimize import lsq_linear
import numpy as np

from .structures import SHCDesign, SHCInference, SHCInputs
from ...exceptions import (MlsynthConfigError, MlsynthDataError,
                           MlsynthEstimationError)


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





def cwz_conformal_test(
    pre_intervention_residuals: np.ndarray,
    post_intervention_residuals: np.ndarray,
    q: float = 1.0,
    scheme: str = "moving_block",
    num_permutations: int | None = None,
    levels: Tuple[float, ...] = (0.01, 0.05, 0.10),
    random_state: int = 0,
) -> dict:
    r"""Exact permutation conformal test of Chernozhukov, Wuthrich & Zhu (2021).

    This is the *exact* conformal inference of CWZ (2021), as opposed to the
    Chen-Yang-Yang (2024, footnote 21) with-replacement residual bootstrap in
    :func:`shc_conformal_test`. It tests the sharp null
    :math:`H_0: \delta_t = 0` over the post window using the CWZ statistic
    (their Definition 1)

    .. math::

       S_q(u) = \Bigl( \tfrac{1}{\sqrt{n}}
                 \sum_{t = T_o + 1}^{T_o + n} |u_t|^q \Bigr)^{1/q},

    evaluated on the trailing ``n`` positions of the full residual vector
    :math:`u = (\hat\varepsilon_1^0, \dots, \hat\varepsilon_{T_o}^0,
    \hat\delta_1, \dots, \hat\delta_n)` of length :math:`T = T_o + n`. The
    reference distribution is obtained by *permuting* ``u`` (CWZ Definition 2,
    Figure 2), not by resampling from the pre-period pool:

    * ``scheme="moving_block"`` -- the :math:`T` cyclic shifts
      :math:`\Pi_{\rightarrow}` (:math:`\pi_j(i) = i + j \bmod T`), valid under
      stationary weak dependence. The set is fully enumerated, so the test is
      deterministic and its p-values lie on the :math:`1/T` grid.
    * ``scheme="iid"`` -- random permutations from :math:`\Pi_{\mathrm{all}}`,
      exact under exchangeability; ``num_permutations`` are drawn (the identity
      is always included so :math:`\hat p \ge 1/|\Pi|`).

    Parameters
    ----------
    pre_intervention_residuals : np.ndarray
        Pre-period residuals :math:`\hat\varepsilon_t^0`, shape ``(T_o,)``.
    post_intervention_residuals : np.ndarray
        Post-period residuals (estimated gaps), shape ``(n,)``.
    q : float, optional
        Norm exponent of the test statistic. Default ``1.0`` (CWZ's :math:`S_1`,
        which matches the bootstrap statistic in :func:`shc_conformal_test`).
    scheme : {"moving_block", "iid"}, optional
        Permutation family. Default ``"moving_block"``.
    num_permutations : int or None, optional
        Number of permutations for ``scheme="iid"`` (>= 2). Ignored for
        ``"moving_block"`` (always ``T``). Defaults to ``1000`` for ``"iid"``.
    levels : tuple of float, optional
        Significance levels for critical values / reject decisions.
    random_state : int, optional
        Seed for the ``"iid"`` permutation RNG (unused for ``"moving_block"``).

    Returns
    -------
    dict
        Keys ``test_statistic``, ``p_value`` (:math:`\Pr(S^* \ge S)`),
        ``critical_values``, ``reject``, ``null_distribution``,
        ``num_permutations``, ``scheme``, ``q``, ``levels``.

    Raises
    ------
    MlsynthDataError
        If either residual array is empty.
    MlsynthConfigError
        If ``q <= 0``, ``scheme`` is unknown, or ``num_permutations < 2``.
    """
    pre = np.asarray(pre_intervention_residuals, dtype=float).ravel()
    post = np.asarray(post_intervention_residuals, dtype=float).ravel()
    if pre.size == 0:
        raise MlsynthDataError("pre_intervention_residuals cannot be empty.")
    if post.size == 0:
        raise MlsynthDataError("post_intervention_residuals cannot be empty.")
    if not q > 0:
        raise MlsynthConfigError("q must be positive.")
    if scheme not in ("moving_block", "iid"):
        raise MlsynthConfigError(
            f"scheme must be 'moving_block' or 'iid', got {scheme!r}."
        )

    n = post.size
    T0 = pre.size
    u = np.concatenate([pre, post])           # full residual vector, length T
    T = u.size
    inv_sqrt_n = 1.0 / np.sqrt(n)

    def stat(block: np.ndarray) -> float:
        return float((inv_sqrt_n * np.sum(np.abs(block) ** q)) ** (1.0 / q))

    test_statistic = stat(post)

    if scheme == "moving_block":
        # The T cyclic shifts; trailing-n block of np.roll(u, -j) for j=0..T-1.
        # j = 0 is the identity (the observed statistic), so it is included.
        null_distribution = np.array(
            [stat(np.roll(u, -j)[T0:]) for j in range(T)], dtype=float
        )
        num_permutations = T
    else:  # scheme == "iid"
        if num_permutations is None:
            num_permutations = 1000
        if num_permutations < 2:
            raise MlsynthConfigError("num_permutations must be >= 2 for 'iid'.")
        rng = np.random.default_rng(random_state)
        stats = np.empty(num_permutations, dtype=float)
        stats[0] = test_statistic                     # identity always included
        for i in range(1, num_permutations):
            stats[i] = stat(rng.permutation(u)[T0:])
        null_distribution = stats

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
        "num_permutations": int(num_permutations),
        "scheme": scheme,
        "q": float(q),
        "levels": tuple(levels),
    }


MAX_REFERENCE_BLOCKS = 60
"""Default cap on refitted blocks, so the pool's cost does not scale with ``N``.

Each block costs one matching solve. A panel with a few hundred blocks would
otherwise make every fit two orders of magnitude slower than the point estimate
it reports, which is a poor default for a quantity most callers read at the 5%
or 10% level. Sixty blocks is ``60 * n`` residuals -- 240 at ``n = 4`` -- which
resolves those levels; pass ``stride=1`` for the full pool when the 1% level
matters.
"""


def block_oos_residuals(
    inputs: SHCInputs,
    design: SHCDesign,
    *,
    stride: Optional[int] = None,
    use_augmented: bool = False,
) -> Tuple[np.ndarray, dict]:
    r"""Out-of-sample residuals over each historical block's own post-window.

    The conformal test compares the post-period statistic against a reference
    distribution drawn from pre-period residuals, and the permutation argument
    needs those residuals to be the same object as the statistic. The statistic
    is the raw outcome minus an SHC prediction the treated block did not inform.
    An in-sample kernel-smoother residual is not that: it omits the matching
    error and omits the treated block's own noise, so it understates the scale
    the null should sit at.

    This builds the pool the statistic's own way. Block ``j`` is treated as if
    it were the treated block: its pre-window in :math:`\hat\ell` space is
    matched by a simplex over the blocks that share no observation with it, and
    the residual is taken over ``j``'s own post-window in raw outcome units,

    .. math::

        \hat\varepsilon_{j,t} = y_{j+m+t} - \bigl(L^{post}_{\cdot,
        \mathcal{D}_j} \hat w^{(j)}\bigr)_t, \qquad t = 1, \dots, n,

    with :math:`\mathcal{D}_j = \{k : |k - j| \ge m + n\}`. Excluding the
    overlapping blocks and not merely ``j`` itself is what makes the residual
    out of sample: the blocks are formed at stride one, so block ``j \pm 1``
    shares :math:`m + n - 1` of ``j``'s observations.

    Parameters
    ----------
    inputs : SHCInputs
        Supplies the outcome, the split and the block count.
    design : SHCDesign
        Supplies the fitted latent trend, which is reused; only the matching
        program is re-solved per block.
    stride : int, optional
        Evaluate every ``stride``-th block. Each block costs one matching solve.
        ``None`` (the default) picks the smallest stride that keeps the count at
        or under :data:`MAX_REFERENCE_BLOCKS`, so the cost does not grow with
        the panel; ``1`` evaluates every block.
    use_augmented : bool
        Solve the ridge-augmented (ASHC) program instead of the simplex one.

    Returns
    -------
    pool : np.ndarray
        The residuals, ``n_blocks * n`` of them, block-major.
    info : dict
        ``n_blocks``, ``blocks`` (the indices evaluated) and ``donor_sets``
        (the donor indices used for each), for tests and diagnostics.

    Raises
    ------
    MlsynthEstimationError
        If no block has any non-overlapping donor, which happens when the
        pre-period is barely longer than one block.
    """
    from ..datautils import build_donor_segments
    from .kernels import solve_shc_qp

    m, n, T0, N = inputs.m, inputs.n, inputs.T0, inputs.N
    ell_hat = np.asarray(design.latent_pre, dtype=float).ravel()
    L_full, L_post, _ell_eval = build_donor_segments(ell_hat, m, T0, n)

    if stride is None:
        stride = max(1, -(-N // MAX_REFERENCE_BLOCKS))      # ceil division
    stride = int(stride)
    if stride < 1:
        raise MlsynthEstimationError(f"stride must be >= 1, got {stride}.")

    residuals: list = []
    blocks: list = []
    donor_sets: list = []
    for j in range(0, N, stride):
        donors = [k for k in range(N) if abs(k - j) >= m + n]
        if len(donors) < 2:
            continue
        idx = np.asarray(donors, dtype=int)
        w, _ = solve_shc_qp(L_full[:, idx], L_full[:, j],
                            use_augmented=use_augmented)
        if w is None:                       # pragma: no cover - solver guard
            continue
        own_post = inputs.y[j + m:j + m + n]
        if own_post.size != n:              # pragma: no cover - tail guard
            continue
        residuals.extend(list(own_post - L_post[:, idx] @ w))
        blocks.append(int(j))
        donor_sets.append([int(k) for k in donors])

    if not residuals:
        raise MlsynthEstimationError(
            "no out-of-sample reference residual could be built: every "
            f"historical block overlaps every other (N={N}, block length "
            f"m+n={m + n}). Shorten m, or use reference_pool='smoother' and "
            "read the test as miscalibrated."
        )
    return (np.asarray(residuals, dtype=float),
            {"n_blocks": len(blocks), "blocks": blocks, "stride": stride,
             "donor_sets": donor_sets})


def run_conformal_inference(
    inputs: SHCInputs,
    design: SHCDesign,
    observed: np.ndarray,
    counterfactual: np.ndarray,
    *,
    method: str = "bootstrap",
    permutation_scheme: str = "moving_block",
    num_permutations: int | None = None,
    q: float = 1.0,
    miscoverage_rate: float = 0.10,
    num_resamples: int = 1000,
    levels: Sequence[float] = (0.01, 0.05, 0.10),
    random_state: int = 0,
    reference_pool: str = "block_oos",
    reference_stride: Optional[int] = None,
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
    method : {"bootstrap", "exact"}
        ``"bootstrap"`` (default) is the Chen-Yang-Yang (2024) with-replacement
        residual bootstrap (:func:`shc_conformal_test`); ``"exact"`` is the
        Chernozhukov-Wuthrich-Zhu (2021) permutation test
        (:func:`cwz_conformal_test`).
    permutation_scheme : {"moving_block", "iid"}
        Permutation family for ``method="exact"``.
    num_permutations : int or None
        Permutation count for ``method="exact"`` with ``permutation_scheme="iid"``.
    q : float
        Norm exponent of the exact-test statistic.
    miscoverage_rate : float
        ``1 - coverage`` for the Andrews-Genton bands (0.10 -> 90%).
    num_resamples, levels, random_state
        Forwarded to the selected test.

    Raises
    ------
    MlsynthConfigError
        If ``method`` is not ``"bootstrap"`` or ``"exact"``.
    """
    m = inputs.m
    T0 = inputs.T0

    # The reference pool. "block_oos" builds it the way the statistic is built
    # -- raw outcome minus a prediction the block did not inform -- which is
    # what the permutation argument needs. "smoother" is the paper's literal
    # reading, y_t - ell_hat_t over the whole pre-period; it is retained for
    # reproducing published numbers and it over-rejects, because an in-sample
    # smoother residual is tighter than the out-of-sample error it calibrates.
    pool_used, note = reference_pool, ""
    if reference_pool == "block_oos":
        try:
            pre_residuals, pool_info = block_oos_residuals(
                inputs, design, stride=reference_stride)
            n_reference = int(pool_info["n_blocks"])
        except MlsynthEstimationError as exc:
            # A panel too short for an out-of-sample residual still has a point
            # estimate to report, so the fit is not taken down for the sake of
            # its p-value. The smoother pool runs instead, the caller is warned,
            # and the result records which pool produced the number.
            note = str(exc)
            warnings.warn(
                f"SHC: {note} Falling back to the in-sample smoother pool, "
                "which over-rejects; read the test as miscalibrated.",
                UserWarning, stacklevel=2,
            )
            pool_used = "smoother"
            pre_residuals = inputs.y[:T0] - np.asarray(design.latent_pre).ravel()
            n_reference = int(pre_residuals.size)
    elif reference_pool == "smoother":
        pre_residuals = inputs.y[:T0] - np.asarray(design.latent_pre).ravel()
        n_reference = int(pre_residuals.size)
    else:
        raise MlsynthConfigError(
            f"reference_pool must be 'block_oos' or 'smoother', got "
            f"{reference_pool!r}."
        )
    post_residuals = observed[m:] - counterfactual[m:]

    if method == "bootstrap":
        test = shc_conformal_test(
            pre_intervention_residuals=pre_residuals,
            post_intervention_residuals=post_residuals,
            num_resamples=num_resamples,
            levels=tuple(levels),
            random_state=random_state,
        )
        method_label = "conformal_permutation"
        n_draws = test["num_resamples"]
    elif method == "exact":
        test = cwz_conformal_test(
            pre_intervention_residuals=pre_residuals,
            post_intervention_residuals=post_residuals,
            q=q,
            scheme=permutation_scheme,
            num_permutations=num_permutations,
            levels=tuple(levels),
            random_state=random_state,
        )
        method_label = f"conformal_exact_{test['scheme']}"
        n_draws = test["num_permutations"]
    else:
        raise MlsynthConfigError(
            f"method must be 'bootstrap' or 'exact', got {method!r}."
        )

    lower, upper = ag_conformal(
        actual_outcomes_pre_treatment=observed[:m],
        predicted_outcomes_pre_treatment=counterfactual[:m],
        predicted_outcomes_post_treatment=counterfactual[m:],
        miscoverage_rate=miscoverage_rate,
        pad_value=np.nan,
    )

    return SHCInference(
        method=method_label,
        test_statistic=test["test_statistic"],
        p_value=test["p_value"],
        critical_values=test["critical_values"],
        reject=test["reject"],
        num_resamples=n_draws,
        null_distribution=test["null_distribution"],
        conformal_lower=lower[m:],
        conformal_upper=upper[m:],
        confidence_level=1.0 - miscoverage_rate,
        levels=tuple(test.get("levels", tuple(levels))),
        scheme=test.get("scheme", "iid_with_replacement"),
        reference_pool=str(pool_used),
        n_reference=n_reference,
        reference_note=note,
    )
