"""Inference for SCMO: the CWZ conformal test and Abadie's permutation test.

Two procedures, either of which serves every weighting scheme.

The conformal test (Chernozhukov-Wuethrich-Zhu, in the multi-outcome form of
Sun, Ben-Michael & Feller 2025, Online Appendix A) is the default, and is the
one that inverts to a confidence interval.

The permutation test at the end of the module is Abadie's: refit with every
unit in turn playing the treated one and rank the treated unit's
post-to-pre-treatment RMSPE ratio among them. It is what Tian, Lee & Panchenko
(2026) report p-values from, and it produces a rank, not an interval.

For a constant-effect null ``H0: tau = tau0`` the synthetic-control weights do
not depend on the post-period outcome (the matching matrix ``Z`` is built from
pre-period information), so the adjusted residual is simply the gap shifted by
``tau0`` in the post-period. The per-period test statistic is::

    S_q(u_t) = ( (1/sqrt(K)) * sum_k |u_tk|^q )^{1/q},   default q = 1,

and the conformal p-value ranks the post-treatment statistic against the
distribution of pre-treatment (moving-block) statistics. Inverting the test
over a grid of ``tau0`` yields a confidence interval for the ATT. With a single
predicted outcome (``K = 1``) the statistic reduces to ``|gap|``.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ...exceptions import MlsynthConfigError
from .structures import PlaceboInference


def conformal_inference(
    y: np.ndarray, counterfactual: np.ndarray, T0: int,
    alpha: float = 0.1, q: float = 1.0, n_grid: int = 600,
) -> Tuple[float, float, Tuple[float, float]]:
    """CWZ conformal ATT, p-value, and confidence interval from a gap series.

    Tests the *average* post-treatment effect (the scalar case of Sun-Ben-Michael-
    Feller Online Appendix A / Chernozhukov-Wuethrich-Zhu 2021): the test statistic
    is the post-period mean gap, and the reference distribution is the set of
    pre-treatment moving-block means of the same length. Inverting the test over
    ``tau0`` yields the ATT confidence interval. (For a single predicted outcome
    the per-period ``S_q`` reduces to ``|gap|``; ``q`` only bites with ``K > 1``
    outcomes.)

    Parameters
    ----------
    y, counterfactual : np.ndarray
        Observed treated outcome and estimated counterfactual, shape ``(T,)``.
    T0 : int
        Number of pre-treatment periods.
    alpha : float
        Miscoverage rate (e.g. 0.1 -> 90% interval).
    q : float
        Norm exponent (kept for the multi-outcome generalization; inert for K=1).
    n_grid : int
        Resolution of the test-inversion grid for the confidence interval.

    Returns
    -------
    att : float
        Mean post-treatment gap.
    p_value : float
        Conformal p-value for the sharp null of no average effect.
    ci : tuple of float
        ``(lower, upper)`` confidence interval for the ATT.
    """
    gap = np.asarray(y, dtype=float) - np.asarray(counterfactual, dtype=float)
    pre, post = gap[:T0], gap[T0:]
    L = post.shape[0]
    L = L if T0 >= L else max(1, T0 // 2)
    n_blocks = T0 - L + 1
    block_means = np.array([pre[s:s + L].mean() for s in range(n_blocks)])   # signed
    ref = np.abs(block_means)                                                # |avg gap| under no effect

    att = float(np.mean(post))

    def p_of(tau0: float) -> float:
        return (1.0 + np.sum(ref >= abs(att - tau0))) / (n_blocks + 1.0)

    p_value = p_of(0.0)

    spread = abs(att) + 4.0 * (ref.max() + np.std(pre)) + 1e-9
    grid = np.linspace(att - spread, att + spread, n_grid)
    keep = np.array([p_of(t) >= alpha for t in grid])
    ci = (float(grid[keep].min()), float(grid[keep].max())) if keep.any() else (float("nan"), float("nan"))
    return att, p_value, ci


# --- permutation (placebo) inference ---------------------------------------

TWO_SIDED, GREATER, LESS = "two-sided", "greater", "less"
_ALTERNATIVES = (TWO_SIDED, GREATER, LESS)


def rmspe_ratio(post: float, pre: float, eta: float = 0.0) -> float:
    """Post-to-pre-treatment RMSPE ratio, with the appendix's ``eta`` guard.

    ``eta`` is added to both sides (Tian-Lee-Panchenko 2026, Online Appendix
    B.3.3, footnote 14), which keeps a unit whose pre-treatment RMSPE is near
    zero from taking an arbitrarily large ratio; the application sets it to a
    small multiple of the outcome's cross-sectional SD. A unit with no
    pre-treatment error and some post-treatment gap has an infinite ratio and
    ranks first; one with neither has a ratio of zero and ranks last.
    """
    if eta < 0:
        raise MlsynthConfigError(f"eta must be non-negative; got {eta}.")
    numerator, denominator = float(post) + eta, float(pre) + eta
    if denominator == 0:
        return float("inf") if numerator > 0 else 0.0
    return numerator / denominator


def _directional(gap: np.ndarray, alternative: str) -> np.ndarray:
    """Keep the part of the gap the alternative is about.

    ``gap`` is mlsynth's ``observed - counterfactual``. The papers write the
    effect the other way round (counterfactual - observed), so their one-sided
    test for a negative effect is ``greater`` here.
    """
    if alternative == GREATER:
        return np.maximum(gap, 0.0)
    if alternative == LESS:
        return np.minimum(gap, 0.0)
    return gap


def permutation_inference(
    inputs, scheme: str, *, demean: bool = False, augment=None,
    ridge_lambda=None, weights: str = "simplex", pcr_rank=None,
    pcr_cumvar: float = 0.95, metric_weights=None,
    metric_weighting: str = "column", eta: float = 0.0,
    alternative: str = TWO_SIDED,
):
    """Abadie's permutation test for one weighting scheme.

    Refits the scheme with every unit in turn playing the treated one, forms
    each unit's post-to-pre-treatment RMSPE ratio, and reads the treated unit's
    rank as the p-value: ``p = #{i : r_i >= r_treated} / N`` (Tian-Lee-Panchenko
    2026, Online Appendix B.3.3). The same ranking applied period by period
    gives the per-period p-values the appendix plots.

    Parameters
    ----------
    inputs : SCMOInputs
        Prepared panel; the placebo loop reuses its matching matrix.
    scheme : str
        ``concatenated``, ``averaged`` or ``separate``.
    demean, augment, ridge_lambda, weights, pcr_rank, pcr_cumvar, metric_weights, metric_weighting
        Passed to the fit, so every placebo is estimated the way the treated
        unit was.
    eta : float, default 0
        Guard added to both RMSPEs (see :func:`rmspe_ratio`).
    alternative : {"two-sided", "greater", "less"}
        Which part of the gap the test statistic keeps.

    Returns
    -------
    PlaceboInference
    """
    if alternative not in _ALTERNATIVES:
        raise MlsynthConfigError(
            f"alternative must be one of {_ALTERNATIVES}; got {alternative!r}.")
    from .estimation import col_scale_for, fit_placebo    # local: estimation imports structures

    col_scale = col_scale_for(inputs, scheme, weights, metric_weights,
                              metric_weighting)
    N, T0, T = inputs.Y.shape[0], inputs.T0, inputs.T
    n_post = T - T0
    pre = np.empty(N)
    post = np.empty(N)
    post_gaps = np.empty((N, n_post))
    per_period_ratios = np.empty((N, n_post))
    for i in range(N):
        donors = np.delete(np.arange(N), i)
        pre_rmse, gap = fit_placebo(
            inputs, i, donors, scheme, demean, augment, ridge_lambda,
            weights, pcr_rank, pcr_cumvar, col_scale)
        post_gaps[i] = gap[T0:]                 # signed, for the aggregate index
        post_gap = _directional(gap[T0:], alternative)
        pre[i] = pre_rmse
        post[i] = float(np.sqrt(np.mean(post_gap ** 2))) if n_post else 0.0
        for t in range(n_post):
            per_period_ratios[i, t] = rmspe_ratio(abs(post_gap[t]), pre[i], eta)

    ratios = np.array([rmspe_ratio(post[i], pre[i], eta) for i in range(N)])
    treated = inputs.treated_idx
    p_value = float(np.mean(ratios >= ratios[treated]))
    per_period_p = np.array([
        float(np.mean(per_period_ratios[:, t] >= per_period_ratios[treated, t]))
        for t in range(n_post)])
    return PlaceboInference(
        p_value=p_value, treated_ratio=float(ratios[treated]), ratios=ratios,
        pre_rmspe=pre, post_rmspe=post, per_period_p=per_period_p,
        per_period_ratios=per_period_ratios, post_gaps=post_gaps,
        alternative=alternative, eta=float(eta))
