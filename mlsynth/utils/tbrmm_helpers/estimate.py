"""Measure a TBRMM design once the experiment has run.

The design half of TBRMM scores partitions on pretest data alone. When the panel
also carries a post window, the chosen partition can be read on it, and the
estimator for that is the augmented difference-in-differences of Li and Van den
Bulte (2022). Their equation (2.4) fits

.. math::

   y_{1t} = \\delta_1 + \\delta_2\\, \\bar{y}_{co,t} + e_{1t},
   \\qquad t = 1, \\dots, T_1

by least squares on the pretreatment periods, where :math:`y_{1t}` is a treated
unit and :math:`\\bar{y}_{co,t}` the average of the control units. Forcing
:math:`\\delta_2 = 1` recovers plain difference-in-differences, so the free scale
is the augmentation. The counterfactual is the fitted projection through the
post window and the effect is what the treated unit did above it.

Their Appendix C, case (i), carries this to several treated units: fit the
regression once per treated unit, take each unit's ATT, and pool by averaging.
That is what this module computes, and it is why the pooled effect here is the
mean of the market-level effects by construction. A pooled number that is the
average of its parts is what lets a market-level breakdown be reported beside a
headline without the two contradicting each other.

The regression is the same one TBR fits. TBR sums the treated geos into one
series first, which buys precision and gives up the breakdown; this keeps the
breakdown and pays for it in precision, since a single geo is noisier against
the same control average than the group is.
"""
from __future__ import annotations

from typing import Any, List, Sequence, Tuple

import numpy as np

from ...exceptions import MlsynthEstimationError
from .structures import TBRMMEffect, TBRMMMarketEffect


def _fit_adid(y_pre: np.ndarray, x_pre: np.ndarray) -> Tuple[float, float, float]:
    """Least squares for equation (2.4); returns ``(delta1, delta2, rmse)``.

    Parameters
    ----------
    y_pre : np.ndarray
        One treated unit's pretest outcomes, shape ``(T0,)``.
    x_pre : np.ndarray
        The control group's pretest average, shape ``(T0,)``.

    Returns
    -------
    tuple of float
        The intercept, the free control scale, and the pretest residual RMSE.

    Raises
    ------
    MlsynthEstimationError
        If the pretest design matrix is rank deficient, which happens when the
        control average is constant over the pretest and no scale is identified.
    """
    design = np.column_stack([np.ones_like(x_pre), x_pre])
    if np.linalg.matrix_rank(design) < 2:
        raise MlsynthEstimationError(
            "the control group's pretest average is constant, so the augmented "
            "DiD scale is not identified; widen the pretest window or the "
            "control group")
    delta, *_ = np.linalg.lstsq(design, y_pre, rcond=None)
    resid = y_pre - design @ delta
    return float(delta[0]), float(delta[1]), float(np.sqrt(np.mean(resid ** 2)))


def measure_design(
    pre: np.ndarray,
    post: np.ndarray,
    units: Sequence[Any],
    treatment_units: Sequence[Any],
    control_units: Sequence[Any],
) -> Tuple[TBRMMEffect, np.ndarray, np.ndarray]:
    """Read one design on the realized post window, market by market.

    Parameters
    ----------
    pre : np.ndarray
        Pretest outcomes, shape ``(T0, N)``, columns ordered as ``units``.
    post : np.ndarray
        Realized outcomes, shape ``(T2, N)``, same column order.
    units : sequence
        The panel's geo labels, in column order.
    treatment_units, control_units : sequence
        The design's two groups, as labels.

    Returns
    -------
    tuple
        The :class:`TBRMMEffect`, and the two trajectories the result contract
        wants: the treated markets' average observed path and the average of
        their counterfactuals, each over the whole panel (pretest then post), so
        the post-period gap averages to the pooled ATT.

    Raises
    ------
    MlsynthEstimationError
        If either group is empty, or a pretest fit is not identified.
    """
    index = {u: j for j, u in enumerate(units)}
    treated_cols = [index[u] for u in treatment_units]
    control_cols = [index[u] for u in control_units]
    if not treated_cols:
        raise MlsynthEstimationError("a design with no treated geo cannot be measured")
    if not control_cols:
        raise MlsynthEstimationError("a design with no control geo cannot be measured")

    x_pre = pre[:, control_cols].mean(axis=1)
    x_post = post[:, control_cols].mean(axis=1)

    market_effects: List[TBRMMMarketEffect] = []
    observed_paths: List[np.ndarray] = []
    counterfactual_paths: List[np.ndarray] = []

    for unit, col in zip(treatment_units, treated_cols):
        d1, d2, rmse = _fit_adid(pre[:, col], x_pre)
        cf_pre = d1 + d2 * x_pre
        cf_post = d1 + d2 * x_post
        gap_post = post[:, col] - cf_post
        baseline = float(np.mean(cf_post))
        market_effects.append(TBRMMMarketEffect(
            unit=unit,
            att=float(np.mean(gap_post)),
            att_percent=(100.0 * float(np.mean(gap_post)) / baseline
                         if abs(baseline) > 1e-12 else None),
            total_effect=float(np.sum(gap_post)),
            delta1=d1,
            delta2=d2,
            rmse_fit=rmse,
        ))
        observed_paths.append(np.concatenate([pre[:, col], post[:, col]]))
        counterfactual_paths.append(np.concatenate([cf_pre, cf_post]))

    atts = np.array([m.att for m in market_effects], dtype=float)
    treated_series = np.mean(np.column_stack(observed_paths), axis=1)
    control_series = np.mean(np.column_stack(counterfactual_paths), axis=1)

    pooled = float(atts.mean())
    pooled_baseline = float(np.mean(control_series[pre.shape[0]:]))
    effect = TBRMMEffect(
        att=pooled,
        att_percent=(100.0 * pooled / pooled_baseline
                     if abs(pooled_baseline) > 1e-12 else None),
        total_effect=float(sum(m.total_effect for m in market_effects)),
        n_post=int(post.shape[0]),
        n_treated=len(market_effects),
        market_effects=market_effects,
    )
    return effect, treated_series, control_series
