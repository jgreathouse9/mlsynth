"""Triage for a control market that went wrong after the design was locked.

A pre-committed design (MAREX and its relatives) fixes the control weights at
the start of the experiment from pre-period data alone, and the estimate is
then a comparison against the weighted average those weights define. If an
outside event hits one of the control markets while the experiment runs -- a
competitor launch, a store closure, a pricing change nobody told the analyst
about -- the weighted average moves by the size of the event times the weight
that market carries, and the reported effect moves with it. For a shock
:math:`\\pi` entering control market :math:`k` with weight :math:`v_k`,

.. math::

    \\hat{\\tau}_{\\text{observed}} - \\hat{\\tau}_{\\text{clean}} = -v_k \\pi .

The identity is arithmetic, not an approximation, which is what makes it
useful under time pressure: three of the questions an analyst asks when the
call comes in have exact answers available before anyone re-estimates
anything.

How much weight does the market carry (``exposure``)? What does an event of a
given size cost (``bias``)? And, running it backwards, how large would the
event have to have been to consume the whole measured effect
(``breakdown_shock``)? The third is the one that usually settles the matter.
A market at :math:`v_k = 0.04` needs an event twenty-five times the reported
effect to overturn it, and most incidents are not that; a market at
:math:`v_k = 0.6` needs an event under twice the effect, and many are.

Repairing the estimate is a different and much harder problem -- it needs the
size of the event, which is what nobody has -- and the correction arms for it
are studied under ``benchmarks/studies/contaminated_control/``. This module
answers the prior question of whether a repair is needed at all.

The concentration summaries alongside them (``max_weight``,
``effective_sample_size``, ``herfindahl``, ``n_carrying_weight``) say how
badly the design could ever be hurt this way. A design spread over an
effective twelve markets has no single market that can move the answer much.
One that put 70% of its control weight in a single market has staked the
experiment on nothing going wrong there, and that is a property of the design,
knowable at :math:`T_0`, before any shock arrives.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ..exceptions import MlsynthDataError
from .microsynth_helpers.diagnostics import effective_sample_size, max_weight

__all__ = ["ContaminationReport", "contamination_report", "control_exposure"]

#: Weights are read off a solver, so they satisfy the simplex constraints to
#: solver precision and not to machine precision. The same tolerance decides
#: when a weight counts as carried at all; MAREX uses ``1e-8`` when it builds
#: its own weight maps.
WEIGHT_TOL = 1e-8


class ContaminationReport(BaseModel):
    """What one contaminated control market can do to a locked design.

    Attributes
    ----------
    market : int
        Index of the market asked about, into the weight vector supplied.
    name : str, optional
        Label of that market when one was available.
    exposure : float
        The control weight :math:`v_k` the design gave it, in :math:`[0, 1]`.
    carries_weight : bool
        Whether that weight exceeds :data:`WEIGHT_TOL`. False means the market
        is not in the control group's support and the event is irrelevant to
        the estimate.
    bias : float, optional
        :math:`-v_k \\pi`, the amount the reported effect is off by, for the
        shock supplied. ``None`` when no shock was supplied.
    breakdown_shock : float, optional
        :math:`|\\hat{\\tau}| / v_k`, the smallest shock whose bias consumes
        the whole measured effect. ``inf`` when the market carries no weight
        above :data:`WEIGHT_TOL`, since then no shock of any size moves the
        estimate. ``None`` when no effect was supplied.
    max_weight : float
        Largest control weight in the design, the worst case over markets.
    effective_sample_size : float
        :math:`1 / \\sum_j v_j^2`. Equal weights over :math:`n` markets give
        :math:`n`; a design resting on one market gives 1.
    herfindahl : float
        :math:`\\sum_j v_j^2`, the reciprocal of the effective sample size.
    n_carrying_weight : int
        How many markets carry weight above :data:`WEIGHT_TOL`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid",
                              arbitrary_types_allowed=True)

    market: int = Field(..., description="Index into the weight vector.")
    name: Optional[str] = Field(None, description="Label of that market.")
    exposure: float = Field(..., description="Control weight v_k in [0, 1].")
    carries_weight: bool = Field(
        ..., description="Whether v_k exceeds the solver tolerance.")
    bias: Optional[float] = Field(
        None, description="-v_k * shock: the error in the reported effect.")
    breakdown_shock: Optional[float] = Field(
        None, description="|att| / v_k: the shock that consumes the effect.")
    max_weight: float = Field(
        ..., description="Largest control weight in the design.")
    effective_sample_size: float = Field(
        ..., description="1 / sum(v^2).")
    herfindahl: float = Field(..., description="sum(v^2).")
    n_carrying_weight: int = Field(
        ..., description="Markets carrying weight above tolerance.")


def _validate_weights(weights: Any) -> np.ndarray:
    """Return ``weights`` as a 1-D simplex vector or raise."""
    try:
        v = np.asarray(weights, dtype=float)
    except (TypeError, ValueError) as exc:  # pragma: no cover - numpy message
        raise MlsynthDataError(
            f"Control weights must be numeric; got {weights!r}.") from exc
    if v.ndim != 1:
        raise MlsynthDataError(
            "Control weights must be one-dimensional; got an array of shape "
            f"{v.shape}.")
    if v.size == 0:
        raise MlsynthDataError(
            "Control weights must contain at least one market; got an empty "
            "array.")
    if not np.all(np.isfinite(v)):
        raise MlsynthDataError(
            "Control weights must all be finite; found "
            f"{int((~np.isfinite(v)).sum())} non-finite entry(ies).")
    if np.any(v < -WEIGHT_TOL):
        raise MlsynthDataError(
            "Control weights must be non-negative; found a negative weight "
            f"of {float(v.min()):.6g}.")
    total = float(v.sum())
    if abs(total - 1.0) > 1e-6:
        raise MlsynthDataError(
            f"Control weights must sum to 1; they sum to {total:.6g}.")
    return np.clip(v, 0.0, None)


def _validate_scalar(value: Any, label: str) -> float:
    x = float(value)
    if not np.isfinite(x):
        raise MlsynthDataError(f"{label} must be finite; got {value!r}.")
    return x


def contamination_report(
    weights: Any,
    market: int,
    shock: Optional[float] = None,
    att: Optional[float] = None,
    name: Optional[str] = None,
) -> ContaminationReport:
    """Exposure, cost and breakdown point for one control market.

    Parameters
    ----------
    weights : array-like
        The design's control weights :math:`v`, non-negative and summing to 1.
        Markets left out of the support may be omitted; zeros do not change
        any quantity reported here.
    market : int
        Index of the market the incident happened in.
    shock : float, optional
        Size of the event, on the outcome's scale, as an additive shift to
        that market's post-period level. Supply it to get ``bias``.
    att : float, optional
        The effect the design reported. Supply it to get ``breakdown_shock``.
    name : str, optional
        Label to carry onto the report.

    Returns
    -------
    ContaminationReport

    Raises
    ------
    MlsynthDataError
        If the weights are not a finite non-negative vector summing to 1, if
        ``market`` is not an index into it, or if ``shock`` / ``att`` is not
        finite.

    Examples
    --------
    >>> rep = contamination_report([0.52, 0.31, 0.17], market=2,
    ...                            shock=-0.9, att=0.4)
    >>> round(rep.exposure, 2), round(rep.bias, 4)
    (0.17, 0.153)
    >>> round(rep.breakdown_shock, 2)
    2.35
    """
    v = _validate_weights(weights)
    k = int(market)
    if k < 0 or k >= v.size:
        raise MlsynthDataError(
            f"market index {market!r} is out of range for {v.size} control "
            "weights.")
    exposure = float(v[k])
    carries = exposure > WEIGHT_TOL

    bias = None
    if shock is not None:
        bias = -exposure * _validate_scalar(shock, "shock")

    breakdown = None
    if att is not None:
        tau = abs(_validate_scalar(att, "att"))
        breakdown = tau / exposure if carries else float("inf")

    ess = effective_sample_size(v)
    return ContaminationReport(
        market=k,
        name=name,
        exposure=exposure,
        carries_weight=carries,
        bias=bias,
        breakdown_shock=breakdown,
        max_weight=max_weight(v),
        effective_sample_size=ess,
        herfindahl=float((v ** 2).sum()),
        n_carrying_weight=int((v > WEIGHT_TOL).sum()),
    )


def _control_weight_map(result: Any) -> Mapping[str, float]:
    """Pull the control-weight map off a design result."""
    design_weights = getattr(result, "design_weights", None)
    stats = getattr(design_weights, "summary_stats", None) or {}
    for key in ("control_weights_agg", "control_weights"):
        if isinstance(stats.get(key), Mapping):
            return stats[key]
    raise MlsynthDataError(
        "This result carries no control-weight map; expected "
        "design_weights.summary_stats['control_weights_agg'], as an "
        "experimental-design result (MAREX) populates.")


def control_exposure(
    result: Any,
    market: str,
    shock: Optional[float] = None,
    att: Optional[float] = None,
) -> ContaminationReport:
    """:func:`contamination_report` for a named market on a design result.

    Reads the control weights off the standardized design surface, so it works
    on any :class:`~mlsynth.config_models.DesignResult` that populates
    ``design_weights.summary_stats["control_weights_agg"]``. A market absent
    from that map carries no control weight -- it was treated, or the design
    did not use it -- and is reported with zero exposure, which is the answer
    to the question asked.

    Parameters
    ----------
    result : DesignResult
        A fitted design.
    market : str
        Label of the market the incident happened in.
    shock, att : float, optional
        As in :func:`contamination_report`. When ``att`` is omitted and the
        result carries one, the result's own ATT is used.

    Returns
    -------
    ContaminationReport
    """
    weight_map = _control_weight_map(result)
    labels: Sequence[str] = [str(key) for key in weight_map]
    values = [float(weight_map[key]) for key in weight_map]
    target = str(market)
    if target in labels:
        k = labels.index(target)
    else:
        k = len(values)
        values.append(0.0)
    if att is None:
        report = getattr(result, "report", None)
        effects = getattr(report, "effects", None)
        att = getattr(effects, "att", None)
    return contamination_report(
        np.asarray(values, dtype=float), market=k, shock=shock, att=att,
        name=target,
    )
