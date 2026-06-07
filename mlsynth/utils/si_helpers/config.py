"""Configuration for the SI estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import Any, List, Literal, Optional
from pydantic import Field, model_validator
from ...exceptions import MlsynthConfigError
from ...config_models import BaseEstimatorConfig


class SIConfig(BaseEstimatorConfig):
    """Configuration for the Synthetic Interventions (SI) estimator.

    Implements:

        Agarwal, A., Shah, D., & Shen, D. (2026). "Synthetic Interventions:
        Extending Synthetic Controls to Multiple Treatments." Operations
        Research 74(2):840-859.

    SI estimates the focal target unit's counterfactual outcome under each
    alternative intervention in ``inters`` via SI-PCR: regress the target's
    pre-treatment control outcomes onto the rank-``k`` denoised donor pool, then
    apply the weights to the donor pool's post-intervention outcomes.

    Parameters
    ----------
    inters : list of str
        Binary indicator columns naming the alternative interventions; for
        each, the units flagged ``1`` form that intervention's donor pool.
    rank_method : {"donoho", "usvt", "cumvar", "fixed"}
        Spectral-rank rule for the donor pre-matrix. ``"donoho"`` (default)
        reproduces the paper's exact Gavish-Donoho optimal hard threshold
        (evaluated at ``ratio = T0 / Nd``); ``"usvt"`` is the same threshold at
        the canonical ``min/max`` aspect ratio; ``"cumvar"`` keeps enough
        components for ``cumvar_threshold`` of the spectral energy; ``"fixed"``
        uses ``rank``.
    rank : int or None
        Explicit spectral rank ``k`` for ``rank_method="fixed"``.
    cumvar_threshold : float
        Cumulative-energy target in ``(0, 1]`` for ``rank_method="cumvar"``.
    bias_correct : bool
        Use the bias-corrected SI-PCR estimator (default ``True``), which fits
        weights on a rank-complete donor subset and enables asymptotic-normality
        intervals (Section 4.3). ``False`` gives plain SI-PCR (eq. 10), point
        estimate only.
    variance : {"double", "units", "time_iv"}
        Noise-variance estimator behind the interval. ``"double"`` (default)
        matches the paper's code (a d.o.f.-weighted combination); ``"units"`` is
        the main-text eq. 14; ``"time_iv"`` uses the donor post-period residual.
    interval : {"confidence", "prediction"}
        Interval type. ``"confidence"`` (default) is the eq.-13 CI for the
        counterfactual mean; ``"prediction"`` is the wider interval the paper's
        case study uses for coverage validation.
    alpha : float
        Two-sided significance level for the intervals.
    display_graphs : bool
        Show the observed-vs-counterfactual plot after fitting.
    """

    inters: List[str] = Field(
        ..., min_length=1,
        description="Binary indicator columns naming the alternative "
                    "interventions (donor pool = units flagged 1).",
    )
    rank_method: Literal["donoho", "usvt", "cumvar", "fixed"] = Field(
        default="donoho",
        description="Spectral-rank rule: paper's Gavish-Donoho ('donoho'), "
                    "min/max Gavish-Donoho ('usvt'), cumulative-energy "
                    "('cumvar'), or explicit ('fixed').",
    )
    rank: Optional[int] = Field(
        default=None, ge=1,
        description="Explicit spectral rank k for rank_method='fixed'.",
    )
    cumvar_threshold: float = Field(
        default=0.95, gt=0.0, le=1.0,
        description="Cumulative-energy target for rank_method='cumvar'.",
    )
    bias_correct: bool = Field(
        default=True,
        description="Use the bias-corrected SI-PCR estimator (enables intervals).",
    )
    variance: Literal["double", "units", "time_iv"] = Field(
        default="double",
        description="Noise-variance estimator: paper's weighted ('double'), "
                    "main-text eq. 14 ('units'), or post-period ('time_iv').",
    )
    interval: Literal["confidence", "prediction"] = Field(
        default="confidence",
        description="Interval type: confidence (eq. 13) or prediction.",
    )
    alpha: float = Field(
        default=0.05, gt=0.0, lt=1.0,
        description="Two-sided significance level for the intervals.",
    )

    @model_validator(mode="after")
    def _check_si_params(cls, values: Any) -> Any:
        if values.rank_method == "fixed" and values.rank is None:
            raise MlsynthConfigError(
                "rank_method='fixed' requires an explicit positive `rank`."
            )
        return values
