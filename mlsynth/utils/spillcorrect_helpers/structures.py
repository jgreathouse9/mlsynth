"""Inputs, intermediate fits and results of SPILLCORRECT."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ...config_models import BaseEstimatorResults


@dataclass(frozen=True)
class CorrectionInputs:
    """The realized panel on the design's units and periods, and the roles.

    ``Y`` holds one row per design unit, in the order of the composite's
    ``members``, and one column per period of the design's timeline (the
    fitting, blank and post windows).
    """

    Y: np.ndarray
    labels: List[str]
    time_labels: Optional[List[Any]]
    wt: np.ndarray                 # treated composite weights, by row
    wc: np.ndarray                 # control composite weights, by row
    pool: List[int]                # rows of the controls the design drew from
    corrected: List[int]           # rows of the affected controls with weight
    skipped: List[int]             # rows of the affected controls without weight
    clean: List[int]               # rows of the controls no spillover reaches
    T0: int
    fit_end: int
    blank_periods: int
    post_periods: int


@dataclass(frozen=True)
class Correction:
    """One correction of the contaminated controls.

    ``cleaned`` is the panel with each corrected control's outcomes replaced
    from ``start`` on; ``spillover`` the observed less the rebuilt series, one
    row per corrected control, from ``start`` on.
    """

    cleaned: np.ndarray
    spillover: np.ndarray
    cleaning_rmspe: np.ndarray     # each rebuilding fit's RMSPE, in levels
    start: int
    composite_weight: Optional[np.ndarray] = None   # inclusive: l_k
    omega_det: Optional[float] = None               # inclusive: det Omega


class SpillCorrectEstimate(BaseModel):
    """One estimate of the design's effect: corrected, uncorrected or restricted.

    ``effect`` and ``counterfactual`` cover the post periods; ``placebo`` is the
    same contrast over the design's blank (held-out) periods, where no effect
    and, after a correction from the end of the fitting window, no spillover
    remains. ``ci`` and ``p_value`` are the design's inference recomputed on
    this estimate's panel; both are ``None`` for a design without inference.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    att: float = Field(..., description="Mean post-period effect.")
    effect: np.ndarray = Field(..., description="Post-period effect path, shape (post_periods,).")
    counterfactual: np.ndarray = Field(..., description="Post-period control composite, shape (post_periods,).")
    placebo: np.ndarray = Field(..., description="The contrast over the blank periods, shape (blank_periods,).")
    ci: Optional[np.ndarray] = Field(default=None, description="Post-period pointwise band, shape (post_periods, 2).")
    p_value: Optional[float] = Field(default=None, description="Global test p-value.")
    inference: Optional[Any] = Field(default=None, exclude=True,
                                     description="The design's full inference object on this panel.")


class SpillCorrectResults(BaseEstimatorResults):
    """SPILLCORRECT's result: an ``EffectResult`` for the corrected estimate.

    The standardized sub-models describe the estimate ``method`` selects. The
    three arms sit beside them: ``corrected`` (the selected correction, or the
    restricted re-solve under ``method='restricted'``), ``uncorrected`` (the
    design's own estimate on the realized panel) and ``restricted`` (the design
    re-solved without the affected controls, when ``design_config`` was given).
    """

    method: str = Field(..., description="'iterative', 'inclusive' or 'restricted'.")
    scale: bool = Field(..., description="Whether each rebuilding fit was scale-free.")
    window: str = Field(..., description="'blank_and_post' or 'post'.")
    correction_start: int = Field(..., description="First corrected period (index into the design's timeline).")
    corrected: SpillCorrectEstimate = Field(..., description="The estimate the standardized sub-models describe.")
    uncorrected: SpillCorrectEstimate = Field(..., description="The design's estimate on the realized panel.")
    restricted: Optional[SpillCorrectEstimate] = Field(
        default=None, description="The design re-solved without the affected controls.")
    restricted_weights: Optional[Dict[str, float]] = Field(
        default=None, description="The restricted design's control composite.")
    corrected_units: List[str] = Field(..., description="Affected controls with control weight, corrected.")
    skipped_units: List[str] = Field(..., description="Affected controls without control weight: not corrected, kept out of the donors.")
    donor_units: List[str] = Field(..., description="The clean controls each rebuilding fit drew from.")
    n_corrected: int = Field(..., description="Number of controls corrected.")
    spillover: Dict[str, np.ndarray] = Field(
        ..., description="Each corrected control's estimated spillover path from correction_start on.")
    cleaning_rmspe: Dict[str, float] = Field(
        ..., description="Each rebuilding fit's RMSPE over the fitting window, in levels.")
    control_weights: Dict[str, float] = Field(..., description="The design's control composite.")
    composite_weight: Optional[Dict[str, float]] = Field(
        default=None, description="Inclusive: each corrected control's level weight on the treated composite.")
    omega_det: Optional[float] = Field(default=None, description="Inclusive: determinant of the cross-weight system.")
