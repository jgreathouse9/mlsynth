"""Configuration for the DTWSC estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field, model_validator

from ...config_models import BaseEstimatorConfig
from ...exceptions import MlsynthConfigError


class DTWSCConfig(BaseEstimatorConfig):
    """Configuration for the DTWSC estimator.

    Implements Cao, J. & Chadefaux, T. (2025), *Dynamic Synthetic Controls:
    Accounting for Varying Speeds in Comparative Case Studies*, Political
    Analysis 33:18-31.

    Standard synthetic control assumes every unit reacts to a common shock at
    the same speed. When donors adapt faster or slower than the treated unit,
    matching them period-by-period lines up the wrong moments in each series,
    which both worsens the pre-treatment fit and biases the effect. DTWSC
    learns each donor's speed from the pre-period by dynamic time warping,
    resamples the donor onto the treated unit's clock, and only then fits the
    synthetic control. Speed differences the donor already had are removed;
    speed changes caused by the intervention are preserved, so the effect
    survives.

    The defaults reproduce the reference implementation
    (github.com/conflictlab/dsc).
    """

    k: int = Field(
        default=4,
        ge=2,
        description=(
            "Length of the sliding window used to carry pre-treatment speeds "
            "into the post-treatment period. Shorter windows adapt faster but "
            "match on less shape; must not exceed the post-period length."
        ),
    )
    warp: bool = Field(
        default=True,
        description=(
            "If False, skip the warping entirely and fit the synthetic control "
            "on the raw donors -- the standard-SC baseline, useful for showing "
            "what the warp buys on a given panel."
        ),
    )
    filter_width: int = Field(
        default=5,
        ge=3,
        description=(
            "Savitzky-Golay window (must be odd) used to smooth each series "
            "before differentiating. Alignment runs on curvature, not level, "
            "so units at different levels with the same turning points are "
            "judged similar. Set ``smooth=False`` to align on the raw series."
        ),
    )
    smooth: bool = Field(
        default=True,
        description=(
            "Apply the Savitzky-Golay second-derivative transform before "
            "aligning. The warp is always applied to the untransformed "
            "outcome; this only affects what the alignment sees."
        ),
    )
    poly_order: int = Field(
        default=3, ge=1,
        description="Polynomial order of the Savitzky-Golay filter.",
    )
    buffer: int = Field(
        default=0, ge=0,
        description=(
            "Extra donor periods past the treatment date made visible to the "
            "first-phase alignment. Only meaningful with "
            "``match_method='open.end'``."
        ),
    )
    n_burn: int = Field(
        default=3, ge=0,
        description=(
            "Periods of overlap either side of the cutoff that the "
            "second-phase search sees but does not report, so the post-period "
            "speeds are not distorted by the boundary."
        ),
    )
    ma: int = Field(
        default=3, ge=1,
        description="Centred moving-average width applied to the learned "
                    "pre-period speeds.",
    )
    default_margin: int = Field(
        default=3, ge=0,
        description=(
            "Initial slack added to each candidate pre-period window so the "
            "alignment has room; grown automatically when a window is too "
            "short to span the query."
        ),
    )
    n_q: int = Field(default=1, ge=1,
                     description="Stride of the post-period query window.")
    n_r: int = Field(default=1, ge=1,
                     description="Stride of the pre-period reference window.")
    dist_quant: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description=(
            "Drop post-period windows whose best match costs more than this "
            "quantile of all match costs. 1.0 (default) keeps every window."
        ),
    )
    n_iqr: float = Field(
        default=3.0, ge=0.0,
        description=(
            "Inter-quartile multiplier for discarding outlying speeds before "
            "averaging across windows."
        ),
    )
    step_pattern1: Literal["symmetricP1", "asymmetricP2"] = Field(
        default="symmetricP1",
        description="Step pattern for the first-phase (pre-period) alignment.",
    )
    step_pattern2: Literal["symmetricP1", "asymmetricP2"] = Field(
        default="asymmetricP2",
        description="Step pattern for the second-phase (post-period) search.",
    )
    inference: Literal["none", "placebo"] = Field(
        default="none",
        description=(
            "'placebo' runs Cao & Chadefaux's placebo procedure: every donor "
            "is re-analysed as if it had been treated, with the real treated "
            "unit removed from each placebo pool, and the pointwise quantiles "
            "of the resulting gaps form the band. Also reports the paper's "
            "ICC-corrected efficiency test on the log MSE ratio. Costs one "
            "full warp per donor per perturbed pool, so it is off by default."
        ),
    )
    placebo_pairs: int = Field(
        default=0,
        ge=0,
        description=(
            "How many two-donor-drop perturbations of the pool to add to the "
            "placebo runs, in lexicographic order. The paper uses 100; the "
            "default 0 runs the cheaper leave-one-out construction only."
        ),
    )
    alpha: float = Field(
        default=0.05,
        gt=0.0,
        lt=1.0,
        description="Band coverage is 1 - alpha, pointwise. The paper uses 0.05.",
    )
    mse_window: Optional[int] = Field(
        default=10,
        ge=1,
        description=(
            "Post-treatment periods entering the efficiency test's MSE, "
            "counted from the treatment date. The paper uses 10. None uses "
            "every post-treatment period."
        ),
    )
    match_method: Literal["fixed", "open.end"] = Field(
        default="fixed",
        description=(
            "'fixed' aligns the donor's pre-period as a closed interval; "
            "'open.end' lets the alignment stop short and grows ``buffer`` "
            "until the donor can span the treated unit's pre-period."
        ),
    )

    @model_validator(mode="after")
    def _check_warping_parameters(self) -> "DTWSCConfig":
        if self.filter_width % 2 == 0:
            raise MlsynthConfigError(
                f"DTWSC: 'filter_width' must be odd, got {self.filter_width}."
            )
        if self.filter_width <= self.poly_order:
            raise MlsynthConfigError(
                "DTWSC: 'filter_width' must exceed 'poly_order' "
                f"({self.filter_width} <= {self.poly_order})."
            )
        return self
