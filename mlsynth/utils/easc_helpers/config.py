"""Configuration for EASC, the exposure-augmented synthetic control (Spruk 2026).

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import Field, model_validator

from ...config_models import BaseEstimatorConfig
from ...exceptions import MlsynthConfigError, MlsynthDataError

#: The phase name reserved for the whole post-treatment period.
OVERALL_PHASE = "post"


def _ordered(start: Any, end: Any) -> bool:
    """``start <= end`` where the labels can be compared; ``True`` where they cannot."""
    try:
        return bool(start <= end)
    except TypeError:
        return True


class EASCConfig(BaseEstimatorConfig):
    """Configuration for the exposure-augmented synthetic control (EASC).

    Spruk (2026) studies an observed common shock -- a commodity price, an
    exchange rate -- that moves with the latent factors before treatment and
    departs from them afterwards. Pre-treatment fit then says almost nothing
    about the synthetic unit's exposure to the shock. EASC corrects the
    synthetic control's exposure imbalance with a cross-donor regression of
    outcome changes on a pre-determined exposure index, reports the diagnostics
    that detect the problem, and gives bias-aware intervals with a breakdown
    value.
    """

    exposure: str = Field(
        ...,
        description="Column holding each unit's exposure to the shock (e.g. oil rents as a share of GDP). The index s_j is the unit's mean of this column over exposure_window; it must be measured before treatment.",
    )
    exposure_window: Optional[Tuple[Any, Any]] = Field(
        default=None,
        description="First and last period (inclusive time labels) over which the exposure column is averaged. Must end before treatment. None averages over every pre-treatment period.",
    )
    shock: str = Field(
        ...,
        description="Column holding the observed common shock x_t (e.g. the log oil price), identical across units in each period. Used for the leverage diagnostics and the bias bound, not for the estimate.",
    )
    method: Literal["easc", "ebsc_min", "easc_tuned"] = Field(
        default="easc",
        description="Headline estimator. 'easc' corrects the synthetic-control weights (the paper's EASC); 'ebsc_min' is the best-fitting simplex weights at the smallest attainable exposure imbalance, uncorrected; 'easc_tuned' corrects the frontier point chosen by the paper's equation (12) (EASC*). All four fits, including plain SC, are returned either way.",
    )
    n_factors: int = Field(
        default=2,
        description="Number k of latent factors estimated from the donors for the leverage diagnostics and the bias bound. The EASC estimate itself does not depend on it; the breakdown value does.",
    )
    n_subperiods: int = Field(
        default=3,
        description="Number of equal pre-period sub-periods whose growth enters the exposure regression as controls for donors' latent loadings (Assumption 3). 0 drops the controls.",
    )
    base_period: Optional[Any] = Field(
        default=None,
        description="Base period b of the exposure regression's outcome changes Y_jt - Y_jb (a pre-treatment time label). None uses the last pre-treatment period.",
    )
    frontier_points: int = Field(
        default=41,
        description="Number of exposure tolerances on the exposure-fit frontier, evenly spaced from the smallest attainable imbalance to the synthetic-control imbalance.",
    )
    phases: Optional[Dict[str, Tuple[Any, Any]]] = Field(
        default=None,
        description="Post-treatment phases for the inference, as name -> (first period, last period), inclusive time labels. The whole post-period is always reported under the name 'post'.",
    )
    inference: bool = Field(
        default=True,
        description="Whether to compute the in-space placebos, the bias-aware intervals and the breakdown values for SC and EASC.",
    )
    alpha: float = Field(
        default=0.05,
        description="Miscoverage of the bias-aware intervals (0.05 for 95 percent intervals).",
    )
    m_grid: List[float] = Field(
        default_factory=lambda: [0.0, 0.5, 1.0],
        description="Values of the sensitivity parameter M (the share of the pre-treatment RMSPE treated as systematic) at which intervals are reported.",
    )
    headline_m: float = Field(
        default=0.0,
        description="The M used for the interval in the standardized inference results (the EASC effect over the whole post-period). M = 0 ignores bias; the breakdown value says how large M can be before the conclusion changes.",
    )
    correction_se: Literal["period_mean", "phase"] = Field(
        default="period_mean",
        description="How the correction's sampling error enters EASC's noise scale: 'period_mean' adds |EI| times the phase mean of the per-period jackknife standard errors (the reference code, and the paper's tables); 'phase' uses the jackknife standard error of the phase average of phi (the paper's text). The first is never smaller.",
    )

    @model_validator(mode="after")
    def _check_easc(self) -> "EASCConfig":
        for name in ("exposure", "shock"):
            column = getattr(self, name)
            if column not in self.df.columns:
                raise MlsynthDataError(f"EASC: the {name} column {column!r} is not in the DataFrame.")
        if self.n_factors < 1:
            raise MlsynthConfigError(f"n_factors must be >= 1; got {self.n_factors}.")
        if self.n_subperiods < 0:
            raise MlsynthConfigError(f"n_subperiods must be >= 0; got {self.n_subperiods}.")
        if self.frontier_points < 2:
            raise MlsynthConfigError(f"frontier_points must be >= 2; got {self.frontier_points}.")
        if not 0.0 < self.alpha < 1.0:
            raise MlsynthConfigError(f"alpha must lie strictly between 0 and 1; got {self.alpha}.")
        if not self.m_grid:
            raise MlsynthConfigError("m_grid must hold at least one value of M.")
        if any(m < 0 for m in self.m_grid) or self.headline_m < 0:
            raise MlsynthConfigError("The sensitivity parameter M must be >= 0.")
        if self.exposure_window is not None and not _ordered(*self.exposure_window):
            raise MlsynthConfigError(f"exposure_window must run forwards; got {self.exposure_window}.")
        if self.phases is not None:
            if not self.phases:
                raise MlsynthConfigError("phases, when given, must name at least one phase.")
            if OVERALL_PHASE in self.phases:
                raise MlsynthConfigError(
                    f"The phase name {OVERALL_PHASE!r} is reserved for the whole post-period.")
            for name, (start, end) in self.phases.items():
                if not _ordered(start, end):
                    raise MlsynthConfigError(f"Phase {name!r} must run forwards; got {(start, end)}.")
        return self
