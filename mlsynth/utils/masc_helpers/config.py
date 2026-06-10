"""Configuration for the MASC estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import Field
from ...config_models import BaseEstimatorConfig


class MASCConfig(BaseEstimatorConfig):
    """Configuration for the MASC estimator (Kellogg et al. 2021).

    Kellogg, Mogstad, Pouliot & Torgovitsky (2021). *Combining
    Matching and Synthetic Control to Trade Off Biases from
    Extrapolation and Interpolation.* JASA 116(536), 1804-1816.
    The estimator forms a convex combination ``phi * matching +
    (1 - phi) * SC`` with the number of neighbours ``m`` and the
    weight ``phi`` jointly chosen by rolling-origin cross-validation.
    """

    covariates: Optional[List[str]] = Field(
        default=None,
        description=(
            "Optional predictor columns added to the SC objective in "
            "place of (or alongside) pre-period outcomes. Each covariate "
            "is row-standardised across all units before the QP -- the "
            "FSCM/MAREX-style ``V = diag(1 / var)`` preconditioning that "
            "matches Abadie's default heuristic V. Per-fold aggregation "
            "is over the fold-specific pre-window unless "
            "``covariate_windows`` overrides it."
        ),
    )
    covariate_windows: Optional[dict] = Field(
        default=None,
        description=(
            "Optional inclusive ``(start, end)`` aggregation window per "
            "covariate label. Covariates without a window are averaged "
            "over each fold's full pre-period."
        ),
    )
    m_grid: Optional[List[int]] = Field(
        default=None,
        description=(
            "Candidate nearest-neighbour counts for cross-validation. "
            "Defaults to ``1..J`` (all values), mirroring the R "
            "reference's default."
        ),
    )
    min_preperiods: Optional[int] = Field(
        default=None, ge=2,
        description=(
            "Smallest fold length used in cross-validation. Folds run "
            "from ``min_preperiods`` to ``treatment_period - 2``. "
            "Defaults to ``ceil(treatment_period / 2)`` per the R "
            "reference. Mutually exclusive with ``set_f``."
        ),
    )
    set_f: Optional[List[int]] = Field(
        default=None,
        description=(
            "Explicit list of fold lengths to use (each integer is the "
            "last pre-treatment period included in that fold's "
            "training window). Mutually exclusive with "
            "``min_preperiods``."
        ),
    )
    fold_weights: Optional[List[float]] = Field(
        default=None,
        description=(
            "Optional positive relative weights for each fold (length "
            "matches the fold count). Normalised to sum to 1. Defaults "
            "to equal weights."
        ),
    )
    forecast_minlength: int = Field(
        default=1, ge=1,
        description=(
            "First post-fold period used for forecast error "
            "evaluation. Fold ``f`` forecasts period ``f + "
            "forecast_minlength``."
        ),
    )
    forecast_maxlength: int = Field(
        default=1, ge=1,
        description=(
            "Last post-fold period used for forecast error "
            "evaluation. Forecast window is capped at "
            "``treatment_period - 1``."
        ),
    )
    solver: Optional[str] = Field(
        default=None,
        description=(
            "cvxpy solver name forwarded to the SC simplex QP. "
            "Defaults to CLARABEL when unset."
        ),
    )
    match_on: Literal["outcomes", "covariates"] = Field(
        default="outcomes",
        description=(
            "Feature space for the nearest-neighbour match. 'outcomes' "
            "(default) matches on the pre-treatment outcome path (the R "
            "reference's default ``Wbar``); 'covariates' matches on the "
            "row-standardised predictor block (the reference's "
            "``solve.covmatch``, used in the Kellogg et al. (2020) Basque "
            "application). Requires ``covariates`` when set to "
            "'covariates'."
        ),
    )
    sc_backend: Literal["mscmt", "bilevel"] = Field(
        default="mscmt",
        description=(
            "Predictor-weight (V) optimiser for the covariate SC step. "
            "'mscmt' (default) uses the MSCMT global search, matching "
            "Abadie's synth() / the Kellogg et al. (2020) reference; "
            "'bilevel' uses the Malo et al. solver shared with FSCM. "
            "Both jointly optimise V and W; they can differ on hard "
            "predictor sets. Ignored when no covariates are supplied."
        ),
    )
