"""Configuration for the FSCM estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import Field
from ...config_models import BaseEstimatorConfig


class FSCMConfig(BaseEstimatorConfig):
    """
    Configuration for the Forward-Selected Synthetic Control Method (FSCM).

    FSCM grows a nested donor sequence by forward stepwise selection on the
    training half of the pre-period (greedy on in-sample RMSPE), then chooses
    the donor count by minimizing out-of-sample RMSPE on the held-out test
    half (two-interval-time cross-validation). The final simplex weights are
    refit on the full pre-period over the selected donors.

    References
    ----------
    Cerulli, Giovanni. 2024.
    "Optimal initial donor selection for the synthetic control method."
    Economics Letters, 244: 111976.
    https://doi.org/10.1016/j.econlet.2024.111976
    """

    forward_selection: bool = Field(
        default=True,
        description=(
            "If True, run Cerulli's forward stepwise donor selection with "
            "rolling-origin out-of-sample validation (each candidate fit by "
            "the bilevel solver). If False, take the full bilevel solve over "
            "all donors with no selection."
        ),
    )

    covariates: Optional[List[str]] = Field(
        default=None,
        description=(
            "Optional covariate columns for predictor matching (Abadie's "
            "specification). Each covariate is averaged over its window and "
            "enters the bilevel lower-level objective; the predictor weights "
            "V are optimized on the full pool and reused. Selection and "
            "cross-validation scores are measured on the outcome."
        ),
    )

    covariate_windows: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Per-covariate averaging window as an inclusive (start, end) range "
            "of time labels, e.g. {'lnincome': (1980, 1988), 'beer': (1984, "
            "1988)}. Covariates not listed are averaged over the full "
            "pre-treatment period. Mirrors Abadie's Proposition 99 spec."
        ),
    )

    match_periods: Optional[List[Any]] = Field(
        default=None,
        description=(
            "Optional 'special predictor' periods: specific pre-treatment "
            "time labels (e.g. [1975, 1980, 1988]) whose outcome value is "
            "matched directly, as in Abadie's Proposition 99 specification."
        ),
    )

    cv_split: float = Field(
        default=0.5,
        gt=0.0,
        lt=1.0,
        description=(
            "Fraction of the pre-treatment period used as the training window; "
            "the remaining tail is the test window for cross-validation. "
            "0.5 reproduces Cerulli's equal split."
        ),
    )

    max_donors: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Cap on the number of forward-selection steps. Defaults to the "
            "full donor pool; lower it to bound runtime in high dimensions."
        ),
    )
