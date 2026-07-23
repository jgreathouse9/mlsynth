"""Configuration for the CSCM (flexible count synthetic control) estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from ...config_models import BaseEstimatorConfig
from ...exceptions import MlsynthConfigError, MlsynthDataError


class CSCMConfig(BaseEstimatorConfig):
    """Configuration for the flexible count synthetic control (CSCM).

    Implements Bonander (2021, *Epidemiology* 32(4):e18-e19), a synthetic
    control for count data and other non-negative outcomes. The donor weights
    keep the non-negativity constraint (so the counterfactual stays
    non-negative, valid for counts) but drop the adding-up (sum-to-one)
    constraint, penalising departures from the classic simplex weights by
    ``lambda * ||w - w_scm||^2``. The effect is reported as a rate ratio, with
    a cross-fitted, bias-corrected confidence interval (Chernozhukov, Wuthrich
    & Zhu 2021).
    """

    K: int = Field(
        default=2,
        ge=2,
        description=(
            "Number of cross-fitting folds for the rate-ratio inference. K=2 "
            "has better coverage on short pre-periods; K=3 needs a longer "
            "pre-period (Bonander 2021, App. 3). The t-CI has K-1 degrees of "
            "freedom, so small K gives a wide interval."
        ),
    )
    n_lambda: int = Field(
        default=20,
        ge=1,
        description="Length of the penalty (lambda) path searched by CV.",
    )
    lambda_min_ratio: float = Field(
        default=1e-16,
        gt=0.0,
        description=(
            "Ratio of the smallest to largest lambda on the path. Small values "
            "allow near-unpenalised (heavily relaxed) fits."
        ),
    )
    min_1se: bool = Field(
        default=False,
        description=(
            "Select lambda by the one-standard-error rule (more shrinkage "
            "toward the simplex SCM) instead of the CV minimum."
        ),
    )
    ci_level: float = Field(
        default=0.95,
        gt=0.0,
        lt=1.0,
        description="Confidence level for the rate-ratio interval.",
    )
    v_method: Literal["poisson_ridge", "uniform"] = Field(
        default="poisson_ridge",
        description=(
            "Predictor-importance matrix V. 'poisson_ridge' (default) weights "
            "each balance feature by |coef|/sum|coef| from a leave-one-out "
            "Poisson ridge of the controls' post-period mean on the features "
            "(Bonander 2021, App. 2). 'uniform' weights all features equally "
            "(1/(k+T0)), the simple default."
        ),
    )

    @model_validator(mode="after")
    def _check_nonnegative_outcome(self):
        vals = np.asarray(self.df[self.outcome], dtype=float)
        finite = vals[np.isfinite(vals)]
        if finite.size and float(finite.min()) < 0.0:
            raise MlsynthDataError(
                "CSCM requires a non-negative outcome (count/rate data); "
                f"column {self.outcome!r} has negative values."
            )
        return self

    @model_validator(mode="after")
    def _check_k(self):
        # ge=2 is enforced by Field; keep an explicit translated error too.
        if self.K < 2:  # pragma: no cover - Field ge=2 rejects K<2 first
            raise MlsynthConfigError("K must be an integer >= 2.")
        return self
