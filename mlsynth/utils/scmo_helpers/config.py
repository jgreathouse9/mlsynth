"""Configuration for the SCMO estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from pydantic import Field
from ...config_models import BaseEstimatorConfig


class SCMOConfig(BaseEstimatorConfig):
    """Configuration for the Synthetic Control with Multiple Outcomes (SCMO) estimator."""

    addout: Union[str, List[str]] = Field(
        default_factory=list,
        description="Auxiliary outcome variable(s) for outcome stacking. Used to build the matching spec when `spec` is not given.",
    )
    method: str = Field(
        default="TLP",
        description="Legacy method selector: 'TLP', 'SBMF', or 'BOTH'. Maps to schemes when `schemes` is not given.",
        pattern="^(TLP|SBMF|BOTH)$",
    )
    conformal_alpha: float = Field(
        default=0.1,
        description="Miscoverage rate for conformal prediction intervals (e.g., 0.1 for 90% CI).",
        gt=0, lt=1,
    )
    spec: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Spec-driven matching matrix: {'year': int|list[int], 'vars': {name: column | (column, op)}}, op in {level,log,per_capita,raw}. If None, built from outcome+addout over the pre-period.",
    )
    schemes: Optional[List[str]] = Field(
        default=None,
        description="Weighting schemes to run: any of 'concatenated','averaged','separate','MA'. If None, derived from `method` (TLP->concatenated, SBMF->averaged, BOTH->[concatenated,averaged,MA]).",
    )
    demean: bool = Field(
        default=False,
        description="Intercept-shift the counterfactual (Doudchenko-Imbens / Sun-Ben-Michael-Feller level adjustment).",
    )
    conformal_q: float = Field(
        default=1.0,
        description="Norm exponent q of the CWZ conformal test statistic S_q (1 = average effect; larger targets sparse/large effects across outcomes).",
        gt=0,
    )
