"""Configuration for the SCMO estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import Field, model_validator
from ...config_models import BaseEstimatorConfig
from ...exceptions import MlsynthConfigError


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
    augment: Optional[Literal["ridge"]] = Field(
        default=None,
        description="If 'ridge', augment each scheme's simplex SC fit with the bilevel ridge-augmented solver (Ben-Michael Augmented SCM; the augsynth progfunc='ridge' default). None -> plain simplex. Combine with demean=True to match augsynth's fixedeff=True.",
    )
    ridge_lambda: Optional[float] = Field(
        default=None,
        description="Fixed ridge penalty for augment='ridge'; None selects it by leave-one-period-out cross-validation (augsynth's 1-SE rule).",
    )
    weights: Literal["simplex", "pcr"] = Field(
        default="simplex",
        description="Weight solver on the matching matrix. 'simplex' -> convex SC weights (default). 'pcr' -> denoised unconstrained principal-component-regression weights (mRSC, Amjad-Misra-Shah-Shen 2019; reuses the pcr/core HSVT+PCR kernel). Intended for the concatenated scheme; weights may be negative and need not sum to one.",
    )
    pcr_rank: Optional[int] = Field(
        default=None,
        description="HSVT truncation rank for weights='pcr'. None selects it by the cumulative-variance rule (pcr_cumvar); an explicit integer forces a fixed rank.",
    )
    pcr_cumvar: float = Field(
        default=0.95,
        description="Cumulative-variance target for the weights='pcr' rank rule when pcr_rank is None.",
        gt=0, le=1,
    )

    @model_validator(mode="after")
    def _check_augment(self):
        if self.ridge_lambda is not None:
            if self.augment != "ridge":
                raise MlsynthConfigError(
                    "ridge_lambda is only used with augment='ridge'.")
            if self.ridge_lambda <= 0:
                raise MlsynthConfigError(
                    f"ridge_lambda must be > 0; got {self.ridge_lambda}.")
        if self.weights == "pcr" and self.augment == "ridge":
            raise MlsynthConfigError(
                "weights='pcr' is incompatible with augment='ridge' "
                "(they are alternative weight solvers).")
        if self.pcr_rank is not None:
            if self.weights != "pcr":
                raise MlsynthConfigError(
                    "pcr_rank is only used with weights='pcr'.")
            if self.pcr_rank < 1:
                raise MlsynthConfigError(
                    f"pcr_rank must be a positive integer; got {self.pcr_rank}.")
        return self
