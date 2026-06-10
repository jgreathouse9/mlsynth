"""Configuration for the PROXIMAL estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from pydantic import Field, model_validator
from ...exceptions import MlsynthConfigError
from ...config_models import BaseEstimatorConfig


class PROXIMALConfig(BaseEstimatorConfig):
    """Configuration for the Proximal Inference (PROXIMAL) estimator."""    # Override counterfactual_color from Base to match PROXIMAL's default and usage
    counterfactual_color: Union[str, List[str]] = Field(default_factory=lambda: ["grey", "red", "blue"], description="Color(s) for counterfactual lines in plots. Can be a single color string or a list of color strings for multiple counterfactuals.")

    methods: List[str] = Field(..., min_length=1, description='Which estimators to run. Any of "PI", "PIS", "PIPost", "SPSC", "DR", "PIPW". The estimator runs exactly these, and each method\'s required inputs must be present.')
    donors: List[Union[str, int]] = Field(..., min_length=1, description="List of donor unit identifiers. Must not be empty.")
    surrogates: List[Union[str, int]] = Field(default_factory=list, description="List of surrogate unit identifiers.")
    vars: Dict[str, List[str]] = Field(default_factory=dict, description='Dictionary specifying proxy variables. Requires "donorproxies" when PI/PIS/PIPost is requested; requires "surrogatevars" when PIS/PIPost is requested. SPSC needs no proxies.')

    spsc_detrend: bool = Field(default=True, description="Whether SPSC detrends the treated outcome against a B-spline time trend (SPSC-DT vs SPSC-NoDT).")
    spsc_lambda: Optional[float] = Field(default=None, description="log10 ridge penalty for SPSC. If None, selected by leave-one-out cross-validation.")
    spsc_spline_df: int = Field(default=5, ge=3, description="Degrees of freedom of the SPSC detrend B-spline basis.")
    spsc_basis_degree: int = Field(default=1, ge=1, description="Degree of the polynomial sieve applied to the SPSC treated-outcome instrument. 1 is the linear single proxy; >=2 is the nonparametric (series) SPSC that over-identifies a nonlinear bridge.")
    spsc_conformal: bool = Field(default=False, description="Whether to compute SPSC conformal prediction intervals for the per-period treatment effect.")
    spsc_conformal_periods: Optional[List[int]] = Field(default=None, description="Absolute post-period indices to cover with SPSC conformal intervals. If None, every post-treatment period is covered.")

    @model_validator(mode='after')
    def check_methods_and_vars(cls, values: Any) -> Any:
        valid_methods = {"PI", "PIS", "PIPost", "SPSC", "DR", "PIPW"}
        methods = list(values.methods)
        unknown = [m for m in methods if m not in valid_methods]
        if unknown:
            raise MlsynthConfigError(
                f"Unknown PROXIMAL method(s) {unknown}. Valid choices: 'PI', 'PIS', 'PIPost', 'SPSC', 'DR', 'PIPW'."
            )

        vars_dict = values.vars
        needs_donorproxies = any(m in methods for m in ("PI", "PIS", "PIPost", "DR", "PIPW"))
        needs_surrogates = any(m in methods for m in ("PIS", "PIPost"))

        if needs_donorproxies and not (isinstance(vars_dict.get("donorproxies"), list) and vars_dict.get("donorproxies")):
            raise MlsynthConfigError(
                "Config 'vars' must contain a non-empty list for 'donorproxies' when PI/PIS/PIPost is requested."
            )
        if needs_surrogates:
            if not values.surrogates:
                raise MlsynthConfigError("PIS/PIPost require a non-empty 'surrogates' list.")
            if not (isinstance(vars_dict.get("surrogatevars"), list) and vars_dict.get("surrogatevars")):
                raise MlsynthConfigError(
                    "Config 'vars' must contain a non-empty list for 'surrogatevars' when PIS/PIPost is requested."
                )
        return values
