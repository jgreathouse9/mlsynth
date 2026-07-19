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
    spsc_att_degree: int = Field(default=0, ge=0, description="Polynomial degree of the SPSC ATT basis in post-treatment time (the reference att.ft). 0 is a constant ATT; 1 is a linear effect path (the authors' California example). With att_degree>0 the headline att is the mean of the fitted effect path; the per-period path and its CIs are in the fit metadata.")
    spsc_detrend_basis: str = Field(default="bspline", pattern="^(bspline|poly)$", description="SPSC detrend trend family: 'bspline' (cubic B-spline, df=spsc_spline_df) or 'poly' (polynomial [1, t, ..., t^spsc_detrend_degree]; degree 1 is the linear (1,t) trend of the authors' California example).")
    spsc_detrend_degree: int = Field(default=1, ge=1, description="Polynomial degree of the SPSC detrend trend when spsc_detrend_basis='poly'.")
    spsc_conformal: bool = Field(default=False, description="Whether to compute SPSC conformal prediction intervals for the per-period treatment effect.")
    spsc_conformal_periods: Optional[List[int]] = Field(default=None, description="Absolute post-period indices to cover with SPSC conformal intervals. If None, every post-treatment period is covered.")

    outcome_instruments: List[Union[str, int]] = Field(default_factory=list, description='Over-identified DR ("DR-OID") and over-identified PI ("PIOID"): unit identifiers whose outcome series instrument the outcome bridge h (the distinct set of donor units used as instruments). Required when "DR-OID" or "PIOID" is requested.')
    pioid_hac_lag: int = Field(default=10, ge=0, description='Over-identified PI ("PIOID"): Newey-West / Bartlett truncation lag for the ATT standard error. Defaults to 10, the lag Shi et al. (2026)\'s NC_nocov_gmm uses; the point estimate is unaffected. Other PROXIMAL methods keep their own HAC bandwidth.')
    pioid_simplex: bool = Field(default=False, description='Over-identified PI ("PIOID"): constrain the donor coefficients to the simplex (omega>=0, sum=1) under the Z\'Z metric -- the authors\' cPI (constrained proximal inference). No GMM standard error is reported for the constrained fit (the paper\'s constrained inference is by permutation).')
    pioid_band: bool = Field(default=False, description='Over-identified PI ("PIOID"): also compute a per-period counterfactual prediction band (Shi et al. 2026, Section 3.2). Off by default; only the unconstrained GMM fit supports it.')
    pioid_band_method: str = Field(default="gmm", pattern="^(gmm|conformal)$", description='PIOID per-period band route: "gmm" (Section 3.2.3 delta method on the joint (tau, omega) sandwich, the default) or "conformal" (Section 3.2.1 / Chernozhukov-Wuthrich-Zhu 2021 split-conformal on the pre-period residuals).')
    pioid_band_level: float = Field(default=0.90, gt=0.0, lt=1.0, description="PIOID per-period band nominal coverage (e.g. 0.90 for a 90% band).")
    pioid_overid_test: bool = Field(default=True, description='Over-identified PI ("PIOID"): report the Hansen (1982) J-test of the over-identifying restrictions -- a proxy-validity falsification check on the pre-period moment conditions Z_t(Y_t - W_t omega). Computed only for the unconstrained GMM fit when strictly over-identified (more instruments than donors); a small p-value is evidence the proximal identifying assumptions fail for some instrument.')
    pioid_overid_hac_lag: int = Field(default=0, ge=0, description="PIOID over-identification (J) test: Bartlett/Newey-West bandwidth for the moment covariance in the J-statistic. Separate from pioid_hac_lag (which governs the ATT standard error): the SE wants a wide bandwidth for conservative coverage, but the J-test wants one matched to the moment dependence -- 0 is exactly calibrated under the paper's classical i.i.d.-error setting (nominal size, best power) while a wide bandwidth over-smooths the test toward conservatism. Raise it to match genuine serial correlation in the moments.")
    treatment_instruments: List[Union[str, int]] = Field(default_factory=list, description='Over-identified DR ("DR-OID"): the (smaller) subset of unit identifiers whose outcome series enter the treatment bridge q. Required when "DR-OID" is requested.')
    dr_oid_ridge: float = Field(default=0.0, ge=0.0, description='Over-identified DR ("DR-OID"): L2 penalty on the treatment-bridge coefficients (excluding intercept), regularising the flat exp(Z beta) valley that arises with few instruments. 0 reproduces the unregularised optimum.')
    dr_oid_n_starts: int = Field(default=8, ge=1, description='Over-identified DR ("DR-OID"): multistart restarts for the treatment-bridge solve (a basin-disagreement diagnostic is reported in the fit metadata).')

    @model_validator(mode='after')
    def check_methods_and_vars(cls, values: Any) -> Any:
        valid_methods = {"PI", "PIS", "PIPost", "SPSC", "DR", "PIPW", "DR-OID", "PIOID"}
        methods = list(values.methods)
        unknown = [m for m in methods if m not in valid_methods]
        if unknown:
            raise MlsynthConfigError(
                f"Unknown PROXIMAL method(s) {unknown}. Valid choices: 'PI', 'PIS', 'PIPost', 'SPSC', 'DR', 'PIPW', 'DR-OID', 'PIOID'."
            )
        if "DR-OID" in methods:
            if not values.outcome_instruments:
                raise MlsynthConfigError("DR-OID requires a non-empty 'outcome_instruments' list (the proxy pool for the outcome bridge).")
            if not values.treatment_instruments:
                raise MlsynthConfigError("DR-OID requires a non-empty 'treatment_instruments' list (the subset for the treatment bridge).")
        if "PIOID" in methods and not values.outcome_instruments:
            raise MlsynthConfigError(
                "PIOID requires a non-empty 'outcome_instruments' list (the distinct set of "
                "donor units instrumenting the outcome bridge)."
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
