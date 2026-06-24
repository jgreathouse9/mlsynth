"""Configuration for the CMBSTS estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import Any, List, Optional
from pydantic import Field, model_validator
from ...exceptions import MlsynthConfigError
from ...config_models import BaseEstimatorConfig

VALID_COMPONENTS = ("trend", "slope", "seasonal", "cycle")


class CMBSTSConfig(BaseEstimatorConfig):
    """Configuration for Causal Multivariate Bayesian Structural Time Series.

    Implements the multivariate Bayesian structural time series causal model of
    Menchetti and Bojinov (2022, Annals of Applied Statistics 16(1): 414-435),
    the multivariate extension of Brodersen et al. (2015) ``CausalImpact``. A
    group of :math:`d` outcome series is modelled jointly; the counterfactual is
    forecast from the posterior predictive distribution and the effect is
    observed-minus-predicted, per series, with credible bands. Under partial
    interference the group's non-treated members carry the spillover.

    The treated unit (``treat`` indicator) is series one; ``group_units`` are the
    other jointly-modelled series (e.g. a competitor brand). ``control_units``
    contribute their outcome paths as regressors and ``covariates`` add exogenous
    columns; together they form the regression block selected by a spike-and-slab
    prior.
    """

    group_units: Optional[List[Any]] = Field(
        default=None,
        description=(
            "Labels of the other units modelled jointly with the treated unit "
            "(the multivariate outcome group, e.g. a competitor brand under "
            "partial interference). None -> univariate (treated unit only)."
        ),
    )
    control_units: Optional[List[Any]] = Field(
        default=None,
        description=(
            "Labels of control units whose outcome paths enter the regression "
            "block as predictors. Must not overlap the treated/group units."
        ),
    )
    control_selection: str = Field(
        default="explicit",
        description=(
            "'explicit' uses ``control_units`` as given; 'dtw' screens "
            "``control_pool`` (or all eligible units) by dynamic time warping "
            "to the treated series and keeps the closest ``n_controls`` "
            "(requires the optional ``fastdtw`` package)."
        ),
    )
    control_pool: Optional[List[Any]] = Field(
        default=None,
        description="Candidate units for control_selection='dtw'. None -> all eligible units.",
    )
    n_controls: Optional[int] = Field(
        default=None,
        ge=1,
        description="Number of controls to keep when control_selection='dtw'.",
    )
    covariates: Optional[List[str]] = Field(
        default=None,
        description=(
            "Exogenous covariate columns in ``df`` (read from the treated unit's "
            "rows; e.g. weekend/holiday dummies, a frozen price). They must be "
            "unaffected by the intervention (Assumption 3)."
        ),
    )
    components: List[str] = Field(
        default_factory=lambda: ["trend", "seasonal"],
        description=(
            "Structural components, a subset of "
            "{'trend','slope','seasonal','cycle'}. 'slope' upgrades the local "
            "level to a local linear trend and requires 'trend'."
        ),
    )
    seas_period: Optional[int] = Field(
        default=None, ge=2, description="Seasonal period S (required if 'seasonal' is used).",
    )
    cycle_period: Optional[int] = Field(
        default=None, ge=2, description="Cycle period (required if 'cycle' is used).",
    )
    niter: int = Field(default=1000, ge=10, description="Total Gibbs iterations (including burn-in).")
    burn: Optional[int] = Field(
        default=None, ge=0, description="Burn-in iterations to discard. None -> niter // 10.",
    )
    ci_alpha: float = Field(
        default=0.05, gt=0.0, lt=1.0, description="Two-sided level for credible bands (0.05 -> 95%).",
    )
    horizon: Optional[int] = Field(
        default=None, ge=1,
        description=(
            "Summarize the effect over the first ``horizon`` post-intervention "
            "periods (e.g. 30 for a one-month horizon on daily data). None -> "
            "the whole post window. The reference package's ``horizon`` argument."
        ),
    )
    prior_scale: float = Field(
        default=0.01, gt=0.0,
        description=(
            "Scale factor on the pre-period outcome covariance for the "
            "Inverse-Wishart prior scale matrices S_eps, S_r (reference default 0.01)."
        ),
    )
    prior_rho: float = Field(
        default=0.0, gt=-1.0, lt=1.0,
        description=(
            "Prior cross-series correlation in the scale matrices (the supermarket "
            "study uses -0.8 to encode substitution)."
        ),
    )
    nu0: Optional[int] = Field(
        default=None, description="Inverse-Wishart degrees of freedom. None -> d + 2.",
    )
    excl_dates: Optional[str] = Field(
        default=None,
        description="Optional 0/1 column in ``df`` flagging post-period dates to drop from the effect summary.",
    )
    seed: Optional[int] = Field(default=None, description="Seed for the numpy Generator inside the sampler.")
    verbose: bool = Field(default=False, description="Print a progress message during sampling.")

    @model_validator(mode="after")
    def _check_cmbsts(self) -> "CMBSTSConfig":
        comps = list(self.components or [])
        if not comps:
            raise MlsynthConfigError("CMBSTS: 'components' must list at least one component.")
        bad = [c for c in comps if c not in VALID_COMPONENTS]
        if bad:
            raise MlsynthConfigError(
                f"CMBSTS: unknown components {bad}; allowed: {list(VALID_COMPONENTS)}.")
        if not any(c in comps for c in ("trend", "seasonal", "cycle")):
            raise MlsynthConfigError(
                "CMBSTS: need at least one of 'trend', 'seasonal', 'cycle' (a level for the series).")
        if "slope" in comps and "trend" not in comps:
            raise MlsynthConfigError("CMBSTS: 'slope' requires 'trend'.")
        if "seasonal" in comps and self.seas_period is None:
            raise MlsynthConfigError("CMBSTS: 'seasonal' component requires 'seas_period'.")
        if "cycle" in comps and self.cycle_period is None:
            raise MlsynthConfigError("CMBSTS: 'cycle' component requires 'cycle_period'.")
        if self.burn is not None and self.burn >= self.niter:
            raise MlsynthConfigError("CMBSTS: 'burn' must be smaller than 'niter'.")
        if self.control_selection not in ("explicit", "dtw"):
            raise MlsynthConfigError("CMBSTS: 'control_selection' must be 'explicit' or 'dtw'.")
        if self.control_selection == "dtw" and self.n_controls is None:
            raise MlsynthConfigError("CMBSTS: control_selection='dtw' requires 'n_controls'.")
        if self.group_units and self.control_units:
            overlap = set(map(str, self.group_units)) & set(map(str, self.control_units))
            if overlap:
                raise MlsynthConfigError(
                    f"CMBSTS: units appear in both group_units and control_units: {sorted(overlap)}.")
        return self
