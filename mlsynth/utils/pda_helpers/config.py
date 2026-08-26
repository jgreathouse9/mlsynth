"""Configuration for the PDA estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import Field, field_validator, model_validator
from ...config_models import BaseEstimatorConfig
from ...exceptions import MlsynthConfigError


class PDAConfig(BaseEstimatorConfig):
    """Configuration for the Panel Data Approach (PDA) estimator."""
    method: str = Field(default="fs", description="Type of PDA to use: 'LASSO', 'l2', 'fs', or 'hcw' (original Hsiao-Ching-Wan best-subset).", pattern="^(LASSO|l2|fs|hcw)$")
    methods: Optional[List[str]] = Field(default=None, description="Optional list of PDA variants to run; overrides `method` when set.")
    hcw_criterion: str = Field(default="AICc", pattern="^(AICc|AIC|BIC)$", description="HCW only: model-selection criterion for the best-subset donor search ('AICc' default, matching pampe / HCW Table XVI; also 'AIC' or 'BIC').")
    hcw_nvmax: Optional[int] = Field(default=None, gt=0, description="HCW only: largest donor-subset size searched (pampe's nvmax). None searches up to all donors (bounded by the pre-period OLS df). Cap it for large donor pools, since best-subset is combinatorial.")
    hcw_backend: str = Field(default="fw", pattern="^(fw|scip)$", description="HCW only: best-subset search engine. 'fw' (default) is the exact Furnival-Wilson branch-and-bound, which certifies the optimum for small pools and otherwise returns the best incumbent with an optimality gap. 'scip' uses the optional SCIP mixed-integer solver (requires pyscipopt) to certify the optimum at larger pool sizes.")
    tau: Optional[float] = Field(default=None, description="User-specified treatment effect value (used as tau_l2 for 'l2' method).")
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0, description="Significance level for confidence intervals and ATE inference.")
    lasso_criterion: str = Field(default="cv", pattern="^(cv|mbic)$", description="LASSO only: rule for the penalty. 'cv' (default) cross-validates it with an intercept, as Li & Bell (2017) describe. 'mbic' reproduces fsPDA's lasso.BIC -- Shi & Huang's modified BIC log(sigma^2) + H log(log N) log(T1)/T1 k over their grid seq(0.01, 1, by = 0.01), fitted without an intercept and with glmnet's column scaling.")
    lasso_mbic_const: float = Field(default=2.0, gt=0.0, description="LASSO only: the constant H in the modified BIC, used when lasso_criterion='mbic'. Their default is 2; the paper describes tuning it 'to allow Lasso to take in more variables', and a smaller value selects more donors.")
    fs_intercept: bool = Field(default=False, description="Forward-selection only: include a constant in the donor regression. False (default) matches Shi & Huang's simulation (no intercept, valid size on mean-zero factor data); True matches the released fsPDA R package (intercept, for panels with genuine level differences).")
    lrvar_lag: Optional[int] = Field(default=None, description="Bartlett-kernel truncation lag for the long-run variance in the post-selection t-test, read by the fs, hcw and LASSO methods. Supplying it selects the released fsPDA package's test, Z = ATE / sqrt(lrvar(gap, lag) / T2), for all three; a supplied value must be a non-negative integer no larger than floor(sqrt(T2)). None (default) leaves each method on its own paper's inference: fs and hcw on the prewhitened Newey-West variance Shi & Huang use in their applications, LASSO on Li & Bell's two-component variance.")
    l2_standardize: bool = Field(default=True, description="L2-relaxation only: standardise (demean + unit-variance scale) the treated and control series before solving, matching the authors' released L2relax (the default). The penalty is scale-sensitive, so standardisation is recommended; set False for the raw-scale variant.")
    prediction_intervals: bool = Field(default=False, description="Attach Jiang, Li, Shen & Zhou (2025) bootstrap prediction intervals for the per-period treatment effect and counterfactual to every fitted PDA variant. Equal-tailed and symmetric intervals are returned; each variant reports whether the post-selection OLS HAC sandwich studentization was used or the sigma^2-only fallback (e.g. for the dense L2-relaxation in high dimensions).")
    pi_n_boot: int = Field(default=999, ge=2, description="Number of bootstrap replications for the prediction intervals (only used when prediction_intervals is True).")
    pi_seed: Optional[int] = Field(default=0, description="Seed for the prediction-interval bootstrap RNG (reproducible by default; set None for a fresh draw).")
    pi_dependent: bool = Field(default=True, description="Resample the pre-period prediction error with the dependent wild bootstrap of Jiang et al. (2025) Algorithm 2.1 -- Bartlett-correlated multipliers whose dependence range is a bandwidth in T0, so a persistent error is resampled as a persistent error. False uses the ordinary i.i.d. standard-normal multipliers of their Remark 2.2, which is cheaper and valid only when the errors are already independent. The choice compounds in the cumulative band, which accumulates the period errors, so drawing persistent errors as independent understates how fast the running total's uncertainty grows. True (the paper's algorithm) by default.")
    cumulative_band: bool = Field(default=False, description="Attach a simultaneous (sup-t) band for the CUMULATIVE effect path -- the running total over post-periods, with one shared critical value so the whole path is covered at 1 - alpha at once, which is how a cumulative path is read. Built from the replicate paths the prediction-interval bootstrap already produces, so it costs no extra refits. Its growth is measured rather than assumed: the replicates are accumulated before the standard error is taken, so independent period errors widen it like sqrt(L) and perfectly correlated ones like L, where adding up per-period interval endpoints would always give the latter. Needs prediction_intervals. Off by default.")
    cumulative_method: Literal["bootstrap", "resample"] = Field(default="bootstrap", description="How the cumulative band is built. 'bootstrap' (default) accumulates the replicate paths the Jiang et al. (2025) prediction-interval bootstrap already produced, so it needs prediction_intervals and costs pi_n_boot refits. 'resample' calibrates on a rolling-origin pass instead -- one refit per origin, roughly a tenth of the cost -- and block-resamples those out-of-sample per-period errors into paths, which is Wheeler's LassoSynth construction generalised to serially correlated periods. It needs no bootstrap, so it does not need prediction_intervals.")
    cumulative_block: int = Field(default=0, ge=0, description="Block length in periods for the post-period draw the cumulative band accumulates, under both cumulative_method settings. 0 (default) means the whole horizon, the longest block the accumulated total is sensitive to. 1 draws periods independently, which is Wheeler's original for the resample method and Algorithm 2.1's out-of-sample draw for the bootstrap method, and is too narrow whenever the period errors are positively autocorrelated -- over six periods a total of AR(0.6) errors is 1.68 times as variable as a total of independent ones, and AR(0.8) 2.02 times. A block longer than the horizon is clamped to it. Under the bootstrap method this reaches the accumulated paths alone: the per-period prediction intervals keep Algorithm 2.1's i.i.d. out-of-sample draw and do not move with it.")
    cumulative_n_sim: int = Field(default=2000, ge=2, description="resample only: number of error paths to draw. Unlike pi_n_boot these cost no refits -- the refits are the rolling origins -- so this buys quantile precision cheaply.")

    @model_validator(mode="after")
    def _check_cumulative_band(self):
        if (self.cumulative_band and not self.prediction_intervals
                and self.cumulative_method == "bootstrap"):
            raise MlsynthConfigError(
                "cumulative_band needs prediction_intervals=True: the band is "
                "built from the replicate paths the Jiang et al. (2025) bootstrap "
                "produces, and with the bootstrap off there are none. Set "
                "cumulative_method='resample' to calibrate on a rolling-origin "
                "pass instead, which needs no bootstrap."
            )
        return self

    @field_validator("pi_dependent", "prediction_intervals", "cumulative_band",
                     mode="before")
    @classmethod
    def _strict_bool(cls, v, info):
        if not isinstance(v, bool):
            raise MlsynthConfigError(
                f"{info.field_name} must be True or False; got {v!r}. Pydantic "
                "would otherwise coerce a string or a number into a boolean, so "
                "a typo becomes a silent choice about how inference is run."
            )
        return v

    @field_validator("cumulative_method", mode="before")
    @classmethod
    def _known_cumulative_method(cls, v):
        """Refuse an unknown construction by name.

        Pydantic would report a ``Literal`` mismatch, but the band is the number a
        reader quotes, so the message names the two constructions and what each
        costs.
        """
        if v not in ("bootstrap", "resample"):
            raise MlsynthConfigError(
                f"cumulative_method must be 'bootstrap' or 'resample'; got {v!r}. "
                "'bootstrap' accumulates the prediction-interval replicates and "
                "needs prediction_intervals; 'resample' calibrates on a "
                "rolling-origin pass and needs no bootstrap."
            )
        return v

    @field_validator("cumulative_block", "cumulative_n_sim", mode="before")
    @classmethod
    def _strict_int(cls, v, info):
        """Refuse a non-integer outright.

        Pydantic would coerce ``2.5`` or ``"3"``, and a silently rounded block
        length is a silent change to how much serial correlation the band carries.
        """
        if isinstance(v, bool) or not isinstance(v, int):
            raise MlsynthConfigError(
                f"{info.field_name} must be an integer; got {v!r}."
            )
        return v
