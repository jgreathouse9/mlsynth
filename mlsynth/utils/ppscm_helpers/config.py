"""Configuration for the PPSCM estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import Any, List, Literal, Optional, Union
from pydantic import Field, field_validator, model_validator
from ...config_models import BaseEstimatorConfig
from ...exceptions import MlsynthConfigError


class PPSCMConfig(BaseEstimatorConfig):
    """Configuration for the Partially Pooled SCM (PPSCM) estimator.

    Implements Ben-Michael, Feller & Rothstein (2022, *JRSS-B*
    84(2):351-381). Targets staggered-adoption designs by minimizing a
    weighted average of the per-treated-unit imbalance ``q_sep`` and
    the average-treated imbalance ``q_pool``, with weighting hyper-
    parameter ``nu``.
    """

    nu: Union[float, Literal["auto"]] = Field(
        default="auto",
        description=(
            "Pooling parameter. Small nu approaches a separate SCM per treated "
            "unit, large nu a fully pooled SCM (nu weights the pooled balance "
            "term). 'auto' (default) uses the triangle-inequality ratio "
            "global_l2 * sqrt(d) / avg_l2 of the separate fit, matching "
            "augsynth's heuristic."
        ),
    )
    fixedeff: bool = Field(
        default=True,
        description=(
            "Include two-way fixed effects (time effect from never-treated "
            "units + per-cohort unit pre-mean) and balance the residuals, as "
            "in augsynth (force=3). False removes only the control time means."
        ),
    )
    n_leads: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Number of post-treatment horizons (relative time) to estimate. "
            "None defaults to the number of post-treatment periods of the last "
            "treated unit."
        ),
    )
    n_lags: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Number of pre-treatment periods to balance. None balances all "
            "pre-treatment periods."
        ),
    )
    time_cohort: bool = Field(
        default=False,
        description=(
            "If True, collapse units sharing an adoption time into one "
            "fully-pooled cohort (one synthetic control per cohort)."
        ),
    )
    donor_weights: Literal["scm", "uniform"] = Field(
        default="scm",
        description=(
            "How donors are weighted. 'scm' solves the partially-pooled QP. "
            "'uniform' puts equal weight on every admissible donor, which is "
            "the comparison Callaway-Sant'Anna and Sun-Abraham make. It is the "
            "'lam' -> infinity limit of the QP in closed form: the barycenter "
            "minimises the ridge penalty over the simplex, so the solved "
            "weights approach it at rate O(1/lam) for every 'nu'. Setting it "
            "reaches the limit exactly and skips the program."
        ),
    )
    base_period: Literal["all_pre", "pre_treatment"] = Field(
        default="all_pre",
        description=(
            "Baseline for the unit fixed effect. 'all_pre' is each unit's mean "
            "over its whole pre-adoption window, which is augsynth's. "
            "'pre_treatment' is the single period before adoption, which is "
            "what Callaway-Sant'Anna normalises against. The choice shifts each "
            "cohort's level without moving the event-study shape."
        ),
    )
    donor_pool: Literal["window", "never_treated", "not_yet_treated"] = Field(
        default="window",
        description=(
            "Which units may serve as donors. 'window' admits any unit "
            "untreated through the cohort's estimation window (augsynth). "
            "'never_treated' and 'not_yet_treated' are the Callaway-Sant'Anna "
            "comparison groups. 'window' and 'never_treated' coincide exactly "
            "when every other cohort adopts inside the window."
        ),
    )
    method: Optional[Literal["callaway_santanna"]] = Field(
        default=None,
        description=(
            "Convenience preset. 'callaway_santanna' sets donor_weights="
            "'uniform', base_period='pre_treatment' and donor_pool="
            "'never_treated', which together reproduce the Callaway-Sant'Anna "
            "and Sun-Abraham estimator to machine precision wherever the donor "
            "pools coincide. Explicitly-set conventions are not overridden."
        ),
    )
    lam: float = Field(
        default=0.0,
        ge=0.0,
        description="L2 regularization on the donor weights.",
    )
    solver: Any = Field(
        default=None,
        description="CVXPY solver. None falls back to OSQP.",
    )
    run_inference: bool = Field(
        default=True,
        description=(
            "Whether to run inference (see ``inference_method``); refits or "
            "reweights the estimator, can be slow for the jackknife."
        ),
    )
    inference_method: str = Field(
        default="jackknife",
        description=(
            "Inference procedure: 'jackknife' (delete-one, refit per unit), "
            "'bootstrap' (augsynth's default Mammen wild/multiplier bootstrap; "
            "reweights the single fit, no refit), or 'influence_function' "
            "(Callaway-Sant'Anna's analytical standard errors, one pass over the "
            "panel and no refit). The augsynth multisynth vignette prints the "
            "bootstrap SEs. 'influence_function' is the interval that goes with "
            "the Callaway-Sant'Anna point estimate, and its derivation assumes "
            "the three conventions that produce that estimate, so it is "
            "available only with donor_weights='uniform', "
            "base_period='pre_treatment', donor_pool='never_treated' and "
            "fixedeff=True; method='callaway_santanna' selects it."
        ),
    )
    cband: bool = Field(
        default=False,
        description=(
            "Tabulate a simultaneous (uniform) band over event-time horizons "
            "alongside the pointwise one, by Mammen multiplier bootstrap on the "
            "influence functions. One critical value covers every horizon at "
            "1 - alpha, which is the level a reader assumes when they read the "
            "path as a path; the pointwise band read that way covers less. Needs "
            "inference_method='influence_function'. Off by default."
        ),
    )
    n_boot: int = Field(
        default=1000, ge=1,
        description="Bootstrap replications when ``inference_method='bootstrap'``.",
    )
    seed: int = Field(
        default=0, description="RNG seed for the bootstrap multipliers.",
    )
    alpha: float = Field(
        default=0.05,
        gt=0.0,
        lt=1.0,
        description="Significance level for the Wald confidence band.",
    )
    conformal_horizon: Optional[int] = Field(
        default=None,
        description=(
            "Post-periods to accumulate for a per-unit conformal band on the "
            "CUMULATIVE effect -- the total each treated unit gained, not its "
            "per-period or time-averaged effect. ``None`` (default) leaves it "
            "off and nothing changes. Unlike VanillaSC, where "
            "``inference='conformal_cumulative'`` selects one mode, this band is "
            "an additional object: ``inference_method`` still chooses the bootstrap "
            "or jackknife used for the ATT. Calibration needs non-overlapping "
            "windows of this length in the pre-period, so roughly "
            "``T0 >= horizon / (alpha * 0.7)`` are required before the band is "
            "finite."
        ),
    )

    @field_validator("conformal_horizon")
    @classmethod
    def _validate_conformal_horizon(cls, v):
        if v is None:
            return v
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise ValueError("conformal_horizon must be an integer >= 1 or None.")
        return v

    conformal_min_train_frac: float = Field(
        default=0.3,
        description=(
            "Where the cumulative band's calibration starts, as a fraction of the "
            "pre-period: the first rolling origin sits at "
            "``max(10, conformal_min_train_frac * T0)``, so every calibration fit "
            "has that many periods to train on. It trades two things the caller "
            "has to weigh and the library cannot. Periods spent on the warm-up are "
            "periods not available for calibration, and a ``1 - alpha`` band needs "
            "at least ``ceil(1/alpha) - 1`` windows before it is finite. Against "
            "that, a fit trained on fewer periods than there are donors can "
            "interpolate its training window, and its out-of-sample error is then "
            "not the error of the fit being calibrated -- inflating the scores and "
            "widening the band. On a panel with 120 pre-periods and 60 donors the "
            "default starts at period 36, which puts the first four origins in "
            "that regime; raising it to 0.5 starts at 60 and removes them, at the "
            "cost of windows. Read ``cumulative_windows`` off each unit to see "
            "what a given choice bought."
        ),
    )

    @field_validator("conformal_min_train_frac")
    @classmethod
    def _validate_conformal_min_train_frac(cls, v):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(
                "conformal_min_train_frac must be a number in the open interval "
                f"(0, 1); got {v!r}."
            )
        if not 0.0 < float(v) < 1.0:
            raise ValueError(
                "conformal_min_train_frac must lie in the open interval (0, 1); "
                f"got {v!r}. Zero would leave the start to the minimum training "
                "length alone, and one would leave no pre-period to calibrate on."
            )
        return float(v)

    covariates: Optional[List[str]] = Field(
        default=None,
        description=(
            "Auxiliary covariates to balance alongside the pre-treatment "
            "outcomes (augsynth::multisynth Sec 5.2). Each covariate is "
            "z-scored against the never-treated controls and rescaled to the "
            "control-outcome standard deviation, so covariate and outcome "
            "imbalance share a scale; the covariate imbalance is then stacked "
            "into the pooled and separate QP terms. Time-varying covariates "
            "are aggregated to their mean over periods before the first "
            "adoption. None (default) balances outcomes only."
        ),
    )

    cumulative_band: bool = Field(
        default=False,
        description=(
            "Attach a simultaneous (sup-t) band for the CUMULATIVE effect path "
            "to the aggregate inference -- the running total over horizons, with "
            "one shared critical value so the whole path is covered at "
            "``1 - alpha`` at once, which is how a cumulative path is read. "
            "Built from the replicate paths ``inference_method`` already "
            "produces, so it costs no extra refits. Its growth is measured "
            "instead of assumed: the replicates are accumulated before the "
            "standard error is taken, so independent period errors widen it "
            "like ``sqrt(L)`` and perfectly correlated ones like ``L``, where "
            "adding up per-period interval endpoints would always give the "
            "latter. Needs ``run_inference``. Off by default."
        ),
    )

    @model_validator(mode="after")
    def _apply_method_preset(self):
        """Expand ``method`` into the three conventions it names.

        Only fields the caller left at their default are set, so an explicit
        convention alongside the preset is honoured and not silently reversed --
        someone asking for the CS estimator with augsynth's donor pool is asking
        a coherent question, and the answer is not the preset's.
        """
        if self.method == "callaway_santanna":
            fields = self.model_fields_set
            if "donor_weights" not in fields:
                object.__setattr__(self, "donor_weights", "uniform")
            if "base_period" not in fields:
                object.__setattr__(self, "base_period", "pre_treatment")
            if "donor_pool" not in fields:
                object.__setattr__(self, "donor_pool", "never_treated")
            # The preset names an estimator, and an estimator is a point
            # estimate and an interval. It is selected only where the three
            # conventions actually landed on Callaway-Sant'Anna, so an explicit
            # convention alongside the preset does not turn into a
            # configuration error about a field the caller never set.
            if ("inference_method" not in fields
                    and self.donor_weights == "uniform"
                    and self.base_period == "pre_treatment"
                    and self.donor_pool == "never_treated"
                    and self.fixedeff):
                object.__setattr__(self, "inference_method", "influence_function")
        return self

    @model_validator(mode="after")
    def _check_inference_method(self):
        if self.inference_method not in ("jackknife", "bootstrap",
                                         "influence_function"):
            raise MlsynthConfigError(
                "inference_method must be 'jackknife', 'bootstrap' or "
                f"'influence_function'; got {self.inference_method!r}.")
        if self.inference_method == "influence_function":
            required = {"donor_weights": "uniform",
                        "base_period": "pre_treatment",
                        "donor_pool": "never_treated"}
            wrong = [f"{k}={getattr(self, k)!r} (needs {v!r})"
                     for k, v in required.items() if getattr(self, k) != v]
            if not self.fixedeff:
                wrong.append("fixedeff=False (needs True)")
            if wrong:
                raise MlsynthConfigError(
                    "inference_method='influence_function' is the "
                    "Callaway-Sant'Anna standard error, and its derivation "
                    "assumes the conventions that produce their point estimate: "
                    "equal donor weights against a never-treated comparison "
                    "group, normalised on the period before adoption. "
                    + "; ".join(wrong) + ". Solved SCM weights carry an "
                    "estimation term of their own and a not-yet-treated pool "
                    "changes composition over time, so the formula does not "
                    "apply; use inference_method='jackknife' or 'bootstrap' "
                    "there.")
        if self.cband and self.inference_method != "influence_function":
            raise MlsynthConfigError(
                "cband tabulates its simultaneous critical value from the "
                "Callaway-Sant'Anna influence functions, so it needs "
                "inference_method='influence_function'; got "
                f"{self.inference_method!r}.")
        if self.cband and not self.run_inference:
            raise MlsynthConfigError(
                "cband needs run_inference=True: with inference off there are "
                "no influence functions to tabulate a critical value from.")
        if self.cumulative_band and not self.run_inference:
            raise MlsynthConfigError(
                "cumulative_band needs run_inference=True: the band is built "
                "from the replicate paths the jackknife or bootstrap produces, "
                "and with inference off there are none.")
        return self
