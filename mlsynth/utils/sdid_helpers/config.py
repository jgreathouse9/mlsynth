"""Configuration for the SDID estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import Field, field_validator, model_validator
from ...config_models import BaseEstimatorConfig
from ...exceptions import MlsynthConfigError


class SDIDConfig(BaseEstimatorConfig):
    """Configuration for the Synthetic Difference-in-Differences (SDID) estimator.

    Implements Arkhangelsky, Athey, Hirshberg, Imbens & Wager (2021)'s SDID
    with the event-study aggregation of Ciccia (2024, arXiv:2407.09565).
    Inherits the standard ``df`` / ``outcome`` / ``treat`` / ``unitid`` /
    ``time`` panel-data interface from :class:`BaseEstimatorConfig`.

    Setting ``subgroup`` switches on the synthetic triple-difference (SC-DDD)
    mode of Zhuang (2024, arXiv:2409.12353): the outcome is demeaned by the
    non-target subgroup within each treatment-group-by-time cell, reducing the
    triple difference to a difference-in-differences that SDID then estimates.
    """

    vce: Literal["placebo", "jackknife", "bootstrap", "noinference"] = Field(
        default="placebo",
        description=(
            "Variance estimator for the ATT, following Arkhangelsky et al. "
            "(2021): 'placebo' (Algorithm 4, the default and the only method "
            "defined for a single treated unit), 'jackknife' (Algorithm 3, "
            "fixed-weights leave-one-out), 'bootstrap' (Algorithm 2, clustered "
            "resampling with weights re-fit), or 'noinference' to skip variance "
            "estimation. 'jackknife' and 'bootstrap' are implemented for the "
            "block (single adoption period) design and return NaN for a single "
            "treated unit, matching the synthdid R package."
        ),
    )
    B: int = Field(
        default=500,
        ge=0,
        description=(
            "Number of resamples for the variance estimator (placebo iterations "
            "or bootstrap replications; ignored by 'jackknife' and "
            "'noinference'). Set to 0 to skip resample-based inference. The "
            "paper uses B = 500."
        ),
    )
    seed: int = Field(
        default=1400,
        description="Random seed used for the placebo / bootstrap resampling.",
    )
    zeta: Optional[float] = Field(
        default=None,
        description=(
            "Ridge penalty on the unit weights, ``synthdid``'s ``zeta.omega``: "
            "the unit-weight objective carries the term "
            "``T_pre * zeta^2 * ||omega||^2``. None (the default) computes it "
            "from the data as Arkhangelsky et al. (2021) prescribe, "
            "``(N_tr * T_post)^(1/4) * sigma_hat`` with ``sigma_hat`` the "
            "standard deviation of the first-differenced control outcomes; "
            "that is the recommended value and nothing changes unless this is "
            "set. Give a number to override it -- published SDID analyses do "
            "not all use the default, and de Brabander, Juodis & Miyazato "
            "Szini (2025) set it to 0 throughout their Brexit study, a "
            "specification that is otherwise not expressible. A number rather "
            "than a flag because the quantity is a number; 0 is only the most "
            "common override. Time weights are unaffected -- their penalty is "
            "``zeta.lambda``, which SDID sets from the machine epsilon and "
            "which this does not touch."
        ),
    )
    intercept_adjust: bool = Field(
        default=False,
        description=(
            "Whether the counterfactual exposed in ``time_series`` is shifted by "
            "the SDID intercept (the constant level difference between the treated "
            "unit and its weighted donors). When False (default), the raw "
            "weighted-donor series is reported, as in Arkhangelsky et al. (2021) "
            "Figure 1; when True it is level-matched to the treated unit over the "
            "pre-period. The point estimate and inference are unaffected either way."
        ),
    )
    subgroup: Optional[str] = Field(
        default=None,
        description=(
            "Synthetic triple-difference (SC-DDD) mode, Zhuang (2024, "
            "arXiv:2409.12353). Column naming a within-unit subgroup dimension "
            "(e.g. an age band). When set, the outcome is demeaned by the "
            "non-target subgroup within each treatment-group-by-time cell -- "
            "reducing the triple difference to a difference-in-differences -- and "
            "SDID is run on the transformed outcome over the target subgroup. "
            "Requires ``target_subgroup``. None (default) -> ordinary SDID."
        ),
    )
    target_subgroup: Optional[Any] = Field(
        default=None,
        description=(
            "SC-DDD mode: the value of the ``subgroup`` column identifying the "
            "policy-exposed (target) subgroup whose effect is estimated; all "
            "other subgroup values are the non-target controls that are demeaned "
            "out. Required when ``subgroup`` is set."
        ),
    )

    covariates: Optional[
        Dict[Literal["adjust", "match", "optimized"], List[str]]
    ] = Field(
        default=None,
        description=(
            "Time-varying covariates, keyed by how they are used. The two "
            "methods are different estimators, so the key is required rather "
            "than defaulted. "
            "'adjust' -- Kranz (2022, xsynthdid): the covariate coefficients "
            "from 'outcome ~ covariates | unit + time', fit on the rows with "
            "no treatment in force, are subtracted from the outcome across the "
            "whole panel and ordinary SDID runs on the result; the weight "
            "problem never sees a covariate. "
            "'match' -- de Brabander, Juodis & Miyazato Szini (2025, eqs. "
            "11-12): covariates enter the unit-weight problem instead. The "
            "treated unit's demeaned pre-treatment outcomes are stacked with "
            "its covariate means, the donors likewise, and a diagonal V "
            "weighting those rows is chosen by nested optimisation against the "
            "pre-treatment outcome fit; the outcome is left alone. "
            "'optimized' -- Arkhangelsky et al. (2021, footnote 4), and the "
            "default in Stata's ``sdid``: the coefficients are estimated "
            "*jointly* with the unit and time weights by minimising a single "
            "objective, rather than in a separate first step, so the "
            "covariates and the weights determine each other. Use it to "
            "reproduce a paper that wrote ``covariates(x)`` without naming a "
            "type. The Stata authors caveat their own default -- it \"has been "
            "observed to be problematic at times (refer to Kranz (2022))\" and "
            "is sensitive to covariates with high dispersion -- so 'adjust' is "
            "the better-behaved choice when you are not matching a published "
            "specification. "
            "Keys may be given together -- residualising for seasonality "
            "while matching donors on income is coherent -- but a column may "
            "not appear under more than one. Columns must be numeric. None "
            "(default) -> no covariates. A bare list is rejected: 'covariates' "
            "previously meant 'adjust', and silently reinterpreting it would "
            "change which estimator runs; pass {'adjust': [...]} for that."
        ),
    )

    match_pre_periods: Optional[Union[Literal["all", "half", "last"], int]] = Field(
        default=None,
        description=(
            "How many pre-treatment outcome periods enter the covariate "
            "matching problem, for covariates={'match': ...}. Not a tuning "
            "knob: it decides whether the covariates can do anything at all. "
            "Kaul, Klossner, Pfeifer & Schieler (2022) show that when every "
            "pre-treatment outcome is among the predictors, the nested V "
            "search gives the covariate rows zero weight and the covariates "
            "are irrelevant; de Brabander et al. (2025) adopt that result and "
            "report three schemes in response. 'last' (the default) uses only "
            "the final pre-treatment period and is the scheme under which "
            "covariates demonstrably bind; 'half' uses the later half; 'all' "
            "uses every pre-period and reproduces Kaul's irrelevance exactly. "
            "An integer n uses the last n pre-treatment periods. Only "
            "meaningful alongside a 'match' key, and rejected without one."
        ),
    )

    @field_validator("covariates", mode="before")
    @classmethod
    def _reject_the_old_list_form(cls, v):
        # Before this option existed, covariates was a list and meant the Kranz
        # adjustment. Pydantic would reject a list here anyway, but with a type
        # dump that does not tell an upgrading caller what to do instead.
        if isinstance(v, (list, tuple, set)):
            raise MlsynthConfigError(
                "covariates must be a dict keyed by method, not a list. Pass "
                "{'adjust': [...]} for the Kranz (2022) outcome adjustment, "
                "which is what a bare list used to mean, or {'match': [...]} "
                "for covariate matching in the unit-weight problem. Both keys "
                "may be given together."
            )
        return v

    @field_validator("zeta")
    @classmethod
    def _check_zeta(cls, v):
        # A bad penalty does not fail loudly downstream: a negative one makes
        # the weight objective non-convex and a NaN one propagates silently
        # into every weight, so both are caught here rather than surfacing as
        # an unexplained fit.
        if v is None:
            return v
        if not math.isfinite(v):
            raise MlsynthConfigError(
                f"zeta must be finite; got {v}. Leave it as None (the default) "
                "to compute the penalty from the data.")
        if v < 0:
            raise MlsynthConfigError(
                f"zeta must be non-negative; got {v}. It is the coefficient on "
                "a squared-norm penalty, so a negative value would reward "
                "large weights. Pass 0 to switch the penalty off.")
        return v

    @model_validator(mode="after")
    def _check_covariates(self) -> "SDIDConfig":
        if self.covariates is None:
            return self
        if not self.covariates:
            raise MlsynthConfigError(
                "covariates is an empty dict; pass None (the default) to run "
                "SDID without covariates, or {'adjust': [...]} / "
                "{'match': [...]} to use them.")

        import pandas.api.types as ptypes

        for key, cols in self.covariates.items():
            if not cols:
                raise MlsynthConfigError(
                    f"covariates['{key}'] is an empty list; drop the key, or "
                    "pass None to run SDID without covariates.")
            missing = [c for c in cols if c not in self.df.columns]
            if missing:
                raise MlsynthConfigError(
                    f"covariate column(s) {missing} under '{key}' are not in "
                    f"df; available: {list(self.df.columns)}.")
            if self.outcome in cols:
                raise MlsynthConfigError(
                    f"the outcome column '{self.outcome}' is listed as a "
                    f"covariate under '{key}'. Under 'adjust' that would zero "
                    "the outcome out; under 'match' it duplicates the "
                    "pre-treatment outcomes already being matched on.")
            if self.treat in cols:
                raise MlsynthConfigError(
                    f"the treatment column '{self.treat}' is listed as a "
                    f"covariate under '{key}'; it would absorb the effect "
                    "being estimated.")
            non_numeric = [c for c in cols
                           if not ptypes.is_numeric_dtype(self.df[c])]
            if non_numeric:
                raise MlsynthConfigError(
                    f"covariate column(s) {non_numeric} under '{key}' are not "
                    "numeric. Both methods are linear in the covariates, so "
                    "each column must be a numeric measurement; encode "
                    "categoricals as indicators yourself if that is intended.")

        from itertools import combinations
        both = sorted({c for a, b in combinations(self.covariates, 2)
                       for c in (set(self.covariates[a])
                                 & set(self.covariates[b]))})
        if both:
            # Defensible either way -- one could argue for residualising on a
            # variable and still matching on its level -- so this is a recorded
            # decision, not an oversight. Refused because the common case is a
            # typo, and because the two uses double-count the same variation
            # with no way for the caller to see it happen.
            raise MlsynthConfigError(
                f"covariate column(s) {both} appear under both of two "
                "covariate methods. Each method uses the variable differently, "
                "so applying two to one column double-counts the same "
                "variation; pick one use per column.")

        # SC-DDD collapses the panel over the subgroup dimension before SDID
        # runs, so a (unit, time) covariate has no single row to attach to.
        # True of both methods, not only the one this was written for.
        if self.subgroup is not None:
            raise MlsynthConfigError(
                "covariates and subgroup cannot be combined: SC-DDD mode "
                "transforms the outcome over the subgroup dimension, leaving "
                "no (unit, time) panel for either covariate method to be "
                "defined on. Adjust the outcome yourself before passing it in "
                "if you need both.")
        return self

    @model_validator(mode="after")
    def _check_match_pre_periods(self) -> "SDIDConfig":
        if self.match_pre_periods is None:
            return self
        if not (self.covariates or {}).get("match"):
            raise MlsynthConfigError(
                "match_pre_periods is set but no 'match' covariates were "
                "given, so it would have no effect. Pass "
                "covariates={'match': [...]}, or drop match_pre_periods.")
        if isinstance(self.match_pre_periods, bool):
            raise MlsynthConfigError(
                "match_pre_periods must be 'all', 'half', 'last' or a "
                f"positive integer; got {self.match_pre_periods!r}.")
        if isinstance(self.match_pre_periods, int) and self.match_pre_periods < 1:
            raise MlsynthConfigError(
                "match_pre_periods must be at least 1 when given as an "
                f"integer; got {self.match_pre_periods}.")
        return self

    @model_validator(mode="after")
    def _check_ddd(self) -> "SDIDConfig":
        if self.subgroup is None:
            if self.target_subgroup is not None:
                raise MlsynthConfigError(
                    "target_subgroup is set but subgroup is not; provide both to "
                    "enable SC-DDD mode, or neither for ordinary SDID.")
            return self
        if self.subgroup not in self.df.columns:
            raise MlsynthConfigError(
                f"subgroup column '{self.subgroup}' is not in df.")
        if self.target_subgroup is None:
            raise MlsynthConfigError(
                "SC-DDD mode requires target_subgroup (the exposed subgroup value) "
                "when subgroup is set.")
        col = self.df[self.subgroup]
        if not (col == self.target_subgroup).any():
            raise MlsynthConfigError(
                f"target_subgroup {self.target_subgroup!r} does not appear in "
                f"column '{self.subgroup}'.")
        if (col != self.target_subgroup).sum() == 0:
            raise MlsynthConfigError(
                f"column '{self.subgroup}' has no non-target rows to demean by; "
                "SC-DDD needs at least one non-target subgroup value.")
        return self
