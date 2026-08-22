"""Configuration for the VanillaSC estimator.

Co-located with the VanillaSC helper package. The shared
:class:`~mlsynth.config_models.BaseEstimatorConfig` remains central (it is
common to every estimator); only the per-estimator config lives here, next to
the code it configures. Re-exported from :mod:`mlsynth.config_models` for
backward compatibility.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from pydantic import (BaseModel, ConfigDict, Field, field_validator,
                      model_validator)

from ...config_models import BaseEstimatorConfig

# Recognized string values for ``VanillaSCConfig.inference`` (besides the bools
# ``True`` -> in-space placebo and ``False`` -> disabled). Kept as the single
# source of truth so an unknown/misspelled value fails loudly at config time
# rather than silently returning no inference.
VALID_INFERENCE_METHODS = frozenset(
    {"placebo", "scpi", "conformal", "conformal_split", "conformal_cumulative",
     "lto", "ttest", "eiv", "jackknife_plus", "none"}
)

#: Spellings normalised onto a canonical method name. ``augsynth`` writes
#: ``inf_type="jackknife+"``, so accept that literally -- a user porting an R
#: script should not have to discover that we renamed it.
INFERENCE_ALIASES = {"jackknife+": "jackknife_plus",
                     "jackknife-plus": "jackknife_plus"}


class StaggeredSCMSpec(BaseModel):
    """Covariate (multi-feature) matching spec for staggered VanillaSC.

    Mirrors the ``scpi`` package's ``scdataMulti`` specification, applied
    uniformly to every treated unit. The treated units are detected from the
    treatment indicator -- the spec never names them. When a ``staggered_spec``
    is attached to a multi-treated panel, ``VanillaSC`` builds one synthetic
    control per treated unit on these features and reproduces the Cattaneo-Feng-
    Palomba-Titiunik (2025) point estimates and prediction intervals for all
    three causal predictands.
    """

    model_config = ConfigDict(extra="forbid")

    features: List[str] = Field(
        ...,
        description="Matching features (outcome-like series), shared across all "
                    "treated units. Must include the outcome.",
    )
    cov_adj: Optional[List[Any]] = Field(
        default=None,
        description="Covariate-adjustment terms per feature (e.g. "
                    "['constant', 'trend']). None -> no adjustment.",
    )
    constant: bool = Field(
        default=False,
        description="Include a global constant (scpi's 'constant').",
    )
    cointegrated: bool = Field(
        default=False,
        description="Difference the data for cointegration (scpi's "
                    "'cointegrated_data').",
    )
    w_constr: Literal["simplex", "ols", "lasso", "ridge", "L1-L2"] = Field(
        default="simplex",
        description="Donor weight constraint (scpi's weight-constraint family), "
                    "applied uniformly to every treated unit. 'simplex' (convex "
                    "Abadie SC, default), 'ols' (unconstrained), 'lasso' "
                    "(L1 <= 1, Chernozhukov et al.), 'ridge' (L2 shrinkage, "
                    "Amjad et al.), 'L1-L2' (Arkhangelsky et al.).",
    )

    @field_validator("features")
    @classmethod
    def _features_nonempty(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("staggered_spec.features must be non-empty.")
        return v


class VanillaSCConfig(BaseEstimatorConfig):
    """Configuration for the VanillaSC estimator (standard SCM, bilevel engine).

    The ordinary single-treated synthetic control, built on the self-contained
    bilevel machinery. With no covariates it reduces to the well-posed convex
    outcome-matching problem; with covariates it routes through the bilevel
    predictor-weight (``V``) optimisation, with a selectable, reliable backend.

    Parameters
    ----------
    backend : {"auto", "outcome-only", "malo", "mscmt", "regression", "penalized"}
        Predictor-weight backend. ``"auto"`` (default) uses ``"outcome-only"``
        (convex simplex fit on pre-treatment outcomes) when no covariates are
        given, and ``"mscmt"`` (global differential-evolution ``V`` search)
        when they are. ``"malo"`` is the Malo et al. (2024) corner search,
        ``"penalized"`` the Abadie-L'Hour (2021) unique/sparse estimator.
    covariates : list of str, optional
        Predictor columns. Each is averaged over its window (see
        ``covariate_windows``) and scaled to unit variance, then matched via
        the bilevel program. ``None`` -> outcome-only matching.
    covariate_windows : dict, optional
        Per-covariate inclusive ``(start, end)`` averaging window of time
        labels (Abadie's special-predictor spec). Covariates not listed are
        averaged over the full pre-treatment period.
    fit_window : tuple, optional
        Inclusive ``(start, end)`` window restricting the outcome fit (the
        dependent SSR, MSCMT's ``times.dep``) to a sub-range of the
        pre-treatment period; predictor matching is unaffected. ``None``
        (default) fits the full pre-period. This is how Abadie-Gardeazabal's
        Basque study fits ``gdpcap`` over 1960-1969 rather than 1955-1969.
    canonical_v : bool or {"min.loss.w", "max.order"}
        Canonicalise the (non-identified) predictor weights for ``mscmt``
        (MSCMT ``determine_v``). The reported ``v_agreement`` is small when
        ``V`` is well identified and large when it is fragile. Default False.
    seed : int
        RNG seed for the ``mscmt`` differential-evolution search.
    mscmt_maxiter, mscmt_popsize : int
        Differential-evolution budget for the ``mscmt`` backend.
    inference : bool or {"placebo", "scpi", "lto"}
        Inference method. ``True``/``"placebo"`` (default) runs Abadie in-space
        placebo inference (refit treating each donor as pseudo-treated; the
        p-value ranks the treated unit's post/pre RMSPE ratio). ``"scpi"`` runs
        Cattaneo-Feng-Titiunik (2021) prediction intervals (in-sample
        simulation + out-of-sample location-scale; exact for the simplex /
        outcome-only synthetic control). ``"lto"`` runs the Lei-Sudijono (2025)
        leave-two-out refined placebo test (O(J^2) reference comparisons; finer
        granularity and non-zero size when ``alpha < 1/N``). ``False`` skips
        inference.
    alpha : float
        Level. For placebo, the confidence statement; for SCPI, used as both
        the in-sample (alpha1) and out-of-sample (alpha2) levels, giving a
        prediction interval with coverage approximately ``1 - 2*alpha``.
    scpi_sims : int
        Number of Gaussian draws for the SCPI in-sample simulation.
    scpi_e_method : {"gaussian", "empirical"}
        Out-of-sample location-scale tabulation for SCPI.
    lto_max_pairs : int, optional
        Cap on the number of donor pairs evaluated by the ``"lto"`` test
        (deterministic subsample via ``seed``). ``None`` (default) uses all
        ``J*(J-1)/2`` pairs; set a cap to keep the O(J^2) cost tractable with
        slow backends.
    """

    backend: Literal["auto", "outcome-only", "malo", "mscmt", "regression",
                     "penalized"] = Field(
        default="auto",
        description="Predictor-weight backend (see class docstring).",
    )
    covariates: Optional[List[str]] = Field(
        default=None,
        description="Predictor columns; None -> outcome-only matching.",
    )
    covariate_windows: Optional[Dict[Any, Any]] = Field(
        default=None,
        description="Per-covariate inclusive (start, end) averaging window.",
    )
    bias_correct: bool = Field(
        default=False,
        description=(
            "Apply the Abadie-L'Hour (2021) correction for residual predictor "
            "imbalance: subtract (X1 - X0 W)' beta_t from the gap, with beta_t "
            "the ridge slope of the donor outcome on the donor predictors. "
            "Reach for it when the synthetic control does not balance the "
            "predictors well and the leftover imbalance plausibly moves the "
            "outcome -- it is what Stata's `allsynth bcorrect` does, and on "
            "Wiltshire's (2023) panel it roughly doubles the estimate. "
            "Requires `covariates`; the correction is identically zero when "
            "the predictors already balance."),
    )
    bias_correct_ridge: float = Field(
        default=1e-2, ge=0.0,
        description=(
            "Ridge penalty on the bias-correction slope. Ignored unless "
            "`bias_correct`. The penalty is what keeps a weakly-explanatory "
            "predictor set from injecting noise instead of removing bias: an "
            "unregularised slope is large exactly when the predictors explain "
            "the outcome poorly. 0 recovers ordinary least squares; a large "
            "value shrinks the correction to nothing, recovering the "
            "uncorrected fit."),
    )
    fit_window: Optional[Tuple[Any, Any]] = Field(
        default=None,
        description="Inclusive (start, end) time window over which the synthetic "
                    "control is fit to the treated outcome -- the dependent SSR, "
                    "MSCMT's ``times.dep``. Restricts the pre-period outcome "
                    "optimisation to this sub-range (predictors are still matched "
                    "over their own covariate_windows). None (default) uses the "
                    "full pre-treatment period; this is how Abadie-Gardeazabal's "
                    "Basque study fits gdpcap over 1960-1969 rather than 1955-1969.",
    )

    @field_validator("fit_window")
    @classmethod
    def _validate_fit_window(cls, v):
        if v is None:
            return v
        if len(v) != 2:
            raise ValueError(
                "fit_window must be a (start, end) pair of time labels.")
        start, end = v
        if start is not None and end is not None and start > end:
            raise ValueError(
                f"fit_window start ({start}) must not exceed end ({end}).")
        return v
    canonical_v: Union[bool, str] = Field(
        default=False,
        description="Canonicalise mscmt predictor weights ('min.loss.w'/'max.order').",
    )
    seed: int = Field(default=0, description="RNG seed for the mscmt DE search.")
    mscmt_maxiter: int = Field(
        default=300, ge=1, description="mscmt differential-evolution max iterations.",
    )
    mscmt_popsize: int = Field(
        default=15, ge=1, description="mscmt differential-evolution population size.",
    )
    mscmt_tol: float = Field(
        default=1e-6, gt=0.0,
        description="Relative tolerance stopping the mscmt outer search: it ends "
                    "when the population's spread in pre-fit MSPE falls below "
                    "this fraction of the mean. It is an estimate-precision "
                    "choice -- at the default the donor weights are within about "
                    "1e-5 of where an exhaustive search leaves them, three orders "
                    "finer than the four decimals the replications compare to. "
                    "Tighten it to spend more of the budget on digits below that.",
    )
    mscmt_prune_shady: bool = Field(
        default=True,
        description="mscmt: drop shady donors (Becker-Kloessner sunny-donor "
                    "reduction) before the outer search. Leaves the optimum "
                    "unchanged; shrinks the inner solve.",
    )
    augment: Optional[Literal["ridge"]] = Field(
        default=None,
        description="Augmentation layer: 'ridge' turns the fit into Augmented "
                    "SCM (Ben-Michael, Feller & Rothstein 2021) -- a ridge "
                    "bias-correction over the simplex base; None -> plain SCM.",
    )
    ridge_lambda: Optional[float] = Field(
        default=None, ge=0.0,
        description="Fixed ridge penalty for augment='ridge'; None -> select by "
                    "leave-one-period-out CV (augsynth's 1-SE rule).",
    )
    jackknife_conservative: bool = Field(
        default=False,
        description="With inference='jackknife_plus', use augsynth's "
                    "conservative branch: bracket the per-drop predictions by "
                    "their min and max and widen by a quantile of the absolute "
                    "held-out errors, instead of taking quantiles of "
                    "``est +/- |err|``. Wider, and augsynth's own default is "
                    "False.",
    )
    residualize: bool = Field(
        default=False,
        description="With augment='ridge' and covariates: stack covariates as "
                    "matching rows (False, augsynth parallel default) or regress "
                    "them out and match on residuals (True, residualize=TRUE).",
    )
    inference: Union[bool, str] = Field(
        default=True,
        description="Inference: True/'placebo' (in-space placebo), 'scpi' "
                    "(Cattaneo-Feng-Titiunik prediction intervals), 'conformal' "
                    "(Chernozhukov-Wuthrich-Zhu test-inversion intervals, the "
                    "augsynth default for ASCM), 'conformal_split' (the "
                    "constant-width split-conformal band, matching R Synth's "
                    "synth_inference), 'lto' (Lei-Sudijono leave-two-"
                    "out refined placebo), 'ttest' (Chernozhukov-Wuthrich-Zhu "
                    "2025 debiased SC t-test for the ATT), 'eiv' (Hirshberg 2021 "
                    "error-in-variables normal/t prediction intervals), "
                    "'jackknife_plus' (augsynth's ``inf_type=\"jackknife+\"`` for "
                    "ridge ASCM -- leave-one-pre-period-out refits; requires "
                    "augment='ridge'), or False.",
    )
    @field_validator("inference")
    @classmethod
    def _validate_inference(cls, v):
        """Reject unknown ``inference`` values so a typo never silently
        disables inference; normalize strings to lower case."""
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            low = v.strip().lower()
            low = INFERENCE_ALIASES.get(low, low)
            if low not in VALID_INFERENCE_METHODS:
                raise ValueError(
                    f"inference={v!r} is not a recognized inference method. "
                    f"Use a bool (True -> in-space placebo, False -> off) or one "
                    f"of: {', '.join(sorted(VALID_INFERENCE_METHODS))}."
                )
            return low
        raise ValueError(
            "inference must be a bool or one of "
            f"{', '.join(sorted(VALID_INFERENCE_METHODS))}."
        )

    alpha: float = Field(
        default=0.05, gt=0.0, lt=1.0,
        description="Level (placebo confidence / SCPI alpha1 = alpha2).",
    )
    scpi_sims: int = Field(
        default=200, ge=1,
        description="Gaussian draws for the SCPI in-sample simulation.",
    )
    scpi_e_method: Literal["gaussian", "empirical"] = Field(
        default="gaussian",
        description="SCPI out-of-sample location-scale tabulation.",
    )
    scpi_cointegrated: bool = Field(
        default=False,
        description="Single-unit SCPI only. When True, fit the in-sample and "
                    "out-of-sample uncertainty models on first differences of "
                    "the donor design (scpi's 'cointegrated_data=True'), for "
                    "cointegrated (I(1)) outcome/donor levels. The point "
                    "counterfactual is unchanged; only the prediction bands move.",
    )
    plot_bands: Literal["pointwise", "simultaneous", "both"] = Field(
        default="pointwise",
        description="Which SCPI prediction-interval band(s) to shade on the "
                    "observed-vs-counterfactual plot: 'pointwise' (default, the "
                    "per-period band), 'simultaneous' (the joint-coverage band), "
                    "or 'both'. The simultaneous band is only available for "
                    "inference='scpi'; other inference modes always show the "
                    "pointwise band.",
    )
    w_constr: Optional[Union[str, Dict[str, Any]]] = Field(
        default=None,
        description="Donor weight-constraint family for the SINGLE-treated fit "
                    "(scpi's w_constr): 'simplex' (the Abadie sum-to-one convex "
                    "hull), 'ols' (unconstrained), 'lasso' (L1 <= Q), 'ridge' "
                    "(L2 <= Q, budget estimated from the data) or 'L1-L2'. A "
                    "dict passes the family plus an explicit budget, e.g. "
                    "{'name': 'lasso', 'Q': 0.5}. None (default) keeps the "
                    "bilevel engine's simplex fit, so behaviour is unchanged "
                    "unless the option is set. Which constraint is used is a "
                    "modelling choice, not a property of the panel: on a "
                    "multi-treated panel the same families are reached through "
                    "staggered_spec.w_constr, and setting both is an error.",
    )
    staggered_spec: Optional[StaggeredSCMSpec] = Field(
        default=None,
        description="Staggered adoption only. Covariate (multi-feature) matching "
                    "spec (scpi's scdataMulti specification). When set on a "
                    "multi-treated panel, VanillaSC reproduces the Cattaneo-Feng-"
                    "Palomba-Titiunik (2025) point estimates and prediction "
                    "intervals for all three predictands with covariates. None "
                    "(default) -> outcome-only staggered matching.",
    )
    scpi_compat: bool = Field(
        default=False,
        description="Staggered adoption only. When True, the cross-unit "
                    "(TSUA / TAUA) in-sample prediction intervals reproduce the "
                    "scpi package's published numbers bit-for-bit, including its "
                    "1/iota**2 scaling of the time-aggregated in-sample term "
                    "(scdataMulti divides the predictand matrix P by the number "
                    "of treated units iota, and scpi_in_diag divides the "
                    "simulated draws by iota a second time). The statistically "
                    "correct scaling for the average of iota independent per-unit "
                    "errors is a single 1/iota, which is the default (False). "
                    "Use True only to match scpi exactly.",
    )
    lto_max_pairs: Optional[int] = Field(
        default=None, ge=1,
        description="Cap on donor pairs for the 'lto' test (None -> all pairs).",
    )
    oracle_weights: Optional[Dict[Any, float]] = Field(
        default=None,
        description="User-specified donor weights ``{donor_id: weight}``. When "
                    "given, the weight optimization is skipped and these weights "
                    "are used directly (e.g. the 'oracle' known-weights case). "
                    "Donors omitted from the map get weight 0. Supported with "
                    "inference=False or inference='ttest'.",
    )
    ttest_K: Union[int, Literal["auto"]] = Field(
        default=3,
        description="Cross-fitting folds for inference='ttest' (Chernozhukov-"
                    "Wuthrich-Zhu 2025): an int >= 2, or 'auto' to select K from "
                    "the SC-residual persistence and the RAE formula per their "
                    "Sec 3.2. K=3 is the small-T0 benchmark; larger K tightens "
                    "the interval.",
    )

    conformal_type: Literal["iid", "block"] = Field(
        default="iid",
        description="Permutation scheme for inference='conformal'. 'iid' draws "
                    "random permutations of the residual path, which assumes "
                    "the errors are exchangeable. 'block' uses the T cyclic "
                    "shifts of the path (the moving-block scheme), preserving "
                    "serial dependence and matching scinference's default; it "
                    "is deterministic, and its p-value cannot fall below 1/T "
                    "because one shift is the observed path.",
    )

    conformal_n_perm: Optional[int] = Field(
        default=None,
        description="Permutation draws for the conformal_type='iid' p-value. "
                    "None (default) reuses scpi_sims. scinference draws 5000, "
                    "and at 200 the Monte-Carlo error on a p-value near 0.02 is "
                    "about half its own size, so raise this before reading an "
                    "iid p-value closely. Ignored by conformal_type='block', "
                    "which enumerates its reference set and draws nothing.",
    )

    conformal_finite_sample: bool = Field(
        default=False,
        description="Report the conformal p-value as (1 + #{stat >= observed}) "
                    "/ (1 + n_perm) instead of the plain mean. This is "
                    "scinference's convention for iid permutations, and it is "
                    "the form that is valid in finite samples: n draws cannot "
                    "evidence a p-value below 1/(n+1), which the plain mean "
                    "reports as 0. The default is the plain mean, which is "
                    "augsynth's and what the ASCM cases reproduce.",
    )

    conformal_grid: Optional[Any] = Field(
        default=None,
        description="Candidate effects the per-period test inversion sweeps, "
                    "used unchanged for every post period (scinference's "
                    "ci_grid). The reported endpoints are grid points, so this "
                    "sets the resolution of the band. None (default) builds a "
                    "grid per period from the panel's own noise scale, which "
                    "adapts where a fixed grid cannot.",
    )

    @field_validator("conformal_n_perm")
    @classmethod
    def _validate_conformal_n_perm(cls, v):
        if v is None:
            return v
        if not isinstance(v, (int, np.integer)) or bool(v < 1):
            raise ValueError("conformal_n_perm must be an integer >= 1 or None.")
        return int(v)

    @field_validator("conformal_grid")
    @classmethod
    def _validate_conformal_grid(cls, v):
        if v is None:
            return v
        arr = np.asarray(v, dtype=float).ravel()
        if arr.size == 0:
            raise ValueError("conformal_grid must hold at least one candidate effect.")
        if not np.isfinite(arr).all():
            raise ValueError("conformal_grid must be finite.")
        return arr

    conformal_horizon: Optional[int] = Field(
        default=None,
        description="Post-periods to accumulate for inference="
                    "'conformal_cumulative'. The band is for the SUM of the "
                    "effect over this many periods, calibrated on out-of-sample "
                    "windows of the same length. None (default) accumulates the "
                    "whole post-period, the total effect of the intervention. A "
                    "long horizon leaves few non-overlapping calibration windows "
                    "in the pre-period, and the band widens accordingly.",
    )

    conformal_method: Literal["split", "resample"] = Field(
        default="split",
        description="How the cumulative band is calibrated. 'split' (default) "
                    "takes the finite-sample order statistic of the m "
                    "rolling-origin window sums, which exists only once m reaches "
                    "ceil(1/alpha) - 1 and otherwise returns an infinite "
                    "half-width. 'resample' keeps the same windows per period, "
                    "giving m * horizon values, and block-resamples them into "
                    "accumulated paths; the band is then the quantile of those "
                    "totals. It stays finite on pre-periods where the order "
                    "statistic does not exist, and it carries the serial "
                    "correlation of the period errors into the total instead of "
                    "reading it off a single summed score.",
    )
    conformal_block: int = Field(
        default=0,
        description="conformal_method='resample' only: block length in periods "
                    "for the circular block bootstrap. 0 (default) means the whole "
                    "horizon; 1 draws periods independently, which understates the "
                    "spread of a total whenever the period errors are positively "
                    "autocorrelated.",
    )
    conformal_n_sim: int = Field(
        default=2000,
        description="conformal_method='resample' only: number of accumulated paths "
                    "to draw. These cost no refits -- the refits are the "
                    "calibration origins -- so this buys quantile precision "
                    "cheaply.",
    )
    conformal_seed: int = Field(
        default=0,
        description="conformal_method='resample' only: seed for the draw, so the "
                    "band is reproducible.",
    )

    @field_validator("conformal_block", "conformal_n_sim", "conformal_seed",
                     mode="before")
    @classmethod
    def _validate_conformal_draw(cls, v, info):
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(f"{info.field_name} must be an integer.")
        if info.field_name == "conformal_block" and v < 0:
            raise ValueError("conformal_block must be >= 0 (0 means the horizon).")
        if info.field_name == "conformal_n_sim" and v < 2:
            raise ValueError("conformal_n_sim must be >= 2 to take a quantile.")
        return v

    @field_validator("conformal_horizon")
    @classmethod
    def _validate_conformal_horizon(cls, v):
        if v is None:
            return v
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise ValueError("conformal_horizon must be an integer >= 1 or None.")
        return v

    @field_validator("ttest_K")
    @classmethod
    def _validate_ttest_K(cls, v):
        if v == "auto":
            return v
        if isinstance(v, bool) or not isinstance(v, int) or v < 2:
            raise ValueError("ttest_K must be an integer >= 2 or 'auto'.")
        return v
    penalized_cv: Literal["holdout", "loo", "pensynth"] = Field(
        default="holdout",
        description="Lambda selector for backend='penalized'. 'holdout'/'loo' "
                    "are Abadie-L'Hour time-split criteria; 'pensynth' is van "
                    "Kesteren's cv_pensynth (fit on covariates, validate on the "
                    "held-out pre-period outcome path; needs covariates). "
                    "Ignored when 'penalized_lambda' is set.",
    )
    penalized_lambda: Optional[float] = Field(
        default=None, ge=0.0,
        description="Fixed Abadie-L'Hour penalty for backend='penalized'. When "
                    "set (>= 0), it is used directly and cross-validation "
                    "('penalized_cv') is skipped; larger values approach "
                    "nearest-neighbour matching, an infinitesimal value gives "
                    "the lexicographic (fit-preserving) tie-break, and 0 is the "
                    "unpenalized synthetic control. 'None' (default) selects "
                    "lambda by cross-validation. Ignored for other backends. "
                    "Covariates are not required: with none supplied the "
                    "pre-treatment outcome lags are the matching variables, "
                    "which is the case where the penalty matters most (a donor "
                    "pool large relative to the pre-period makes the "
                    "unpenalized optimum non-unique). Note the inner QP is a "
                    "first-order method that under-converges for very small "
                    "lambda relative to the fit term, so Abadie-L'Hour's "
                    "at-most-K+1 sparsity bound may not be attained there.",
    )

    # Recognized scpi weight-constraint families, shared with StaggeredSCMSpec.
    _W_CONSTR_NAMES = ("simplex", "ols", "lasso", "ridge", "L1-L2")

    @field_validator("w_constr")
    @classmethod
    def _validate_w_constr(cls, v):
        if v is None:
            return v
        if isinstance(v, str):
            name = v
        elif isinstance(v, dict):
            name = v.get("name", "simplex")
        else:
            raise ValueError(
                "w_constr must be a family name or a dict carrying one; got "
                f"{type(v).__name__}."
            )
        valid = ("simplex", "ols", "lasso", "ridge", "L1-L2")
        if name not in valid:
            raise ValueError(
                f"w_constr name must be one of {valid}; got {name!r}."
            )
        return v

    @model_validator(mode="after")
    def _w_constr_is_exclusive(self):
        """``w_constr`` routes the fit through scpi's constrained program, which
        is a different estimator from the bilevel predictor-weight engine and
        from user-supplied weights. Silently preferring one over the other is
        the failure mode this option exists to remove, so the clashes raise."""
        if self.w_constr is None:
            return self
        clashes = []
        if self.covariates:
            clashes.append(
                "covariates (the scpi constrained program matches on the "
                "outcome; for predictor matching use a bilevel `backend`)")
        if self.backend != "auto":
            clashes.append(
                f"backend={self.backend!r} (a predictor-weight backend; "
                "w_constr constrains the donor weights instead)")
        if self.oracle_weights is not None:
            clashes.append(
                "oracle_weights (weights supplied, so no program is solved)")
        if self.staggered_spec is not None:
            clashes.append(
                "staggered_spec, which carries its own w_constr")
        if clashes:
            raise ValueError(
                "w_constr cannot be combined with " + "; ".join(clashes) + "."
            )
        return self

    @model_validator(mode="after")
    def _bias_correction_needs_predictors(self):
        """The correction is a function of predictor imbalance, so with no
        predictors there is nothing to correct. Raise rather than silently
        returning the raw gap, which is what asking for it and not getting it
        would otherwise look like."""
        if self.bias_correct and not self.covariates:
            raise ValueError(
                "bias_correct=True needs covariates: the correction subtracts "
                "(X1 - X0 W)' beta from the gap, and with no predictors that "
                "term does not exist. Supply `covariates`, or leave "
                "bias_correct=False."
            )
        return self
