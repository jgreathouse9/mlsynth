"""Configuration for the MAREX estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union
import pandas as pd
import warnings
from pydantic import Field, field_validator, model_validator
from ...exceptions import MlsynthConfigError, MlsynthDataError
from ...config_models import BaseMAREXConfig


class MAREXConfig(BaseMAREXConfig):
    """Configuration for the Synthetic Experiment Design estimator (MAREX) in mlsynth."""

    T0: Optional[int] = Field(
        default=None,
        description="Number of pre-treatment periods. Alternative to ``post_col``.",
    )
    post_col: Optional[str] = Field(
        default=None,
        description=(
            "Optional 0/1 (or boolean) column identifying post-treatment periods. "
            "When supplied, ``T0`` is derived as the count of pre-treatment periods "
            "and an explicit ``T0`` argument is ignored."
        ),
    )
    cluster: Optional[str] = Field(
        default=None,
        description="Column name for cluster membership. Each cluster is a "
        "distinct experimental design with its own representativeness target "
        "(its predictor mean Xbar_k) -- this is part of the objective, not a "
        "constraint, and the per-cluster m_min/m_max IS the stratum quota. It is "
        "the MAREX analog of SYNDES's `arm`, but baked into the objective rather "
        "than run as separate solves, so the geographic restrictions compose "
        "with it (they apply within each cluster). None (default) = one global "
        "cluster (a single design against the whole-population mean).")
    design: str = Field(default="standard", description="Design type: 'standard', 'weakly_targeted', 'penalized', 'unit_penalized'.")
    covariates: Optional[List[str]] = Field(default=None, description="Covariate columns matched on alongside the fitting-window outcomes (the paper's X = [Y^E ; Z]). Each is averaged per unit over the fitting window, the first T0 - blank_periods periods, so a time-varying covariate's blank-window values stay out of the design.")
    covariate_weight: Union[float, Dict[str, float]] = Field(
        default=1.0,
        description=(
            "Weight on the covariate predictors relative to one period of the "
            "outcome, applied after standardisation. A number weights every "
            "covariate alike; a mapping from covariate name to weight gives "
            "each its own, the diagonal H of Vives-i-Bastida (2022, program 1), "
            "and must name exactly the listed covariates. Weights are >= 0, and "
            "0 drops a covariate from the match."))
    standardize: bool = Field(default=False, description="Scale each design predictor to unit variance across units (the paper's Walmart normalisation).")
    program_type: str = Field(default="MIQP", description="Optimization type: 'QP' or 'MIQP'.")

    # --- NEW display option ---
    display_graph: bool = Field(default=False, description="Whether to display plots.")

    beta: float = Field(
        default=1e-6,
        description=(
            "Weakly-targeted design: weight on tying the synthetic control to "
            "the synthetic treated instead of to the population mean."))
    control_penalty: float = Field(
        default=0.0,
        description=(
            "Weakly-targeted design: Abadie and L'Hour's penalty on the "
            "controls, beta * control_penalty * sum_j v_j ||B w - X_j||^2, "
            "which charges each control market its distance from the "
            "synthetic treated unit and makes the synthetic control sparse "
            "and local (Vives-i-Bastida 2022, program 1, uses 0.1). The "
            "objective is convex only while beta * control_penalty * "
            "(1 + control_penalty) <= 1; larger values are refused. 0 leaves "
            "the weakly-targeted design as it was."))
    lambda1: float = Field(
        default=0.0,
        description=(
            "Penalized design: weight on each selected unit's distance from "
            "the cluster mean, summed over the treated weights."))
    lambda2: float = Field(
        default=0.0,
        description=(
            "Penalized design: weight on the same distance for the control "
            "side."))
    xi: float = Field(
        default=0.0,
        description=(
            "Unit-level design (eq. 10): weight on each candidate treated "
            "unit's own reproducibility by the remaining controls. Enters "
            "linearly in the treated weights, the per-unit minima being "
            "evaluated once before the solve."))
    lambda1_unit: float = Field(
        default=0.0,
        description=(
            "Unit-level design: weight on each selected treated unit's "
            "distance from the population target."))
    lambda2_unit: float = Field(
        default=0.0,
        description=(
            "Accepted only as 0. The paper's control-side unit-level penalty "
            "pairs the treated weights against the control weights through a "
            "distance matrix, which is bilinear in two decision variables and "
            "leaves no convex program. Use lambda1_unit, or the 'penalized' "
            "design, which weighs each side against the target separately."))
    costs: Optional[List[float]] = None
    budget: Optional[Union[int, Dict[int, int]]] = None

    blank_periods: Optional[int] = Field(
        default=None,
        description=(
            "Number of held-out blank periods at the tail of the pre-period used "
            "for placebo inference and noise-scale estimation. When unspecified "
            "and ``inference=True``, defaults to ``max(1, floor(0.3 * T0))`` — "
            "the same 30% pre-tail convention as the other MAREX-family "
            "estimators. When ``inference=False`` (no placebo needed), defaults "
            "to ``0`` (the optimizer fits on the entire pre-period)."
        ),
    )
    m_eq: Optional[int] = Field(default=None)
    m_min: Optional[int] = Field(default=None)
    m_max: Optional[int] = Field(default=None)
    exclusive: bool = Field(default=True)
    relaxed: bool = Field(default=False, description="Relax the MIQP (continuous z) and discretise post hoc.")
    backend: Literal["mip", "exact"] = Field(
        default="mip",
        description=(
            "How the design is solved. 'mip' (the default) hands the mixed-"
            "integer program to SCIP. 'exact' searches treated sets directly: "
            "naming the treated set splits the standard, penalized and "
            "unit-penalized objectives into two independent simplex fits, so "
            "each candidate costs two small quadratic programs instead of a "
            "branch-and-bound node, and lower bounds on the two fits, which "
            "keep the penalties, rule out most candidates before either fit. "
            "The weakly-targeted "
            "objective couples the two fits; each candidate is then solved by "
            "Newton steps over the faces of the control simplex until a "
            "Frank-Wolfe certificate closes, and discarded early once its "
            "certified lower bound clears the best design found. Above twenty "
            "million treated sets in a cluster, the standard, penalized and "
            "unit-penalized designs are searched by a certified branch and "
            "bound instead of set by set, and the weakly-targeted design is "
            "refused. It returns the same optimum and the same result. It "
            "does not take "
            "a cost budget, relaxed=True, a negative beta, or the "
            "branch-and-bound settings (solver, warm_start, time_limit), and "
            "each cluster is searched on its own, so no border conflict or "
            "stratum quota may span two clusters."
        ),
    )
    certify: bool = Field(
        default=False,
        description=(
            "With backend='exact', also return a certificate "
            "(``MAREXResults.certificate``): a lower bound on the optimal "
            "objective and an upper bound on the returned design's, both exact "
            "for the stored data and rounded outward, so the design is optimal "
            "to within their difference whatever the floating-point rounding "
            "in the search. It prices every candidate a second time with a "
            "certified bound and fits the few that bound does not clear in "
            "exact arithmetic. The standard, penalized and unit-penalized "
            "designs only: the weakly-targeted objective does not split into a "
            "treated fit and a control fit."
        ),
    )
    solver: Any = Field(default=None)
    verbose: bool = Field(default=False)
    warm_start: Optional[List] = Field(
        default=None,
        description="Treated unit labels to seed the exact MIQP as a SCIP "
        "partial-solution MIP start (e.g. LEXSCM's top candidate). A hint only -- "
        "the proven optimum is unchanged; it speeds the branch-and-bound and, with "
        "`time_limit`, lets MAREX refine the seed within a budget. MIQP-only.")
    time_limit: Optional[float] = Field(
        default=None, gt=0,
        description="Wall-clock cap (seconds) on the SCIP solve; returns the best "
        "incumbent found instead of proving optimality. MIQP-only.")

    # --- NEW inference options ---
    inference: bool = Field(default=False, description="Whether to run post-fit inference (placebo CI/p-values).")
    unit_inference: bool = Field(
        default=False,
        description="Requires inference=True. Adds inference for each treated "
                    "unit: its own synthetic control, fitted on the fitting "
                    "periods from its cluster's untreated units (Abadie and "
                    "Zhao's eq. 10), and MAREX's inference on its contrast; "
                    "Bonferroni-adjusted p-values and a joint test across the "
                    "treated units; and eq. (11)'s aggregate, the weighted "
                    "sum of the per-unit contrasts, beside the shared one. A "
                    "unit's band is about twice as wide as the aggregate's, "
                    "and on a 'standard' design eq. (11) trails the shared "
                    "contrast; 'unit_penalized' designs choose treated units "
                    "their clusters reproduce well.")
    T_post: Optional[int] = Field(default=None, description="Number of post-intervention periods for inference.")
    global_test: Literal["subsets", "cyclic"] = Field(
        default="subsets",
        description="The global test of no effect. 'subsets' (default) is "
                    "Abadie and Zhao's eq. (17): the mean absolute post "
                    "contrast against the T_post-subsets of the blank and post "
                    "periods, exact when those periods are exchangeable. "
                    "'cyclic' is Chernozhukov, Wuthrich and Zhu's (2021) "
                    "moving-block test with their Remark 1 statistic: the "
                    "magnitude of the mean post contrast against the same "
                    "statistic on every cyclic shift of the blank and post "
                    "contrast. It holds its level when the contrast is "
                    "serially correlated, where eq. (17) over-rejects; the "
                    "smallest p-value it can reach is "
                    "1 / (blank_periods + T_post). It changes the global "
                    "p-value of every contrast MAREX tests, and nothing else.",
    )

    # --- cumulative-effect band ---
    # The blank window is a contiguous stretch the fit never saw, so the effects
    # there are an out-of-sample error series already. That is what a band for a
    # running total needs, and it costs no refits: VanillaSC and PDA have to slide
    # an origin across the pre-period to manufacture the same thing.
    cumulative_band: bool = Field(
        default=False,
        description="Whether to add a band for the TOTAL post-period effect, "
                    "drawn from the blank-period effects. Requires "
                    "inference=True and a blank window. The existing pointwise "
                    "band answers 'how big was the effect in week 7'; this "
                    "answers 'how much did the intervention add in total', which "
                    "is not obtainable by arithmetic on the pointwise endpoints.",
    )
    cumulative_method: Literal["resample", "inversion"] = Field(
        default="resample",
        description="cumulative_band only: how the band for the total is "
                    "built. 'resample' (default) block-resamples the "
                    "blank-period effects into paths, so it uses only errors "
                    "observed before the post period and is limited by how "
                    "many horizon-length windows the blank stretch holds. "
                    "'inversion' tests constant effects (Chernozhukov, "
                    "Wuthrich and Zhu 2021, Remark 1): it refits a synthetic "
                    "control for the synthetic treated series on every period "
                    "from the untreated units, and compares the signed post "
                    "sum of the residuals with its cyclic shifts over the "
                    "blank and post periods. It reads the post outcomes, so "
                    "no window count limits it; the smallest p-value it can "
                    "reach is 1 / (blank_periods + T_post). The "
                    "cumulative_block, cumulative_n_sim and cumulative_seed "
                    "settings apply to 'resample' alone.",
    )
    cumulative_block: int = Field(
        default=0,
        description="cumulative_band only: block length in periods for the "
                    "circular block bootstrap of the blank-period effects. 0 "
                    "(default) means the whole horizon; 1 draws periods "
                    "independently, which understates the spread of a total "
                    "whenever the period errors are positively autocorrelated.",
    )
    cumulative_n_sim: int = Field(
        default=2000,
        description="cumulative_band only: number of accumulated paths to draw. "
                    "These cost no solves, so this buys quantile precision "
                    "cheaply.",
    )
    cumulative_seed: int = Field(
        default=0,
        description="cumulative_band only: seed for the draw, so the band is "
                    "reproducible.",
    )

    # --- geographic design restrictions ---
    # MAREX already covers region-clustering (`cluster`), stratum quotas
    # (`m_min`/`m_max`), cost/budget, and same-region donors natively. These add
    # the remaining SYNDES/LEXSCM capabilities as constraints on the MIP. Only
    # supported in the exact MIQP (not `relaxed=True`, whose post-hoc rounding
    # cannot guarantee them).
    to_be_treated: Optional[List] = Field(
        default=None,
        description="Units forced into the treated set; must be disjoint from "
        "not_to_be_treated.")
    not_to_be_treated: Optional[List] = Field(
        default=None,
        description="Units forbidden from treatment; they remain donors.")
    adjacency: Optional[pd.DataFrame] = Field(
        default=None,
        description="Square border/spillover matrix (DataFrame keyed by unit "
        "label); off-diagonal pairs above `spillover_threshold` may not both be "
        "treated, and (with `exclude_bordering_donors`) are dropped from each "
        "other's donor pool.")
    spillover_threshold: float = Field(
        default=0.0,
        description="Off-diagonal `adjacency` entries strictly above this mark a "
        "conflicting (bordering) pair.")
    exclude_bordering_donors: bool = Field(
        default=False,
        description="Drop a treated market's bordering neighbours from its "
        "(within-cluster) control pool. Requires `adjacency`.")
    size_col: Optional[str] = Field(
        default=None,
        description="Per-unit-constant size column; units outside "
        "[min_size, max_size] lose treatment eligibility (stay donors).")
    min_size: Optional[float] = Field(default=None, description="Lower size bound.")
    max_size: Optional[float] = Field(default=None, description="Upper size bound.")
    cluster_col: Optional[str] = Field(
        default=None,
        description="Per-unit-constant column whose value defines an interference "
        "cluster: at most one treated market per cluster value (a no-two-from-one-"
        "cluster rule). Distinct from `cluster` (the design grouping); enforced as "
        "conflict constraints on the treated set, the same as SYNDES's "
        "cluster_col.")
    stratum_col: Optional[str] = Field(
        default=None,
        description="Per-unit-constant column defining coverage strata for a "
        "treated-set quota via `min_per_stratum` / `max_per_stratum`. (For a "
        "per-region representativeness target use `cluster`; this is a coverage "
        "constraint on one design.)")
    min_per_stratum: Optional[int] = Field(
        default=None, ge=1,
        description="At least this many treated markets in every stratum that has "
        "a treatable member. Requires `stratum_col`.")
    max_per_stratum: Optional[int] = Field(
        default=None, ge=1,
        description="At most this many treated markets per stratum. Requires "
        "`stratum_col`.")

    # --- solution pool + power-based recommendation (mirrors SYNDES) ---
    top_K: int = Field(
        default=1, ge=1,
        description=(
            "Size of the returned solution pool. ``1`` (default) returns only the "
            "MSE-optimal design and no pool. ``>1`` enumerates the top-K distinct "
            "designs via no-good cuts (forbid each chosen treated set and re-solve "
            "for the next-best), each scored on a minimum-detectable-effect (MDE) "
            "power curve, and attaches them as ``results.pool`` plus a composite "
            "``results.recommendation`` -- the SYNDES-style menu."),
    )
    power_weight: float = Field(
        default=0.51, gt=0.0,
        description=("Weight on power (MDE) in the composite recommendation score, "
                     "normalised against ``fit_weight`` to sum to one."),
    )
    fit_weight: float = Field(
        default=0.49, gt=0.0,
        description=("Weight on fit (the design objective) in the composite "
                     "recommendation score, normalised against ``power_weight``."),
    )
    max_shortlist: int = Field(
        default=5, ge=1,
        description="Maximum number of designs in results.recommendation.shortlist.",
    )
    alpha: float = Field(
        default=0.05, gt=0.0, lt=1.0,
        description="Two-sided level for inference and the MDE power curve: "
                    "the pointwise and cumulative bands are at 1 - alpha, and "
                    "the default blank window holds at least ceil(1/alpha) - 1 "
                    "periods, the fewest at which the pointwise band exists.",
    )
    power_target: float = Field(
        default=0.80, gt=0.0, lt=1.0,
        description="Target power (1 - beta) for the MDE.",
    )

    @field_validator("lambda2_unit")
    @classmethod
    def _refuse_lambda2_unit(cls, v):
        """Reject the bilinear control-side penalty at construction."""
        if v:
            raise MlsynthConfigError(
                "lambda2_unit must be 0. It pairs the treated weights against "
                "the control weights through a distance matrix, which is "
                "bilinear in two decision variables and leaves no convex "
                "program to solve. Use lambda1_unit, which is linear in the "
                "treated weights, or the 'penalized' design, whose lambda1 "
                "and lambda2 weigh each side against the population target "
                f"separately. Got {v}."
            )
        return v

    @field_validator("unit_inference", mode="before")
    @classmethod
    def _validate_unit_inference(cls, v):
        if not isinstance(v, bool):
            raise MlsynthConfigError(
                f"unit_inference must be True or False; got {v!r}.")
        return v

    @model_validator(mode="after")
    def _unit_inference_needs_inference(self) -> "MAREXConfig":
        if self.unit_inference and not self.inference:
            raise MlsynthConfigError(
                "unit_inference=True needs inference=True: each unit's "
                "inference is calibrated on the blank periods inference "
                "holds out.")
        return self

    @field_validator("cumulative_band", mode="before")
    @classmethod
    def _validate_cumulative_request(cls, v):
        if not isinstance(v, bool):
            raise MlsynthConfigError(
                f"cumulative_band must be True or False; got {v!r}.")
        return v

    @field_validator("global_test", mode="before")
    @classmethod
    def _known_global_test(cls, v):
        """Refuse an unknown test by name, not as a Literal mismatch."""
        if not isinstance(v, str) or v not in ("subsets", "cyclic"):
            raise MlsynthConfigError(
                f"global_test must be 'subsets' or 'cyclic'; got {v!r}. "
                "'subsets' is the paper's eq. (17); 'cyclic' compares the "
                "post contrast with the cyclic shifts of the blank and post "
                "contrast.")
        return v

    @field_validator("cumulative_method", mode="before")
    @classmethod
    def _known_cumulative_method(cls, v):
        """Refuse an unknown construction by name, not as a Literal mismatch."""
        if not isinstance(v, str) or v not in ("resample", "inversion"):
            raise MlsynthConfigError(
                f"cumulative_method must be 'resample' or 'inversion'; got "
                f"{v!r}. 'resample' draws paths from the blank-period effects; "
                "'inversion' tests constant effects against the post outcomes.")
        return v

    @field_validator("cumulative_block", "cumulative_n_sim", "cumulative_seed",
                     mode="before")
    @classmethod
    def _validate_cumulative_draw(cls, v, info):
        # ``mode="before"`` so pydantic does not coerce "3" to 3 first: a string
        # where an integer belongs is a caller error, not a formatting detail.
        if isinstance(v, bool) or not isinstance(v, int):
            raise MlsynthConfigError(f"{info.field_name} must be an integer.")
        if info.field_name == "cumulative_block" and v < 0:
            raise MlsynthConfigError(
                "cumulative_block must be >= 0 (0 means the horizon).")
        if info.field_name == "cumulative_n_sim" and v < 2:
            raise MlsynthConfigError(
                "cumulative_n_sim must be >= 2 to take a quantile.")
        return v

    @model_validator(mode="after")
    def _validate_covariate_weight(self) -> "MAREXConfig":
        """Weights are non-negative; a mapping names exactly the covariates."""
        w = self.covariate_weight
        values = list(w.values()) if isinstance(w, dict) else [w]
        bad = [v for v in values if not v >= 0]
        if bad:
            raise MlsynthConfigError(
                f"covariate_weight must be >= 0; got {w!r}.")
        if isinstance(w, dict):
            if not self.covariates:
                raise MlsynthConfigError(
                    "covariate_weight is a mapping by covariate name, but no "
                    "covariates are listed.")
            missing = [c for c in self.covariates if c not in w]
            unknown = [c for c in w if c not in self.covariates]
            if missing or unknown:
                raise MlsynthConfigError(
                    f"covariate_weight must name exactly the covariates "
                    f"{self.covariates}; missing {missing}, unknown {unknown}.")
        return self

    @model_validator(mode="after")
    def _validate_control_penalty(self) -> "MAREXConfig":
        """The control penalty is weakly-targeted only, and convex only below a bound."""
        lam = self.control_penalty
        if lam == 0:
            return self
        if lam < 0:
            raise MlsynthConfigError(
                f"control_penalty must be >= 0; got {lam}.")
        if self.design != "weakly_targeted":
            raise MlsynthConfigError(
                f"control_penalty charges each control its distance from the "
                f"synthetic treated unit inside the weakly_targeted design; "
                f"design={self.design!r} has no such term.")
        kappa = self.beta * lam * (1.0 + lam)
        if kappa > 1.0 + 1e-12:
            raise MlsynthConfigError(
                f"beta * control_penalty * (1 + control_penalty) = {kappa:.6g} "
                f"exceeds 1. Past that bound the weakly-targeted objective with "
                f"the control penalty is not convex, so neither backend can "
                f"certify a minimum. Lower beta or control_penalty.")
        return self

    @model_validator(mode="after")
    def _validate_backend(self) -> "MAREXConfig":
        """Refuse, before any solve, what the exact backend cannot solve."""
        if self.backend != "exact":
            return self
        problems = []
        if self.relaxed:
            problems.append(
                "relaxed=True solves the continuous relaxation and rounds it, "
                "while the exact backend solves the integer design directly")
        if self.design == "weakly_targeted" and self.beta < 0:
            problems.append(
                f"beta={self.beta} makes the weakly-targeted coupling concave, "
                f"so the design has no minimum to certify; beta must be >= 0")
        if self.costs is not None or self.budget is not None:
            problems.append(
                "a cost budget constrains the treated weights, which the "
                "exact backend's per-candidate fit does not take")
        for name in ("warm_start", "time_limit", "solver"):
            if getattr(self, name) is not None:
                problems.append(
                    f"{name} configures SCIP's branch-and-bound, which the "
                    f"exact backend does not run")
        if problems:
            raise MlsynthConfigError(
                "backend='exact' cannot honour this configuration: "
                + "; ".join(problems) + ". Use backend='mip'.")
        return self

    @model_validator(mode="after")
    def _validate_certify(self) -> "MAREXConfig":
        """The certificate covers the exact backend's separable designs."""
        if not self.certify:
            return self
        if self.backend != "exact":
            raise MlsynthConfigError(
                "certify=True certifies the exact backend's search; set "
                "backend='exact'.")
        if self.design == "weakly_targeted":
            raise MlsynthConfigError(
                "certify=True splits each candidate into a treated fit and a "
                "control fit, and the weakly-targeted design couples them; "
                "certify the standard, penalized or unit_penalized design.")
        return self

    @model_validator(mode="after")
    def _validate_geo_quota(self) -> "MAREXConfig":
        if (self.min_per_stratum is not None or self.max_per_stratum is not None) \
                and self.stratum_col is None:
            raise MlsynthConfigError(
                "min_per_stratum / max_per_stratum require `stratum_col`.")
        if (self.min_per_stratum is not None and self.max_per_stratum is not None
                and self.min_per_stratum > self.max_per_stratum):
            raise MlsynthConfigError(
                f"min_per_stratum ({self.min_per_stratum}) cannot exceed "
                f"max_per_stratum ({self.max_per_stratum}).")
        return self

    @model_validator(mode="after")
    def validate_design_params(cls, values: Any) -> Any:
        df = values.df
        T0 = values.T0
        design = values.design
        program_type = values.program_type
        cluster_col = values.cluster
        time_col = values.time

        # --- program_type ---
        if program_type not in {"QP", "MIQP"}:
            raise MlsynthDataError(f"program_type must be 'QP' or 'MIQP', got '{program_type}'")

        # --- T0 / post_col ---
        n_periods = df[time_col].nunique()
        if values.post_col is not None:
            if values.post_col not in df.columns:
                raise MlsynthConfigError(
                    f"post_col '{values.post_col}' is not present in df."
                )
            # Derive T0 from post_col (count of pre-treatment periods).
            post_by_time = (
                df[[time_col, values.post_col]]
                .drop_duplicates(subset=[time_col])
                .set_index(time_col)[values.post_col]
            )
            if post_by_time.isna().any():
                raise MlsynthDataError(
                    "post_col must be defined for every time period in the panel."
                )
            post_mask = post_by_time.astype(bool).to_numpy()
            if post_mask.all():
                raise MlsynthConfigError(
                    "post_col marks every period as post-treatment; no pre-period."
                )
            T0_from_post = int((~post_mask).sum())
            if T0 is not None and T0 != T0_from_post:
                warnings.warn(
                    f"T0={T0} ignored: derived T0={T0_from_post} from post_col "
                    f"'{values.post_col}'.",
                    UserWarning,
                )
            values.T0 = T0_from_post
            T0 = T0_from_post
        if T0 is not None and (T0 <= 0 or T0 > n_periods):
            raise MlsynthDataError(f"T0 must be between 1 and {n_periods}.")

        # --- design ---
        valid_designs = {"standard", "weakly_targeted", "penalized", "unit_penalized"}
        if design not in valid_designs:
            raise MlsynthDataError(
                f"design must be one of {sorted(valid_designs)}, got '{design}'")

        # --- time axis ---
        # Time handling is delegated to ``geoex_dataprep`` (the same balanced-
        # panel prep GeoLift uses), invoked in ``prepare_marex_panel``: it sorts
        # the time index, so any orderable time — integer, datetime, or ISO-date
        # string as the geoex pipeline supplies — is accepted, and it enforces a
        # strongly balanced panel. No dtype-specific check is needed here.

        # --- cluster handling ---
        if cluster_col is not None:
            if cluster_col not in df.columns:
                raise MlsynthDataError(f"Cluster column '{cluster_col}' not found in df.")

            col = df[cluster_col]

            # Check for all missing
            if col.isna().all():
                raise MlsynthDataError(f"Cluster column '{cluster_col}' contains only missing values.")

            # Convert to integer codes if non-integer
            if not pd.api.types.is_integer_dtype(col):
                warnings.warn(
                    f"Cluster column '{cluster_col}' contains non-integers; converting to codes.",
                    UserWarning
                )
                df[cluster_col] = pd.Categorical(col).codes
                col = df[cluster_col]

            # Ensure each unit is in only one cluster
            unit_to_clusters = df.groupby(values.unitid, observed=True)[cluster_col].apply(lambda x: set(x.dropna()))
            non_invariant = unit_to_clusters[unit_to_clusters.apply(len) != 1]
            if not non_invariant.empty:
                raise MlsynthDataError(f"Units with multiple cluster assignments: {non_invariant.to_dict()}")

            # --- m_eq / m_min / m_max validation ---
            cluster_sizes = df.groupby(cluster_col, observed=True).size()
            if values.m_eq is not None and values.m_eq > cluster_sizes.max():
                raise MlsynthDataError(
                    f"m_eq ({values.m_eq}) cannot be greater than max cluster size ({cluster_sizes.max()})"
                )
            if values.m_min is not None and values.m_min < 1:
                raise MlsynthDataError("m_min must be >= 1")
            if values.m_max is not None and values.m_max > cluster_sizes.min():
                raise MlsynthDataError(
                    f"m_max ({values.m_max}) cannot be greater than min cluster size ({cluster_sizes.min()})"
                )
            if values.m_min is not None and values.m_max is not None and values.m_min > values.m_max:
                raise MlsynthDataError(
                    f"m_min ({values.m_min}) cannot be greater than m_max ({values.m_max})"
                )

        # --- costs validation ---
        if values.costs is not None and not all(c > 0 for c in values.costs):
            raise MlsynthDataError("All values in 'costs' must be strictly positive.")

        # --- budget validation ---
        if values.budget is not None:
            if isinstance(values.budget, int):
                if values.budget <= 0:
                    raise MlsynthDataError("Scalar 'budget' must be strictly positive.")
            elif isinstance(values.budget, dict):
                clusters = df[cluster_col].unique() if cluster_col is not None else []
                missing_clusters = [c for c in clusters if c not in values.budget]
                if missing_clusters:
                    raise MlsynthDataError(f"Budget dict missing entries for cluster(s): {missing_clusters}")
                if any(b <= 0 for b in values.budget.values()):
                    raise MlsynthDataError("All values in 'budget' dict must be strictly positive.")
            else:
                raise MlsynthDataError(f"'budget' must be int or dict, got {type(values.budget)}")

        values.df = df

        # --- inference / blank_periods validation ---
        n_periods = values.df[values.time].nunique()
        T0_eff = values.T0 if values.T0 is not None else n_periods - 1
        if values.inference:
            max_post = n_periods - T0_eff
            if values.T_post is None:
                values.T_post = max_post
            elif values.T_post <= 0 or values.T_post > max_post:
                raise MlsynthDataError(
                    f"T_post must be between 1 and {max_post} "
                    f"(T0={T0_eff}, total periods={n_periods})"
                )
            # The default blank window aims for 30% of the pre-period, the
            # convention of the other MAREX-family estimators (LEXSCM, SYNDES,
            # PANGEO), held between two floors: enough blanks for the conformal
            # quantile to exist at ``alpha``, and enough fitting periods for the
            # panel's factors. An explicit ``blank_periods`` is taken as given.
            if values.blank_periods is None:
                from ..datautils import geoex_dataprep
                from .windows import default_blank_periods
                Ywide = geoex_dataprep(values.df, values.unitid, values.time,
                                       values.outcome)["Ywide"]
                values.blank_periods = default_blank_periods(
                    Ywide.to_numpy(dtype=float)[:T0_eff].T, values.alpha)
        else:
            # No inference requested: do not silently carve out a blank
            # window from the optimization (would change the synthetic
            # design relative to the natural "fit on full pre-period" case).
            if values.blank_periods is None:
                values.blank_periods = 0

        if values.blank_periods < 0 or values.blank_periods >= T0_eff:
            raise MlsynthDataError(
                f"blank_periods must satisfy 0 <= blank_periods < T0 "
                f"(T0={T0_eff}, got {values.blank_periods})."
            )

        # --- geographic design restrictions ---
        _restr = (values.to_be_treated is not None
                  or values.not_to_be_treated is not None
                  or values.adjacency is not None
                  or values.exclude_bordering_donors
                  or values.size_col is not None
                  or values.min_size is not None or values.max_size is not None)
        if _restr and values.relaxed:
            raise MlsynthConfigError(
                "design restrictions are only supported in the exact MIQP, not "
                "with relaxed=True (its post-hoc rounding cannot guarantee them)."
            )
        if (values.warm_start is not None or values.time_limit is not None) \
                and values.relaxed:
            raise MlsynthConfigError(
                "warm_start / time_limit apply to the exact MIQP only, not "
                "relaxed=True."
            )
        units = set(df[values.unitid].unique())
        if values.warm_start is not None:
            unknown_ws = [u for u in values.warm_start if u not in units]
            if unknown_ws:
                raise MlsynthConfigError(
                    "warm_start contains units not in df: "
                    f"{sorted(map(str, set(unknown_ws)))}."
                )
        forced = list(values.to_be_treated or [])
        forbidden = list(values.not_to_be_treated or [])
        unknown = [u for u in (forced + forbidden) if u not in units]
        if unknown:
            raise MlsynthConfigError(
                "to_be_treated/not_to_be_treated contain units not in df: "
                f"{sorted(map(str, set(unknown)))}."
            )
        clash = set(forced) & set(forbidden)
        if clash:
            raise MlsynthConfigError(
                "units cannot be both to_be_treated and not_to_be_treated: "
                f"{sorted(map(str, clash))}."
            )
        if values.size_col is not None and values.size_col not in df.columns:
            raise MlsynthConfigError(
                f"size_col '{values.size_col}' is not a column of df."
            )
        if (values.min_size is not None or values.max_size is not None) \
                and values.size_col is None:
            raise MlsynthConfigError("min_size / max_size require size_col.")
        if values.min_size is not None and values.max_size is not None \
                and values.min_size > values.max_size:
            raise MlsynthConfigError(
                f"min_size ({values.min_size}) must be <= max_size "
                f"({values.max_size})."
            )
        if values.exclude_bordering_donors and values.adjacency is None:
            raise MlsynthConfigError(
                "exclude_bordering_donors requires an adjacency matrix."
            )

        return values
