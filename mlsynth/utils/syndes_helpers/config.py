"""Configuration for the SYNDES estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import Any, List, Literal, Optional
import warnings
import pandas as pd
from pydantic import Field, model_validator
from ...exceptions import MlsynthConfigError
from ...config_models import BaseMAREXConfig


class SYNDESConfig(BaseMAREXConfig):
    """Configuration for the Synthetic Design (SYNDES) estimator.

    Implements the three MIP formulations of Doudchenko, Khosravi,
    Pouget-Abadie, Lahaie, Lubin, Mirrokni, Spiess, and Imbens (2021),
    *"Synthetic Design: An Optimization Approach to Experimental
    Design with Synthetic Controls"* (arXiv:2112.00278). The estimator
    jointly chooses

    * which units to treat (binary assignment ``D``), and
    * the synthetic-control weights ``w`` used to build the
      counterfactual,

    by minimising a single mean-squared-error objective. Three
    formulations are exposed, each with a different geometry over the
    treated/control sample-variance terms (Theorem 1 of the paper):

    * ``"per_unit"``        --  separate SC weights for each treated
                                 unit (paper's "per-unit" problem).
    * ``"two_way_global"``  --  single weight vector applied
                                 symmetrically to treated and control
                                 (paper's "two-way global" problem).
    * ``"one_way_global"``  --  ``"two_way_global"`` with equal
                                 weights pinned on the treated set
                                 (paper's "one-way global" problem).
    * ``"two_way_global_annealed"`` -- simulated-annealing relaxation
                                 of ``two_way_global`` (mlsynth-specific
                                 extension; not in the paper).

    Parameters
    ----------
    K : int or None
        Number of treated units. Required for ``per_unit`` and
        ``one_way_global``. May be ``None`` for ``two_way_global``
        (Doudchenko et al. 2021, paragraph after eq. 9, note that the
        K-constraint is mathematically optional in the symmetric
        formulation); when ``None`` the MIP picks the cardinality of
        the treated set endogenously, with at least one treated and
        one control unit.
    mode : str
        Paper-aligned mode name (see above).
    lam : float or None
        Penalty on the squared weights. ``None`` defaults to the
        sample variance of the pre-treatment outcomes (Section 6 of
        the paper).
    T0 : int or None
        Number of pre-treatment periods. If neither ``T0`` nor
        ``post_col`` is supplied, the **entire panel is treated as
        pre-treatment** (design-only / planning mode -- no post period,
        so no ATT/inference is produced).
    post_col : str or None
        Optional 0/1 column identifying post-treatment periods.
    alpha : float
        Two-sided significance level for the permutation test.
    run_inference : bool
        Whether to run the moving-block permutation test
        (Chernozhukov-Wuethrich-Zhu (2021) style; see Appendix A.4
        of the paper).
    solver : Any
        CVXPY-compatible MIP solver. Defaults to SCIP.
    display_graph : bool
        Whether to plot the design.
    verbose : bool
        Solver verbosity.
    """

    K: Optional[int] = Field(
        default=None, gt=0,
        description="Number of treated units. Required for per_unit "
        "and one_way_global; optional for two_way_global.",
    )
    mode: Literal[
        "per_unit",
        "two_way_global",
        "one_way_global",
        "two_way_global_annealed",
    ] = Field(
        default="two_way_global",
        description=(
            "Paper-aligned formulation: per-unit / two-way / one-way "
            "global from Doudchenko et al. (2021), or the simulated-"
            "annealing relaxation of two-way global (mlsynth-specific "
            "extension; not in the paper)."
        ),
    )
    lam: Optional[float] = Field(
        default=None, ge=0,
        description="L2 penalty on weights. Defaults to the pre-period sample variance.",
    )
    T0: Optional[int] = Field(
        default=None, gt=0,
        description="Number of pre-treatment periods when post_col is not supplied.",
    )
    post_col: Optional[str] = Field(
        default=None,
        description="Optional 0/1 column identifying post-treatment periods.",
    )
    alpha: float = Field(default=0.10, gt=0.0, lt=1.0,
                          description="Permutation test significance level.")
    run_inference: bool = Field(default=True,
                                 description="Run post-period inference when post data are present.")
    solver: Any = Field(default="SCIP",
                         description="CVXPY-compatible mixed-integer solver "
                         "(ignored for the annealed mode).")
    relaxed_max_iter: int = Field(
        default=40, gt=0,
        description="Outer annealing iterations for mode='two_way_global_annealed'.",
    )
    relaxed_decay: float = Field(
        default=0.97, gt=0.0, lt=1.0,
        description="Geometric decay factor for the annealed solver's temperature.",
    )
    gap_limit: Optional[float] = Field(
        default=0.05, ge=0.0, lt=1.0,
        description=(
            "Optimality-gap tolerance handed to the MIP solver. The default "
            "of 0.05 (5%) follows Abadie & Zhao (2026, eq. 10 discussion, "
            "p. 13): 'we do not strictly require optimality of {w*, v*}, "
            "provided {w*, v*} is feasible and the design produces "
            "approximate balance.' Their bias bounds (Theorems 1-2) are "
            "written in terms of the residual fit, not the QP gap, so a "
            "5%-suboptimal solution inherits the same econometric "
            "guarantees as a proven-optimal one. Set to ``None`` for full "
            "optimality (can be very slow on long panels). For SCIP this "
            "is plumbed through as ``scip_params={'limits/gap': value}``."
        ),
    )
    time_limit: Optional[float] = Field(
        default=60.0, gt=0.0,
        description=(
            "Wall-clock cap on the MIP solve in seconds. Default 60s "
            "trades a guaranteed-near-optimal return time against rare "
            "early termination when the dual bound is far from the "
            "incumbent. Same justification as ``gap_limit``: near-optimal "
            "feasibility is sufficient for the paper's bias bounds. Set "
            "to ``None`` for no cap."
        ),
    )
    display_graph: bool = Field(default=False,
                                 description="Whether to render the SYNDES design plot.")
    verbose: bool = Field(default=False,
                           description="Solver verbosity flag.")
    costs: Optional[List[float]] = Field(
        default=None,
        description=(
            "Optional per-unit cost vector of length N (same ordering "
            "as the sorted ``unitid`` column). When supplied with "
            "``budget``, the MIP adds the constraint ``sum_i c_i D_i "
            "<= budget`` (Doudchenko et al. 2021 section 1: 'enforce "
            "a budget constraint if there is a varying cost to treat "
            "specific units'). Both fields must be set together."
        ),
    )
    budget: Optional[float] = Field(
        default=None, gt=0,
        description=(
            "Upper bound on the total cost of the treated set. "
            "Required when ``costs`` is supplied; ignored otherwise."
        ),
    )
    top_K: int = Field(
        default=1, ge=1,
        description=(
            "Size of the returned solution pool. ``1`` (default) returns only "
            "the MSE-optimal design. ``>1`` enumerates the top-K distinct "
            "designs by no-good cuts (forbidding each chosen treated set and "
            "re-solving for the next-best), ranked by MSE, and attaches them as "
            "``results.pool`` -- a menu of near-optimal options re-scored on MDE "
            "and cost, since the MIP optimises fit alone. Fewer than ``top_K`` "
            "are returned if the feasible region is exhausted."
        ),
    )
    arm: Optional[str] = Field(
        default=None,
        description=(
            "Optional categorical column naming each unit's treatment arm. "
            "When given, SYNDES solves its design independently within each "
            "arm's units and returns SYNDESMultiArmResults (a dict of per-arm "
            "results); when None (default), a single SYNDESResults is "
            "returned. K (if set) then applies per arm."
        ),
    )
    power_weight: float = Field(
        default=0.51, gt=0.0,
        description=(
            "Weight on power (minimum detectable effect) in the GeoLift-style "
            "composite score that picks ``results.recommendation`` from the "
            "``top_K`` pool. Normalised against ``fit_weight`` to sum to one; "
            "the default 0.51 vs 0.49 slightly prefers power over fit."
        ),
    )
    fit_weight: float = Field(
        default=0.49, gt=0.0,
        description=(
            "Weight on fit (the MIP objective) in the composite recommendation "
            "score. Normalised against ``power_weight`` to sum to one."
        ),
    )
    max_shortlist: int = Field(
        default=5, gt=0,
        description="Maximum number of designs in results.recommendation.shortlist.",
    )
    # ---- design restrictions (geography / clustering / size / forcing) ----
    # Same vocabulary as GEOLIFT/LEXSCM; translated to linear constraints on the
    # MIP assignment vector D. Not supported with mode='two_way_global_annealed'
    # (no MIP) or an 'arm' column (restrictions are global, not per-arm).
    to_be_treated: Optional[List] = Field(
        default=None,
        description="Units forced into the treated set (D_i = 1). Must not "
        "exceed K, and must be disjoint from not_to_be_treated.",
    )
    not_to_be_treated: Optional[List] = Field(
        default=None,
        description="Units forbidden from treatment (D_i = 0); they remain "
        "available as control donors.",
    )
    cluster_col: Optional[str] = Field(
        default=None,
        description="Per-unit-constant column; units sharing a cluster interfere "
        "(treating both is forbidden), combined with `adjacency` by logical OR.",
    )
    adjacency: Optional[pd.DataFrame] = Field(
        default=None,
        description="Square spillover matrix (DataFrame indexed/columned by unit "
        "label). Off-diagonal pairs strictly above `spillover_threshold` "
        "interfere; combined with `cluster_col` by logical OR.",
    )
    spillover_threshold: float = Field(
        default=0.0,
        description="Off-diagonal `adjacency` entries strictly above this mark a "
        "conflicting (spillover) pair.",
    )
    stratum_col: Optional[str] = Field(
        default=None,
        description="Per-unit-constant column defining strata for coverage quotas "
        "via `min_per_stratum` / `max_per_stratum`.",
    )
    min_per_stratum: Optional[int] = Field(
        default=None,
        description="Minimum treated units in every stratum that contains a "
        "treatable unit. Requires stratum_col.",
    )
    max_per_stratum: Optional[int] = Field(
        default=None,
        description="Maximum treated units in any stratum. Requires stratum_col.",
    )
    size_col: Optional[str] = Field(
        default=None,
        description="Per-unit-constant size column; units outside "
        "[min_size, max_size] lose treatment eligibility (stay donors).",
    )
    min_size: Optional[float] = Field(
        default=None, description="Lower size bound for treatment eligibility.",
    )
    max_size: Optional[float] = Field(
        default=None, description="Upper size bound for treatment eligibility.",
    )
    # ---- donor-side restrictions (which units may be a treated unit's donor) ----
    donor_region_col: Optional[str] = Field(
        default=None,
        description="Per-unit-constant column; a unit may serve as a donor for a "
        "treated unit only if they share this column's value (e.g. region). "
        "Couples to the assignment via the control weights, so it works in every "
        "mode (in the global modes it forces the treated set into one region).",
    )
    exclude_bordering_donors: bool = Field(
        default=False,
        description="When True, a treated unit's spillover neighbours (from "
        "`cluster_col` / `adjacency` + `spillover_threshold`) are dropped from "
        "its donor pool (the Vives-i-Bastida exclusion restriction). Requires a "
        "conflict source.",
    )
    donor_exclusion: Optional[pd.DataFrame] = Field(
        default=None,
        description="Explicit donor-exclusion matrix (DataFrame indexed/columned "
        "by unit label); entry [i, j] > 0 forbids unit j as a donor for treated "
        "unit i. Combined with `donor_region_col` / `exclude_bordering_donors` by "
        "union. The most flexible escape hatch.",
    )
    selection: Optional[Literal["in_sample", "holdout", "ic"]] = Field(
        default=None,
        description=(
            "Design-selection rule applied to the ``top_K`` candidate pool. "
            "``None`` (default) infers ``'holdout'`` when ``holdout_frac`` is "
            "set, otherwise ``'in_sample'`` -- so existing configs are "
            "unchanged. ``'in_sample'`` keeps the MIP fit optimum (Doudchenko "
            "et al. 2021). ``'holdout'`` validates candidates on a held-out tail "
            "and needs ``holdout_frac``. ``'ic'`` ranks candidates by an "
            "information criterion (in-sample contrast SSR + ``2 sigma^2 df`` "
            "with ``df = active control donors - 1``, after Pouliot-Xie-Liu) "
            "over the whole pre-window -- no data split, preferable when the "
            "pre-period is short. ``'holdout'`` and ``'ic'`` need ``top_K >= 2`` "
            "and a MIP mode."
        ),
    )
    holdout_frac: Optional[float] = Field(
        default=None, gt=0.0, lt=1.0,
        description=(
            "Out-of-sample design selection. When ``None`` (default) the winning "
            "design is the MIP's in-sample optimum (Doudchenko et al. 2021 "
            "behaviour, unchanged). When set to a fraction in (0, 1), SYNDES "
            "learns the ``top_K`` candidate pool on the leading ``1 - "
            "holdout_frac`` of the pre-period and selects the candidate whose "
            "*held-out* contrast error on the trailing ``holdout_frac`` is "
            "smallest -- a train/validate guard against overfitting transient "
            "pre-period co-movement (e.g. ``0.3`` for a 70/30 split). Requires "
            "``top_K >= 2`` (a pool to validate) and a MIP mode (not the "
            "annealed relaxation). Power and inference are computed exactly as in "
            "the in-sample path."
        ),
    )

    @model_validator(mode="after")
    def _check_syndes_params(cls, values: Any) -> Any:
        df = values.df
        n_units = df[values.unitid].nunique()
        n_periods = df[values.time].nunique()

        # Resolve the design-selection rule (None infers from holdout_frac).
        resolved = values.selection or (
            "holdout" if values.holdout_frac is not None else "in_sample"
        )
        # Mutual-exclusion / coherence between selection and holdout_frac.
        if values.selection == "in_sample" and values.holdout_frac is not None:
            raise MlsynthConfigError(
                "selection='in_sample' conflicts with holdout_frac; clear "
                "holdout_frac or set selection='holdout'."
            )
        if values.selection == "ic" and values.holdout_frac is not None:
            raise MlsynthConfigError(
                "selection='ic' does not use holdout_frac (it needs no data "
                "split); clear holdout_frac."
            )
        if resolved == "holdout" and values.holdout_frac is None:
            raise MlsynthConfigError(
                "selection='holdout' requires holdout_frac in (0, 1)."
            )
        # Pool-based selectors need a candidate pool and a MIP mode.
        if resolved in ("holdout", "ic"):
            if values.mode == "two_way_global_annealed":
                raise MlsynthConfigError(
                    f"selection={resolved!r} is not supported for "
                    "mode='two_way_global_annealed' (it has no candidate pool); "
                    "use a MIP mode."
                )
            if values.top_K is None or values.top_K < 2:
                raise MlsynthConfigError(
                    f"selection={resolved!r} requires top_K >= 2 (a candidate "
                    f"pool to rank); got top_K={values.top_K!r}."
                )

        # ---- design restrictions ----
        _restr_set = (
            values.to_be_treated is not None or values.not_to_be_treated is not None
            or values.cluster_col is not None or values.adjacency is not None
            or values.stratum_col is not None or values.min_per_stratum is not None
            or values.max_per_stratum is not None or values.size_col is not None
            or values.min_size is not None or values.max_size is not None
            or values.donor_region_col is not None
            or values.exclude_bordering_donors
            or values.donor_exclusion is not None
        )
        if _restr_set:
            if values.mode == "two_way_global_annealed":
                raise MlsynthConfigError(
                    "design restrictions are not supported for "
                    "mode='two_way_global_annealed' (it has no MIP assignment "
                    "vector); use a MIP mode."
                )
            if values.arm is not None:
                raise MlsynthConfigError(
                    "design restrictions are not supported together with an "
                    "'arm' column (restrictions are global, not per-arm)."
                )
        units = set(df[values.unitid].unique())
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
        if values.K is not None and len(forced) > values.K:
            raise MlsynthConfigError(
                f"to_be_treated ({len(forced)}) cannot exceed K ({values.K})."
            )
        for col in ("cluster_col", "stratum_col", "size_col", "donor_region_col"):
            name = getattr(values, col)
            if name is not None and name not in df.columns:
                raise MlsynthConfigError(f"{col} '{name}' is not a column of df.")
        if values.exclude_bordering_donors and values.cluster_col is None \
                and values.adjacency is None:
            raise MlsynthConfigError(
                "exclude_bordering_donors requires a conflict source "
                "(cluster_col and/or adjacency)."
            )
        if values.min_per_stratum is not None and values.min_per_stratum < 1:
            raise MlsynthConfigError(
                f"min_per_stratum must be >= 1; got {values.min_per_stratum}."
            )
        if values.max_per_stratum is not None and values.max_per_stratum < 1:
            raise MlsynthConfigError(
                f"max_per_stratum must be >= 1; got {values.max_per_stratum}."
            )
        if (values.min_per_stratum is not None or values.max_per_stratum is not None) \
                and values.stratum_col is None:
            raise MlsynthConfigError(
                "min_per_stratum / max_per_stratum require stratum_col."
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

        if values.arm is not None and values.arm in df.columns:
            # K applies within each arm, so validate against the smallest arm.
            arm_sizes = df.groupby(values.arm)[values.unitid].nunique()
            n_units = int(arm_sizes.min()) if len(arm_sizes) else n_units
            if values.costs is not None:
                raise MlsynthConfigError(
                    "costs/budget are not supported together with an 'arm' "
                    "column (the cost vector is global, not per-arm)."
                )

        if values.K is not None:
            if values.K >= n_units:
                raise MlsynthConfigError(
                    "K must be strictly less than the number of unique units in df."
                )
        else:
            if values.mode != "two_way_global":
                raise MlsynthConfigError(
                    "K=None is only supported for mode='two_way_global'; "
                    f"got mode={values.mode!r}."
                )

        # --- T0 / post_col resolution (mirrors MAREXConfig) ---
        if values.post_col is not None:
            if values.post_col not in df.columns:
                raise MlsynthConfigError(
                    f"post_col '{values.post_col}' is not present in df."
                )
            post_by_time = (
                df[[values.time, values.post_col]]
                .drop_duplicates(subset=[values.time])
                .set_index(values.time)[values.post_col]
            )
            if post_by_time.isna().any():
                raise MlsynthConfigError(
                    "post_col must be defined for every time period in the panel."
                )
            post_mask = post_by_time.astype(bool).to_numpy()
            if post_mask.all():
                raise MlsynthConfigError(
                    "post_col marks every period as post-treatment; no pre-period."
                )
            T0_from_post = int((~post_mask).sum())
            if values.T0 is not None and values.T0 != T0_from_post:
                warnings.warn(
                    f"T0={values.T0} ignored: derived T0={T0_from_post} from "
                    f"post_col '{values.post_col}'.",
                    UserWarning,
                )
            values.T0 = T0_from_post
        if values.T0 is not None and values.T0 > n_periods:
            raise MlsynthConfigError(
                "T0 cannot exceed the number of unique time periods in df."
            )
        # Neither T0 nor post_col is allowed: the whole panel is then treated
        # as pre-treatment (design-only / planning mode, no post period).

        if (values.costs is None) != (values.budget is None):
            raise MlsynthConfigError(
                "costs and budget must be supplied together (or both None)."
            )
        if values.costs is not None:
            n_units = df[values.unitid].nunique()
            if len(values.costs) != n_units:
                raise MlsynthConfigError(
                    f"costs must have length {n_units}; got {len(values.costs)}."
                )
            if any(c < 0 for c in values.costs):
                raise MlsynthConfigError("costs must be non-negative.")

        return values
