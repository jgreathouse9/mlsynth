from typing import List, Optional, Any, Dict, Union, Literal
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator
from mlsynth.exceptions import MlsynthDataError, MlsynthConfigError
import warnings

class BaseMAREXConfig(BaseModel):
    """
    Base configuration for synthetic experiment designs.
    Contains fields common to all synthetic experiment-based estimators.
    """

    df: pd.DataFrame = Field(..., description="Input panel data (units x time).")
    outcome: str = Field(..., description="Column name for the outcome variable.")
    unitid: str = Field(..., description="Column name for the unit identifier.")
    time: str = Field(..., description="Column name for the time period.")

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"

    @model_validator(mode="after")
    def check_df_columns(cls, values: Any) -> Any:
        df = values.df
        outcome = values.outcome
        unitid = values.unitid
        time = values.time

        if df.empty:
            raise MlsynthDataError("Input DataFrame 'df' cannot be empty.")

        # Ensure required columns exist
        required_columns = {outcome, unitid, time}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise MlsynthDataError(
                f"Missing required columns in DataFrame: {', '.join(sorted(missing_columns))}"
            )

        # Check for missing values in required columns
        missing_info = {col: int(df[col].isna().sum()) for col in required_columns if df[col].isna().any()}
        if missing_info:
            details = ", ".join([f"{col}: {count}" for col, count in missing_info.items()])
            raise MlsynthDataError(
                f"Missing values detected in required columns -> {details}. "
                "Please clean or impute these values before passing to BaseMAREXConfig."
            )

        # Check for uniqueness of (unitid, time) pairs
        duplicate_count = df.duplicated(subset=[unitid, time]).sum()
        if duplicate_count > 0:
            raise MlsynthDataError(
                f"Duplicate (unitid, time) pairs found: {duplicate_count}. "
                f"Each (unitid, time) combination must be unique in panel data."
            )

        # Ensure time is sorted within each unit (auto-fix if not)
        if not df.sort_values([unitid, time]).equals(df):
            warnings.warn(
                f"DataFrame was not sorted by [{unitid}, {time}] — auto-sorting applied.",
                UserWarning
            )
            df = df.sort_values([unitid, time]).reset_index(drop=True)
            values.df = df  # overwrite with sorted DataFrame

        return values



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
    cluster: Optional[str] = Field(default=None, description="Column name for cluster membership.")
    design: str = Field(default="standard", description="Design type: 'standard', 'weakly_targeted', 'penalized', 'unit_penalized'.")
    covariates: Optional[List[str]] = Field(default=None, description="Time-invariant covariate columns matched on alongside pre-period outcomes (the paper's X = [Y^E ; Z]).")
    covariate_weight: float = Field(default=1.0, description="Scale applied to covariate predictors relative to outcomes.")
    standardize: bool = Field(default=False, description="Scale each design predictor to unit variance across units (the paper's Walmart normalisation).")
    program_type: str = Field(default="MIQP", description="Optimization type: 'QP' or 'MIQP'.")

    # --- NEW display option ---
    display_graph: bool = Field(default=False, description="Whether to display plots.")

    beta: float = Field(default=1e-6)
    lambda1: float = Field(default=0.0)
    lambda2: float = Field(default=0.0)
    xi: float = Field(default=0.0)
    lambda1_unit: float = Field(default=0.0)
    lambda2_unit: float = Field(default=0.0)
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
    solver: Any = Field(default=None)
    verbose: bool = Field(default=False)

    # --- NEW inference options ---
    inference: bool = Field(default=False, description="Whether to run post-fit inference (placebo CI/p-values).")
    T_post: Optional[int] = Field(default=None, description="Number of post-intervention periods for inference.")

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

        # --- consecutive time check ---
        time_vals = df[time_col].sort_values().unique()
        time_dtype = df[time_col].dtype
        if pd.api.types.is_numeric_dtype(time_dtype):
            if not np.all(np.diff(time_vals) == 1):
                raise MlsynthDataError(f"Time periods in '{time_col}' are not consecutive: {time_vals}")
        elif pd.api.types.is_datetime64_any_dtype(time_dtype):
            diffs = np.diff(time_vals)
            if len(diffs) > 0 and not np.all(diffs == diffs[0]):
                raise MlsynthDataError(f"Datetime time periods in '{time_col}' are not consecutive.")

        else:
            raise MlsynthDataError(f"Unsupported dtype for time column '{time_col}': {time_dtype}")

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
            unit_to_clusters = df.groupby(values.unitid)[cluster_col].apply(lambda x: set(x.dropna()))
            non_invariant = unit_to_clusters[unit_to_clusters.apply(len) != 1]
            if not non_invariant.empty:
                raise MlsynthDataError(f"Units with multiple cluster assignments: {non_invariant.to_dict()}")

            # --- m_eq / m_min / m_max validation ---
            cluster_sizes = df.groupby(cluster_col).size()
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
            # Default blank window to the same 30% pre-tail convention used by
            # the other MAREX-family estimators (LEXSCM, SYNDES, PANGEO).
            if values.blank_periods is None:
                values.blank_periods = max(1, int(0.3 * T0_eff))
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

        return values









class BaseEstimatorConfig(BaseModel):
    """
    Base Pydantic model for estimator configurations.
    Includes common fields required by most or all estimators.
    """
    df: pd.DataFrame = Field(..., description="Input panel data as a pandas DataFrame.")
    outcome: str = Field(..., description="Name of the outcome variable column in the DataFrame.")
    treat: str = Field(..., description="Name of the treatment indicator column in the DataFrame.")
    unitid: str = Field(..., description="Name of the unit identifier column in the DataFrame.")
    time: str = Field(..., description="Name of the time period column in the DataFrame.")
    display_graphs: bool = Field(default=True, description="Whether to display plots of results.")
    save: Union[bool, str] = Field(default=False, description="Configuration for saving plots. If False (default), plots are not saved. If True, plots are saved with default names. If a string, it's used as the base filename for saved plots.")
    counterfactual_color: List[str] = Field(default_factory=lambda: ["red"],description="Color(s) for counterfactual line(s) in plots.")
    treated_color: str = Field(default="black", description="Color for the treated unit line in plots.")

    class Config:
        arbitrary_types_allowed = True
        extra = 'forbid' # Forbid extra fields not defined in the model

    @model_validator(mode='after')
    def check_df_and_columns(cls, values: Any) -> Any:
        df = values.df
        outcome = values.outcome
        treat = values.treat
        unitid = values.unitid
        time = values.time

        if df.empty:
            raise MlsynthDataError("Input DataFrame 'df' cannot be empty.")

        required_columns = {outcome, treat, unitid, time}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise MlsynthDataError(
                f"Missing required columns in DataFrame 'df': {', '.join(sorted(list(missing_columns)))}"
            )
        return values

# TSSCConfig has been relocated to mlsynth/utils/tssc_helpers/config.py
# (co-located with the estimator's helpers). It is re-exported from this
# module for backward compatibility via the module-level __getattr__ below.

# Placeholder for other estimator configs - to be added sequentially

# FDIDConfig has been relocated to mlsynth/utils/fdid_helpers/config.py
# (co-located with the estimator's helpers). It is re-exported from this
# module for backward compatibility via the module-level __getattr__ below.


    # Note: The original GSC __init__ docstring mentioned 'save', but it was commented out
    # in the implementation. If 'save' functionality specific to GSC is re-added,
    # it might need to be defined here if different from BaseEstimatorConfig.save.



# VanillaSCConfig has been relocated to mlsynth/utils/vanillasc_helpers/config.py
# (co-located with the estimator's helpers). It is re-exported from this
# module for backward compatibility via the module-level __getattr__ below.


class PANGEOConfig(BaseModel):
    """Configuration for the PANGEO experimental-design estimator.

    Parallel-trends supergeo design (in the Supergeo / Chen et al. 2023
    lineage): partitions each treatment arm's geos into supergeo pairs
    whose treatment/control halves are maximally parallel over the
    pre-period, via a cvxpy/HiGHS set-partitioning MIP. A prospective
    *design* method -- it returns supergeo pairs + a treatment/control
    assignment, not a treatment effect -- so it takes a single categorical
    ``arm`` column rather than a binary ``treat`` indicator.

    Parameters
    ----------
    df : pd.DataFrame
        Historical (pre-treatment) balanced long panel.
    outcome : str
        Historical outcome column (e.g. sales).
    arm : str
        Single categorical column naming each geo's eligible treatment arm
        (e.g. values ``A``/``B``/``C``). Arms occupy non-overlapping geos;
        the design runs independently within each arm.
    unitid : str
        Unit (geo) identifier column.
    time : str
        Time-period column.
    post_col : str, optional
        0/1 indicator column marking post-treatment periods (0 = pre). When
        given, the design is built on the pre rows alone -- identical to the
        design-only result -- and the realized difference-in-differences ATT
        is additionally computed on the post rows (``results.effects``).
    weight_col : str, optional
        Per-unit aggregation weight (e.g. population), constant within a
        unit. Makes both the supergeo design and the ATT population-weighted.
    max_supergeo_size : int
        Q -- the maximum size of either supergeo within a pair. Set ``1``
        to recover classic matched pairs.
    min_pairs : int
        Minimum number of supergeo pairs per arm.
    """

    class Config:
        arbitrary_types_allowed = True

    df: pd.DataFrame = Field(..., description="Historical pre-treatment panel.")
    outcome: str = Field(..., description="Historical outcome column (e.g. sales).")
    arm: str = Field(..., description="Categorical column of each geo's arm.")
    unitid: str = Field(..., description="Unit (geo) identifier column.")
    time: str = Field(..., description="Time-period column.")
    post_col: Optional[str] = Field(
        default=None,
        description="0/1 indicator of post-treatment periods. If given, the "
                    "design uses pre rows only and a realized DiD ATT is "
                    "computed on the post rows (results.effects).")
    weight_col: Optional[str] = Field(
        default=None,
        description="Per-unit aggregation weight column (e.g. population); "
                    "makes the design and ATT population-weighted.")
    max_supergeo_size: Optional[int] = Field(
        default=None, ge=1,
        description="Q: max size of either supergeo within a pair. If None "
                    "(default), Q is selected automatically by minimising the "
                    "program-level MDE over feasible Q (see metadata "
                    "'q_sweep').")
    min_pairs: int = Field(
        default=1, ge=1,
        description="Minimum number of supergeo pairs per arm.")
    objective: Literal["ss_res", "r2", "weighted"] = Field(
        default="ss_res",
        description="Per-pair parallelism cost: 'ss_res' (absolute DiD "
                    "residual SS; scale-dependent), 'r2' (1-R^2; scale-free), "
                    "or 'weighted' (recency-weighted residual SS).")
    recency_decay: float = Field(
        default=0.97, gt=0.0, le=1.0,
        description="Geometric recency-weight decay for objective='weighted' "
                    "(period t weight = recency_decay**(T0-1-t)).")
    frac_E: float = Field(
        default=0.7, gt=0.0, lt=1.0,
        description="Fraction of the pre-period used as the estimation window "
                    "the split is optimised over; the remaining tail is held "
                    "out as a blank window whose residuals give honest, "
                    "out-of-sample variance for the MDE and conformal CIs "
                    "(mirrors LEXSCM / SPCD).")
    covariates: Optional[List[str]] = Field(
        default=None,
        description="Optional baseline (time-invariant) covariate columns to "
                    "balance across each supergeo pair, in addition to "
                    "pre-period parallelism. Each unit's covariate value is "
                    "its mean over the panel; the per-pair cost gains a "
                    "standardized SMD^2 imbalance term "
                    "sum_m w_m ((cbar_A - cbar_B)/s_m)^2, keeping the outer "
                    "selection a linear MILP.")
    covariate_weights: Optional[Dict[str, float]] = Field(
        default=None,
        description="Per-covariate weight on the standardized SMD^2 imbalance "
                    "penalty (default 1.0 each). Larger up-weights balancing "
                    "that covariate relative to raw parallelism.")
    standardize_covariates: bool = Field(
        default=True,
        description="Standardize each covariate by its cross-unit std before "
                    "the SMD^2 imbalance (puts covariates on a common scale).")
    compute_power: bool = Field(
        default=True,
        description="Attach a program- and arm-level MDE / power analysis to "
                    "the result (AR(1) serial-correlation-corrected DiD; the "
                    "program-level MDE is the headline metric).")
    power_target: float = Field(
        default=0.80, gt=0.0, lt=1.0,
        description="Target power the stored MDEs are computed at.")
    power_alpha: float = Field(
        default=0.05, gt=0.0, lt=1.0,
        description="Two-sided significance level for the MDE.")
    power_post_periods: List[int] = Field(
        default_factory=lambda: list(range(2, 13)),
        description="Post-period horizons (X) to evaluate the MDE at "
                    "(default 2..12).")
    att_augment: bool = Field(
        default=True,
        description="Use the augmented DiD (free control-scale coefficient "
                    "delta_2, Li & Van den Bulte 2022) for the realized ATT; "
                    "False forces delta_2=1 (plain DiD).")
    att_trend: bool = Field(
        default=True,
        description="Include a linear time-trend regressor in the realized-"
                    "ATT counterfactual (stationarizes a drifting gap).")
    display_graphs: bool = Field(
        default=True,
        description="Plot treatment vs control aggregate trajectories per arm.")
    save: Union[bool, str] = Field(
        default=False, description="Save the plot (True or a filename base).")


class FSCMConfig(BaseEstimatorConfig):
    """
    Configuration for the Forward-Selected Synthetic Control Method (FSCM).

    FSCM grows a nested donor sequence by forward stepwise selection on the
    training half of the pre-period (greedy on in-sample RMSPE), then chooses
    the donor count by minimizing out-of-sample RMSPE on the held-out test
    half (two-interval-time cross-validation). The final simplex weights are
    refit on the full pre-period over the selected donors.

    References
    ----------
    Cerulli, Giovanni. 2024.
    "Optimal initial donor selection for the synthetic control method."
    Economics Letters, 244: 111976.
    https://doi.org/10.1016/j.econlet.2024.111976
    """

    forward_selection: bool = Field(
        default=True,
        description=(
            "If True, run Cerulli's forward stepwise donor selection with "
            "rolling-origin out-of-sample validation (each candidate fit by "
            "the bilevel solver). If False, take the full bilevel solve over "
            "all donors with no selection."
        ),
    )

    covariates: Optional[List[str]] = Field(
        default=None,
        description=(
            "Optional covariate columns for predictor matching (Abadie's "
            "specification). Each covariate is averaged over its window and "
            "enters the bilevel lower-level objective; the predictor weights "
            "V are optimized on the full pool and reused. Selection and "
            "cross-validation scores are measured on the outcome."
        ),
    )

    covariate_windows: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Per-covariate averaging window as an inclusive (start, end) range "
            "of time labels, e.g. {'lnincome': (1980, 1988), 'beer': (1984, "
            "1988)}. Covariates not listed are averaged over the full "
            "pre-treatment period. Mirrors Abadie's Proposition 99 spec."
        ),
    )

    match_periods: Optional[List[Any]] = Field(
        default=None,
        description=(
            "Optional 'special predictor' periods: specific pre-treatment "
            "time labels (e.g. [1975, 1980, 1988]) whose outcome value is "
            "matched directly, as in Abadie's Proposition 99 specification."
        ),
    )

    cv_split: float = Field(
        default=0.5,
        gt=0.0,
        lt=1.0,
        description=(
            "Fraction of the pre-treatment period used as the training window; "
            "the remaining tail is the test window for cross-validation. "
            "0.5 reproduces Cerulli's equal split."
        ),
    )

    max_donors: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Cap on the number of forward-selection steps. Defaults to the "
            "full donor pool; lower it to bound runtime in high dimensions."
        ),
    )



class LEXSCMConfig(BaseMAREXConfig):
    """Configuration for LEXSCM - Fast Synthetic Experiment Design pipeline."""

    # =========================================================
    # IDENTIFICATION DESIGN
    # =========================================================

    candidate_col: str = Field(
        ...,
        description="REQUIRED: Column indicating units eligible for treatment selection "
                    "(boolean or 0/1, constant within unit)."
    )

    m: int = Field(
        ...,
        gt=0,
        description="REQUIRED: Number of units selected per treated tuple (m)."
    )

    post_col: Optional[str] = Field(
        default=None,
        description="Optional 0/1 indicator for post-treatment period."
    )
    
    weight_col: Optional[str] = Field(
        default=None,
        description="Weight column."
    )

    unit_cost_col: Optional[str] = Field(
        default=None,
        description="Column name containing the per-unit treatment cost. "
                    "Value must be constant within each unit."
    )

    budget: Optional[float] = Field(
        default=None,
        gt=0,
        description="Hard total budget for the sum of treatment costs of the "
                    "selected treated units."
    )

    seed: int = 42

    frac_E: float = Field(
        default=0.7,
        description="Fraction of pre-treatment period used for estimation window E."
    )

    # =========================================================
    # SYNTHETIC CONTROL SPECIFICATION
    # =========================================================

    covariates: Optional[List[str]] = Field(
        default=None,
        description="Optional covariates included in synthetic control."
    )

    lambda_penalty: float = Field(
        default=0.1,
        description="Penalty for control mismatch in QP."
    )

    # =========================================================
    # SEARCH / COMPUTATIONAL BUDGET
    # =========================================================

    top_K: int = Field(
        default=20,
        description="Number of top candidate treated tuples returned by the "
                    "Stage 1 search (exact enumeration or multi-start local search)."
    )

    top_P: int = Field(
        default=10,
        description="Deprecated and unused by the rebuilt Stage 1 search "
                    "(retained for backward-compatible configs). The multi-start "
                    "local search sets its own seed count internally."
    )

    # =========================================================
    # POWER / INFERENCE (MDE)
    # =========================================================

    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)

    n_post_grid: List[int] = Field(
        default_factory=lambda: list(range(2, 9)),
        description="Post-treatment horizons used for MDE detectability curves."
    )

    n_sims: int = Field(
        default=1000,
        description="Monte Carlo simulations for null distribution in MDE."
    )

    post_imputation: Literal["mean", "max", "double_max"] = Field(default="mean")
    test_statistic: Literal["mean_abs", "mean", "rms"] = Field(default="mean_abs")
    mde_horizon: Literal[
        "early_mean",
        "early_min",
        "late"
    ] = Field(
        default="late",
        description=(
            "Defines how detectability (MDE) is aggregated across post-treatment horizons.\n\n"
            "- 'early_mean': average MDE across early windows (e.g., 2–4 weeks)\n"
            "- 'early_min': minimum MDE across early windows (most optimistic detectability)\n"
            "- 'late': uses longest horizon (e.g., 8-week MDE; recommended default)\n\n"
            "The 'late' option is recommended for operational experiments because it provides a "
            "conservative detectability bound under sustained treatment exposure."
        )
    )
    max_shortlist: int = Field(default=5, gt=0)

    power_target: float = Field(
        default=0.8, gt=0.0, lt=1.0,
        description="Target power for the minimum-detectable-effect search."
    )

    imbalance_tol: float = Field(
        default=0.25, ge=0.0,
        description="Relative slack above the best achievable pre-treatment "
                    "imbalance defining the validity gate for design selection."
    )

    # =========================================================
    # INTERNAL / SYSTEM
    # =========================================================

    display_graph: bool = Field(
        default=False,
        description="Display plot."
    )

    verbose: bool = Field(
        default=True,
        description="Print progress logs."
    )

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"




class TASCConfig(BaseEstimatorConfig):
    """Configuration for the Time-Aware Synthetic Control (TASC) estimator.

    Implements the state-space model of Rho, Illick, Narasipura, Abadie, Hsu,
    and Misra (2026, arXiv:2601.03099) with EM learning (Kalman filter + RTS
    smoother on the E-step, closed-form MLE on the M-step) and Kalman-with-
    infinite-variance counterfactual inference.
    """

    d: int = Field(
        ...,
        ge=1,
        description=(
            "Hidden state dimension. Should satisfy d << min(n_donors, T) so "
            "that H X retains a low-rank signal structure."
        ),
    )
    n_em_iter: int = Field(
        default=50,
        ge=1,
        description="Number of EM iterations N1 in Algorithm 2 (EM_pre).",
    )
    em_tol: Optional[float] = Field(
        default=None,
        gt=0,
        description=(
            "Optional convergence tolerance on the maximum absolute change in "
            "(A, H) between successive EM iterations. If None, EM runs for the "
            "full ``n_em_iter`` iterations."
        ),
    )
    diagonal_Q: bool = Field(
        default=True,
        description=(
            "If True, the M-step constrains the state-noise covariance Q to be "
            "diagonal (the paper's default in Algorithm 7). If False, the full "
            "symmetric covariance is updated."
        ),
    )
    diagonal_R: bool = Field(
        default=True,
        description=(
            "If True, the M-step constrains the observation-noise covariance R "
            "to be diagonal (the paper's default in Algorithm 7). If False, the "
            "full symmetric covariance is updated."
        ),
    )
    alpha: float = Field(
        default=0.05,
        gt=0,
        lt=1,
        description=(
            "Significance level for the posterior-based counterfactual "
            "confidence intervals (computed from h_1' P_t^s h_1)."
        ),
    )
    seed: Optional[int] = Field(
        default=None,
        description="Optional random seed used for EM initialization tie-breaks.",
    )

    @model_validator(mode="after")
    def _check_tasc_dim(self) -> "TASCConfig":
        if self.d < 1:
            raise MlsynthConfigError("'d' (hidden state dimension) must be >= 1.")
        return self




class SBCConfig(BaseEstimatorConfig):
    """Configuration for the Synthetic Business Cycle (SBC) estimator.

    Implements:

        Shi, Z., Xi, J., & Xie, H. (2025). "A Synthetic Business Cycle
        Approach to Counterfactual Analysis with Nonstationary
        Macroeconomic Data." arXiv:2505.22388.

    Parameters
    ----------
    h : int
        Hamilton-filter forecasting horizon (paper's recommendation:
        roughly two to four years; default ``h=2``).
    p : int
        Number of self-lags used by the Hamilton filter (paper default
        ``p=2``).
    weights_mode : {"simplex", "unrestricted"}
        Synthetic-control variant for the cycle imputation step.
        ``"simplex"`` (default) matches the paper's Eq. (3): non-negative
        weights summing to 1, no intercept. ``"unrestricted"`` runs an
        OLS with intercept (Doudchenko-Imbens vertical regression style).
    display_graphs : bool
        Display the observed-vs-counterfactual plot after the fit.
    """

    h: int = Field(default=2, ge=1, description=(
        "Hamilton-filter forecasting horizon. Default 2 follows Hamilton "
        "(2018) and Shi/Xi/Xie (2025)."
    ))
    p: int = Field(default=2, ge=1, description=(
        "Number of self-lags used by the Hamilton filter."
    ))
    weights_mode: Literal["simplex", "unrestricted"] = Field(
        default="simplex",
        description=(
            "Weighting scheme for the SCM step on cycles. 'simplex' is "
            "the paper's default; 'unrestricted' is the vertical-regression "
            "alternative discussed in Section 2.1."
        ),
    )
    display_graphs: bool = Field(default=False, description=(
        "Show the SBC counterfactual plot after fitting."
    ))








class HSCConfig(BaseEstimatorConfig):
    """Configuration for the Harmonic Synthetic Control (HSC) estimator.

    Implements:

        Liu, Z., & Xu, Y. (2026). "The Harmonic Synthetic Control Method."

    HSC matches donors under a frequency-dependent metric and absorbs the
    treated unit's low-frequency residual into a smooth component that is
    forecast forward. A single allocation parameter ``rho in [0, 1]``,
    selected by rolling-origin cross-validation, interpolates between
    synthetic control on ``q``-th differences (``rho -> 0``) and synthetic
    control on levels with an intercept/trend (``rho -> 1``).

    Parameters
    ----------
    q : int
        Smoothness order of the difference operator (1 or 2). ``q=1``
        controls a stochastic trend / random-walk drift; ``q=2`` controls a
        local-linear trend. Default 1.
    rho_grid : list of float
        Candidate allocation values in ``[0, 1]`` searched by CV. Default
        ``[0.0, 0.2, 0.5, 0.8, 0.97]``.
    cv_splits : int
        Number of rolling-origin folds (sklearn ``TimeSeriesSplit``). Default 3.
    ridge : float
        Relative ridge coefficient on the donor weights for QP conditioning.
        Default ``1e-6``.
    forecaster : {"arima110", "last"}
        Forecaster for the smooth residual. ``"arima110"`` (default) is a
        closed-form ARIMA(1, 1, 0); ``"last"`` carries the last value forward.
    display_graphs : bool
        Show the observed-vs-counterfactual plot after fitting.

    Notes
    -----
    Following Liu & Xu (2026), HSC ships the *point* estimator only; uncertainty
    quantification is deliberately out of scope, so there are no inference
    options here.
    """

    q: Literal[1, 2] = Field(
        default=1,
        description="Smoothness order of the difference operator (1 or 2).",
    )
    rho_grid: List[float] = Field(
        default_factory=lambda: [0.0, 0.2, 0.5, 0.8, 0.97],
        description="Candidate allocation values in [0, 1] searched by CV.",
    )
    cv_splits: int = Field(
        default=3, ge=2,
        description="Rolling-origin CV folds (sklearn TimeSeriesSplit).",
    )
    ridge: Union[float, Literal["sdid"]] = Field(
        default=1e-6,
        description="Donor-weight ridge. A float is a relative coefficient "
                    "(ridge * trace(X'WX)/N). The string 'sdid' uses the "
                    "data-driven SDID-style penalty zeta^2 T0 with "
                    "zeta = T_post^{1/4} sigma_dX (Liu & Xu 2026 sec. 7), "
                    "which diversifies the donor weights.",
    )
    forecaster: Literal["arima110", "last"] = Field(
        default="arima110",
        description="Smooth-residual forecaster: ARIMA(1,1,0) or last-value.",
    )
    display_graphs: bool = Field(
        default=False,
        description="Show the HSC counterfactual plot after fitting.",
    )

    @model_validator(mode="after")
    def _check_hsc_params(cls, values: Any) -> Any:
        grid = values.rho_grid
        if not grid:
            raise MlsynthConfigError("rho_grid must contain at least one value.")
        if any((r < 0.0 or r > 1.0) for r in grid):
            raise MlsynthConfigError("All rho_grid values must lie in [0, 1].")
        if not isinstance(values.ridge, str) and values.ridge < 0.0:
            raise MlsynthConfigError("ridge must be non-negative (or 'sdid').")
        return values


class BVSSConfig(BaseEstimatorConfig):
    """Configuration for the Bayesian Synthetic Control with Soft Simplex (BVS-SS).

    Implements:

        Xu, Y., & Zhou, Q. (2025). "Bayesian Synthetic Control with a
        Soft Simplex Constraint." arXiv:2503.06454.

    Parameters
    ----------
    n_iter : int
        Total Gibbs iterations (including burn-in).
    burn_in : int
        Number of warm-up iterations discarded before reporting.
    kappa1, kappa2 : float
        Gamma hyperparameters for the prior on the observation
        precision phi. Paper Section 5.1 uses kappa1=kappa2=1.
    theta : float
        Bernoulli prior inclusion probability per donor. Paper's
        empirical Section 6 uses theta in 0.2--0.25.
    tau_a, tau_b : float
        Gamma-prior shape (a1) and rate (a2) for the soft-constraint
        variance tau. Paper Section 5.1 uses (0.01, 0.1).
    n_tau : int
        Number of MH steps for tau per outer iteration.
    tau_min : float
        Numerical floor on tau; proposals below are reflected.
    ci_alpha : float
        Two-sided significance level for credible intervals (default
        0.05 gives 95% bands).
    init_phi, init_tau : float
        Initial values for phi and tau.
    init_mu : list, optional
        Initial weight vector of length N. Defaults to the uniform
        simplex mu_i = 1 / N.
    display_graphs : bool
        Display the observed-vs-counterfactual plot after the fit.
    verbose : bool
        Show a tqdm progress bar during MCMC.
    seed : int, optional
        Seed for the numpy.random.Generator used inside the sampler.
    """

    n_iter: int = Field(default=2000, ge=10)
    burn_in: int = Field(default=1000, ge=0)
    kappa1: float = Field(default=1.0, gt=0)
    kappa2: float = Field(default=1.0, gt=0)
    theta: float = Field(default=0.25, gt=0.0, lt=1.0)
    tau_a: float = Field(default=0.01, gt=0)
    tau_b: float = Field(default=0.1, gt=0)
    n_tau: int = Field(default=11, ge=1)
    tau_min: float = Field(default=1e-6, gt=0)
    ci_alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    init_phi: float = Field(default=0.8, gt=0)
    init_tau: float = Field(default=1.0, gt=0)
    init_mu: Optional[List[float]] = Field(default=None)
    display_graphs: bool = Field(default=False)
    verbose: bool = Field(default=False)
    seed: Optional[int] = Field(default=None)


class SPCDConfig(BaseMAREXConfig):
    """
    Configuration for the Synthetic Principal Component Design (SPCD) estimator.

    Implements Lu, Li, Ying, Blanchet (2022), "Synthetic Principal Component
    Design: Fast Covariate Balancing with Synthetic Controls",
    arXiv:2211.15241v1.

    Parameters
    ----------
    variant : {"spcd", "norm_spcd"}
        Iteration-box choice. ``"spcd"`` uses Eq. (4)/(7) of the paper;
        ``"norm_spcd"`` uses Eq. (5)/(8). The paper's Section 4
        experiments use ``"norm_spcd"`` with closed-form weights.
    weights : {"empirical", "exact"}
        Final-weight-step choice. ``"empirical"`` uses Eq. (9) — the
        closed-form approximation used in all of the paper's
        experiments. ``"exact"`` solves Eq. (6) via cvxpy.
    alpha_ridge : float or None, optional
        Ridge term ``alpha`` in Eq. (2), playing the role of the noise
        variance ``sigma``. If ``None``, it is chosen by out-of-sample
        pre-period balance over a noise-scale grid
        (``select_alpha_by_holdout``), since the post-period RMSE is a
        jumpy function of ``alpha`` when ``N > T_pre``. Pass a value
        (e.g. a known noise variance) to bypass selection.
    lam_balance : float or None, optional
        Sum-zero penalty ``lambda`` in Eq. (2). Auto-estimated if
        ``None``. Theorem 1 requires this to be "large enough".
    beta : float or None, optional
        Iteration step parameter ``beta`` in Eqs. (4)/(5)/(7)/(8).
        Auto-estimated from the spectrum if ``None``.
    max_iter : int
        Maximum iterations for the SPCD/NormSPCD while loop.
    T0 : int or None, optional
        Number of pre-treatment periods.
    post_col : str or None, optional
        Column indicating post-treatment periods.
    solver : Any, optional
        CVXPY-compatible solver. Used only when ``weights="exact"``.
    display_graph : bool
        Whether to display the synthetic treated/control plot.
    verbose : bool
        Solver verbosity.

    Notes
    -----
    Algorithms 3 and 4 of the paper (Appendix 3.2) are abstract
    meta-versions used in the proof of Theorem 3 (global convergence).
    They correspond to the same iterations as Algorithms 1 and 2 acting
    on a generic Hermitian perturbed rank-1 matrix and are not exposed
    as separate user options here.
    """

    variant: Literal["spcd", "norm_spcd"] = Field(
        default="norm_spcd",
        description="SPCD iteration variant. 'spcd' uses Eq. (4)/(7); "
                    "'norm_spcd' uses Eq. (5)/(8).",
    )
    weights: Literal["empirical", "exact"] = Field(
        default="empirical",
        description="Final-weight-step choice. 'empirical' uses Eq. (9); "
                    "'exact' solves Eq. (6) via cvxpy.",
    )
    alpha_ridge: Optional[float] = Field(
        default=None,
        ge=0,
        description="Ridge term alpha in Eq. (2) (the noise variance sigma). "
                    "If None, selected by out-of-sample pre-period balance "
                    "over a noise-scale grid (select_alpha_by_holdout).",
    )
    lam_balance: Optional[float] = Field(
        default=None,
        ge=0,
        description="Sum-zero penalty lambda in Eq. (2). Auto-estimated if None.",
    )
    beta: Optional[float] = Field(
        default=None,
        ge=0,
        description="Iteration step parameter beta in Eqs. (4)/(5)/(7)/(8). "
                    "Auto-estimated if None.",
    )
    max_iter: int = Field(
        default=200,
        gt=0,
        description="Maximum iterations for the SPCD/NormSPCD while loop.",
    )
    T0: Optional[int] = Field(
        default=None,
        gt=0,
        description="Number of pre-treatment periods when post_col is not supplied.",
    )
    post_col: Optional[str] = Field(
        default=None,
        description="Optional 0/1 or boolean column identifying post-treatment periods.",
    )
    arm: Optional[str] = Field(
        default=None,
        description="Optional categorical column naming each unit's treatment "
                    "arm. When given, SPCD solves its design independently "
                    "within each arm's units and returns SPCDMultiArmResults "
                    "(a dict of per-arm SPCDResults); when None (default), a "
                    "single SPCDResults is returned.",
    )
    solver: Any = Field(
        default=None,
        description="CVXPY-compatible solver, only used when weights='exact'.",
    )
    display_graph: bool = Field(default=False, description="Whether to display SPCD plots.")
    verbose: bool = Field(default=False, description="Whether to print solver progress.")

    # ------------------------------------------------------------------
    # Inference and power analysis (LEXSCM-style E/B holdout split).
    # ------------------------------------------------------------------
    enable_inference: bool = Field(
        default=True,
        description="Run conformal inference + Monte Carlo power analysis. "
                    "Trains the design on the first holdout_frac_E of pretreatment "
                    "and uses the remaining periods as out-of-sample residuals.",
    )
    holdout_frac_E: float = Field(
        default=0.7,
        ge=0.1,
        le=0.95,
        description="Fraction of pretreatment periods used for the SPCD design fit. "
                    "The remaining 1 - holdout_frac_E periods form the holdout window.",
    )
    inference_alpha: float = Field(
        default=0.05,
        gt=0.0,
        lt=1.0,
        description="Two-sided significance level for conformal CI and MDE.",
    )
    power_target: float = Field(
        default=0.8,
        gt=0.0,
        lt=1.0,
        description="Target statistical power for the MDE search.",
    )
    mde_n_sims: int = Field(
        default=5000, gt=0,
        description="Monte Carlo draws for the null distribution in the MDE.",
    )
    mde_n_trials: int = Field(
        default=400, gt=0,
        description="Trials per tau grid point for empirical power estimation.",
    )
    mde_horizon_grid: Optional[List[int]] = Field(
        default=None,
        description="Optional list of post-treatment horizons for the "
                    "detectability curve. If None, no curve is computed.",
    )
    inference_seed: int = Field(
        default=1400,
        description="Seed for the Monte Carlo MDE machinery.",
    )
    min_blank_size: int = Field(
        default=5, gt=0,
        description="Minimum holdout-window size below which inference is "
                    "skipped with a warning (design is still fit on the "
                    "estimation window).",
    )
    pooled_weights: Literal["size", "equal"] = Field(
        default="size",
        description="Weighting for the multi-arm pooled average-effect MDE "
                    "(only used when an 'arm' column is set). 'size' weights "
                    "each arm by its unit count (population-average effect); "
                    "'equal' weights arms equally.",
    )
    covariates: Optional[List[str]] = Field(
        default=None,
        description="Optional covariate columns to balance on *in addition* "
                    "to the pre-treatment outcomes. Each unit's per-covariate "
                    "pre-period mean is z-scored across units and folded into "
                    "the SPCD iteration matrix as a covariate-balance term "
                    "(M += covariate_weight * scale * X X^T). None (default) "
                    "balances on outcomes only. Time-invariant covariates "
                    "(e.g. last year's market share) collapse to their value.",
    )
    covariate_weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Relative weight of covariate balance vs outcome balance "
                    "in the SPCD Gram matrix. 0 ignores covariates; 1 gives "
                    "covariates equal 'energy' to the outcomes; >1 upweights "
                    "covariate balance. Only used when 'covariates' is set.",
    )

    @model_validator(mode="after")
    def check_spcd_params(cls, values: Any) -> Any:
        df = values.df
        n_periods = df[values.time].nunique()

        if values.post_col is not None and values.post_col not in df.columns:
            raise MlsynthConfigError(f"post_col '{values.post_col}' is not present in df.")

        if values.T0 is not None and values.T0 > n_periods:
            raise MlsynthConfigError("T0 cannot exceed the number of unique time periods in df.")

        if values.covariates:
            missing = [c for c in values.covariates if c not in df.columns]
            if missing:
                raise MlsynthConfigError(
                    f"covariates not present in df: {missing}."
                )
            non_numeric = [
                c for c in values.covariates
                if not pd.api.types.is_numeric_dtype(df[c])
            ]
            if non_numeric:
                raise MlsynthConfigError(
                    f"SPCD covariates must be numeric; non-numeric: {non_numeric}."
                )

        return values


class NSCConfig(BaseEstimatorConfig):
    """Configuration for the Nonlinear Synthetic Control (NSC) estimator.

    Implements Tian (2023), *"The Synthetic Control Method with
    Nonlinear Outcomes"* (arXiv:2306.01967). The estimator generalises
    Abadie-Diamond-Hainmueller (2010) synthetic control to nonlinear
    outcome functions by

    * dropping the non-negativity restriction on donor weights,
    * adding a pairwise-distance-weighted L1 penalty plus an L2
      penalty in the weight-fitting objective, and
    * scaling the tuning parameters by the eigenvalues of
      :math:`Z_0 Z_0'` so they can be cross-validated on ``[0, 1]``.

    Parameters
    ----------
    a : float or None
        Dimensionless L1-discrepancy tuning parameter on ``[0, 1]``.
        Higher values concentrate weight on units close to the
        treated one in pretreatment matching variables (paper eq.
        (7)). ``None`` triggers coordinate-descent CV.
    b : float or None
        Dimensionless L2 tuning parameter on ``[0, 1]``. Higher
        values spread weights more evenly across donors. ``None``
        triggers coordinate-descent CV.
    cv_grid_size : float
        Step of the CV grid on ``[0, 1]``. Paper default is 0.1.
    cv_target : {"controls", "treated"}
        CV target. ``"controls"`` (paper default) leaves each donor
        out in turn and predicts it from the others; ``"treated"``
        scores on the treated unit's pretreatment fit.
    cv_max_iterations : int
        Hard cap on coordinate-descent iterations for the CV sweep.
    covariates : list of str, optional
        Optional covariate columns to use as additional matching
        variables alongside the pretreatment outcomes; collapsed to
        per-unit pretreatment means before being stacked into ``Z_0``.
    alpha : float
        Two-sided significance level for the Doudchenko-Imbens
        confidence intervals.
    run_inference : bool
        Whether to compute the Doudchenko-Imbens variance estimator
        and the per-period / ATT CIs.
    display_graphs : bool
        Whether to render the diagnostic NSC plot.
    """

    a: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Dimensionless L1-discrepancy tuning parameter; "
        "None triggers CV.",
    )
    b: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Dimensionless L2 tuning parameter; None triggers CV.",
    )
    cv_grid_size: float = Field(
        default=0.1, gt=0.0, le=0.5,
        description="CV grid step on [0, 1].",
    )
    cv_target: Literal["controls"] = Field(
        default="controls",
        description=(
            "CV target. Only the R-faithful 'controls' target is supported: "
            "for each donor, fit weights on its pre-period using the other "
            "donors and score on its held-out post-period MSPE."
        ),
    )
    cv_max_iterations: int = Field(
        default=3, ge=1, le=20,
        description="Coordinate-descent iterations for the CV sweep.",
    )
    covariates: Optional[List[str]] = Field(
        default=None,
        description="Optional covariate columns stacked into the matching matrix.",
    )
    standardize: bool = Field(
        default=True,
        description=(
            "Centre each matching-variable column to mean 0 and rescale to "
            "sample sd 1 (R's scale()). Default True matches the reference "
            "NSC implementation; set False only for back-compat."
        ),
    )
    alpha: float = Field(
        default=0.05, gt=0.0, lt=1.0,
        description="Two-sided significance level for Doudchenko-Imbens CIs.",
    )
    run_inference: bool = Field(
        default=True,
        description="Whether to run Doudchenko-Imbens variance estimation.",
    )
    seed: int = Field(
        default=123,
        description=(
            "Seed for the extra-donor draws in the R-faithful CV and "
            "leave-one-control inference loops. Matches the reference "
            "R script's set.seed(123)."
        ),
    )

    @model_validator(mode='after')
    def check_nsc_params(cls, values: Any) -> Any:
        if values.covariates is not None:
            unknown = [c for c in values.covariates if c not in values.df.columns]
            if unknown:
                raise MlsynthConfigError(
                    f"covariates references unknown columns: {unknown}"
                )
        return values

class SDIDConfig(BaseEstimatorConfig):
    """Configuration for the Synthetic Difference-in-Differences (SDID) estimator.

    Implements Arkhangelsky, Athey, Hirshberg, Imbens & Wager (2021)'s SDID
    with the event-study aggregation of Ciccia (2024, arXiv:2407.09565).
    Inherits the standard ``df`` / ``outcome`` / ``treat`` / ``unitid`` /
    ``time`` panel-data interface from :class:`BaseEstimatorConfig`.
    """

    B: int = Field(
        default=500,
        ge=0,
        description=(
            "Number of placebo iterations for the variance estimator. "
            "Set to 0 to skip placebo inference (att_se / p_value will be NaN). "
            "The paper uses B = 500."
        ),
    )
    seed: int = Field(
        default=1400,
        description="Random seed used for the placebo resampling.",
    )


class SparseSCConfig(BaseEstimatorConfig):
    """Configuration for the Sparse Synthetic Control (SparseSC) estimator.

    Implements the L1-penalized predictor-weighting SCM variant of
    Vives-i-Bastida and collaborators (port of the MATLAB
    ``sparse_synth.m`` driver) for the canonical Abadie, Diamond, and
    Hainmueller (2010) framework.

    Like every other ``mlsynth`` estimator this one is fed a single
    long-format ``df`` with one row per (unit, time). Predictors are
    constructed under the hood from the long frame: each column listed
    in ``covariates`` is collapsed to its pre-treatment mean per unit,
    and each entry of ``outcome_lag_periods`` adds the outcome at that
    specific pre-treatment period as a predictor.
    """

    covariates: Optional[List[str]] = Field(
        default=None,
        description=(
            "Names of columns in ``df`` to use as predictors. For each "
            "covariate, the per-unit pre-treatment mean is taken as the "
            "predictor value. The first covariate is the *anchor*: its "
            "V-weight is pinned to 1."
        ),
    )
    outcome_lag_periods: Optional[List[Any]] = Field(
        default=None,
        description=(
            "Optional list of pre-treatment time labels (as found in the "
            "``time`` column) whose outcome values become additional "
            "predictors -- the canonical ADH lagged-outcome predictors. "
            "Appended after ``covariates`` in the predictor matrix."
        ),
    )
    T0_train: Optional[int] = Field(
        default=None,
        ge=2,
        description=(
            "End of the training block within the pre-treatment period "
            "(exclusive). Validation runs on [T0_train, T0_total). "
            "Defaults to floor(T0_total * 0.75)."
        ),
    )
    lambda_grid: Optional[List[float]] = Field(
        default=None,
        description=(
            "L1 penalty grid for predictor selection. Defaults to "
            "[0] + numpy.logspace(-4, 0, 50) -- the MATLAB default."
        ),
    )
    standardize: bool = Field(
        default=True,
        description=(
            "Standardize each predictor across all units before fitting."
        ),
    )
    outer_loss_window: str = Field(
        default="training",
        description=(
            "Which pre-treatment block the outer V-objective evaluates "
            "the outcome MSE over. 'training' (default) matches the "
            "paper's Algorithm 1 line 4 ('for the training data') and "
            "the MATLAB driver sparse_synth.m; this gives the pre-fit "
            "shown in paper Figure 3 and the Table 1 estimates. "
            "'validation' takes the page-4 L_V definition literally "
            "(Y_val in the outer objective); useful for ablation but "
            "produces much worse in-sample fit."
        ),
    )
    solver: Any = Field(
        default=None,
        description="CVXPY solver for the inner W-weight QP. Defaults to OSQP.",
    )
    max_outer_iter: int = Field(
        default=500,
        ge=10,
        description=(
            "Max iterations of the outer L-BFGS-B optimization of "
            "V-weights per lambda. With the analytical envelope-theorem "
            "gradient and ftol=1e-12 (the analytical-mode default), "
            "L-BFGS-B may need several hundred iterations to converge "
            "fully on hard predictor sets; each iteration is microseconds "
            "without the finite-difference multiplier, so 500 is cheap."
        ),
    )
    run_inference: bool = Field(
        default=True,
        description="Whether to run the post-estimation inference procedure.",
    )
    inference_method: Literal["conformal", "placebo", "none"] = Field(
        default="conformal",
        description=(
            "Which inference procedure to run when ``run_inference`` "
            "is True. ``conformal`` (default) builds a moving-block "
            "conformal CI for the ATT in the spirit of Chernozhukov, "
            "Wuethrich and Zhu (2021), calibrated on the validation "
            "residuals; ``placebo`` runs the Abadie-style placebo "
            "permutation; ``none`` skips inference entirely (equivalent "
            "to ``run_inference=False``)."
        ),
    )
    conformal_window: Literal["validation", "pre"] = Field(
        default="validation",
        description=(
            "Residual block used to calibrate the conformal CI. "
            "``validation`` uses only the held-out validation periods "
            "[T0_train, T0_total); ``pre`` uses the full pre-treatment "
            "block [0, T0_total). Validation is smaller but truly "
            "out-of-sample under the chosen V."
        ),
    )
    alpha: float = Field(
        default=0.05,
        gt=0.0,
        lt=1.0,
        description="Two-sided significance level for the ATT CI.",
    )
    n_placebo: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Number of placebo donors to use. ``None`` uses every donor."
        ),
    )
    placebo_resweep: bool = Field(
        default=False,
        description=(
            "If True, re-run the full lambda sweep for each placebo. "
            "Slow but most faithful to the actual fit; ``False`` reuses "
            "the lambda selected on the actual treated unit."
        ),
    )
    seed: int = Field(
        default=1400,
        description="Random seed for the placebo subsample.",
    )
    use_analytical_grad: bool = Field(
        default=False,
        description=(
            "Use the envelope-theorem closed-form Jacobian inside the "
            "outer L-BFGS-B sweep. The analytical Jacobian is exact "
            "(verified against finite differences to ~1e-7) and yields "
            "a 5-10x speedup, but the clean gradient lets L-BFGS-B "
            "settle at the first critical point near the cold init "
            "on the non-convex L1-penalized V-objective; the FD path's "
            "implicit gradient noise tends to find better local optima "
            "at non-zero lambda. Off by default for correctness; opt in "
            "when running large placebo sweeps where exact local optimum "
            "matters less than throughput."
        ),
    )
    warm_start: bool = Field(
        default=False,
        description=(
            "Reuse the previous lambda's V-solution as the initialiser "
            "for the next lambda in the sweep. Warm-starting can save "
            "outer iterations, but on rank-deficient designs it can "
            "also push v into a poorly-conditioned region that breaks "
            "the inner Clarabel solve and can cause L-BFGS-B to settle "
            "in a different local optimum than the canonical "
            "MATLAB-style cold init. Off by default."
        ),
    )


class MicroSynthConfig(BaseEstimatorConfig):
    """Configuration for the MicroSynth estimator.

    Implements Robbins & Davenport (2021, *J. Stat. Software*),
    "microsynth: Synthetic Control Methods for Disaggregated and
    Micro-Level Data in R". A user-level balancing estimator: solve a
    constrained QP for non-negative simplex weights on control users
    that exactly balance covariate moments against the treated group's
    moments, then read off the ATT as the weighted-mean outcome
    difference.

    Unlike aggregate-unit SCM estimators in :mod:`mlsynth`, MicroSynth
    operates at the individual-user level with many treated units and
    a large donor pool of controls. The dual ascent solver scales with
    the number of balancing constraints (``d + 1``), not with the
    number of controls, making it tractable for ``N_C`` in the
    millions on a single machine.
    """

    covariates: List[str] = Field(
        ...,
        description=(
            "Column names in ``df`` to use as balancing covariates. "
            "These must be time-invariant per unit (a single value "
            "per user); time-varying features should be collapsed by "
            "the caller (e.g., to pre-treatment means) before passing."
        ),
    )
    outcome_lag_periods: Optional[List[Any]] = Field(
        default=None,
        description=(
            "Optional list of pre-treatment time labels (as found in "
            "the ``time`` column) whose outcome values become "
            "additional balancing constraints -- the canonical lagged-"
            "outcome predictors. Appended after ``covariates``."
        ),
    )
    standardize_covariates: bool = Field(
        default=True,
        description=(
            "Standardize each covariate to unit SD across all units "
            "before fitting. Improves numerical conditioning of the "
            "dual problem; does not change the final weights."
        ),
    )
    balance_tol: float = Field(
        default=1e-4,
        gt=0.0,
        description=(
            "Maximum absolute standardized mean difference per "
            "covariate accepted as 'balanced' after weighting. Used "
            "for the feasibility diagnostic."
        ),
    )
    max_iter: int = Field(
        default=500,
        ge=10,
        description="L-BFGS-B maximum iterations for the dual problem.",
    )
    gtol: float = Field(
        default=1e-8,
        gt=0.0,
        description="L-BFGS-B gradient tolerance.",
    )
    run_inference: bool = Field(
        default=True,
        description="Whether to compute a bootstrap confidence interval.",
    )
    n_bootstrap: int = Field(
        default=500,
        ge=2,
        description="Bootstrap replications for CI.",
    )
    seed: int = Field(
        default=1400,
        description="Random seed for the bootstrap.",
    )


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
            "Whether to run the paper's delete-one jackknife inference "
            "(refits the estimator dropping each unit; can be slow)."
        ),
    )
    alpha: float = Field(
        default=0.05,
        gt=0.0,
        lt=1.0,
        description="Significance level for the Wald confidence band.",
    )


class SequentialSDIDConfig(BaseEstimatorConfig):
    """Configuration for the Sequential Synthetic Difference-in-Differences estimator.

    Implements Arkhangelsky & Samkov (2025, arXiv:2404.00164v2). Operates on
    cohort-level aggregates and is robust to violations of parallel trends
    induced by interactive fixed effects. Inherits the standard ``df`` /
    ``outcome`` / ``treat`` / ``unitid`` / ``time`` panel interface from
    :class:`BaseEstimatorConfig`.
    """

    eta: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Non-negative regularization for the two QPs. Larger values shrink "
            "the unit weights toward ``omega_j proportional to pi_j`` (the "
            "stacked-DiD imputation limit of Remark 2.2)."
        ),
    )
    mode: Literal["ssdid", "sdid_imputation"] = Field(
        default="ssdid",
        description=(
            "Estimator mode. 'ssdid' is the paper's main estimator with a "
            "finite ``eta``. 'sdid_imputation' forces the limit ``eta -> "
            "infinity``, recovering the imputation-style sequential DiD of "
            "Remark 2.2 (Borusyak et al. 2024-style)."
        ),
    )
    K: Optional[int] = Field(
        default=None,
        ge=0,
        description=(
            "Maximum event-time horizon to estimate. ``None`` (default) "
            "auto-sets ``K = T - a_max`` so every estimable effect fits "
            "inside the panel."
        ),
    )
    a_min: Optional[int] = Field(
        default=None,
        description=(
            "Earliest treated cohort (1-based time index) to include. "
            "``None`` (default) uses the earliest adopting cohort."
        ),
    )
    a_max: Optional[int] = Field(
        default=None,
        description=(
            "Latest treated cohort (1-based time index) to include. ``None`` "
            "(default) uses the latest finitely-adopting cohort."
        ),
    )
    n_bootstrap: int = Field(
        default=500,
        ge=0,
        description=(
            "Number of Bayesian-bootstrap iterations for SE/CI (Section 2.3 "
            "of the paper). Set to 0 to skip inference."
        ),
    )
    alpha: float = Field(
        default=0.05,
        gt=0.0,
        lt=1.0,
        description="Significance level for the Wald confidence band.",
    )
    seed: int = Field(
        default=1400,
        description="Random seed for the bootstrap.",
    )


class SHCConfig(BaseEstimatorConfig):
    m: int = Field(default=1, description="Length of the evaluation window.")
    bandwidth_grid: Optional[List[float]] = Field(default=None, description="Bandwidth grid for LOOCV.")
    use_augmented: bool = Field(default=False, description="Use Augmented SHC (ASHC) variant.")

    @model_validator(mode="after")
    def check_shc_params(self) -> "SHCConfig":
        if not isinstance(self.use_augmented, bool):
            raise MlsynthConfigError("'use_augmented' must be a boolean.")

        if self.m <= 0:
            raise MlsynthConfigError("'m' must be a positive integer.")

        if self.bandwidth_grid is not None:
            if not self.bandwidth_grid:
                raise MlsynthConfigError("'bandwidth_grid' cannot be an empty list.")
            if not all(isinstance(h, (int, float)) for h in self.bandwidth_grid):
                raise MlsynthConfigError("All elements in 'bandwidth_grid' must be numeric.")
            if not all(h > 0 for h in self.bandwidth_grid):
                raise MlsynthConfigError("All bandwidth values must be strictly positive.")

        return self




class MLSCConfig(BaseModel):
    """Configuration for the Multi-Level Synthetic Control (mlSC) estimator.

    Implements the data-driven hierarchical-aggregation estimator of
    Bottmer (2025, "Synthetic Control with Disaggregated Data"). Unlike the
    rest of mlsynth's estimators, mlSC operates on a *two-level* panel: an
    aggregate-level DataFrame (e.g. state-by-time) and a disaggregate-level
    DataFrame (e.g. county-by-time, with a column linking each county to
    its parent state). Treatment is assigned at the aggregate level and the
    estimand is the aggregate-level ATT, but the disaggregate data enters
    the donor pool with a ridge-type penalty that shrinks the disaggregated
    weights toward "population-share times aggregate weight" — recovering
    classical SC in the large-penalty limit and fully-disaggregated SC at
    penalty zero.

    v1 implements only the heuristic and fixed-lambda penalty-selection
    paths from Section 5.2 of the paper; cross-validation will follow.
    """

    df_agg: pd.DataFrame = Field(
        ..., description="Aggregate-level long-form panel (e.g. states x time)."
    )
    df_disagg: pd.DataFrame = Field(
        ...,
        description=(
            "Disaggregate-level long-form panel (e.g. counties x time). Must "
            "contain a column identifying each disaggregate unit's parent "
            "aggregate unit (see ``agg_id``)."
        ),
    )
    outcome: str = Field(
        ..., description="Outcome column name (must exist in both dataframes)."
    )
    time: str = Field(
        ..., description="Time-period column name (must exist in both dataframes)."
    )
    treat: str = Field(
        ...,
        description=(
            "Binary 0/1 treatment indicator column (must exist in both "
            "dataframes). Treatment is assigned at the aggregate level; each "
            "disaggregate unit's treat value must equal its parent aggregate's "
            "treat value at every period."
        ),
    )
    unitid_agg: str = Field(
        ...,
        description="Aggregate-unit identifier column in ``df_agg`` (e.g. 'state').",
    )
    unitid_disagg: str = Field(
        ...,
        description=(
            "Disaggregate-unit identifier column in ``df_disagg`` "
            "(e.g. 'county_fips')."
        ),
    )
    agg_id: str = Field(
        ...,
        description=(
            "Column in ``df_disagg`` mapping each disaggregate unit to its "
            "parent aggregate unit. Values must match ``unitid_agg`` labels "
            "in ``df_agg``."
        ),
    )
    weight_col: Optional[str] = Field(
        default=None,
        description=(
            "Optional column in ``df_disagg`` giving population weights "
            "``v_sc`` for the aggregation rule ``Y_st = sum_c v_sc Y_sct``. "
            "Within each aggregate the weights are normalized to sum to 1. "
            "If None, uniform weights ``1 / C_s`` are used (the paper's "
            "simulation default)."
        ),
    )
    lambda_est: Literal["heuristic", "fixed"] = Field(
        default="heuristic",
        description=(
            "Penalty-selection rule. 'heuristic' uses the Appendix-B closed "
            "form ``lambda = 2 * sigma_eps^2 / sigma_y^2`` estimated from "
            "the disaggregate pre-treatment panel (Appendix G). 'fixed' uses "
            "``lambda_val`` directly. Cross-validation is planned for v2."
        ),
    )
    lambda_val: float = Field(
        default=1e-4,
        ge=0.0,
        description="Penalty value used when ``lambda_est == 'fixed'``.",
    )
    solver: Any = Field(
        default=None,
        description=(
            "CVXPY-compatible solver. ``None`` (default) falls back to SCS, "
            "which ships with cvxpy and handles the QP comfortably."
        ),
    )
    display_graphs: bool = Field(
        default=True, description="Whether to display the counterfactual plot."
    )
    save: Union[bool, str, Dict[str, str]] = Field(
        default=False,
        description=(
            "Plot save configuration, identical to BaseEstimatorConfig.save."
        ),
    )
    counterfactual_color: Union[str, List[str]] = Field(
        default_factory=lambda: ["red"],
        description="Counterfactual line color(s) in the plot.",
    )
    treated_color: str = Field(
        default="black", description="Treated-unit line color in the plot."
    )

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"

    @model_validator(mode="after")
    def _check_mlsc_panels(self) -> "MLSCConfig":
        df_a = self.df_agg
        df_d = self.df_disagg
        if df_a.empty:
            raise MlsynthDataError("'df_agg' is empty.")
        if df_d.empty:
            raise MlsynthDataError("'df_disagg' is empty.")

        agg_required = {self.outcome, self.time, self.treat, self.unitid_agg}
        agg_missing = agg_required - set(df_a.columns)
        if agg_missing:
            raise MlsynthDataError(
                f"'df_agg' is missing required columns: {sorted(agg_missing)}."
            )

        disagg_required = {
            self.outcome,
            self.time,
            self.treat,
            self.unitid_disagg,
            self.agg_id,
        }
        if self.weight_col is not None:
            disagg_required.add(self.weight_col)
        disagg_missing = disagg_required - set(df_d.columns)
        if disagg_missing:
            raise MlsynthDataError(
                f"'df_disagg' is missing required columns: {sorted(disagg_missing)}."
            )

        # Time alignment: same set of periods in both panels.
        t_agg = set(df_a[self.time].unique())
        t_disagg = set(df_d[self.time].unique())
        if t_agg != t_disagg:
            sym = (t_agg ^ t_disagg)
            raise MlsynthDataError(
                "Aggregate and disaggregate panels must cover the same time "
                f"periods. Mismatching periods: {sorted(list(sym))[:10]}."
            )

        # Each disaggregate unit must map to exactly one aggregate.
        per_unit_aggs = (
            df_d.groupby(self.unitid_disagg)[self.agg_id]
            .nunique()
        )
        offenders = per_unit_aggs[per_unit_aggs > 1]
        if not offenders.empty:
            raise MlsynthDataError(
                "Each disaggregate unit must belong to exactly one aggregate "
                f"unit. Offending units: {offenders.index.tolist()[:5]}."
            )

        # Disaggregate agg_id labels must be a subset of aggregate unit labels.
        disagg_aggs = set(df_d[self.agg_id].unique())
        agg_units = set(df_a[self.unitid_agg].unique())
        orphan = disagg_aggs - agg_units
        if orphan:
            raise MlsynthDataError(
                "Disaggregate units reference aggregate labels missing from "
                f"'df_agg': {sorted(orphan)[:5]}."
            )

        return self


class RESCMConfig(BaseEstimatorConfig):
    """Configuration for the Relaxed/Balanced SCM (RESCM) estimator.

    Pick one or more named corner-case estimators of the RESCM convex program
    via ``methods`` (e.g. ``["SC", "LINF", "RELAX_L2"]``); the first listed
    drives the convenience aliases on the returned ``RESCMResults``. Valid
    names and aliases come from the registry in
    :mod:`mlsynth.utils.laxscm_helpers.specs` (``METHOD_SPECS``).
    """

    methods: List[str] = Field(
        default_factory=lambda: ["SC", "LINF", "RELAX_L2"],
        description=(
            "Named RESCM estimators to fit (e.g. 'SC', 'LASSO', 'L2', 'ENET', "
            "'LINF', 'L1LINF', 'RELAX_L2', 'RELAX_ENTROPY', 'RELAX_EL'; aliases "
            "allowed). The first listed drives the result aliases."
        ),
    )
    tau: Optional[Union[float, Literal["heuristic"]]] = Field(
        default=None,
        description=(
            "Relaxation parameter for the SCM-relaxation methods. ``None`` "
            "selects it by time-series cross-validation (slow). ``\"heuristic\"`` "
            "skips CV and uses the Bickel-Ritov-Tsybakov universal penalty "
            "``sd(y) * sqrt(2 T0 log 2J)`` (with 2x/4x feasibility fallbacks); "
            "much faster, at the cost of a defensible-but-not-optimal tau."
        ),
    )
    n_splits: Optional[int] = Field(
        default=None, ge=2,
        description="Number of CV folds for tau selection (relaxation methods).",
    )
    n_taus: Optional[int] = Field(
        default=None, ge=1,
        description="Grid size for the cross-validated tau search.",
    )
    solver: Any = Field(
        default="CLARABEL",
        description="CVXPY solver name (e.g. 'CLARABEL', 'ECOS', 'OSQP', 'SCS').",
    )
    alpha: float = Field(
        default=0.05, gt=0.0, lt=1.0,
        description="Significance level for confidence intervals and ATE inference.",
    )
    @model_validator(mode="after")
    def _validate_methods(self):
        from mlsynth.utils.laxscm_helpers.specs import normalize_method

        if not self.methods:
            raise MlsynthConfigError("`methods` must list at least one RESCM estimator.")
        normalized = []
        for m in self.methods:
            try:
                normalized.append(normalize_method(m))
            except ValueError as e:
                raise MlsynthConfigError(str(e)) from e
        object.__setattr__(self, "methods", normalized)
        return self

    class Config:
        extra = "forbid"  # Unknown fields will raise a validation error
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

    @model_validator(mode="after")
    def _check_syndes_params(cls, values: Any) -> Any:
        df = values.df
        n_units = df[values.unitid].nunique()
        n_periods = df[values.time].nunique()

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


class SIVConfig(BaseEstimatorConfig):
    """Configuration for the Synthetic Instrumental Variables (SIV) estimator.

    Implements Gulek and Vives-i-Bastida (2024), "Synthetic IV
    Estimation in Panels". SIV is a two-step procedure for panels
    with an instrumental variable: a per-unit synthetic-control fit
    on the pre-period builds debiased outcome / treatment / instrument
    series, and a just-identified 2SLS on those debiased series in the
    post-period delivers a causal effect estimate that is robust to
    both unobserved factor structure and treatment endogeneity given
    a partially-valid instrument.

    Parameters
    ----------
    instrument : str
        Name of the instrument column in ``df``.
    T0 : int or None
        Number of pre-treatment periods. Either ``T0`` or ``post_col``
        must be supplied.
    post_col : str or None
        Optional 0/1 column identifying post-treatment periods.
    T0_train : int or None
        Optional end of the training block inside the pre-period
        (exclusive); the remaining pre-periods form the "blank" block
        used by the ensemble CV and the split-conformal inference.
        Defaults to ``floor(0.75 * T0)``.
    weight_constraint : {"simplex", "l1_ball"}
        SC weight constraint per unit. ``"simplex"`` (default) matches
        the paper's empirical applications; ``"l1_ball"`` is the
        regularised relaxation analysed in Section 3.
    l1_C : float
        L1-ball radius; ignored when ``weight_constraint == "simplex"``.
    mode : {"siv", "projected", "ensemble"}
        Which estimator the orchestrator reports as the primary
        ``theta_hat``. The other variants are always computed and
        returned in ``results.estimates`` for diagnostics.
    ensemble_alpha : float or None
        Override the CV-selected blend weight in ``ensemble`` mode.
        ``None`` (default) triggers the validation-block CV from
        Section 5.1.
    inference_method : {"asymptotic", "conformal", "none"}
        ``"asymptotic"`` uses the IV sandwich SE (valid under
        Theorem 4); ``"conformal"`` runs the split-conformal
        permutation test of Section 5.2.
    alpha : float
        Two-sided significance level for the CI.
    n_permutations : int
        Maximum number of permutations enumerated when building the
        conformal distribution. Ignored under ``asymptotic``.
    """

    instrument: str = Field(
        ..., description="Instrument column name."
    )
    T0: Optional[int] = Field(
        default=None, gt=0,
        description="Number of pre-treatment periods (alternative to post_col).",
    )
    post_col: Optional[str] = Field(
        default=None,
        description="Optional 0/1 column identifying post-treatment periods.",
    )
    T0_train: Optional[int] = Field(
        default=None, ge=2,
        description="End of the training block inside the pre-period.",
    )
    weight_constraint: Literal["simplex", "l1_ball"] = Field(
        default="simplex",
        description="SC weight constraint per unit.",
    )
    l1_C: float = Field(
        default=1.0, gt=0.0,
        description="L1-ball radius for weight_constraint='l1_ball'.",
    )
    mode: Literal["siv", "projected", "ensemble"] = Field(
        default="siv",
        description="Primary estimator variant reported in results.",
    )
    ensemble_alpha: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Override CV-selected ensemble blend weight.",
    )
    inference_method: Literal["asymptotic", "conformal", "none"] = Field(
        default="conformal",
        description="Post-estimation inference procedure.",
    )
    alpha: float = Field(
        default=0.05, gt=0.0, lt=1.0,
        description="Two-sided significance level for the CI.",
    )
    n_permutations: int = Field(
        default=5000, ge=100,
        description="Max permutations for the split-conformal test.",
    )
    seed: int = Field(
        default=1400,
        description="Random seed for conformal permutation sampling.",
    )
    display_graph: bool = Field(
        default=False,
        description="Whether to plot the event-study coefficients.",
    )

    @model_validator(mode="after")
    def _check_columns_and_periods(cls, values: Any) -> Any:
        df = values.df
        if values.instrument not in df.columns:
            raise MlsynthConfigError(
                f"instrument '{values.instrument}' is not in df."
            )
        if values.T0 is None and values.post_col is None:
            raise MlsynthConfigError(
                "Either T0 or post_col must be supplied."
            )
        if values.post_col is not None and values.post_col not in df.columns:
            raise MlsynthConfigError(
                f"post_col '{values.post_col}' is not in df."
            )
        return values


# --- Pydantic Models for Standardized Estimator Results ---

class MlsynthResult(BaseModel):
    """Common base for every ``fit()`` return value in mlsynth.

    mlsynth has exactly two output families, because panel causal inference
    has exactly two modes:

    * :class:`EffectResult` -- an *observational report*: measure a treatment
      effect on already-observed data (ATT, counterfactual, weights,
      inference).
    * :class:`DesignResult` -- a *research design*: choose which units to
      treat before any intervention. A design resolves to the same effect
      report once outcomes exist (see :attr:`DesignResult.report`).

    Every estimator returns one of these two types, so library behaviour is
    simple and predictable: ``isinstance(result, MlsynthResult)`` always
    holds, and the two faces share serialization and validation config.
    """

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"
        json_encoders = {
            np.ndarray: lambda arr: [None if pd.isna(x) else x for x in arr.tolist()]
            if arr is not None
            else None
        }


class EffectsResults(BaseModel):
    """Standardized model for reporting treatment effects."""
    att: Optional[float] = Field(default=None, description="Average Treatment Effect on the Treated.")
    att_percent: Optional[float] = Field(default=None, description="Percentage Average Treatment Effect on the Treated.")
    att_std_err: Optional[float] = Field(default=None, description="Standard error of the ATT estimate.") # Added
    additional_effects: Optional[Dict[str, Any]] = Field(default=None, description="Dictionary for other estimator-specific effects.")

    class Config:
        extra = 'allow' # Allow other effect measures to be added dynamically

class FitDiagnosticsResults(BaseModel):
    """Standardized model for reporting goodness-of-fit diagnostics."""
    rmse_pre: Optional[float] = Field(default=None, description="Root Mean Squared Error in the pre-treatment period.") # Renamed
    r_squared_pre: Optional[float] = Field(default=None, description="R-squared value in the pre-treatment period.") # Renamed
    rmse_post: Optional[float] = Field(default=None, description="Root Mean Squared Error in the post-treatment period (often std of post-treatment gap).") # Added
    additional_metrics: Optional[Dict[str, Any]] = Field(default=None, description="Dictionary for other fit metrics.")

    class Config:
        extra = 'allow'

class TimeSeriesResults(BaseModel):
    """Standardized model for reporting key time series vectors."""
    observed_outcome: Optional[np.ndarray] = Field(default=None, description="Observed outcome vector for the treated unit.")
    counterfactual_outcome: Optional[np.ndarray] = Field(default=None, description="Estimated counterfactual outcome vector.")
    estimated_gap: Optional[np.ndarray] = Field(default=None, description="Estimated treatment effect vector (observed - counterfactual).")
    time_periods: Optional[np.ndarray] = Field(default=None, description="Array of time periods corresponding to the series.") # Retaining np.ndarray, TSSC will convert

    class Config:
        arbitrary_types_allowed = True
        extra = 'allow'

class WeightsResults(BaseModel):
    """Standardized model for reporting estimator weights.

    The single weights container for the whole library. ``weights`` are not
    one thing across synthetic-control methods, so this model exposes the
    real variety as optional faces; an estimator populates whichever apply:

    * ``donor_weights`` -- ``{donor label: weight}`` (most SCMs);
    * ``time_weights``  -- ``{period: weight}`` (e.g. SDID's lambda, DSC
      period weights);
    * ``unit_weights``  -- a weight matrix / array (e.g. MCNNM / ISCM unit
      factors, per-unit weight matrices).
    """
    donor_weights: Optional[Dict[str, float]] = Field(default=None, description="Dictionary mapping donor unit names/IDs to their weights.")
    time_weights: Optional[Dict[Any, float]] = Field(default=None, description="Dictionary mapping time periods to weights (e.g. SDID lambda, DSC period weights).")
    unit_weights: Optional[np.ndarray] = Field(default=None, description="Unit weight matrix/array for estimators whose weights are not a donor mapping (e.g. MCNNM/ISCM).")
    summary_stats: Optional[Dict[str, Any]] = Field(default=None, description="Summary statistics about weights (e.g., cardinality).")
    # For estimators returning multiple sets of weights (e.g. TSSC sub-methods), this might be part of a list or dict structure.
    # donor_names is removed as it's incorporated into donor_weights dict keys

    class Config:
        arbitrary_types_allowed = True
        extra = 'allow'

class InferenceResults(BaseModel):
    """Standardized model for reporting statistical inference results."""
    p_value: Optional[float] = Field(default=None, description="P-value for the estimated ATT.")
    ci_lower: Optional[float] = Field(default=None, description="Lower bound of the confidence interval for ATT.") # Renamed
    ci_upper: Optional[float] = Field(default=None, description="Upper bound of the confidence interval for ATT.") # Renamed
    standard_error: Optional[float] = Field(default=None, description="Standard error of the ATT estimate.")
    confidence_level: Optional[float] = Field(default=None, description="Confidence level used for the CI (e.g., 0.95 for 95%).")
    method: Optional[str] = Field(default=None, description="Method used for inference (e.g., 'placebo', 'conformal', 'asymptotic').")
    details: Optional[Any] = Field(default=None, description="More detailed inference results, e.g., full prediction interval arrays for conformal methods.")

    class Config:
        arbitrary_types_allowed = True
        extra = 'allow'

class MethodDetailsResults(BaseModel):
    """Standardized model for reporting details about the specific estimation method/variant used."""
    method_name: Optional[str] = Field(default=None, description="Name of the specific method or variant used (e.g., 'SIMPLEX' for TSSC, 'FDID' for FDID method).") # Renamed
    is_recommended: Optional[bool] = Field(default=None, description="Flag indicating if this method variant was recommended.") # Added
    parameters_used: Optional[Dict[str, Any]] = Field(default=None, description="Key parameters used for this specific result set.")
    # Could also include version of estimator if relevant.

    class Config:
        extra = 'allow'

class BaseEstimatorResults(MlsynthResult):
    """The observational report: standardized result of an effect estimator.

    Aliased as :class:`EffectResult`. Carries the standardized sub-models
    (:class:`EffectsResults`, :class:`TimeSeriesResults`,
    :class:`WeightsResults`, :class:`InferenceResults`,
    :class:`FitDiagnosticsResults`) plus flat convenience accessors
    (``att``, ``att_ci``, ``counterfactual``, ``gap``, ``donor_weights``,
    ``pre_rmse``) so every effect estimator exposes one predictable surface.
    """
    # Core components, all optional as not all estimators produce all parts.
    effects: Optional[EffectsResults] = None
    fit_diagnostics: Optional[FitDiagnosticsResults] = None
    time_series: Optional[TimeSeriesResults] = None
    weights: Optional[WeightsResults] = None
    inference: Optional[InferenceResults] = None
    method_details: Optional[MethodDetailsResults] = None # Details about the specific method/variant if the main result is for one.

    # For estimators returning multiple sets of results (e.g., TSSC, FDID)
    # This field can hold a list or dictionary of BaseEstimatorResults objects.
    # The exact type (List vs Dict) can be decided per estimator or a Union can be used.
    # For simplicity here, using Dict; can be refined.
    sub_method_results: Optional[Dict[str, Any]] = Field(default=None, description="Results for sub-methods or variants, where each value could be another BaseEstimatorResults or a more specific model.")
    # Alternatively, `fit` could return Union[BaseEstimatorResults, List[BaseEstimatorResults], Dict[str, BaseEstimatorResults]]

    # For outputs that don't fit neatly into the standardized fields.
    additional_outputs: Optional[Dict[str, Any]] = Field(default=None, description="Dictionary for any other outputs specific to the estimator not covered by standard fields.")

    # To store the original, unprocessed results dictionary from the estimator.
    raw_results: Optional[Dict[str, Any]] = Field(default=None, exclude=True, description="Original raw results dictionary from the estimator's core logic.")

    # To capture any errors or warnings encountered during fitting.
    execution_summary: Optional[Dict[str, Any]] = Field(default=None, description="Summary of execution, including any errors or warnings.")


    class Config:
        arbitrary_types_allowed = True
        extra = 'forbid' # Generally forbid extra fields at the top level of BaseEstimatorResults itself.
                        # Sub-models can use 'allow' if they need more flexibility.
        json_encoders = {
            np.ndarray: lambda arr: [None if pd.isna(x) else x for x in arr.tolist()] if arr is not None else None
            # This explicitly converts np.nan (which becomes float('nan') in tolist()) to Python None.
        }

    # ------------------------------------------------------------------
    # Flat convenience accessors -- the minimum read contract every effect
    # estimator satisfies, delegating to the standardized sub-models. Adding
    # them here brings *every* estimator that returns a BaseEstimatorResults
    # into compliance at once; subclasses may override (e.g. dispatchers that
    # delegate to a selected variant).
    # ------------------------------------------------------------------
    @property
    def att(self) -> Optional[float]:
        """Average treatment effect on the treated."""
        return self.effects.att if self.effects else None

    @property
    def att_ci(self) -> Optional[tuple]:
        """``(lower, upper)`` confidence interval for the ATT, if available."""
        inf = self.inference
        if inf is not None and inf.ci_lower is not None and inf.ci_upper is not None:
            return (inf.ci_lower, inf.ci_upper)
        return None

    @property
    def counterfactual(self) -> Optional[np.ndarray]:
        """Estimated counterfactual outcome path."""
        return self.time_series.counterfactual_outcome if self.time_series else None

    @property
    def gap(self) -> Optional[np.ndarray]:
        """Estimated gap (observed minus counterfactual)."""
        return self.time_series.estimated_gap if self.time_series else None

    @property
    def donor_weights(self) -> Optional[Dict[str, float]]:
        """Donor weights ``{label: weight}``."""
        return self.weights.donor_weights if self.weights else None

    @property
    def pre_rmse(self) -> Optional[float]:
        """Pre-treatment root-mean-squared fit error."""
        return self.fit_diagnostics.rmse_pre if self.fit_diagnostics else None


# ``EffectResult`` is the canonical, intention-revealing name for the
# observational report. ``BaseEstimatorResults`` is retained as an alias for
# backward compatibility (it is referenced throughout the library and docs).
EffectResult = BaseEstimatorResults


class DesignResult(MlsynthResult):
    """The research design: an experimental-design estimator's output.

    Experimental/design methods (e.g. MAREX, PANGEO, SYNDES, SPCD) choose
    which units to treat *before* any intervention. A design is not disjoint
    from an effect report -- it *resolves* to one: once outcomes are observed,
    the design is realized as an :class:`EffectResult`, exposed via
    :attr:`report` (today's per-estimator ``post_fit`` / ``summary``).

    This is a skeleton for the two-family contract; design estimators are
    migrated onto it after the observational family is validated.
    """

    report: Optional[BaseEstimatorResults] = Field(
        default=None,
        description="The effect report once the design is realized (the "
        "design's `post_fit`/`summary`). Same type the observational "
        "family returns.",
    )
    assignment: Optional[Any] = Field(
        default=None, description="Treatment assignment chosen by the design."
    )
    selected_units: Optional[Any] = Field(
        default=None, description="Units selected for treatment by the design."
    )
    design_weights: Optional[WeightsResults] = Field(
        default=None, description="Synthetic-control weights implied by the design."
    )
    power: Optional[Any] = Field(
        default=None, description="Power / MDE analysis for the design."
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Free-form design diagnostics."
    )

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"


# ---------------------------------------------------------------------------
# Backward-compatibility shims for relocated per-estimator configs.
#
# Per-estimator config classes are being moved next to their helper packages
# (``mlsynth/utils/<name>_helpers/config.py``) to shrink this module. The
# shared bases (BaseEstimatorConfig, BaseMAREXConfig) and the standardized
# result models stay here. A module-level ``__getattr__`` (PEP 562) lazily
# re-exports the relocated classes so existing imports such as
# ``from mlsynth.config_models import VanillaSCConfig`` keep working. The
# import is lazy to avoid a circular import (the relocated module imports
# BaseEstimatorConfig from here).
# ---------------------------------------------------------------------------
_RELOCATED_CONFIGS = {
    "RMSIConfig": "mlsynth.utils.rmsi_helpers.config",
    "SPOTSYNTHConfig": "mlsynth.utils.spotsynth_helpers.config",
    "SNNConfig": "mlsynth.utils.snn_helpers.config",
    "CTSCConfig": "mlsynth.utils.ctsc_helpers.config",
    "ISCMConfig": "mlsynth.utils.iscm_helpers.config",
    "SPILLSYNTHConfig": "mlsynth.utils.spillsynth_helpers.config",
    "DSCARConfig": "mlsynth.utils.dscar_helpers.config",
    "PROXIMALConfig": "mlsynth.utils.proximal_helpers.config",
    "SCMOConfig": "mlsynth.utils.scmo_helpers.config",
    "SIConfig": "mlsynth.utils.si_helpers.config",
    "FMAConfig": "mlsynth.utils.fma_helpers.config",
    "PDAConfig": "mlsynth.utils.pda_helpers.config",
    "CLUSTERSCConfig": "mlsynth.utils.clustersc_helpers.config",
    "DSCConfig": "mlsynth.utils.dsc_helpers.config",
    "MUSCConfig": "mlsynth.utils.musc_helpers.config",
    "MASCConfig": "mlsynth.utils.masc_helpers.config",
    "SpSyDiDConfig": "mlsynth.utils.spsydid_helpers.config",
    "MCNNMConfig": "mlsynth.utils.mcnnm_helpers.config",
    "MSQRTConfig": "mlsynth.utils.msqrt_helpers.config",
    "SSCConfig": "mlsynth.utils.ssc_helpers.config",
    "VanillaSCConfig": "mlsynth.utils.vanillasc_helpers.config",
    "FDIDConfig": "mlsynth.utils.fdid_helpers.config",
    "TSSCConfig": "mlsynth.utils.tssc_helpers.config",
}


def __getattr__(name: str):  # PEP 562 module-level attribute hook
    module_path = _RELOCATED_CONFIGS.get(name)
    if module_path is not None:
        import importlib

        return getattr(importlib.import_module(module_path), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
