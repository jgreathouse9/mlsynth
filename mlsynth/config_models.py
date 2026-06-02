from typing import List, Optional, Any, Dict, Union, Literal
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, model_validator
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

class TSSCConfig(BaseEstimatorConfig):
    """Configuration for the Two-Step Synthetic Control (TSSC) estimator.

    Implements:

        Li, K. T., & Shankar, V. (2023). "A Two-Step Synthetic Control
        Approach for Estimating Causal Effects of Marketing Events."
        Management Science. https://doi.org/10.1287/mnsc.2023.4878

    Parameters
    ----------
    alpha : float
        Two-sided significance level for the Step-1 restriction tests
        (the SC-pretrends test and the two single-restriction tests).
        Default 0.05.
    subsample_size : int or None
        Subsample size ``m`` for the Step-1 subsampling procedure. When
        ``None`` (default) it is set to ``T_1`` (the bootstrap special
        case the paper's simulations validate). For genuine subsampling,
        the paper's rule of thumb is ``m`` between ``T_1/2`` and ``T_1``
        for moderate ``T_1`` (and smaller for large ``T_1``).
    draws : int
        Number of subsampling replications ``B`` for the Step-1 tests and
        bootstrap replications for the per-variant ATT confidence
        intervals. Default 500.
    ci : float
        Confidence level for the per-variant ATT confidence interval.
        Default 0.95.
    seed : int or None
        Seed for the subsampling RNG (reproducibility). Default None.
    """

    alpha: float = Field(default=0.05, gt=0.0, lt=1.0,
                         description="Significance level for Step-1 restriction tests.")
    subsample_size: Optional[int] = Field(default=None, ge=2,
                         description="Subsample size m; None uses T_1 (bootstrap).")
    draws: int = Field(default=500, ge=1, description="Subsampling/bootstrap replications.")
    ci: float = Field(default=0.95, gt=0.0, lt=1.0, description="ATT confidence level.")
    seed: Optional[int] = Field(default=None, description="RNG seed for subsampling.")

# Placeholder for other estimator configs - to be added sequentially

class FMAConfig(BaseEstimatorConfig):
    """Configuration for the Factor Model Approach (FMA) estimator.

    Implements Li & Sonnier (2023), *"Statistical Inference for the
    Factor Model Approach to Estimate Causal Effects in
    Quasi-Experimental Settings"*, JMR 60(3):449-472. FMA estimates
    the ATT for a single treated unit by extracting principal-
    component factors from the control panel, projecting the treated
    unit's pre-period outcomes onto those factors, and using the
    fitted loading to predict the untreated potential outcome in the
    post-period.

    Parameters
    ----------
    stationarity : {"stationary", "nonstationary"}
        Selects the factor-selection criterion: ``"stationary"`` uses
        the paper's modified Bai-Ng (MBN) criterion (Web Appendix
        D.1); ``"nonstationary"`` uses Bai (2004) IPC1 with a
        log-log adjustment for non-stationary factors. Default:
        ``"nonstationary"`` (the paper's recommendation for general
        applied settings).
    preprocessing : {"demean", "standardize"}
        Preprocessing applied to the control panel before PCA.
    n_factors : int or None
        Override the data-driven factor count. ``None`` triggers the
        criterion in ``stationarity``.
    max_factors : int
        Upper bound passed to the factor-selection routine.
    alpha : float
        Two-sided significance level for CIs.
    inference_methods : list of {"asymptotic", "bootstrap", "placebo"}
        Inference procedures to run. Defaults to ``["asymptotic"]``,
        which gives the paper's Theorem 3.1 normal CI for the ATT.
        Add ``"bootstrap"`` to get per-period ATT_t CIs via the Web
        Appendix F residual bootstrap, and ``"placebo"`` to get the
        Web Appendix G control-as-pseudo-treated band.
    n_bootstrap : int
        Number of bootstrap replicates (Web Appendix F). Ignored when
        ``"bootstrap"`` is not in ``inference_methods``.
    bootstrap_seed : int
        Seed for the bootstrap RNG.
    """

    stationarity: Literal["stationary", "nonstationary"] = Field(
        default="nonstationary",
        description="Stationarity assumption for factor selection.",
    )
    preprocessing: Literal["demean", "standardize"] = Field(
        default="demean",
        description="Preprocessing applied to the control panel before PCA.",
    )
    n_factors: Optional[int] = Field(
        default=None, ge=1,
        description="Optional override of the data-driven factor count.",
    )
    max_factors: int = Field(
        default=10, ge=1,
        description="Upper bound on the factor-selection routine.",
    )
    alpha: float = Field(
        default=0.05, gt=0.0, lt=1.0,
        description="Two-sided significance level for CIs.",
    )
    inference_methods: List[str] = Field(
        default_factory=lambda: ["asymptotic"],
        description="Inference procedures: asymptotic / bootstrap / placebo.",
    )
    n_bootstrap: int = Field(
        default=1000, ge=100,
        description="Number of bootstrap replicates (Web Appendix F).",
    )
    bootstrap_seed: int = Field(
        default=0,
        description="Seed for the bootstrap RNG.",
    )

    @model_validator(mode="after")
    def check_fma_params(cls, values: Any) -> Any:
        allowed = {"asymptotic", "bootstrap", "placebo"}
        for m in values.inference_methods:
            if m not in allowed:
                raise MlsynthConfigError(
                    f"inference_methods entry {m!r} is not one of "
                    f"{sorted(allowed)}."
                )
        return values

class PDAConfig(BaseEstimatorConfig):
    """Configuration for the Panel Data Approach (PDA) estimator."""
    method: str = Field(default="fs", description="Type of PDA to use: 'LASSO', 'l2', or 'fs'.", pattern="^(LASSO|l2|fs)$")
    methods: Optional[List[str]] = Field(default=None, description="Optional list of PDA variants to run; overrides `method` when set.")
    tau: Optional[float] = Field(default=None, description="User-specified treatment effect value (used as tau_l2 for 'l2' method).")
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0, description="Significance level for confidence intervals and ATE inference.")
    fs_intercept: bool = Field(default=False, description="Forward-selection only: include a constant in the donor regression. False (default) matches Shi & Huang's simulation (no intercept, valid size on mean-zero factor data); True matches the released fsPDA R package (intercept, for panels with genuine level differences).")

class FDIDConfig(BaseEstimatorConfig):
    """
    Configuration for the Forward Difference-in-Differences (FDID) estimator.
    Inherits all common configuration parameters from BaseEstimatorConfig.

    Additional Parameters
    ---------------------
    plot_did : bool, default=True
        Whether to display a plot for the standard DID estimator.
        Has no effect on FDID or ADID plots.
    """
    verbose: bool = Field(default=True, description="Whether to save intermediary Forward Selection Results.")



class GSCConfig(BaseEstimatorConfig):
    """Configuration for the Generalized Synthetic Control (GSC) estimator."""
    denoising_method: str = Field(
        default="non-convex",
        description="Method for the denoising algorithm: 'auto', 'convex', or 'non-convex'.",
        pattern="^(auto|convex|non-convex)$"
    )
    target_rank: Optional[int] = Field(
        default=None,
        description="Optional user-specified rank for the denoising algorithm. If None, rank is estimated internally.",
        ge=1
    )
    # Note: The original GSC __init__ docstring mentioned 'save', but it was commented out
    # in the implementation. If 'save' functionality specific to GSC is re-added,
    # it might need to be defined here if different from BaseEstimatorConfig.save.



class CLUSTERSCConfig(BaseEstimatorConfig):
    """Configuration for the Cluster-based Synthetic Control (CLUSTERSC) estimator.

    CLUSTERSC packages two robust synthetic-control families behind a
    single interface:

    * **PCR-RSC** -- SVD-based donor clustering plus Principal Component
      Regression weight estimation (Amjad, Shah, Shen 2018; Agarwal,
      Shah, Shen, Song 2021). Frequentist QP via :mod:`cvxpy` or
      Bayesian Robust Synthetic Control posterior (Amjad, Shah, Shen 2018).
    * **RPCA-SC** -- robust low-rank donor denoising (PCP -- Candes,
      Li, Ma, Wright 2011; or HQF -- Wang, Li, So, Liu 2023) followed
      by simplex SCM weights.

    Parameters
    ----------
    method : {"pcr", "rpca", "both"}
        Estimation family to run. ``"both"`` runs PCR and RPCA in
        parallel; ``primary`` selects which one exposes ``att`` /
        ``counterfactual`` / ``gap`` on the result object.
    primary : {"pcr", "rpca"}
        Which fit drives the result aliases when ``method = "both"``.
    pcr_objective : {"OLS", "SIMPLEX"}
        Inner SCM weight objective for the PCR family.
    clustering : bool
        Whether to apply SVD-based donor clustering before the PCR fit.
    estimator : {"frequentist", "bayesian"}
        Frequentist QP versus Bayesian posterior for the PCR family.
    rpca_method : {"PCP", "HQF"}
        Robust-PCA decomposition for the RPCA family.
    lambda_penalty, p, q : float or None
        Elastic-net-style regularization knobs forwarded to the
        PCR inner solver.
    alpha : float
        Two-sided level for the Bayesian credible interval (PCR
        estimator = "bayesian" only).
    """

    method: Literal["pcr", "rpca", "both"] = Field(
        default="pcr",
        description="Family to run: pcr, rpca, or both.",
    )
    primary: Literal["pcr", "rpca"] = Field(
        default="pcr",
        description="Which fit drives the result aliases when method='both'.",
    )
    pcr_objective: Literal["OLS", "SIMPLEX"] = Field(
        default="OLS",
        description="Inner SCM weight objective for the PCR family.",
    )
    clustering: bool = Field(
        default=True,
        description="Whether to apply SVD-based donor clustering before the PCR fit.",
    )
    estimator: Literal["frequentist", "bayesian"] = Field(
        default="frequentist",
        description="Frequentist QP or Bayesian posterior for the PCR family.",
    )
    rpca_method: Literal["PCP", "HQF"] = Field(
        default="PCP",
        description="Robust-PCA decomposition for the RPCA family.",
    )
    lambda_penalty: Optional[float] = Field(
        default=None, ge=0.0,
        description="Elastic-net regularization strength (lambda).",
    )
    p: Optional[float] = Field(
        default=None, ge=0.0,
        description="Norm parameter p for the elastic-net penalty.",
    )
    q: Optional[float] = Field(
        default=None, ge=0.0,
        description="Secondary norm parameter q (mixed-norm).",
    )
    alpha: float = Field(
        default=0.05, gt=0.0, lt=1.0,
        description="Two-sided credible-interval level (Bayesian PCR only).",
    )
    rank: Optional[int] = Field(
        default=None, ge=1,
        description="Explicit HSVT truncation rank `r` (Rho et al. 2025, Algorithm 2). "
                    "If unset, rank_method controls selection.",
    )
    rank_method: Literal["cumvar", "fixed", "usvt"] = Field(
        default="cumvar",
        description="Rank-selection rule: 'cumvar' (paper default; smallest r "
                    "with cumulative spectral energy >= cumvar_threshold), 'fixed' "
                    "(use explicit `rank`), or 'usvt' (Chatterjee 2015 / Donoho-Gavish).",
    )
    cumvar_threshold: float = Field(
        default=0.95, gt=0.0, le=1.0,
        description="Cumulative-variance target when rank_method='cumvar'. "
                    "Paper Section 6.1 uses 0.95.",
    )
    standardize_for_rank: bool = Field(
        default=True,
        description="If True (default), the cumvar / USVT rank rules operate on the "
                    "column-standardised donor matrix so the spectral comparison "
                    "is not dominated by uncentered level information. The HSVT "
                    "step itself still consumes the raw matrix.",
    )
    project_denoised: bool = Field(
        default=False,
        description="If True, project f_hat through the HSVT-denoised donor matrix "
                    "in both pre and post periods (Rho et al. 2025 Algorithm 4 "
                    "Step 5, paper-strict). Default False uses the raw donor "
                    "outcomes for the counterfactual, matching the practical "
                    "behaviour of Amjad-Shah-Shen (2018) and legacy mlsynth.",
    )
    k_clusters: Optional[int] = Field(
        default=None, ge=1,
        description="Number of clusters for Algorithm 3. If unset, the silhouette "
                    "coefficient picks k in [2, k_max].",
    )
    k_max: int = Field(
        default=8, ge=2,
        description="Upper bound for the silhouette-driven k-search.",
    )
    n_bayes_samples: int = Field(
        default=1000, ge=1,
        description="Posterior sample count for Bayesian PCR credible bands.",
    )
    random_state: int = Field(
        default=0,
        description="Seed forwarded to k-means and the Bayesian sampler.",
    )
    compute_shen_ci: bool = Field(
        default=True,
        description="When True (default) and the frequentist OLS PCR path is used, "
                    "compute Shen-Ding-Sekhon-Yu (2023) per-period and ATT confidence "
                    "intervals under three sources of randomness (HZ / VT / DR).",
    )
    shen_variance: Literal["homoskedastic", "jackknife", "hrk"] = Field(
        default="homoskedastic",
        description="Variance estimator for the Shen et al. (2023) CIs. "
                    "'homoskedastic' (paper eq 19, default), 'jackknife', or 'hrk' "
                    "(Hartley-Rao-Kish; only valid when max(1 - diag(H_perp)) < 1/2 "
                    "for both projections).",
    )
    fpca_cumvar: float = Field(
        default=0.95, gt=0.0, le=1.0,
        description="Cumulative-variance target for the FPCA truncation in "
                    "RPCA-SC Step 1 (Bayani 2021 recommends >= 0.95).",
    )
    pcp_lambda: Optional[float] = Field(
        default=None, ge=0.0,
        description="PCP sparsity penalty (Candes et al. 2011). Defaults to "
                    "1/sqrt(max(N, T)) when unset.",
    )
    pcp_mu: Optional[float] = Field(
        default=None, gt=0.0,
        description="PCP augmented-Lagrangian penalty mu. Defaults to "
                    "N*T / (4 * sum |Y|) (Bayani 2021).",
    )
    pcp_max_iter: int = Field(
        default=1000, ge=1,
        description="Maximum ADMM iterations for the PCP solver.",
    )
    pcp_tol: float = Field(
        default=1e-9, gt=0.0,
        description="Convergence tolerance for the PCP solver (relative to ||Y||_F).",
    )
    hqf_rank: Optional[int] = Field(
        default=None, ge=1,
        description="Explicit factorisation rank for HQF. If unset, picked by "
                    "hqf_cumvar (Bayani 2021 uses 0.999).",
    )
    hqf_cumvar: float = Field(
        default=0.999, gt=0.0, le=1.0,
        description="Cumulative-variance target for HQF rank selection when "
                    "hqf_rank is None.",
    )
    hqf_lambda: Optional[float] = Field(
        default=None, ge=0.0,
        description="Tikhonov factor for HQF (Wang et al. 2023). Defaults to "
                    "1/sqrt(max(m, n)).",
    )
    hqf_ip: float = Field(
        default=1.0, gt=0.0,
        description="HQF noise-scale adaptation factor (Bayani 2021 default 1.0).",
    )
    hqf_max_iter: int = Field(
        default=1000, ge=1,
        description="Maximum iterations for the HQF solver.",
    )
    cv_lambda: bool = Field(
        default=False,
        description="Leave-one-time-period-out CV for PCP's sparsity penalty "
                    "lambda. Sweeps cv_lambda_multipliers x Candes default, picks "
                    "the value with lowest held-out NNLS MSE. On the California "
                    "Prop 99 panel this halves pre-RMSE vs the Candes default.",
    )
    cv_hqf_rank: bool = Field(
        default=False,
        description="Leave-one-time-period-out CV for HQF's factorisation rank. "
                    "Sweeps integer ranks in [1, min(J, T0-1)] and picks the rank "
                    "with lowest held-out NNLS MSE.",
    )
    compute_cft_pi: bool = Field(
        default=False,
        description="Cattaneo-Feng-Titiunik (2021) prediction intervals for the "
                    "RPCA-SC fit. Two-component (in-sample bootstrap + out-of-sample "
                    "Hoeffding bound). Default False because it requires `cft_sims` "
                    "full refits of the pipeline (~0.5s each).",
    )
    cft_sims: int = Field(
        default=200, ge=10,
        description="Number of bootstrap draws for the CFT in-sample component.",
    )
    cft_alpha: float = Field(
        default=0.05, gt=0.0, lt=1.0,
        description="Two-sided level for the CFT prediction intervals.",
    )
    cft_e_method: Literal["gaussian"] = Field(
        default="gaussian",
        description="Out-of-sample method for the CFT component. Currently "
                    "only 'gaussian' (Hoeffding sub-Gaussian bound) is supported.",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalise_legacy_fields(cls, data: Any) -> Any:
        """Accept the legacy field names and uppercase method strings."""
        if not isinstance(data, dict):
            return data
        # objective -> pcr_objective
        if "objective" in data and "pcr_objective" not in data:
            data["pcr_objective"] = data.pop("objective")
        else:
            data.pop("objective", None)
        # cluster -> clustering
        if "cluster" in data and "clustering" not in data:
            data["clustering"] = data.pop("cluster")
        else:
            data.pop("cluster", None)
        # Frequentist -> estimator
        if "Frequentist" in data and "estimator" not in data:
            data["estimator"] = (
                "frequentist" if data.pop("Frequentist") else "bayesian"
            )
        else:
            data.pop("Frequentist", None)
        # ROB / Robust -> rpca_method
        for k in ("ROB", "Robust"):
            if k in data and "rpca_method" not in data:
                data["rpca_method"] = data.pop(k)
            else:
                data.pop(k, None)
        # Uppercase method tags
        if "method" in data and isinstance(data["method"], str):
            mapping = {"PCR": "pcr", "RPCA": "rpca", "BOTH": "both"}
            data["method"] = mapping.get(data["method"], data["method"])
        return data



class DSCConfig(BaseEstimatorConfig):
    """Configuration for the Distributional Synthetic Control (DSC) estimator.

    DSC (Gunsilius 2023; asymptotic theory: Zhang, Zhang & Zhang 2026)
    fits simplex-constrained weights on the *quantile functions* of
    donor outcomes to reconstruct the treated unit's counterfactual
    *outcome distribution*. Unlike the other mlsynth estimators, DSC
    expects micro-level panel data: each ``(unit, time)`` cell carries
    multiple individual observations, supplied as one row per
    individual in the input DataFrame.

    Parameters
    ----------
    M : int, optional
        Number of quantile-grid points used to approximate the
        2-Wasserstein loss. If ``None``, defaults to
        ``max(200, min_cell_size)`` (Zhang et al. 2026 suggest
        :math:`M = C n` for a constant ``C >= 1``).
    grid_method : {"halton", "sobol", "uniform"}
        Sampling rule for the quantile grid. ``"halton"`` (default)
        and ``"sobol"`` are quasi-Monte Carlo with Koksma-Hlawka error
        :math:`O(\\log M / M)`; ``"uniform"`` is i.i.d. with
        :math:`O(M^{-1/2})` error.
    lambda_method : {"uniform", "recency"}
        Default rule for the pre-period aggregation weights
        :math:`\\lambda_t`. Ignored when ``lambda_weights`` is set.
    lambda_decay : float
        Geometric decay factor for ``lambda_method="recency"``.
        Default ``0.9``.
    lambda_weights : sequence of float, optional
        Caller-supplied length-``T0`` aggregation weights (must be
        non-negative and sum to 1). Useful for Arkhangelsky et al.
        (2021) SDiD-style time weights computed externally.
    qte_quantiles : sequence of float, optional
        Quantile grid in ``(0, 1)`` at which to report the QTE. If
        ``None``, an evenly spaced grid of ``n_qte_points`` quantiles
        is used.
    n_qte_points : int
        Length of the default QTE grid when ``qte_quantiles`` is None.
        Default 99 (every percentile from 0.01 to 0.99).
    random_state : int
        Seed forwarded to the QMC quantile-grid sampler.
    """

    M: Optional[int] = Field(
        default=None, ge=50,
        description="Number of quantile-grid points used to approximate the "
                    "2-Wasserstein loss. Defaults to max(200, min cell size).",
    )
    grid_method: Literal["halton", "sobol", "uniform"] = Field(
        default="halton",
        description="Quantile-grid sampling rule (QMC by default).",
    )
    lambda_method: Literal["uniform", "recency"] = Field(
        default="uniform",
        description="Default rule for pre-period aggregation weights lambda_t.",
    )
    lambda_decay: float = Field(
        default=0.9, gt=0.0, le=1.0,
        description="Geometric decay for lambda_method='recency'.",
    )
    lambda_weights: Optional[List[float]] = Field(
        default=None,
        description="Caller-supplied length-T0 aggregation weights "
                    "(non-negative, sum to 1).",
    )
    qte_quantiles: Optional[List[float]] = Field(
        default=None,
        description="Explicit quantile grid in (0, 1) for QTE reporting.",
    )
    n_qte_points: int = Field(
        default=99, ge=1,
        description="Length of the default QTE grid when qte_quantiles is None.",
    )
    random_state: int = Field(
        default=0,
        description="Seed forwarded to the QMC sampler.",
    )
    compute_inference: bool = Field(
        default=False,
        description="Run the Gunsilius (2023) placebo permutation test "
                    "(Algorithm 1): refit DSC with every donor treated as a "
                    "placebo and rank the real unit's post-period Wasserstein "
                    "distance. Costs J extra pre-period refits.",
    )
    inference_grid_points: int = Field(
        default=200, ge=2,
        description="Number of quantiles used to evaluate the squared "
                    "2-Wasserstein distances in the placebo permutation test.",
    )






class MUSCConfig(BaseEstimatorConfig):
    """Configuration for the Modified Unbiased Synthetic Control (MUSC) estimator.

    MUSC is the Bottmer, Imbens, Spiess & Warnick (2024 JBES)
    modification of the synthetic control method: a single column-
    sums-to-zero linear restriction is added to the standard SC
    quadratic programme so that the resulting estimator is exactly
    unbiased under random assignment of which unit is treated (the
    paper's Lemma 1).
    """

    alpha: float = Field(
        default=0.05, gt=0.0, lt=1.0,
        description=(
            "Two-sided significance level for both the Normal-approximation "
            "and the randomization-based CI surfaced on the MUSC inference "
            "object. Defaults to 0.05 (95% CI)."
        ),
    )
    run_inference: bool = Field(
        default=True,
        description=(
            "Whether to compute the Proposition 1 unbiased variance "
            "estimator (eq. 3.3) and the Section 3.5 randomization-based "
            "confidence interval. The randomization CI requires one "
            "leave-one-out QP refit per donor unit, so disabling it is "
            "useful for exploratory fits on large panels."
        ),
    )
    solver: Optional[str] = Field(
        default=None,
        description=(
            "cvxpy solver name forwarded to the MUSC quadratic programme. "
            "Defaults to None, which routes to cvxpy's default for the "
            "problem class (CLARABEL on cvxpy >= 1.4)."
        ),
    )



class TASCMConfig(BaseEstimatorConfig):
    """Configuration for Temporal-Aggregation SCM (Sun et al. 2024).

    Sun, L., Ben-Michael, E., & Feller, A. (2024). *Temporal Aggregation
    for the Synthetic Control Method.* AEA Papers and Proceedings.
    """

    aggregation_block_size: int = Field(
        ..., ge=2,
        description=(
            "Number of high-frequency periods per aggregated period "
            "(``K`` in the paper). For weekly-into-monthly aggregation "
            "use ``K = 4`` or ``K = 5``; for monthly-into-yearly use "
            "``K = 12``."
        ),
    )
    nu_hat: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description=(
            "Convex-combination weight on the aggregated objective in "
            "``[0, 1]``. ``nu_hat = 0`` recovers plain SCM on the "
            "high-frequency data; ``nu_hat = 1`` recovers plain SCM "
            "on the block-aggregated data; ``nu_hat = 0.5`` (default) "
            "matches the paper's recommended baseline."
        ),
    )
    nu_grid: Optional[List[float]] = Field(
        default=None,
        description=(
            "Optional explicit grid of ``nu`` values for the imbalance-"
            "frontier sweep. Each entry must lie in ``[0, 1]``. Defaults "
            "to ``numpy.linspace(0, 1, 11)``."
        ),
    )
    solver: Optional[str] = Field(
        default=None,
        description=(
            "cvxpy solver name forwarded to the combined-objective QP. "
            "Defaults to CLARABEL when unset."
        ),
    )


class MASCConfig(BaseEstimatorConfig):
    """Configuration for the MASC estimator (Kellogg et al. 2021).

    Kellogg, Mogstad, Pouliot & Torgovitsky (2021). *Combining
    Matching and Synthetic Control to Trade Off Biases from
    Extrapolation and Interpolation.* JASA 116(536), 1804-1816.
    The estimator forms a convex combination ``phi * matching +
    (1 - phi) * SC`` with the number of neighbours ``m`` and the
    weight ``phi`` jointly chosen by rolling-origin cross-validation.
    """

    covariates: Optional[List[str]] = Field(
        default=None,
        description=(
            "Optional predictor columns added to the SC objective in "
            "place of (or alongside) pre-period outcomes. Each covariate "
            "is row-standardised across all units before the QP -- the "
            "FSCM/MAREX-style ``V = diag(1 / var)`` preconditioning that "
            "matches Abadie's default heuristic V. Per-fold aggregation "
            "is over the fold-specific pre-window unless "
            "``covariate_windows`` overrides it."
        ),
    )
    covariate_windows: Optional[dict] = Field(
        default=None,
        description=(
            "Optional inclusive ``(start, end)`` aggregation window per "
            "covariate label. Covariates without a window are averaged "
            "over each fold's full pre-period."
        ),
    )
    m_grid: Optional[List[int]] = Field(
        default=None,
        description=(
            "Candidate nearest-neighbour counts for cross-validation. "
            "Defaults to ``1..J`` (all values), mirroring the R "
            "reference's default."
        ),
    )
    min_preperiods: Optional[int] = Field(
        default=None, ge=2,
        description=(
            "Smallest fold length used in cross-validation. Folds run "
            "from ``min_preperiods`` to ``treatment_period - 2``. "
            "Defaults to ``ceil(treatment_period / 2)`` per the R "
            "reference. Mutually exclusive with ``set_f``."
        ),
    )
    set_f: Optional[List[int]] = Field(
        default=None,
        description=(
            "Explicit list of fold lengths to use (each integer is the "
            "last pre-treatment period included in that fold's "
            "training window). Mutually exclusive with "
            "``min_preperiods``."
        ),
    )
    fold_weights: Optional[List[float]] = Field(
        default=None,
        description=(
            "Optional positive relative weights for each fold (length "
            "matches the fold count). Normalised to sum to 1. Defaults "
            "to equal weights."
        ),
    )
    forecast_minlength: int = Field(
        default=1, ge=1,
        description=(
            "First post-fold period used for forecast error "
            "evaluation. Fold ``f`` forecasts period ``f + "
            "forecast_minlength``."
        ),
    )
    forecast_maxlength: int = Field(
        default=1, ge=1,
        description=(
            "Last post-fold period used for forecast error "
            "evaluation. Forecast window is capped at "
            "``treatment_period - 1``."
        ),
    )
    solver: Optional[str] = Field(
        default=None,
        description=(
            "cvxpy solver name forwarded to the SC simplex QP. "
            "Defaults to CLARABEL when unset."
        ),
    )



class GMMSCEConfig(BaseEstimatorConfig):
    """Configuration for the GMM Synthetic Control Estimator (Fry 2024 JoE).

    Fry, J. (2024). *A method of moments approach to asymptotically
    unbiased Synthetic Controls.* J. of Econometrics 244, 105846.
    The estimator solves a GMM problem with IV moment conditions from
    instrument units, on row-normalised pre-period outcomes. Under a
    linear factor model the SC weights converge to the treated unit's
    factor loadings so the ATT is asymptotically unbiased as
    ``T0 -> infinity``.
    """

    donors: Optional[List[Any]] = Field(
        default=None,
        description=(
            "Unit labels for the donor (control) pool. If omitted and "
            "``auto_select=False``, every never-treated unit serves "
            "as both donor and instrument (the R reference's initial "
            "step, ``YK = YN0``)."
        ),
    )
    instruments: Optional[List[Any]] = Field(
        default=None,
        description=(
            "Unit labels for the instrument pool. Must be never-treated. "
            "If omitted defaults to the donor pool (mirrors the R "
            "reference's initial fit)."
        ),
    )
    fixed_instruments: Optional[List[Any]] = Field(
        default=None,
        description=(
            "Units guaranteed to be used as instruments under "
            "``auto_select=True`` (the R reference's ``YN1`` argument). "
            "For German reunification, Fry uses Penn World Table OECD "
            "countries outside Abadie's candidate pool."
        ),
    )
    auto_select: bool = Field(
        default=False,
        description=(
            "If True, run the Sargan-Hansen J-test-driven model "
            "selection (R reference's ``RandomModelSelection``) to "
            "partition the never-treated units into donors and "
            "instruments. Otherwise use the user-supplied "
            "``donors`` / ``instruments``."
        ),
    )
    meanfit: bool = Field(
        default=True,
        description=(
            "Append a constant column to the instrument matrix (R "
            "reference's ``meanfit=TRUE``). Adds the constant-instrument "
            "moment ``mean_pre(Y0 - YJ w) = 0`` to the GMM system."
        ),
    )
    solver: Optional[str] = Field(
        default=None,
        description=(
            "cvxpy solver name forwarded to the step-1 QP. Defaults to "
            "CLARABEL when unset."
        ),
    )



class OSCEConfig(BaseEstimatorConfig):
    """Configuration for the Orthogonalized SCE (Fry 2026).

    Fry, J. (2026). *Orthogonalized Synthetic Controls.* SSRN.
    The IV-SCE specialization of Fry's general orthogonalization
    framework: simplex donor weights ``delta`` are fit alongside
    orthogonalization coefficients ``eta`` via a penalized GMM step,
    then the ATT is recovered in closed form from the orthogonalized
    moment condition. Asymptotic normality of the ATT holds even when
    ``delta`` is partially identified or on the simplex boundary.
    """

    instruments: list = Field(
        ...,
        description=(
            "Unit labels (matching ``df[unitid]``) to use as instruments "
            "Z_t. Must be never-treated. The donor pool delta is fit on "
            "every other non-treated unit. For Fry's Sweden carbon-tax "
            "application: Finland, Germany, Ireland, Italy, Netherlands, "
            "Norway, United Kingdom."
        ),
    )
    lambda_delta: Optional[float] = Field(
        default=None, ge=0.0,
        description=(
            "Slackness on the pre-period IV moment constraints. When "
            "None (default) the value is auto-tuned by Fry's "
            "``EstimateLambdaDelta`` LP (inflated by ``log(min(T0,T1)) "
            "* log(J) / log(K)``)."
        ),
    )
    lambda_eta: Optional[float] = Field(
        default=None, ge=0.0,
        description=(
            "Slackness on the orthogonality constraints. When None "
            "(default) the value is auto-tuned by Fry's "
            "``EstimateLambdaEta`` LP (inflated by ``log(min(T0,T1))``)."
        ),
    )
    scaled: bool = Field(
        default=True,
        description=(
            "Apply per-period standard-deviation rescaling to "
            "``Y_treated``, ``Y_donors``, and ``Z`` before step 1 "
            "(matches the R reference's ``Scaled = TRUE`` path). "
            "Required for the empirical Sweden replication."
        ),
    )
    solver: Optional[str] = Field(
        default=None,
        description=(
            "cvxpy solver name forwarded to the step-1 QPs. Defaults "
            "to CLARABEL when unset."
        ),
    )



class TROPConfig(BaseEstimatorConfig):
    """Configuration for the Triply Robust Panel (TROP) estimator.

    Athey, Imbens, Qu & Viviano (2026, *Journal of Applied
    Econometrics*). Combines exponential time and unit weights with a
    nuclear-norm-penalised low-rank factor adjustment, fit on control
    observations via weighted least squares; ATT is the average
    treated-cell residual. First mlsynth release implements the
    ``global`` (simultaneous-adoption) path only.
    """

    lambda_time_grid: Optional[List[float]] = Field(
        default=None,
        description=(
            "LOOCV grid for the time-decay rate. 0.0 yields uniform "
            "time weights. Inf is rejected. Default: "
            "[0, 0.1, 0.5, 1, 2, 5]."
        ),
    )
    lambda_unit_grid: Optional[List[float]] = Field(
        default=None,
        description=(
            "LOOCV grid for the unit-decay rate. 0.0 yields uniform "
            "unit weights. Inf is rejected. Default: "
            "[0, 0.1, 0.5, 1, 2, 5]."
        ),
    )
    lambda_nn_grid: Optional[List[float]] = Field(
        default=None,
        description=(
            "LOOCV grid for the nuclear-norm penalty on the low-rank "
            "component. 0 means no penalty (rank free); values >= 1e10 "
            "disable the factor model (L=0). Default: "
            "[0, 0.01, 0.1, 1, 10]."
        ),
    )
    max_iter: int = Field(
        default=100, ge=1,
        description="Outer alternating-minimisation iterations.",
    )
    tol: float = Field(
        default=1e-6, gt=0.0,
        description="Convergence tolerance on the low-rank component.",
    )
    verbose: bool = Field(
        default=False,
        description="Print LOOCV scores for each grid triplet.",
    )


class SpSyDiDConfig(BaseEstimatorConfig):
    """Configuration for the Spatial Synthetic Difference-in-Differences estimator.

    Serenini & Masek (2024). *"Spatial Synthetic
    Difference-in-Differences,"* SSRN 4736857. Extends SDID
    (Arkhangelsky et al. 2021) with a spatial spillover term so the
    estimator separates the direct ATT from the indirect (spillover)
    effect on units exposed via the spatial weight matrix :math:`W`.

    Parameters
    ----------
    spatial_matrix : np.ndarray
        Square :math:`N \\times N` spatial weight matrix. Rows /
        columns must align with ``unit_order`` (or
        ``sorted(df[unitid].unique())`` if ``unit_order`` is None).
        Use the helpers in
        :mod:`mlsynth.utils.spsydid_helpers.spatial` to build ``W``
        from coordinates (k-NN, inverse distance) or from an adjacency
        list (queen / rook contiguity).
    unit_order : list, optional
        Canonical ordering of unit ids matching the rows / columns of
        ``spatial_matrix``. If ``None`` (default), units are ordered
        by ``sorted(df[unitid].unique())``.
    row_standardize_spatial : bool
        Row-standardise ``W`` internally before computing exposure.
        Default True. Skip when the caller has already standardised.
    """

    spatial_matrix: Any = Field(
        ...,
        description="Square (N, N) spatial weight matrix as a numpy array.",
    )
    unit_order: Optional[List[Any]] = Field(
        default=None,
        description="Canonical ordering of unit ids matching the rows / columns "
                    "of spatial_matrix. Defaults to sorted(unique unit ids).",
    )
    row_standardize_spatial: bool = Field(
        default=True,
        description="Row-standardise the spatial matrix internally so each row "
                    "of W sums to 1.",
    )


class CSCConfig(BaseEstimatorConfig):
    """Configuration for the Correlated Synthetic Controls (CSC) estimator.

    Moev, T. (2025). *"Correlated Synthetic Controls,"* arXiv:2507.08918.
    Builds per-treated-unit synthetic controls whose donor weights vary
    with the treated unit's categorical covariates, for the
    many-treated-units / short-pre-period regime.

    Parameters
    ----------
    covariates : list of str
        Categorical covariate columns on the treated units
        (time-invariant). These drive the weight heterogeneity. CSC
        supports categorical covariates only -- bin any continuous
        covariate (e.g. into quantile brackets) before passing it.
    cluster_bootstrap : bool
        If True, run a stratified cluster bootstrap over the treated
        units to obtain CIs for the ATT, per-segment averages, and
        per-unit effects. Default False (point estimates only).
    n_boot : int
        Number of bootstrap resamples when ``cluster_bootstrap=True``.
    alpha : float
        Two-sided level for the bootstrap CIs (e.g. 0.05 -> 95% CI).
    random_state : int
        Seed for the bootstrap RNG.
    """

    covariates: List[str] = Field(
        ...,
        description="Categorical covariate columns driving the weight "
                    "heterogeneity (time-invariant, categorical only).",
    )
    cluster_bootstrap: bool = Field(
        default=False,
        description="Run a stratified cluster bootstrap over treated units "
                    "for confidence intervals.",
    )
    n_boot: int = Field(
        default=500, ge=10,
        description="Number of bootstrap resamples when cluster_bootstrap=True.",
    )
    alpha: float = Field(
        default=0.05, gt=0.0, lt=1.0,
        description="Two-sided level for the bootstrap confidence intervals.",
    )
    random_state: int = Field(
        default=0,
        description="Seed for the bootstrap RNG.",
    )


class MCNNMConfig(BaseEstimatorConfig):
    """Configuration for the MC-NNM estimator.

    Athey, Bayati, Doudchenko, Imbens & Khosravi (2021), *"Matrix
    Completion Methods for Causal Panel Data Models"* (JASA). Imputes the
    treated cells of the outcome matrix via nuclear-norm-regularised
    low-rank matrix completion with unregularised two-way fixed effects
    (SOFT-IMPUTE, threshold chosen by cross-validation). Inherits the
    standard ``df`` / ``outcome`` / ``treat`` / ``unitid`` / ``time``
    interface.

    Parameters
    ----------
    estimate_unit_fe : bool
        Estimate (unregularised) unit fixed effects. Default True.
    estimate_time_fe : bool
        Estimate (unregularised) time fixed effects. Default True.
    n_lambda : int
        Number of candidate singular-value thresholds in the CV grid.
    n_folds : int
        Cross-validation folds over the observed cells.
    inference : bool
        Run a leave-one-control jackknife for the ATT SE / CI. Default
        False (it refits the model once per control unit).
    alpha : float
        Two-sided level for the jackknife confidence interval.
    random_state : int
        Seed for the CV fold assignment.
    """

    estimate_unit_fe: bool = Field(
        default=True, description="Estimate unregularised unit fixed effects.")
    estimate_time_fe: bool = Field(
        default=True, description="Estimate unregularised time fixed effects.")
    n_lambda: int = Field(
        default=40, ge=2,
        description="Number of candidate thresholds in the CV grid.")
    n_folds: int = Field(
        default=5, ge=2, description="Cross-validation folds over observed cells.")
    inference: bool = Field(
        default=False,
        description="Run a leave-one-control jackknife for the ATT SE/CI.")
    alpha: float = Field(
        default=0.05, gt=0.0, lt=1.0,
        description="Two-sided level for the jackknife confidence interval.")
    random_state: int = Field(
        default=0, description="Seed for the CV fold assignment.")


class MSQRTConfig(BaseEstimatorConfig):
    """Configuration for the MSQRT estimator.

    Shen, Song & Abadie (2025), *"Efficiently Learning Synthetic Control
    Models for High-dimensional Disaggregated Data"* (arXiv:2510.22828).
    Stacks all treated units into one matrix regression ``Y = X Theta + E``
    and fits the donor-weight matrix by Multivariate Square-root Lasso
    (nuclear-norm loss + element-wise L1), with the L1 penalty chosen by
    rolling-origin cross-validation on the pre-period. Assumes a block design
    (all treated units adopt at the same period) and inherits the standard
    ``df`` / ``outcome`` / ``treat`` / ``unitid`` / ``time`` interface.

    Parameters
    ----------
    lambda_ : float, optional
        Fixed L1 penalty. If None (default), selected by cross-validation.
    n_lambda : int
        Number of log-spaced candidate penalties in the CV grid.
    cv_initial_train, cv_val_window, cv_step, cv_folds : int, optional
        Rolling-origin CV schedule overrides. Adaptive defaults
        (scaled to the pre-period length) are used when left None.
    inference : bool
        Attach CFPT/scpi prediction intervals (Cattaneo, Feng, Palomba &
        Titiunik 2025) for all four predictands plus simultaneous bands. For
        MSQRT only the out-of-sample error is modelled. Default False.
    alpha : float
        Total miscoverage level for the intervals (default 0.1 -> 90%).
    time_dependence : {"iid", "general"}
        Time-averaging assumption for the time-averaged predictands (TAUS,
        TAUA). ``"iid"`` (default) shrinks the band by ``sqrt(L)``;
        ``"general"`` makes no serial-dependence assumption.
    """

    lambda_: Optional[float] = Field(
        default=None, gt=0.0,
        description="Fixed L1 penalty; None selects it by cross-validation.")
    n_lambda: int = Field(
        default=15, ge=2,
        description="Number of log-spaced candidate penalties in the CV grid.")
    cv_initial_train: Optional[int] = Field(
        default=None, ge=2,
        description="Initial training window for rolling-origin CV.")
    cv_val_window: Optional[int] = Field(
        default=None, ge=1,
        description="Validation window length for rolling-origin CV.")
    cv_step: Optional[int] = Field(
        default=None, ge=1, description="Step between successive CV folds.")
    cv_folds: Optional[int] = Field(
        default=None, ge=1, description="Maximum number of CV folds.")
    inference: bool = Field(
        default=False,
        description="Attach CFPT/scpi prediction intervals to the predictands.")
    alpha: float = Field(
        default=0.1, gt=0.0, lt=1.0,
        description="Total miscoverage level for the prediction intervals.")
    time_dependence: Literal["iid", "general"] = Field(
        default="iid",
        description="Time-averaging assumption for time-averaged predictands.")


class SSCConfig(BaseEstimatorConfig):
    """Configuration for the SSC (Staggered Synthetic Control) estimator.

    Cao, Lu & Wu (2026), *"Synthetic Control Inference for Staggered
    Adoption"* (The Econometrics Journal). Models each unit's untreated
    outcome as an intercept plus a simplex synthetic control on all other
    units (not-yet-treated units are valid donors), jointly estimates every
    unit x time effect by GLS, and reports event-time / overall ATT with
    Andrews (2003) end-of-sample stability inference. Targets staggered
    adoption with a long pre-period (large ``T``, moderate ``N``, small
    ``S``). Inherits the standard ``df`` / ``outcome`` / ``treat`` /
    ``unitid`` / ``time`` interface.

    Parameters
    ----------
    inference : bool
        Attach Andrews end-of-sample bands and p-values to the event-time and
        overall ATT. Default True.
    alpha : float
        Two-sided level for the bands (default 0.1 -> 90% band).
    """

    inference: bool = Field(
        default=True,
        description="Attach Andrews end-of-sample bands/p-values to the ATTs.")
    alpha: float = Field(
        default=0.1, gt=0.0, lt=1.0,
        description="Two-sided level for the end-of-sample bands.")


class RMSIConfig(BaseEstimatorConfig):
    """Configuration for the RMSI estimator.

    Agarwal, Choi & Yuan (2026), *"Robust Matrix Estimation with Side
    Information"* (arXiv:2603.24833). Imputes the treated counterfactual of a
    block-adoption causal panel by a four-component sieve + nuclear-norm matrix
    estimator that exploits unit-level (row) and time-level (column) covariates.
    Inherits the standard ``df`` / ``outcome`` / ``treat`` / ``unitid`` /
    ``time`` interface.

    Parameters
    ----------
    unit_covariates : list of str
        Columns that vary across units and are (approximately) constant within a
        unit; averaged per unit to form the row feature matrix ``X``. May be
        empty (then the row projection captures only the mean).
    time_covariates : list of str
        Columns that vary across periods and are constant across units within a
        period; averaged per period to form the column feature matrix ``Z``.
        May be empty.
    sieve_order : int
        Polynomial sieve order ``J`` (default 2, matching the paper).
    rank : int, optional
        Factor rank for the block recombination; chosen by a relative-magnitude
        singular-value threshold when None.
    """

    unit_covariates: List[str] = Field(
        default_factory=list,
        description="Unit-level covariate columns (row side information).")
    time_covariates: List[str] = Field(
        default_factory=list,
        description="Time-level covariate columns (column side information).")
    sieve_order: int = Field(
        default=2, ge=1, description="Polynomial sieve order J.")
    rank: Optional[int] = Field(
        default=None, ge=1,
        description="Factor rank; relative-threshold estimate when None.")


class SPOTSYNTHConfig(BaseEstimatorConfig):
    """Configuration for the SPOTSYNTH estimator.

    O'Riordan & Gilligan-Lee (2025), *"Spillover detection for donor selection in
    synthetic control models"* (Journal of Causal Inference 13:20240036,
    doi:10.1515/jci-2024-0036). Screens each candidate donor for spillover
    contamination via a pre-intervention forecast test (Algorithm 1), excludes
    the contaminated donors, and fits a simplex synthetic control on the valid
    set. Inherits the standard ``df`` / ``outcome`` / ``treat`` / ``unitid`` /
    ``time`` interface and expects a single treated unit.

    Parameters
    ----------
    selection : {"S1", "S2", "all"}
        Donor-selection rule. ``S1`` keeps the ``n_donors`` donors with the
        smallest forecast error (the analyst fixes how many to keep); ``S2``
        keeps donors whose realised post-intervention value falls inside the
        ``ppi`` posterior predictive interval (controls the false-positive rate);
        ``all`` keeps every donor (the unscreened baseline).
    forecast : {"lag", "loo"}
        Forecast anchor for the screen. ``lag`` (paper Algorithm 1) forecasts the
        first post-intervention point from the last clean pre-intervention
        cross-section -- robust even when most donors are invalid. ``loo`` is a
        leave-one-out variant that scores the mean post-period deviation from the
        other donors' common-factor forecast -- suited to single-contaminant
        panels whose treatment effect builds slowly.
    n_donors : int, optional
        Number of donors to keep under ``S1`` (default: half the pool).
    ppi : float
        Posterior-predictive-interval level for ``S2`` (default 0.8).
    n_factors : int
        Number of donor factors used to regularise the forecast (default 5).
    time_average : int, optional
        Bucket width for time-averaging the data before screening (``lag`` only;
        reduces false negatives with very noisy donors).
    inference : {"bayes", "frequentist"}
        Synthetic-control weight model. ``bayes`` (default) is the authors'
        Bayesian simplex SC -- weights with a ``Dirichlet(dirichlet_alpha)``
        prior, a half-normal prior on the residual sd, pre-period standardisation
        of target and donors, and 95% posterior-predictive credible intervals.
        ``frequentist`` is a fast simplex least-squares point estimate (no
        intervals), useful for large simulations.
    dirichlet_alpha : float
        Dirichlet concentration on the donor weights (paper uses 0.4; ``< 1``
        favours sparse weights).
    ci_level : float
        Credible-interval level for the Bayesian fit (paper uses 0.95).
    n_samples, n_warmup : int
        Posterior draws and warm-up iterations for the Bayesian sampler.
    debias : bool
        If True, also compute the proximal (two-stage / GMM) debiased ATT
        (equation 5), using the screen-excluded donors as proximal controls to
        correct errors-in-variables bias when the kept donors are noisy proxies.
    seed : int
        RNG seed for the Bayesian sampler.
    """

    selection: Literal["S1", "S2", "all"] = Field(
        default="S1", description="Donor-selection rule (S1, S2, or all).")
    forecast: Literal["lag", "loo"] = Field(
        default="lag", description="Forecast anchor (lag = Algorithm 1, or loo).")
    n_donors: Optional[int] = Field(
        default=None, ge=1, description="Number of donors to keep under S1.")
    ppi: float = Field(
        default=0.8, gt=0.0, lt=1.0,
        description="Posterior predictive interval level for S2.")
    n_factors: int = Field(
        default=5, ge=1, description="Number of donor factors in the forecast.")
    time_average: Optional[int] = Field(
        default=None, ge=1, description="Bucket width for time-averaging (lag).")
    inference: Literal["bayes", "frequentist"] = Field(
        default="bayes", description="SC weight model (Bayesian Dirichlet or simplex LS).")
    dirichlet_alpha: float = Field(
        default=0.4, gt=0.0, description="Dirichlet concentration on donor weights.")
    ci_level: float = Field(
        default=0.95, gt=0.0, lt=1.0, description="Credible-interval level.")
    n_samples: int = Field(
        default=4000, ge=200, description="Posterior draws for the Bayesian fit.")
    n_warmup: int = Field(
        default=2000, ge=100, description="Warm-up iterations for the Bayesian fit.")
    debias: bool = Field(
        default=False, description="Compute the proximal (GMM) debiased ATT.")
    seed: int = Field(
        default=0, description="RNG seed for the Bayesian sampler.")


class SNNConfig(BaseEstimatorConfig):
    """Configuration for the Synthetic Nearest Neighbors (SNN) estimator.

    Agarwal, Dahleh, Shah & Shen (2021), *"Causal Matrix Completion"*
    (arXiv:2109.15154). Imputes treated units' untreated potential
    outcomes by MNAR matrix completion (anchor submatrix + principal
    component regression), generalising the Synthetic Interventions /
    synthetic-control approach. Inherits the standard ``df`` / ``outcome``
    / ``treat`` / ``unitid`` / ``time`` interface.

    Parameters
    ----------
    n_neighbors : int
        Number of synthetic neighbours (anchor-row groups) to average.
    max_rank : int, optional
        Fixed PCR truncation rank; overrides the spectral/universal rule.
    spectral_energy : float
        Singular-value energy threshold for spectral rank selection
        (used when ``max_rank`` is None and ``universal_rank`` is False).
    universal_rank : bool
        Use the Donoho-Gavish (2014) universal hard-threshold rank.
        Default True -- well-calibrated for small low-rank panels (e.g.
        Prop 99); set False to use the spectral-energy threshold.
    clip : bool
        Clip imputations to the observed value range.
    inference : bool
        Run a leave-one-control jackknife for the ATT SE / CI.
    alpha : float
        Two-sided level for the jackknife confidence interval.
    random_state : int
        Seed for anchor-row splitting.
    """

    n_neighbors: int = Field(
        default=1, ge=1,
        description="Number of synthetic neighbours (anchor-row groups).",
    )
    max_rank: Optional[int] = Field(
        default=None, ge=1,
        description="Fixed PCR truncation rank (overrides spectral rule).",
    )
    spectral_energy: float = Field(
        default=0.95, gt=0.0, le=1.0,
        description="Singular-value energy threshold for rank selection.",
    )
    universal_rank: bool = Field(
        default=True,
        description="Use the Donoho-Gavish universal hard-threshold rank "
                    "(default; well-calibrated for small low-rank panels). "
                    "Set False to use the spectral-energy threshold instead.",
    )
    clip: bool = Field(
        default=True,
        description="Clip imputations to the observed value range.",
    )
    inference: bool = Field(
        default=False,
        description="Run a leave-one-control jackknife for the ATT SE/CI.",
    )
    alpha: float = Field(
        default=0.05, gt=0.0, lt=1.0,
        description="Two-sided level for the jackknife confidence interval.",
    )
    random_state: int = Field(
        default=0,
        description="Seed for anchor-row splitting.",
    )


class CTSCConfig(BaseEstimatorConfig):
    """Configuration for the Continuous-Treatment Synthetic Control (CTSC).

    Powell, D. (2022). *"Synthetic Control Estimation Beyond Comparative
    Case Studies,"* Journal of Business & Economic Statistics. Generalises
    synthetic control to continuous / multi-valued treatments with no clean
    treated/never-treated split; jointly estimates unit-specific treatment
    slopes and synthetic controls for all units. (The paper calls it "GSC";
    mlsynth uses CTSC to avoid collision with Xu (2017)'s GSC.)

    Parameters
    ----------
    treatment_vars : list of str
        The ``K >= 1`` treatment / explanatory columns (continuous or
        discrete). CTSC estimates an average marginal effect for each.
    population_col : str, optional
        Time-invariant per-unit weight column for the average effect
        (e.g. population). Defaults to uniform weights.
    use_fit_weights : bool
        Use the two-step per-unit fit weights ``Omega_i`` (paper eq. 6).
        Default True.
    inference : bool
        Run the sign-flip Wald test of ``H0: alpha^AE = 0``. Default True.
    n_draws : int
        Rademacher draws for the randomization test.
    random_state : int
        Seed for the randomization-test RNG.

    Notes
    -----
    The base ``treat`` field is unused by CTSC; provide the continuous /
    discrete treatment column(s) via ``treatment_vars`` instead. Pass any
    existing column name for ``treat`` to satisfy the base config.
    """

    treatment_vars: List[str] = Field(
        ...,
        description="The K >= 1 continuous/discrete treatment columns.",
    )
    population_col: Optional[str] = Field(
        default=None,
        description="Per-unit weight column for the average effect "
                    "(default uniform).",
    )
    use_fit_weights: bool = Field(
        default=True,
        description="Use the two-step per-unit fit weights Omega_i.",
    )
    inference: bool = Field(
        default=True,
        description="Run the sign-flip Wald test of H0: alpha^AE = 0.",
    )
    n_draws: int = Field(
        default=2000, ge=100,
        description="Rademacher draws for the randomization test.",
    )
    random_state: int = Field(
        default=0,
        description="Seed for the randomization-test RNG.",
    )


class ISCMConfig(BaseEstimatorConfig):
    """Configuration for the Imperfect Synthetic Controls (ISCM) estimator.

    Powell, D. (2026). *"Imperfect Synthetic Controls,"* Journal of Applied
    Econometrics. Builds synthetic controls for every unit, identifies the
    treatment effect even when the treated unit is outside the convex hull,
    weights units by a data-driven fit metric, and uses Ibragimov-Muller
    inference valid for small donor pools. Inherits the standard ``df`` /
    ``outcome`` / ``treat`` / ``unitid`` / ``time`` interface.

    Parameters
    ----------
    inference : bool
        Run Ibragimov-Muller inference over the per-unit estimates.
        Default True.
    null_value : float
        Null effect ``alpha_0`` for the randomization test. Default 0.
    alpha : float
        Two-sided level for the confidence interval.
    n_draws : int
        Number of Rademacher sign-flip draws for the p-value.
    random_state : int
        Seed for the randomization-test RNG.
    """

    inference: bool = Field(
        default=True,
        description="Run Ibragimov-Muller inference over per-unit estimates.",
    )
    null_value: float = Field(
        default=0.0,
        description="Null effect alpha_0 for the randomization test.",
    )
    alpha: float = Field(
        default=0.05, gt=0.0, lt=1.0,
        description="Two-sided level for the confidence interval.",
    )
    n_draws: int = Field(
        default=10000, ge=100,
        description="Number of Rademacher sign-flip draws for the p-value.",
    )
    random_state: int = Field(
        default=0,
        description="Seed for the randomization-test RNG.",
    )


class SPILLSYNTHConfig(BaseEstimatorConfig):
    """Configuration for the SPILLSYNTH estimator.

    Spillover-aware synthetic-control wrapper. Currently ships the
    Cao & Dowd (2023) estimator under ``method='cd'``: a Ferman-Pinto
    demeaned SCM is fit leave-one-out for every unit, and the
    treatment-and-spillover effect vector is recovered jointly under
    a user-specified spillover structure (Example 3 of the paper:
    each declared affected unit gets its own free spillover
    coefficient). Additional methods (e.g. iSCM, distance-decay
    spillover, P-test inference) will be added behind the same
    ``method`` dispatcher.

    Parameters
    ----------
    method : {"cd"}
        Estimation method. Only ``"cd"`` (Cao-Dowd 2023) is
        implemented for now.
    affected_units : list, optional
        Labels (matching values of ``unitid``) of control units the
        researcher believes are potentially exposed to spillover from
        the treated unit. Each gets its own free spillover coefficient
        in the resulting estimator. If ``None`` (the default), no
        affected units are declared and the estimator reduces to
        vanilla demeaned SCM. The treated unit must NOT appear in this
        list.
    solver : str, optional
        ``cvxpy`` solver name used for the leave-one-out simplex SCM
        fits. Defaults to ``"CLARABEL"``.

    References
    ----------
    Cao, J., & Dowd, C. (2023). *Estimation and Inference for
    Synthetic Control Methods with Spillover Effects.* Working paper.

    Ferman, B., & Pinto, C. (2021). *Synthetic Controls with Imperfect
    Pretreatment Fit.* Quantitative Economics, 12(4), 1197-1221.
    """

    method: Literal["cd"] = Field(
        default="cd",
        description="Spillover-aware SCM method. 'cd' = Cao & Dowd (2023).",
    )
    affected_units: Optional[List[Any]] = Field(
        default=None,
        description=(
            "Labels of control units potentially exposed to spillover. "
            "Required for spillover_structure='per_unit' (when p > 0) "
            "and 'homogeneous'. Ignored for 'distance_decay' (which "
            "uses unit_distances). The treated unit must NOT appear here."
        ),
    )
    spillover_structure: Literal["per_unit", "homogeneous", "distance_decay"] = Field(
        default="per_unit",
        description=(
            "A-matrix construction (Cao-Dowd v3 Examples 1/2/3): "
            "'per_unit' (Example 1, default, each affected unit gets its "
            "own free coefficient), 'homogeneous' (Example 2, shared "
            "coefficient b across affected units), 'distance_decay' "
            "(Example 3, alpha_i = b * exp(-d_i) via unit_distances)."
        ),
    )
    unit_distances: Optional[Dict[Any, float]] = Field(
        default=None,
        description=(
            "Required when spillover_structure='distance_decay'. Maps "
            "unit label to a non-negative scalar distance from the "
            "treated unit. Controls absent from the dict are treated as "
            "infinitely far (zero decay weight)."
        ),
    )
    weighting: Literal["identity", "efficient"] = Field(
        default="identity",
        description=(
            "Weighting matrix in the SP estimator. 'identity' = the "
            "standard W = I estimator. 'efficient' = additionally "
            "compute the GMM-weighted variant of Cao-Dowd v3 "
            "Proposition S.1 using W = sample-Omega^{-1} (lower "
            "asymptotic variance); exposed via results.cd.efficient_fit."
        ),
    )
    solver: Optional[str] = Field(
        default=None,
        description="cvxpy solver name for the leave-one-out SCM fits. "
                    "Defaults to CLARABEL.",
    )

    @model_validator(mode="after")
    def _check_affected_units(cls, values: Any) -> Any:
        au = values.affected_units
        structure = values.spillover_structure
        distances = values.unit_distances

        # Structure-specific requirements.
        if structure == "homogeneous" and (au is None or len(au) == 0):
            raise MlsynthConfigError(
                "SPILLSYNTH: spillover_structure='homogeneous' needs at "
                "least one entry in affected_units."
            )
        if structure == "distance_decay":
            if not isinstance(distances, dict) or len(distances) == 0:
                raise MlsynthConfigError(
                    "SPILLSYNTH: spillover_structure='distance_decay' "
                    "requires a non-empty unit_distances={label: d, ...}."
                )

        if au is None:
            return values
        if len(set(au)) != len(au):
            raise MlsynthConfigError(
                "SPILLSYNTH: affected_units contains duplicate labels."
            )
        # Resolve treated units on the spot to surface helpful errors
        # before .fit() runs. Multiple treated units are allowed (Cao-
        # Dowd v3 Section S.1.2), but none of them may also appear in
        # affected_units.
        df = values.df
        treat_col = values.treat
        unit_col = values.unitid
        treated_rows = set(df.loc[df[treat_col] != 0, unit_col].unique())
        overlap = treated_rows.intersection(au)
        if overlap:
            raise MlsynthConfigError(
                f"SPILLSYNTH: treated units {sorted(overlap, key=str)} cannot "
                "also appear in affected_units."
            )
        present = set(df[unit_col].unique())
        missing = [u for u in au if u not in present]
        if missing:
            raise MlsynthConfigError(
                f"SPILLSYNTH: affected_units {missing} not in df[{unit_col!r}]."
            )
        return values


class DSCARConfig(BaseEstimatorConfig):
    """Configuration for the Dynamic Synthetic Control AR (DSCAR) estimator.

    Zheng & Chen (2024), JRSS-B 86(1):155-176, "Dynamic synthetic
    control method for evaluating treatment effects in auto-regressive
    processes." Extends Abadie-Diamond-Hainmueller (2010) to settings
    with time-varying confounders, an auto-regressive outcome, spatial
    dependence in the residuals, and possibly multiple treated units
    that all turn on at a common intervention time.

    The DSC weights are computed **per post-period** via empirical-
    likelihood maximisation under per-period matching constraints
    (equations 2.7-2.9 of the paper), allowing exact matches as
    :math:`N_{co}, N_{tr} \\to \\infty` with :math:`T` fixed -- the
    asymptotic regime that suits a typical air-pollution / hourly
    panel.

    Parameters
    ----------
    exog_covariates : list of str, optional
        Time-varying exogenous covariate columns to include in the
        per-period matching. ``None`` skips the covariate match (DSC
        then matches on the lagged outcome only).
    lagged_outcome : str, optional
        Column name supplying the externally-computed
        :math:`Y_{i, t-1}` value at the **first** sample period. For
        later periods the lag is read off the panel itself. ``None``
        drops the lag constraint at ``t = 1``.
    placebo_reps : int
        Number of normalised-placebo replications for the SE on the
        DSC ATT (Section 3.2). ``0`` (default) skips placebo inference.
    el_tolerance : float
        Threshold for the QP residual ``mean|Z_1 - Z_0 w|`` at which
        the EL refinement step is attempted; matches the R reference's
        default ``0.01``. Smaller values fall back to QP weights more
        often.
    fdr_alpha : float
        Significance level for the BY-adjusted pre-period
        unconfoundedness test (Section 3.1).
    seed : int
        RNG seed for the placebo draw.

    References
    ----------
    Zheng, X., & Chen, S. X. (2024). *Dynamic synthetic control method
    for evaluating treatment effects in auto-regressive processes.*
    Journal of the Royal Statistical Society Series B, 86(1):155-176.
    """

    exog_covariates: Optional[List[str]] = Field(
        default=None,
        description=(
            "Time-varying exogenous covariate column names. None to "
            "skip the covariate match (lagged-outcome match only)."
        ),
    )
    lagged_outcome: Optional[str] = Field(
        default=None,
        description=(
            "Column carrying Y_{i, t-1} at the first sample period. "
            "Required to include the lag match at t = 1."
        ),
    )
    placebo_reps: int = Field(
        default=0,
        description=(
            "Number of normalised-placebo replications for the DSC "
            "ATT standard error. 0 disables placebo inference."
        ),
    )
    el_tolerance: float = Field(
        default=1e-2,
        description=(
            "Mean-absolute-mismatch threshold for triggering the EL "
            "refinement step. R reference default: 0.01."
        ),
    )
    fdr_alpha: float = Field(
        default=0.05,
        description="Significance level for the BY pre-period test.",
    )
    seed: int = Field(default=0, description="RNG seed for placebo draws.")


class COMPSYNTHConfig(BaseEstimatorConfig):
    """Configuration for the COMPSYNTH estimator.

    Bogatyrev, K. & Stoetzer, L. F. (2026). *"Estimating Treatment Effects
    on Proportions with Synthetic Controls,"* Political Analysis (R package
    ``propsdid``). Common-weights synthetic controls for compositional
    (proportional) outcomes -- one donor (and time) weighting shared across
    all ``K`` proportions, so the per-outcome ATTs sum to zero.

    Parameters
    ----------
    outcomes : list of str
        The ``K >= 2`` compositional outcome columns. Each row's values
        across these columns must be non-negative and sum to one.
    method : {"sdid", "sc"}
        ``"sdid"`` -- synthetic difference-in-differences with common unit
        and time weights (default). ``"sc"`` -- classic synthetic control
        with common unit weights, no time weights, no intercept shift.
    inference : bool
        Run placebo inference for per-outcome SEs/CIs. Default True.
    alpha : float
        Two-sided level for the placebo confidence intervals.
    max_placebo : int, optional
        Cap on the number of control units used as placebos.

    Notes
    -----
    The base ``outcome`` field is unused by COMPSYNTH; provide the ``K``
    proportion columns via ``outcomes`` instead. Pass any existing column
    name for ``outcome`` to satisfy the base config (e.g. ``outcomes[0]``).
    """

    outcomes: List[str] = Field(
        ...,
        description="The K >= 2 compositional outcome columns (sum to 1 "
                    "per unit-time).",
    )
    method: Literal["sdid", "sc"] = Field(
        default="sdid",
        description="'sdid' (common unit+time weights) or 'sc' (common unit "
                    "weights only).",
    )
    inference: bool = Field(
        default=True,
        description="Run placebo inference for per-outcome standard errors.",
    )
    alpha: float = Field(
        default=0.05, gt=0.0, lt=1.0,
        description="Two-sided level for placebo confidence intervals.",
    )
    max_placebo: Optional[int] = Field(
        default=None, ge=2,
        description="Cap on the number of control units used as placebos.",
    )


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


class BASCConfig(BaseEstimatorConfig):
    """Configuration for the Bayesian spike-and-slab synthetic control (BASC).

    BASC builds donor weights from a Gamma "slab" magnitude times a
    Bernoulli "spike" inclusion indicator, renormalized onto the simplex::

        w_j   = (u_j * gamma_j) / sum_k (u_k * gamma_k),
        u_j   ~ Gamma(a_u, b_u),
        gamma_j ~ Bernoulli(pi),     pi ~ Beta(a_pi, b_pi).

    With every ``gamma_j = 1`` the construction reduces to a Dirichlet
    prior on the simplex (the "B-MV" model of Martinez &
    Vives-i-Bastida, 2024), so the Bernoulli spike is the only addition
    for sparse donor selection. The weights are fit on the
    **pre-treatment window only**; the post-treatment counterfactual is
    the donor projection (plus an optional GP trend), so post-treatment
    treated outcomes never enter the weight likelihood.

    Parameters
    ----------
    n_iter : int
        Total MCMC iterations (including burn-in).
    burn_in : int
        Number of warm-up iterations discarded before reporting.
    a_u, b_u : float
        Gamma slab shape / rate. ``a_u`` doubles as the Dirichlet
        concentration on the active simplex (``a_u = 1`` is uniform).
    a_pi, b_pi : float
        Beta hyperparameters for the prior inclusion probability ``pi``.
        ``(1, 1)`` is uniform; raise ``b_pi`` to favor sparser designs.
    sigma2_c0, sigma2_d0 : float
        Inverse-Gamma shape / rate for the observation variance.
    rw_conc : float
        Concentration of the Dirichlet random-walk proposal for the
        active weights (larger = smaller steps, higher acceptance).
    use_gp : bool
        Add a squared-exponential GP trend ``f_t`` to the pre-period fit.
        Off by default: on the data we tested the GP was immaterial to
        the effect estimate and is costly on long panels.
    ci_alpha : float
        Two-sided significance level for credible intervals (default
        0.05 gives 95% bands).
    display_graphs : bool
        Display the observed-vs-counterfactual plot after the fit.
    verbose : bool
        Show a tqdm progress bar during MCMC.
    seed : int, optional
        Seed for the ``numpy.random.Generator`` used inside the sampler.
    """

    n_iter: int = Field(default=4000, ge=10)
    burn_in: int = Field(default=2000, ge=0)
    a_u: float = Field(default=1.0, gt=0)
    b_u: float = Field(default=1.0, gt=0)
    a_pi: float = Field(default=1.0, gt=0)
    b_pi: float = Field(default=1.0, gt=0)
    sigma2_c0: float = Field(default=0.01, gt=0)
    sigma2_d0: float = Field(default=0.01, gt=0)
    rw_conc: float = Field(default=50.0, gt=0)
    use_gp: bool = Field(default=False)
    ci_alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
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


class SCMOConfig(BaseEstimatorConfig):
    """Configuration for the Synthetic Control with Multiple Outcomes (SCMO) estimator."""

    addout: Union[str, List[str]] = Field(
        default_factory=list,
        description="Auxiliary outcome variable(s) for outcome stacking. Used to build the matching spec when `spec` is not given.",
    )
    method: str = Field(
        default="TLP",
        description="Legacy method selector: 'TLP', 'SBMF', or 'BOTH'. Maps to schemes when `schemes` is not given.",
        pattern="^(TLP|SBMF|BOTH)$",
    )
    conformal_alpha: float = Field(
        default=0.1,
        description="Miscoverage rate for conformal prediction intervals (e.g., 0.1 for 90% CI).",
        gt=0, lt=1,
    )
    spec: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Spec-driven matching matrix: {'year': int|list[int], 'vars': {name: column | (column, op)}}, op in {level,log,per_capita,raw}. If None, built from outcome+addout over the pre-period.",
    )
    schemes: Optional[List[str]] = Field(
        default=None,
        description="Weighting schemes to run: any of 'concatenated','averaged','separate','MA'. If None, derived from `method` (TLP->concatenated, SBMF->averaged, BOTH->[concatenated,averaged,MA]).",
    )
    demean: bool = Field(
        default=False,
        description="Intercept-shift the counterfactual (Doudchenko-Imbens / Sun-Ben-Michael-Feller level adjustment).",
    )
    conformal_q: float = Field(
        default=1.0,
        description="Norm exponent q of the CWZ conformal test statistic S_q (1 = average effect; larger targets sparse/large effects across outcomes).",
        gt=0,
    )





class SIConfig(BaseEstimatorConfig):
    """Configuration for the Synthetic Interventions (SI) estimator.

    Implements:

        Agarwal, A., Shah, D., & Shen, D. (2026). "Synthetic Interventions:
        Extending Synthetic Controls to Multiple Treatments." Operations
        Research 74(2):840-859.

    SI estimates the focal target unit's counterfactual outcome under each
    alternative intervention in ``inters`` via SI-PCR: regress the target's
    pre-treatment control outcomes onto the rank-``k`` denoised donor pool, then
    apply the weights to the donor pool's post-intervention outcomes.

    Parameters
    ----------
    inters : list of str
        Binary indicator columns naming the alternative interventions; for
        each, the units flagged ``1`` form that intervention's donor pool.
    rank_method : {"donoho", "usvt", "cumvar", "fixed"}
        Spectral-rank rule for the donor pre-matrix. ``"donoho"`` (default)
        reproduces the paper's exact Gavish-Donoho optimal hard threshold
        (evaluated at ``ratio = T0 / Nd``); ``"usvt"`` is the same threshold at
        the canonical ``min/max`` aspect ratio; ``"cumvar"`` keeps enough
        components for ``cumvar_threshold`` of the spectral energy; ``"fixed"``
        uses ``rank``.
    rank : int or None
        Explicit spectral rank ``k`` for ``rank_method="fixed"``.
    cumvar_threshold : float
        Cumulative-energy target in ``(0, 1]`` for ``rank_method="cumvar"``.
    bias_correct : bool
        Use the bias-corrected SI-PCR estimator (default ``True``), which fits
        weights on a rank-complete donor subset and enables asymptotic-normality
        intervals (Section 4.3). ``False`` gives plain SI-PCR (eq. 10), point
        estimate only.
    variance : {"double", "units", "time_iv"}
        Noise-variance estimator behind the interval. ``"double"`` (default)
        matches the paper's code (a d.o.f.-weighted combination); ``"units"`` is
        the main-text eq. 14; ``"time_iv"`` uses the donor post-period residual.
    interval : {"confidence", "prediction"}
        Interval type. ``"confidence"`` (default) is the eq.-13 CI for the
        counterfactual mean; ``"prediction"`` is the wider interval the paper's
        case study uses for coverage validation.
    alpha : float
        Two-sided significance level for the intervals.
    display_graphs : bool
        Show the observed-vs-counterfactual plot after fitting.
    """

    inters: List[str] = Field(
        ..., min_length=1,
        description="Binary indicator columns naming the alternative "
                    "interventions (donor pool = units flagged 1).",
    )
    rank_method: Literal["donoho", "usvt", "cumvar", "fixed"] = Field(
        default="donoho",
        description="Spectral-rank rule: paper's Gavish-Donoho ('donoho'), "
                    "min/max Gavish-Donoho ('usvt'), cumulative-energy "
                    "('cumvar'), or explicit ('fixed').",
    )
    rank: Optional[int] = Field(
        default=None, ge=1,
        description="Explicit spectral rank k for rank_method='fixed'.",
    )
    cumvar_threshold: float = Field(
        default=0.95, gt=0.0, le=1.0,
        description="Cumulative-energy target for rank_method='cumvar'.",
    )
    bias_correct: bool = Field(
        default=True,
        description="Use the bias-corrected SI-PCR estimator (enables intervals).",
    )
    variance: Literal["double", "units", "time_iv"] = Field(
        default="double",
        description="Noise-variance estimator: paper's weighted ('double'), "
                    "main-text eq. 14 ('units'), or post-period ('time_iv').",
    )
    interval: Literal["confidence", "prediction"] = Field(
        default="confidence",
        description="Interval type: confidence (eq. 13) or prediction.",
    )
    alpha: float = Field(
        default=0.05, gt=0.0, lt=1.0,
        description="Two-sided significance level for the intervals.",
    )

    @model_validator(mode="after")
    def _check_si_params(cls, values: Any) -> Any:
        if values.rank_method == "fixed" and values.rank is None:
            raise MlsynthConfigError(
                "rank_method='fixed' requires an explicit positive `rank`."
            )
        return values

class StableSCConfig(BaseEstimatorConfig):
    """Configuration for the Stable Synthetic Control (StableSC) estimator."""
    granger_alpha: float = Field(default=0.05, description="Significance level for Granger causality tests in the fit method.", ge=0, le=1)
    granger_maxlag: int = Field(default=1, description="Maximum lag for Granger causality tests in the fit method.", ge=1)
    proximity_alpha: float = Field(default=0.05, description="Significance level for proximity mask chi-squared tests in the fit method.", ge=0, le=1)
    rbf_sigma_fit: float = Field(default=20.0, description="RBF kernel sigma (width) used for scoring in the fit method.", gt=0)
    sc_model_type: str = Field(default="SIMPLEX", description="Synthetic control optimization model type. Valid options are 'SIMPLEX', 'OLS', 'MSCa', 'MSCb', 'MSCc'.", pattern="^(SIMPLEX|OLS|MSCa|MSCb|MSCc)$")

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
class EICPConfig(BaseEstimatorConfig):
    """Configuration for the Entrywise Inference for Causal Panels (EICP) estimator.

    Implements the SVD-based imputation and entrywise Gaussian
    confidence intervals of Yan and Wainwright (2024,
    arXiv:2401.13665) for causal panel data under staggered adoption.

    Parameters
    ----------
    rank : int or None, default None
        Truncation rank ``r`` for the SVD steps. If ``None``, selected
        automatically via the Donoho-Gavish optimal singular-value
        threshold (Gavish and Donoho, 2014).
    alpha : float, default 0.05
        Two-sided significance level for the entrywise confidence
        intervals (coverage = ``1 - alpha``).
    estimate_treated_potential_outcomes : bool, default False
        Whether to also impute the treated potential outcomes
        ``N^*_{i,t}`` for untreated cells by applying the algorithm to
        the 180-degree-rotated panel. Identifiability requires at least
        one "always-treated" cohort; if absent, this flag has no effect
        and the observed outcomes are used as the only estimate of
        ``N^*`` on treated cells.
    display_graph : bool, default False
        Whether to render a cohort-mean diagnostic plot via
        :func:`mlsynth.utils.eicp_helpers.plotter.plot_eicp_results`.
    """

    rank: Optional[int] = Field(
        default=None,
        ge=1,
        description="SVD truncation rank. If None, selected by the "
        "Donoho-Gavish optimal SVHT rule.",
    )
    alpha: float = Field(
        default=0.05,
        gt=0.0,
        lt=1.0,
        description="Two-sided significance level for entrywise CIs.",
    )
    estimate_treated_potential_outcomes: bool = Field(
        default=False,
        description="Whether to estimate N^* via the rotated panel.",
    )
    display_graph: bool = Field(
        default=False,
        description="Whether to plot cohort-mean observed vs. counterfactual series.",
    )


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
    """Standardized model for reporting donor weights."""
    donor_weights: Optional[Dict[str, float]] = Field(default=None, description="Dictionary mapping donor unit names/IDs to their weights.")
    summary_stats: Optional[Dict[str, Any]] = Field(default=None, description="Summary statistics about weights (e.g., cardinality).")
    # For estimators returning multiple sets of weights (e.g. TSSC sub-methods), this might be part of a list or dict structure.
    # donor_names is removed as it's incorporated into donor_weights dict keys

    class Config:
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

class BaseEstimatorResults(BaseModel):
    """
    Base Pydantic model for standardized estimator `fit()` method results.
    This model aims to provide a common structure for the primary outputs.
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
