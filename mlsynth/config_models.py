from typing import List, Optional, Any, Dict, Union
import pandas as pd
import numpy as np # Added for potential use in other models, e.g. np.ndarray
from pydantic import BaseModel, Field, model_validator
from typing import Any # Ensure Any is imported for the validator
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
    """
    Configuration for the Synthetic Experiment Design estimator (MAREX) in mlsynth.

    This configuration is specifically intended for **experimental design using synthetic control**.
    Unlike typical causal estimators, users do **not** pre-specify treated units because the
    MAREX class internally optimizes treatment assignment to achieve balance in pre-treatment outcomes.
    This allows experimenters to design experiments even when post-treatment data does not yet exist.

    Key points:
    - `treat` is intentionally omitted: treatment assignment is optimized internally.
    - Users may provide a `clusters` vector to indicate groupings for synthetic treated/control units.
    - `T0` allows limiting pre-treatment periods used for optimization.
    - `design` selects the type of synthetic control optimization:
        - `"base"`: cluster-targeted fit to cluster mean
        - `"weak"`: weakly-targeted fit with optional penalty `beta`
        - `"eq11"`: penalized design with lambda1/lambda2 weights
        - `"unit"`: unit-level penalized design (xi, lambda1_unit, lambda2_unit)
    - This configuration is designed for iterative or exploratory experiment design,
      where treatment may later be applied in a real-world study.
    """

    # Core design parameters
    T0: Optional[int] = Field(default=None, description="Number of pre-treatment periods.")
    cluster: Optional[str] = Field(
        default=None,
        description="Column name in df indicating cluster membership for each unit."
    )

    design: str = Field(default="base", description="Design type: 'base', 'weak', 'eq11', 'unit'.")

    # Penalization parameters
    beta: float = Field(default=1e-6, description="Weak-targeting penalty (used if design='weak').")
    lambda1: float = Field(default=0.0, description="Treated distance penalty (used if design='eq11').")
    lambda2: float = Field(default=0.0, description="Control distance penalty (used if design='eq11').")
    xi: float = Field(default=0.0, description="Unit-level OA.1 penalty (used if design='unit').")
    lambda1_unit: float = Field(default=0.0, description="Unit-level OA.2 penalty (used if design='unit').")
    lambda2_unit: float = Field(default=0.0, description="Unit-level OA.3 penalty (used if design='unit').")

    # Additional SCMEXP options
    blank_periods: int = Field(default=0, description="Number of blank periods at the start of Y_full.")
    m_eq: Optional[int] = Field(default=None, description="Optional exact number of treated units per cluster.")
    m_min: Optional[int] = Field(default=None, description="Optional minimum treated units per cluster.")
    m_max: Optional[int] = Field(default=None, description="Optional maximum treated units per cluster.")
    exclusive: bool = Field(default=True, description="Whether treated units are mutually exclusive across clusters.")
    solver: Any = Field(default=None, description="Optional cvxpy solver to use (e.g., cp.ECOS_BB).")
    verbose: bool = Field(default=False, description="Whether to display solver/logging output.")

    @model_validator(mode="after")
    def validate_design_params(cls, values: Any) -> Any:
        df = values.df
        T0 = values.T0
        design = values.design

        # Validate T0
        n_periods = df[values.time].nunique()
        if T0 is not None and (T0 <= 0 or T0 > n_periods):
            raise MlsynthDataError(f"T0 must be between 1 and the number of time periods ({n_periods}).")

        # Validate design
        valid_designs = {"base", "weak", "eq11", "unit"}
        if design not in valid_designs:
            raise MlsynthDataError(f"design must be one of {valid_designs}; got '{design}'")

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
    """Configuration for the Two-Step Synthetic Control (TSSC) estimator."""
    draws: int = Field(default=500, description="Number of draws for inference.", ge=0) # Changed ge=1 to ge=0
    ci: float = Field(default=0.95, description="Confidence interval level.", ge=0, le=1)
    parallel: bool = Field(default=False, description="Whether to use parallel processing for draws.")
    cores: Optional[int] = Field(default=None, description="Number of cores for parallel processing. Defaults to all available if None and parallel is True.", ge=1)
    scm_weights_args: Optional[Dict[str, Any]] = Field(default=None, description="Additional arguments for SCM weight optimization.")

# Placeholder for other estimator configs - to be added sequentially

class FMAConfig(BaseEstimatorConfig):
    """Configuration for the Factor Model Approach (FMA) estimator."""
    criti: int = Field(default=11, description="Criterion for stationarity assumption: 11 for nonstationary, 10 for stationary.", ge=10, le=11) # Assuming 10 or 11
    DEMEAN: int = Field(default=1, description="Data processing method: 1 for demean, 2 for standardize.", ge=1, le=2) # Assuming 1 or 2

class PDAConfig(BaseEstimatorConfig):
    """Configuration for the Panel Data Approach (PDA) estimator."""
    method: str = Field(default="fs", description="Type of PDA to use: 'LASSO', 'l2', or 'fs'.", pattern="^(LASSO|l2|fs)$")
    tau: Optional[float] = Field(default=None, description="User-specified treatment effect value (used as tau_l2 for 'l2' method).")
    # Note: The original __init__ had validation for method.lower(). Pydantic's pattern handles case-sensitivity.
    # If case-insensitivity is desired, the pattern would be different or a validator would be needed.
    # For now, assuming exact match "LASSO", "l2", "fs".

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
    plot_did: bool = Field(default=True, description="Whether to plot standard DID results.")

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
    """Configuration for the Cluster-based Synthetic Control (CLUSTERSC) estimator."""
    objective: str = Field(default="OLS", description="Constraint for PCR ('OLS', 'SIMPLEX').", pattern="^(OLS|SIMPLEX|)$")
    cluster: bool = Field(default=True, description="Whether to apply clustering for PCR.")
    Frequentist: bool = Field(default=True, description="If True, use Frequentist Robust SCM; False for Bayesian (for PCR method).")
    ROB: str = Field(default="PCP", description="Robust method for RPCA ('PCP' or 'HQF').", pattern="^(PCP|HQF)$") # Parameter name is ROB in code
    method: str = Field(default="PCR", description="Estimation method: 'PCR', 'RPCA', or 'BOTH'.", pattern="^(PCR|RPCA|BOTH)$")
    Robust: Optional[str] = Field(default=None, exclude=True, description="Temporary field to catch phantom 'Robust' param from pytest issue.")

    @model_validator(mode='before')
    @classmethod
    def _handle_legacy_robust_param(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if 'Robust' in data:
                robust_value = data.pop('Robust')
                # If 'ROB' is not already in data or if 'ROB' is present but from a default,
                # let 'Robust' (if it was the one explicitly passed due to the phantom issue) set 'ROB'.
                # This prioritizes the phantom 'Robust' to become the actual 'ROB' value.
                if 'ROB' not in data or data.get('ROB') == cls.model_fields['ROB'].default:
                    data['ROB'] = robust_value
                # If 'ROB' was also explicitly passed and is different from default, it means
                # both 'Robust' (phantom) and 'ROB' (intended) were in the input.
                # In this specific scenario, the original 'ROB' (if not default) would have been overwritten by 'Robust'
                # when config_dict was formed: config_dict = {**base_config_dict, **rpca_config_override}
                # if rpca_config_override was {'Robust': 'HQF'}.
                # The pop('Robust') already removed it. If 'ROB' is already set from rpca_config_override,
                # this logic is fine. The main goal is to ensure 'Robust' doesn't cause validation error
                # and its value is used for 'ROB' if 'ROB' wasn't more specifically provided.
        return data

class PROXIMALConfig(BaseEstimatorConfig):
    """Configuration for the Proximal Inference (PROXIMAL) estimator."""
    # Override counterfactual_color from Base to match PROXIMAL's default and usage
    counterfactual_color: Union[str, List[str]] = Field(default_factory=lambda: ["grey", "red", "blue"], description="Color(s) for counterfactual lines in plots. Can be a single color string or a list of color strings for multiple counterfactuals.")

    donors: List[Union[str, int]] = Field(..., min_length=1, description="List of donor unit identifiers. Must not be empty.")
    surrogates: List[Union[str, int]] = Field(default_factory=list, description="List of surrogate unit identifiers.")
    vars: Dict[str, List[str]] = Field(default_factory=dict, description='Dictionary specifying proxy variables. Requires "donorproxies". If surrogates are used, also requires "surrogatevars".')

    @model_validator(mode='after')
    def check_vars_structure(cls, values: Any) -> Any: # Changed to 'values: Any' for Pydantic v2 compatibility if 'self' is not used
        # In Pydantic v2, for model_validator(mode='after'), the first argument is the model instance (often named 'self' or 'v')
        # However, to be safe and more aligned with examples, using 'values' as a dict if direct attribute access isn't needed.
        # If direct attribute access like `self.vars` is preferred, the first arg should be `self`.
        # Let's assume `values` is the model instance here for direct attribute access.
        
        # Re-accessing from the model instance 'values' (which is 'self' effectively)
        vars_dict = values.vars
        surrogates_list = values.surrogates

        if not vars_dict.get("donorproxies") or not isinstance(vars_dict.get("donorproxies"), list) or not vars_dict["donorproxies"]:
            raise MlsynthConfigError("Config 'vars' must contain a non-empty list for 'donorproxies'.")

        if surrogates_list and (not vars_dict.get("surrogatevars") or not isinstance(vars_dict.get("surrogatevars"), list) or not vars_dict["surrogatevars"]):
            raise MlsynthConfigError("Config 'vars' must contain a non-empty list for 'surrogatevars' when surrogates are provided.")
        return values


class FSCMConfig(BaseEstimatorConfig):
    """
    Configuration for the Forward Selected Synthetic Control Method (FSCM) estimator.
    This estimator supports sparse model selection using mBIC, with optional
    full selection and affine refinement (augmented FSCM).

    References
    ----------
    Shi, Zhentao, and Jingyi Huang. 2023.
    "Forward-selected panel data approach for program evaluation."
    Journal of Econometrics 234 (2): 512–535.
    https://doi.org/10.1016/j.jeconom.2021.04.009

    Cerulli, Giovanni. 2024.
    “Optimal initial donor selection for the synthetic control method.”
    Economics Letters, 244: 111976.
    https://doi.org/10.1016/j.econlet.2024.111976

    Ben-Michael, Eli, Avi Feller, and Jesse Rothstein. 2021.
    "The Augmented Synthetic Control Method."
    Journal of the American Statistical Association 116 (536): 1789–1803.
    https://doi.org/10.1080/01621459.2021.1929245
    """

    use_augmented: bool = Field(
        default=False,
        description=(
            "If True, refine selected sparse weights using Affine Synthetic Control "
            "(Augmented FSCM). Uses ridge regularization tuned via Bayesian optimization."
        )
    )

    full_selection: bool = Field(
        default=True,
        description=(
            "If True, performs full forward selection through all possible models. "
            "If False, stops early once mBIC no longer improves."
        )
    )

    selection_fraction: float = Field(
        default=1.0,
        ge=0.01,
        le=1.0,
        description=(
            "Fraction (0 < fraction ≤ 1.0) of donor pool to consider during forward selection. "
            "Reduces computation time by limiting candidate donor units. "
            "Use values < 1.0 for high-dimensional donor settings."
        )
    )

    bo_n_iter: int = Field(
        default=25,
        ge=1,
        description=(
            "Number of iterations for Bayesian optimization of the ridge penalty "
            "used in affine refinement."
        )
    )

    bo_initial_evals: int = Field(
        default=5,
        ge=1,
        description=(
            "Number of initial random evaluations before fitting the surrogate model "
            "in Bayesian optimization."
        )
    )


class SRCConfig(BaseEstimatorConfig):
    """
    Configuration for the Synthetic Regressing Control (SRC) estimator.
    This estimator currently uses only the common configuration parameters.
    """
    pass

class SCMOConfig(BaseEstimatorConfig):
    """Configuration for the Synthetic Control with Multiple Outcomes (SCMO) estimator."""
    addout: Union[str, List[str]] = Field(default_factory=list, description="Auxiliary outcome variable(s) for outcome stacking.")
    method: str = Field(default="TLP", description="Estimation method: 'TLP', 'SBMF', or 'BOTH'.", pattern="^(TLP|SBMF|BOTH)$")
    conformal_alpha: float = Field(default=0.1, description="Miscoverage rate for conformal prediction intervals (e.g., 0.1 for 90% CI).", gt=0, lt=1)

class SIConfig(BaseEstimatorConfig):
    """Configuration for the Synthetic Interventions (SI) estimator."""
    inters: List[str] = Field(..., min_length=1, description="Required list of binary treatment indicator column names for alternative interventions. Must not be empty.")
    # No default_factory=list, as Pydantic will require it due to '...' and min_length=1.

class StableSCConfig(BaseEstimatorConfig):
    """Configuration for the Stable Synthetic Control (StableSC) estimator."""
    granger_alpha: float = Field(default=0.05, description="Significance level for Granger causality tests in the fit method.", ge=0, le=1)
    granger_maxlag: int = Field(default=1, description="Maximum lag for Granger causality tests in the fit method.", ge=1)
    proximity_alpha: float = Field(default=0.05, description="Significance level for proximity mask chi-squared tests in the fit method.", ge=0, le=1)
    rbf_sigma_fit: float = Field(default=20.0, description="RBF kernel sigma (width) used for scoring in the fit method.", gt=0)
    sc_model_type: str = Field(default="SIMPLEX", description="Synthetic control optimization model type. Valid options are 'SIMPLEX', 'OLS', 'MSCa', 'MSCb', 'MSCc'.", pattern="^(SIMPLEX|OLS|MSCa|MSCb|MSCc)$")

class NSCConfig(BaseEstimatorConfig):
    """
    Configuration for the Nonlinear Synthetic Control (NSC) estimator.
    Hyperparameters (a, b) for optimization are determined via cross-validation.
    """
    a_search_space: Optional[List[float]] = Field(default=None, description="Optional list of values for hyperparameter 'a' to search during cross-validation. If None, internal defaults are used.")
    b_search_space: Optional[List[float]] = Field(default=None, description="Optional list of values for hyperparameter 'b' to search during cross-validation. If None, internal defaults are used.")

    @model_validator(mode='after')
    def check_search_spaces(cls, values: Any) -> Any:
        a_space = values.a_search_space
        b_space = values.b_search_space
        if a_space is not None and not all(isinstance(x, (int, float)) for x in a_space):
            raise MlsynthConfigError("All elements in a_search_space must be numbers.")
        if b_space is not None and not all(isinstance(x, (int, float)) for x in b_space):
            raise MlsynthConfigError("All elements in b_search_space must be numbers.")
        if a_space is not None and not a_space: # Check if list is empty
            raise MlsynthConfigError("a_search_space cannot be an empty list if provided.")
        if b_space is not None and not b_space: # Check if list is empty
            raise MlsynthConfigError("b_search_space cannot be an empty list if provided.")
        return values

class SDIDConfig(BaseEstimatorConfig):
    """Configuration for the Synthetic Difference-in-Differences (SDID) estimator."""
    B: int = Field(default=500, description="Number of placebo iterations for inference.", ge=0) # B can be 0 if no inference desired

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





class RESCMConfig(BaseEstimatorConfig):
    pass


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



class StudyConfig(BaseModel):
    """
    Holds hyperparameters and design characteristics of the SCMEXP study.
    """
    beta: float
    lambda1: float
    lambda2: float
    xi: float
    T0: int
    blank_periods: int
    design: str

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"


class GlobalResults(BaseModel):
    """
    Aggregated global results from the synthetic control fitting,
    before any treatment effect calculations.
    """
    Y_fit: Optional[np.ndarray]           # Full fitted matrix (units x time)
    Y_blank: Optional[np.ndarray]         # Blank / missing periods
    Y_full: Optional[np.ndarray]          # Original observed matrix
    treated_weights_agg: np.ndarray       # Flattened treated weights across all units
    control_weights_agg: np.ndarray       # Flattened control weights across all units
    rmse_clusters: Optional[np.ndarray] = None  # Pre-treatment RMSE for each cluster

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"


class ClusterResults(BaseModel):
    """
    Results for a single cluster in the SCMEXP design.
    """
    members: List[str]                        # unit IDs in this cluster
    cluster_cardinality: int                  # number of units
    rmse: Optional[float] = None              # pre-treatment fit error

    # Synthetic outcomes
    synthetic_treated: np.ndarray
    synthetic_control: np.ndarray

    # Weights
    treated_weights: np.ndarray
    control_weights: np.ndarray
    selection_indicators: np.ndarray

    # Diagnostics
    pre_treatment_means: Optional[np.ndarray] = None

    unit_weight_map: Optional[dict] = None

    # Allow arbitrary types like np.ndarray
    model_config = {
        "arbitrary_types_allowed": True
    }



class MAREXResults(BaseModel):
    """
    Results of a MAREX synthetic experiment design.
    Contains cluster-level results, study configuration, and global pre-treatment results.
    """
    clusters: Dict[str, ClusterResults]
    study: StudyConfig
    globres: GlobalResults

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"
