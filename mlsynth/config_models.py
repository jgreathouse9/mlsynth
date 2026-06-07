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
    "PANGEOConfig": "mlsynth.utils.pangeo_helpers.config",
    "FSCMConfig": "mlsynth.utils.fscm_helpers.config",
    "LEXSCMConfig": "mlsynth.utils.fast_scm_helpers.config",
    "TASCConfig": "mlsynth.utils.tasc_helpers.config",
    "SBCConfig": "mlsynth.utils.sbc_helpers.config",
    "HSCConfig": "mlsynth.utils.hsc_helpers.config",
    "BVSSConfig": "mlsynth.utils.bvss_helpers.config",
    "SPCDConfig": "mlsynth.utils.spcd_helpers.config",
    "NSCConfig": "mlsynth.utils.nsc_helpers.config",
    "SDIDConfig": "mlsynth.utils.sdid_helpers.config",
    "SparseSCConfig": "mlsynth.utils.sparse_sc_helpers.config",
    "MicroSynthConfig": "mlsynth.utils.microsynth_helpers.config",
    "PPSCMConfig": "mlsynth.utils.ppscm_helpers.config",
    "SequentialSDIDConfig": "mlsynth.utils.seq_sdid_helpers.config",
    "SHCConfig": "mlsynth.utils.shc_helpers.config",
    "MLSCConfig": "mlsynth.utils.mlsc_helpers.config",
    "RESCMConfig": "mlsynth.utils.laxscm_helpers.config",
    "SYNDESConfig": "mlsynth.utils.syndes_helpers.config",
    "SIVConfig": "mlsynth.utils.siv_helpers.config",
    "MAREXConfig": "mlsynth.utils.marex_helpers.config",
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
