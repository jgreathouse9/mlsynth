"""Configuration for the MAREX estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import warnings
from pydantic import Field, model_validator
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
        description="Two-sided significance level for the MDE power curve.",
    )
    power_target: float = Field(
        default=0.80, gt=0.0, lt=1.0,
        description="Target power (1 - beta) for the MDE.",
    )

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
