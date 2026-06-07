"""Configuration for the MLSC estimator.

Co-located with the helper package; re-exported from
:mod:`mlsynth.config_models` for backward compatibility.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union
import pandas as pd
from pydantic import BaseModel, Field, model_validator
from ...exceptions import MlsynthDataError


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
