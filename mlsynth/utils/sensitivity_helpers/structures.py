"""Typed collections of native fits and extracted sensitivity trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from pydantic import BaseModel, ConfigDict, Field

from ...exceptions import MlsynthConfigError
from ..vanillasc_helpers.structures import VanillaSCResults
from .numerics import median, std
from .plotter import plot_specification_curve, plot_validation

_DIMENSIONS = {"donors", "fit_window", "fit_exclude", "covariates", "covariate_windows"}


@dataclass(frozen=True)
class FittedPaths:
    """Aligned, finite trajectories and donor weights for one specification.

    ``paths`` has one row per common study period; ``weights`` has one row
    per donor. Both tables include ``spec_id`` for concatenating attempts.
    """

    paths: pd.DataFrame
    weights: pd.DataFrame


class SensitivityResult(BaseModel):
    """A collection of unchanged estimator results and tidy sensitivity tables.

    ``summary`` has one row per attempt; ``failures`` and ``warnings`` preserve
    unsuccessful fits and diagnostics. ``fits`` retains the complete native
    results, including predictor weights and solver diagnostics. ``registry``
    contains resolved configurations without repeated panel data; ``baseline``
    preserves the original configuration, including a copy of the panel.
    ``validation`` records every attempted design/fold, including errors.
    ``validation_paths``, ``validation_weights`` and ``validation_warnings``
    carry spec_id/fold_id keys. ``validation_registry`` and ``validation_fits``
    are nested by those keys; native fold effects describe forecasting gaps,
    not policy effects. The policy ATT remains in the main summary.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)
    summary: pd.DataFrame
    paths: pd.DataFrame
    weights: pd.DataFrame
    failures: pd.DataFrame
    warnings: pd.DataFrame
    registry: dict[str, dict[str, Any]]
    baseline: dict[str, Any]
    metadata: dict[str, Any]
    fits: dict[str, VanillaSCResults]

    validation: pd.DataFrame = Field(default_factory=pd.DataFrame)
    validation_paths: pd.DataFrame = Field(default_factory=pd.DataFrame)
    validation_weights: pd.DataFrame = Field(default_factory=pd.DataFrame)
    validation_warnings: pd.DataFrame = Field(default_factory=pd.DataFrame)
    validation_registry: dict[str, dict[str, dict[str, Any]]] = Field(
        default_factory=dict
    )
    validation_fits: dict[str, dict[str, VanillaSCResults]] = Field(
        default_factory=dict
    )

    def dispersion(self, validation_only: bool = False) -> pd.Series:
        """Describe finite estimates, including warning-bearing fits.

        Sign shares have the number of finite estimates as their denominator.
        They describe registered designs and are not causal probabilities.
        Standard deviation is NaN when fewer than two estimates are finite or
        its value exceeds the floating-point range.
        ``validation_only=True`` describes the subset passing the cutoff
        declared through validation_max_rmse. All original tables are retained.
        """
        table = self.summary
        if validation_only:
            if self.metadata.get("validation_max_rmse") is None:
                raise MlsynthConfigError(
                    "Declare validation_max_rmse before fitting to summarize its subset."
                )
            table = table.loc[table.validation_pass]
        effects = table.loc[np.isfinite(table.att), "att"]
        return pd.Series(
            {
                "attempted": len(table),
                "estimated": len(effects),
                "failed": (table.status == "failed").sum(),
                "with_warnings": (table.status == "warning").sum(),
                "min": effects.min(),
                "median": median(effects),
                "max": effects.max(),
                "std": std(effects),
                "positive_share": (effects > 0).mean() if len(effects) else np.nan,
                "negative_share": (effects < 0).mean() if len(effects) else np.nan,
                "zero_share": (effects == 0).mean() if len(effects) else np.nan,
            }
        )

    def by_dimension(self, dimension: str) -> pd.DataFrame:
        """Group dispersion by one design choice, without attributing causality.

        Other dimensions can vary within each group. An unbalanced registry
        does not identify a variance contribution of the selected dimension.
        Failed designs remain in attempted/failed counts.
        """
        if dimension not in _DIMENSIONS:
            raise MlsynthConfigError(f"dimension must be one of {sorted(_DIMENSIONS)}")
        table = self.summary.copy()
        table.loc[~np.isfinite(table.att), "att"] = np.nan
        table["choice"] = [
            repr(self.registry[name].get(dimension)) for name in table.spec_id
        ]
        return (
            table.groupby("choice", sort=False, dropna=False, observed=True)
            .agg(
                attempted=("spec_id", "size"),
                estimated=("att", "count"),
                failed=("status", lambda x: (x == "failed").sum()),
                min=("att", "min"),
                median=("att", lambda values: median(values.dropna())),
                max=("att", "max"),
            )
            .reset_index()
        )

    def plot(self) -> Figure:
        """Return a specification-curve Figure displaying every attempted design."""
        return plot_specification_curve(self.summary)

    def plot_validation(self) -> Figure:
        """Plot ATT against pooled validation RMSE and name unplottable designs."""
        if not self.metadata.get("validation_windows"):
            raise MlsynthConfigError("Declare validation_windows before fitting.")
        return plot_validation(self.summary, self.metadata.get("validation_max_rmse"))
