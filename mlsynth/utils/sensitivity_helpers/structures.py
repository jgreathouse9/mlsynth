"""Typed collections of native fits and extracted sensitivity trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from pydantic import BaseModel, ConfigDict

from ...exceptions import MlsynthConfigError
from ..vanillasc_helpers.structures import VanillaSCResults
from .plotter import plot_specification_curve

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

    def dispersion(self) -> pd.Series:
        """Describe finite estimates, including warning-bearing fits.

        Sign shares have the number of finite estimates as their denominator.
        They describe registered designs and are not causal probabilities.
        """
        effects = self.summary.att.dropna()
        return pd.Series(
            {
                "attempted": len(self.summary),
                "estimated": len(effects),
                "failed": (self.summary.status == "failed").sum(),
                "with_warnings": (self.summary.status == "warning").sum(),
                "min": effects.min(),
                "median": effects.median(),
                "max": effects.max(),
                "std": effects.std(),
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
                median=("att", "median"),
                max=("att", "max"),
            )
            .reset_index()
        )

    def plot(self) -> Figure:
        """Return a specification-curve Figure displaying every attempted design."""
        return plot_specification_curve(self.summary)
