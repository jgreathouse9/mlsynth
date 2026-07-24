"""Distributionally Robust Synthetic Control (DROSC).

Koo, T. & Guo, Z. (2026). "Distributionally Robust Synthetic Control: Ensuring
Robustness Against Highly Correlated Controls and Weight Shifts." arXiv:2511.02632.

DROSC targets a worst-case treatment effect over the donor weights compatible
with the pre-treatment moments up to a robustness radius ``robustness_lambda``.
When the classical synthetic-control identification holds it recovers the same
effect; when it fails (highly correlated controls, or a shift in the treated-
control relationship) it is a conservative proxy for the non-identifiable effect.
Optional perturbation-based inference returns a (possibly disjoint) union
confidence interval whose limiting law is non-normal.
"""
from __future__ import annotations

from typing import Union

import pandas as pd

from ..config_models import BaseEstimatorResults, DROSCConfig
from ..utils.drosc_helpers import plot_drosc, run_drosc


class DROSC:
    """Distributionally Robust Synthetic Control estimator.

    Parameters
    ----------
    config : DROSCConfig or dict
        Configuration object. See :class:`mlsynth.config_models.DROSCConfig`.
    """

    def __init__(self, config: Union[DROSCConfig, dict]) -> None:
        if isinstance(config, dict):
            config = DROSCConfig(**config)
        self.config = config
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.treat: str = config.treat
        self.unitid: str = config.unitid
        self.time: str = config.time
        self.display_graphs: bool = config.display_graphs
        self.save = config.save
        self.counterfactual_color = config.counterfactual_color
        self.treated_color = config.treated_color

    def fit(self) -> BaseEstimatorResults:
        """Estimate the DROSC effect (and, if requested, its union CI)."""
        results = run_drosc(self.config)
        if self.display_graphs:
            plot_drosc(
                results,
                outcome=self.outcome,
                time=self.time,
                treated_color=self.treated_color,
                counterfactual_color=self.counterfactual_color,
                save=self.save,
            )
        return results
