"""Synthetic Control with Differencing (SCD) estimator.

SCD (Rincon & Song 2026, *"Synthetic Control with Differencing"*,
arXiv:2510.26106; repeated-cross-section inference from Canen & Song 2025)
targets settings where the treated and control "units" are *groups* observed
through repeated cross-sections of individuals -- states in a labour survey,
say -- rather than a single aggregated series each.

The estimator collapses the microdata to survey-weighted group means, applies a
within-group differencing (the "D": off the last pre-period, off the pre-period
average, or none), and fits a simplex synthetic control on the differenced
pre-period. It reports the per-period effect path as an event study. Because
the group means are estimated from individuals, the individual rows re-enter as
influence functions to give :math:`\\sqrt{n}` (number of individuals),
fixed-``T`` confidence bands -- a genuine standard-error band, together with a
weight confidence set for the counterfactuals the data cannot rule out --
rather than the coarse donor-permutation inference of classical SCM.

Data requirements
-----------------

SCD operates on grouped microdata: one row per individual observation, with an
optional survey-weight column. The input is treated as repeated cross-sections
(individuals are not tracked across periods). Treatment is applied at the
unit-time level via the ``treat`` column.

Output
------

:meth:`SCD.fit` returns a :class:`~mlsynth.config_models.BaseEstimatorResults`
whose ``estimated_gap`` is the effect path :math:`\\hat\\theta_t`, ``att`` its
post-period mean, and (when ``compute_inference``) whose ``inference.details``
carry the per-period lower/upper bands and the standard errors.
"""

from __future__ import annotations

from typing import Union

import pandas as pd
from pydantic import ValidationError

from ..config_models import BaseEstimatorResults, SCDConfig
from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
)
from ..utils.scd_helpers.pipeline import run_scd
from ..utils.scd_helpers.plotter import plot_scd
from ..utils.scd_helpers.setup import prepare_scd_inputs


class SCD:
    """Synthetic Control with Differencing estimator.

    Parameters
    ----------
    config : SCDConfig or dict
        Configuration object. See
        :class:`mlsynth.config_models.SCDConfig`.
    """

    def __init__(self, config: Union[SCDConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = SCDConfig(**config)
            except ValidationError as exc:
                raise MlsynthConfigError(f"Invalid SCD configuration: {exc}") from exc
        self.config: SCDConfig = config
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.treat: str = config.treat
        self.unitid: str = config.unitid
        self.time: str = config.time
        self.weight_col = config.weight_col
        self.differencing: str = config.differencing
        self.compute_inference: bool = config.compute_inference
        self.alpha: float = config.alpha
        self.kappa: float = config.kappa
        self.n_grid: int = config.n_grid
        self.grid_radius: float = config.grid_radius
        self.tolerance: float = config.tolerance
        self.random_state: int = config.random_state
        self.display_graphs: bool = config.display_graphs
        self.save = config.save
        self.treated_color: str = config.treated_color
        self.counterfactual_color = config.counterfactual_color

    def fit(self) -> BaseEstimatorResults:
        """Run SCD and return :class:`BaseEstimatorResults`."""
        try:
            inputs = prepare_scd_inputs(
                df=self.df,
                outcome=self.outcome,
                treat=self.treat,
                unitid=self.unitid,
                time=self.time,
                weight_col=self.weight_col,
            )
            results = run_scd(
                inputs=inputs,
                differencing=self.differencing,
                compute_inference=self.compute_inference,
                alpha=self.alpha,
                kappa=self.kappa,
                n_grid=self.n_grid,
                grid_radius=self.grid_radius,
                tolerance=self.tolerance,
                random_state=self.random_state,
                plot_config=self.config.resolved_plot(),
            )
            if self.display_graphs:
                cf_color = self.counterfactual_color
                if isinstance(cf_color, (list, tuple)):
                    cf_color = cf_color[0] if cf_color else "C0"
                plot_scd(
                    results,
                    treated_color=self.treated_color,
                    counterfactual_color=cf_color,
                    save=self.save,
                    outcome_label=self.outcome,
                )
            return results
        except (MlsynthConfigError, MlsynthDataError, MlsynthEstimationError):
            raise
        except Exception as exc:
            raise MlsynthEstimationError(f"SCD estimation failed: {exc}") from exc
