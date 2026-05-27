"""Synthetic Interventions (SI) estimator.

Implements:

    Agarwal, A., Shah, D., & Shen, D. (2026). "Synthetic Interventions:
    Extending Synthetic Controls to Multiple Treatments." Operations Research
    74(2):840-859.

SI extends synthetic control to *multiple* interventions. For a focal target
unit, it estimates the counterfactual outcome the unit would have realised
under each alternative intervention ``d`` by:

* regressing the target's pre-treatment (control) outcomes onto the
  pre-treatment outcomes of the units that actually received ``d`` (the donor
  pool ``I(d)``), then
* applying those weights to the donor pool's *post-treatment* outcomes under
  ``d`` to predict the target's counterfactual under ``d``.

The default variant is **bias-corrected SI-PCR** (Section 4.3): the donor
pre-matrix is denoised by rank-``k`` HSVT, weights are fit on a rank-complete
donor subset, and an asymptotic-normality confidence interval is reported.
"""

from __future__ import annotations

from typing import Union

import pandas as pd
from pydantic import ValidationError

from ..config_models import SIConfig
from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
    MlsynthPlottingError,
)
from ..utils.datautils import balance
from ..utils.si_helpers.orchestration import solve_si
from ..utils.si_helpers.plotter import plot_si
from ..utils.si_helpers.setup import prepare_si_inputs
from ..utils.si_helpers.structures import SIResults


class SI:
    """Synthetic Interventions (SI) estimator.

    Parameters
    ----------
    config : SIConfig or dict
        Configuration object. See :class:`mlsynth.config_models.SIConfig`.

    Returns
    -------
    SIResults
        Per-intervention donor weights, counterfactuals, ATTs, and (with the
        default bias-corrected estimator) asymptotic-normality confidence
        intervals.
    """

    def __init__(self, config: Union[SIConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = SIConfig(**config)
            except ValidationError as exc:
                raise MlsynthConfigError(
                    f"Invalid SI configuration: {exc}"
                ) from exc

        self.config: SIConfig = config
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.unitid: str = config.unitid
        self.time: str = config.time
        self.treat: str = config.treat
        self.inters = list(config.inters)

        for col in self.inters:
            if col not in self.df.columns:
                raise MlsynthConfigError(
                    f"Intervention column '{col}' not found in dataframe."
                )

        self.rank_method: str = config.rank_method
        self.rank = config.rank
        self.cumvar_threshold: float = config.cumvar_threshold
        self.bias_correct: bool = config.bias_correct
        self.variance: str = config.variance
        self.interval: str = config.interval
        self.alpha: float = config.alpha
        self.display_graphs: bool = config.display_graphs

    def fit(self) -> SIResults:
        """Run the SI pipeline over all alternative interventions."""
        try:
            balance(self.df, self.unitid, self.time)
        except MlsynthDataError:
            raise
        except Exception as exc:
            raise MlsynthDataError(f"Error balancing panel data: {exc}") from exc

        try:
            inputs = prepare_si_inputs(
                df=self.df,
                outcome=self.outcome,
                unitid=self.unitid,
                time=self.time,
                treat=self.treat,
                inters=self.inters,
            )
        except (MlsynthDataError, MlsynthConfigError):
            raise
        except Exception as exc:
            raise MlsynthDataError(f"Error preparing SI inputs: {exc}") from exc

        try:
            results = solve_si(
                inputs=inputs,
                rank_method=self.rank_method,
                rank=self.rank,
                cumvar_threshold=self.cumvar_threshold,
                bias_correct=self.bias_correct,
                alpha=self.alpha,
                variance=self.variance,
                interval=self.interval,
            )
        except (MlsynthConfigError, MlsynthDataError, MlsynthEstimationError):
            raise
        except Exception as exc:
            raise MlsynthEstimationError(f"SI estimation failed: {exc}") from exc

        if self.display_graphs:
            try:
                plot_si(results)
            except MlsynthPlottingError:
                raise
            except Exception as exc:
                raise MlsynthPlottingError(f"SI plotting failed: {exc}") from exc

        return results
