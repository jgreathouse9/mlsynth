"""Dynamic Synthetic Control for Auto-Regressive processes (DSCAR).

Zheng, X., & Chen, S. X. (2024). *Dynamic synthetic control method for
evaluating treatment effects in auto-regressive processes.* Journal of
the Royal Statistical Society Series B, 86(1):155-176.

DSCAR extends Abadie-Diamond-Hainmueller (2010) to panels with
time-varying confounders, spatial dependence in the residuals, and an
auto-regressive outcome model. The matching weights are computed
**per post-period** via empirical likelihood (equations 2.7-2.9 of the
paper), so the synthetic control tracks both the current covariate
state and the previous-period potential outcome.

Public API: ``DSCAR(config).fit() -> DSCARResults``.

(The acronym ``DSCAR`` is used in mlsynth to distinguish this method
from the Distributional Synthetic Control of Gunsilius (2023), which
ships under :class:`mlsynth.DSC`.)
"""

from __future__ import annotations

from typing import Union

import pandas as pd
from pydantic import ValidationError

from ..config_models import DSCARConfig
from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
    MlsynthPlottingError,
)
from ..utils.dscar_helpers import (
    DSCARResults,
    prepare_dsc_inputs,
    run_dsc,
)


class DSCAR:
    """Dynamic Synthetic Control for Auto-Regressive processes.

    Parameters
    ----------
    config : DSCARConfig or dict
        See :class:`mlsynth.config_models.DSCARConfig`.
    """

    def __init__(self, config: Union[DSCARConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = DSCARConfig(**config)
            except ValidationError as exc:
                raise MlsynthConfigError(
                    f"Invalid DSCAR configuration: {exc}"
                ) from exc
        self.config: DSCARConfig = config
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.treat: str = config.treat
        self.unitid: str = config.unitid
        self.time: str = config.time
        self.exog_covariates = list(config.exog_covariates or [])
        self.lagged_outcome = config.lagged_outcome
        self.placebo_reps: int = config.placebo_reps
        self.el_tolerance: float = config.el_tolerance
        self.fdr_alpha: float = config.fdr_alpha
        self.seed: int = config.seed
        self.display_graphs: bool = config.display_graphs
        self.save = config.save
        self.counterfactual_color = config.counterfactual_color
        self.treated_color: str = config.treated_color

    def fit(self) -> DSCARResults:
        try:
            inputs = prepare_dsc_inputs(
                df=self.df,
                outcome=self.outcome,
                treat=self.treat,
                unitid=self.unitid,
                time=self.time,
                exog_covariates=self.exog_covariates or None,
                lagged_outcome=self.lagged_outcome,
            )
        except MlsynthDataError:
            raise
        except Exception as exc:
            raise MlsynthDataError(
                f"DSCAR: failed to prepare panel ({exc})."
            ) from exc

        try:
            fit = run_dsc(
                inputs,
                el_tolerance=self.el_tolerance,
                placebo_reps=self.placebo_reps,
                do_fdr_test=True,
                fdr_alpha=self.fdr_alpha,
                seed=self.seed,
            )
        except (MlsynthConfigError, MlsynthDataError):
            raise
        except Exception as exc:
            raise MlsynthEstimationError(
                f"DSCAR: estimation failed ({exc})."
            ) from exc

        results = DSCARResults(inputs=inputs, fit=fit, method="dscar")

        if self.display_graphs:
            try:
                from ..utils.dscar_helpers.plotter import plot_dsc
                save_path = self.save if isinstance(self.save, str) else None
                cf_color = (
                    self.counterfactual_color[0]
                    if isinstance(self.counterfactual_color, list)
                    and self.counterfactual_color
                    else "red"
                )
                plot_dsc(
                    results,
                    treated_color=self.treated_color,
                    counterfactual_color=cf_color,
                    save=save_path,
                )
            except MlsynthPlottingError:
                raise
            except Exception as exc:
                raise MlsynthPlottingError(
                    f"DSCAR: plotting failed ({exc})."
                ) from exc

        return results
