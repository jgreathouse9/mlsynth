"""Distributional Synthetic Control (DSC) estimator.

DSC (Gunsilius, F. (2023). *"Distributional Synthetic Controls."*
Econometrica 91(3):1105-1117) reconstructs the treated unit's
*outcome distribution* under no treatment as a weighted average of
donor units' outcome distributions, where the average is taken in the
2-Wasserstein space. Unlike classical SCM, which targets aggregate
means, DSC delivers the full counterfactual quantile function and
hence the *quantile treatment effect* at any quantile.

The asymptotic optimality of the DSC weights is established by
Zhang, L., Zhang, X., & Zhang, X. (2026). *"Asymptotic Properties of
the Distributional Synthetic Controls."* arXiv:2405.00953. mlsynth's
implementation follows Algorithm 1 of that paper, which formalises
Gunsilius's recipe with explicit quantile-grid sampling and
simplex-constrained Wasserstein regression on the resulting
pseudo-sample matrices.

Data requirements
-----------------

DSC operates on *micro-level* panel data: for each ``(unit, time)``
cell the user supplies multiple individual observations. The input
DataFrame should therefore have one row per ``(unit, time, individual
observation)`` triple.

When only aggregate-level data are available, classical
synthetic-control estimators (e.g. :class:`mlsynth.CLUSTERSC`,
:class:`mlsynth.FMA`) remain the appropriate tool.

Output
------

The :class:`DSCResults` container surfaces:

* ``donor_weights`` -- the aggregated simplex weights
  :math:`\\widehat w = \\sum_t \\lambda_t \\widehat w_t`.
* ``qte_curves`` -- per-post-period :class:`QTECurve` objects with
  the observed and counterfactual quantile functions plus the QTE.
* ``average_qte`` -- QTE averaged over post-periods.
* ``att`` -- single scalar mean-of-QTE summary, for compatibility
  with the rest of mlsynth.
"""

from __future__ import annotations

from typing import Union

import pandas as pd
from pydantic import ValidationError

from ..config_models import DSCConfig
from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
)
from ..utils.dsc_helpers.pipeline import run_dsc
from ..utils.dsc_helpers.plotter import plot_dsc
from ..utils.dsc_helpers.setup import prepare_dsc_inputs
from ..utils.dsc_helpers.structures import DSCResults


class DSC:
    """Distributional Synthetic Control estimator.

    Parameters
    ----------
    config : DSCConfig or dict
        Configuration object. See
        :class:`mlsynth.config_models.DSCConfig`.
    """

    def __init__(self, config: Union[DSCConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = DSCConfig(**config)
            except ValidationError as exc:
                raise MlsynthConfigError(
                    f"Invalid DSC configuration: {exc}"
                ) from exc
        self.config: DSCConfig = config
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.treat: str = config.treat
        self.unitid: str = config.unitid
        self.time: str = config.time
        self.M = config.M
        self.grid_method: str = config.grid_method
        self.lambda_method: str = config.lambda_method
        self.lambda_decay: float = config.lambda_decay
        self.lambda_weights = config.lambda_weights
        self.qte_quantiles = config.qte_quantiles
        self.n_qte_points: int = config.n_qte_points
        self.random_state: int = config.random_state
        self.display_graphs: bool = config.display_graphs
        self.save = config.save
        self.counterfactual_color = config.counterfactual_color
        self.treated_color: str = config.treated_color

    def fit(self) -> DSCResults:
        """Run Algorithm 1 of Zhang et al. (2026) and return :class:`DSCResults`."""
        try:
            inputs = prepare_dsc_inputs(
                df=self.df,
                outcome=self.outcome,
                treat=self.treat,
                unitid=self.unitid,
                time=self.time,
            )
            results = run_dsc(
                inputs=inputs,
                M=self.M,
                grid_method=self.grid_method,
                lambda_method=self.lambda_method,
                lambda_decay=self.lambda_decay,
                lambda_weights=self.lambda_weights,
                qte_quantiles=self.qte_quantiles,
                n_qte_points=self.n_qte_points,
                random_state=self.random_state,
            )
            if self.display_graphs:
                cf_color = self.counterfactual_color
                if isinstance(cf_color, (list, tuple)):
                    cf_color = cf_color[0] if cf_color else "red"
                plot_dsc(
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
            raise MlsynthEstimationError(
                f"DSC estimation failed: {exc}"
            ) from exc
