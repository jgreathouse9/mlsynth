"""Ensemble Synthetic Control (ESC).

A thin, NumPy-first orchestration over :mod:`mlsynth.utils.esc_helpers`. ESC
(Lian & Pu 2026) targets synthetic-control inference under a short pre-treatment
panel, where a single specification extrapolates unstably. It aggregates a
collection of low-complexity LASSO base learners, each fit on a block-subsampled
slice of the pre-period (respecting temporal dependence) crossed with a random
donor subspace, and reads pointwise prediction intervals directly off the
empirical quantiles of the ensemble counterfactuals -- treating SC inference as a
small-sample prediction problem rather than a parametric one.

Demonstrated on California's Proposition 99: the ESC counterfactual reproduces
the classic synthetic-control path (corr ~0.998) while its prediction interval
stays informative and non-truncated under the short panel.
"""

from __future__ import annotations

from typing import List, Union

import pandas as pd
from pydantic import ValidationError

from ..config_models import ESCConfig
from ..exceptions import MlsynthConfigError
from ..utils.datautils import balance
from ..utils.esc_helpers import ESCResults, prepare_esc_inputs, run_esc
from ..utils.esc_helpers.plotter import plot_esc


class ESC:
    """Ensemble Synthetic Control estimator.

    Parameters
    ----------
    config : ESCConfig or dict
        Validated configuration. Beyond the common fields (``df``, ``outcome``,
        ``treat``, ``unitid``, ``time``, ``display_graphs``, ``save``, colors),
        ESC reads ``n_learners``, ``block_length``, ``subspace_ratio``,
        ``block_keep``, ``alpha`` (the prediction-interval level), ``lambda_rule``
        / ``lambda_scope`` (LASSO penalty selection), ``aggregate`` and ``seed``.

    References
    ----------
    Lian, Y., & Pu, J. (2026). Inference of Synthetic Control Method under Short
    Panel. (Ensemble Synthetic Control.)
    """

    def __init__(self, config: Union[ESCConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = ESCConfig(**config)
            except ValidationError as exc:
                raise MlsynthConfigError(f"Invalid ESC configuration: {exc}") from exc
        self.config = config
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.treat: str = config.treat
        self.unitid: str = config.unitid
        self.time: str = config.time
        self.display_graphs: bool = config.display_graphs
        self.save: Union[bool, str, dict] = config.save
        self.counterfactual_color: Union[str, List[str]] = config.counterfactual_color
        self.treated_color: str = config.treated_color

    def fit(self) -> ESCResults:
        """Fit the ensemble and return standardized results.

        Returns
        -------
        ESCResults
            Standardized results: the ensemble counterfactual and gap plus the
            pointwise prediction interval in ``time_series``, the ATT in
            ``effects`` with its ensemble interval in ``inference``, the mean
            donor contributions in ``weights``, and the pre-treatment RMSE in
            ``fit_diagnostics``.
        """
        balance(self.df, self.unitid, self.time)
        inputs = prepare_esc_inputs(
            self.df, unitid=self.unitid, time=self.time,
            outcome=self.outcome, treat=self.treat,
        )
        results = run_esc(inputs, self.config)

        if self.display_graphs:
            plot_esc(
                results, outcome=self.outcome, time=self.time,
                treated_color=self.treated_color,
                counterfactual_color=self.counterfactual_color, save=self.save,
            )
        return results
