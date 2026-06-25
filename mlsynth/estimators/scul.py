"""Synthetic Control Using Lasso (SCUL).

A thin, NumPy-first orchestration over :mod:`mlsynth.utils.scul_helpers`. SCUL
(Hollingsworth & Wing 2022) builds a single-treated-unit synthetic control by a
**lasso** regression of the treated unit's pre-treatment outcome on a
high-dimensional, multi-type donor pool. The weights are unrestricted (negative
weights and an intercept -- extrapolation -- are allowed), so the donor pool may
exceed the pre-period; the lasso penalty is chosen by a rolling-origin
expanding-window cross-validation that respects the time ordering. Inference is
a placebo permutation test on the unit-free post/pre RMSE ratio.

This is a port of the reference R package ``github.com/hollina/scul``, validated
value-for-value on the California (Proposition 99) panel: the rolling-CV penalty
matches ``glmnet`` to ten digits, and -- since the lasso solution is unique for
continuous donors (Tibshirani 2013) -- the weights and synthetic series agree to
solver tolerance.
"""

from __future__ import annotations

from typing import List, Union

import pandas as pd

from ..config_models import SCULConfig
from ..utils.datautils import balance
from ..utils.scul_helpers import (
    SCULResults,
    prepare_scul_inputs,
    run_scul,
)
from ..utils.scul_helpers.plotter import plot_scul


class SCUL:
    """Synthetic Control Using Lasso estimator.

    Parameters
    ----------
    config : SCULConfig or dict
        Validated configuration. Beyond the common fields (``df``, ``outcome``,
        ``treat``, ``unitid``, ``time``, ``display_graphs``, ``save``, colors),
        SCUL reads ``donor_variables`` (extra panel columns widening the pool),
        ``number_initial_periods``, ``training_post_length``, ``cv_option``,
        ``cohensd_threshold`` and ``inference``.

    References
    ----------
    Hollingsworth, A., & Wing, C. (2022). Tactics for design and inference in
    synthetic control studies: An applied example using high-dimensional data.
    """

    def __init__(self, config: Union[SCULConfig, dict]) -> None:
        if isinstance(config, dict):
            config = SCULConfig(**config)
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

    def fit(self) -> SCULResults:
        """Build the lasso synthetic control and return standardized results.

        Returns
        -------
        SCULResults
            Standardized results: the synthetic series and gap in
            ``time_series``, the ATT in ``effects``, the selected donor columns
            in ``weights``, the unit-free Cohen's D pre-fit in
            ``fit_diagnostics``, and (when ``inference``) the placebo p-value in
            ``inference``.
        """
        balance(self.df, self.unitid, self.time)
        inputs = prepare_scul_inputs(
            self.df, unitid=self.unitid, time=self.time, outcome=self.outcome,
            treat=self.treat, donor_variables=self.config.donor_variables,
        )
        results = run_scul(inputs, self.config)

        if self.display_graphs:
            plot_scul(
                results, outcome=self.outcome, time=self.time,
                treated_color=self.treated_color,
                counterfactual_color=self.counterfactual_color, save=self.save,
            )
        return results
