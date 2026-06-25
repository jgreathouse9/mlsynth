"""Synthetic Control with Temporal Aggregation (SCTA).

A thin, NumPy-first orchestration over :mod:`mlsynth.utils.scta_helpers`. SCTA
(Sun, Ben-Michael & Feller 2024) builds a single-treated-unit synthetic control
that jointly balances the disaggregated high-frequency pre-period outcomes and
their temporal aggregates -- block means of ``block_length`` consecutive periods
-- trading the two off through a single weight ``nu`` on a fixed diagonal ``V``.
Aggregating reduces noise in the balancing objective (tighter bias when long-run
signal survives); the joint fit at ``nu=0.5`` is the paper's compromise.

The simplex is solved at the *true* optimum of the temporal-aggregation
objective (mlsynth's active-set QP); ``augment="ridge"`` adds the bilevel
ridge-augmented correction (Augmented SCM). An optional ``frontier`` grid
traces the disaggregated-vs-aggregated imbalance frontier (the paper's Fig. 1).
"""

from __future__ import annotations

from typing import List, Union

import pandas as pd

from ..config_models import SCTAConfig
from ..utils.datautils import balance
from ..utils.scta_helpers import SCTAResults, plot_scta, prepare_scta_inputs, run_scta


class SCTA:
    """Synthetic Control with Temporal Aggregation estimator.

    Parameters
    ----------
    config : SCTAConfig or dict
        Validated configuration. Beyond the common fields (``df``, ``outcome``,
        ``treat``, ``unitid``, ``time``, ``display_graphs``, ``save``, colors),
        SCTA reads ``block_length`` (``K``), ``nu``, ``augment``,
        ``ridge_lambda``, ``demean`` and ``frontier``.

    References
    ----------
    Sun, L., Ben-Michael, E., & Feller, A. (2024). Temporal Aggregation for the
    Synthetic Control Method. AEA Papers and Proceedings, 114: 614-617.
    """

    def __init__(self, config: Union[SCTAConfig, dict]) -> None:
        if isinstance(config, dict):
            config = SCTAConfig(**config)
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

    def fit(self) -> SCTAResults:
        """Aggregate, fit the joint SC, and return standardized results.

        Returns
        -------
        SCTAResults
            Standardized results with the headline fit lifted into the shared
            sub-models and an optional imbalance frontier in ``frontier``.
        """
        balance(self.df, self.unitid, self.time)

        inputs = prepare_scta_inputs(
            self.df, unitid=self.unitid, time=self.time, outcome=self.outcome,
            treat=self.treat, block_length=self.config.block_length,
        )
        fit, frontier = run_scta(
            inputs, nu=self.config.nu, augment=self.config.augment,
            ridge_lambda=self.config.ridge_lambda, demean=self.config.demean,
            conformal_alpha=self.config.conformal_alpha,
            frontier=self.config.frontier,
        )
        results = SCTAResults(inputs=inputs, fit=fit, frontier=frontier)

        if self.display_graphs:
            plot_scta(
                results, outcome=self.outcome, time=self.time,
                treated_color=self.treated_color,
                counterfactual_color=self.counterfactual_color, save=self.save,
            )
        return results
