"""Forward-Selected Synthetic Control (FSCM).

A thin, NumPy-first orchestration over :mod:`mlsynth.utils.fscm_helpers`.
FSCM (Cerulli 2024) treats the number of donors as a complexity parameter
governing a bias--variance trade-off. A forward stepwise selection grows a
nested donor sequence on the training half of the pre-period (greedy on
in-sample RMSPE), and a two-interval-time out-of-sample validation on the
held-out test half picks the donor count that minimizes test RMSPE. The final
simplex weights are refit on the full pre-period over the selected donors.

Optional covariate matching (``covariates=[...]``) augments the SCM objective
with the author's predictor specification; selection and validation scores are
always measured on the outcome.

References
----------
Cerulli, G. (2024). Optimal initial donor selection for the synthetic control
method. Economics Letters, 244, 111976.
https://doi.org/10.1016/j.econlet.2024.111976
"""

from __future__ import annotations

from typing import List, Union

import pandas as pd

from ..config_models import FSCMConfig
from ..utils.datautils import balance
from ..utils.fscm_helpers import (
    FSCMResults,
    plot_fscm,
    prepare_fscm_inputs,
    run_fscm,
)


class FSCM:
    """Forward-Selected Synthetic Control estimator.

    Parameters
    ----------
    config : FSCMConfig or dict
        Validated configuration. Beyond the common fields (``df``, ``outcome``,
        ``treat``, ``unitid``, ``time``, ``display_graphs``, ``save``, colors),
        FSCM reads ``covariates`` (optional predictor columns),
        ``match_periods`` (specific pre-treatment periods whose outcome value
        is matched directly, as in Abadie's special predictors), ``cv_split``
        (training fraction of the pre-period) and ``max_donors`` (cap on
        forward-selection steps).
    """

    def __init__(self, config: Union[FSCMConfig, dict]) -> None:
        if isinstance(config, dict):
            config = FSCMConfig(**config)
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
        self.forward_selection: bool = config.forward_selection
        self.covariates: List[str] = list(config.covariates or [])
        self.covariate_windows: dict = dict(config.covariate_windows or {})
        self.match_periods: list = list(config.match_periods or [])
        self.cv_split: float = config.cv_split
        self.max_donors = config.max_donors

    def fit(self) -> FSCMResults:
        """Select donors, fit the synthetic control, and return results.

        Returns
        -------
        FSCMResults
            Container with the optimal donor set, simplex weights,
            counterfactual, gap, ATT, fit diagnostics, and the
            forward-selection / cross-validation path.
        """
        balance(self.df, self.unitid, self.time)

        inputs = prepare_fscm_inputs(
            self.df,
            unitid=self.unitid,
            time=self.time,
            outcome=self.outcome,
            treat=self.treat,
            covariates=self.covariates,
            covariate_windows=self.covariate_windows,
            match_periods=self.match_periods,
        )

        results = run_fscm(
            inputs,
            forward_selection=self.forward_selection,
            cv_split=self.cv_split,
            max_donors=self.max_donors,
        )

        if self.display_graphs:
            plot_fscm(
                results, outcome=self.outcome, time=self.time,
                treated_color=self.treated_color,
                counterfactual_color=self.counterfactual_color, save=self.save,
            )
        return results
