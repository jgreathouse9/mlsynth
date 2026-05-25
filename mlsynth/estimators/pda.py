"""Panel Data Approach (PDA) for program evaluation.

A thin, NumPy-first orchestration over :mod:`mlsynth.utils.pda_helpers`. PDA
(Hsiao, Ching & Wan 2012) predicts a single treated unit's untreated
counterfactual by a linear regression on the control units fit over the
pre-treatment window, then extrapolates out-of-sample to estimate the ATE.
Three high-dimensional variants are supported, each with the inference theory
from its own paper:

* ``l2``    -- L2-relaxation (Shi & Wang 2024): dense ridge-like coefficients
  via ``min ||beta||_2^2 s.t. ||eta_hat - Sigma_hat beta||_inf <= tau``; ATE
  inference combines pre-residual and post-effect HAC long-run variances.
* ``LASSO`` -- L1 (Li & Bell 2017): sparse donor selection by cross-validated
  LASSO; HAC t-test on the ATE with a first-stage variance term.
* ``fs``    -- forward selection (Shi & Huang 2023): greedy R^2 selection with
  a modified-BIC stopping rule; post-selection HAC t-test (sample splitting).

Set ``method`` for one variant or ``methods`` to run several at once.
"""

from __future__ import annotations

from typing import List, Union

import pandas as pd

from ..config_models import PDAConfig
from ..utils.datautils import balance
from ..utils.pda_helpers import (
    PDAResults,
    assemble_pda_results,
    plot_pda,
    prepare_pda_inputs,
    resolve_methods,
    run_pda,
)


class PDA:
    """Panel Data Approach estimator (l2 / LASSO / fs).

    Parameters
    ----------
    config : PDAConfig or dict
        Validated configuration. Beyond the common fields, PDA reads
        ``method`` / ``methods`` (which variant(s) to run), ``tau`` (the l2
        relaxation parameter; ``None`` selects it by validation), and
        ``alpha`` (CI level).

    References
    ----------
    Hsiao, C., Ching, H. S., & Wan, S. K. (2012). A panel data approach for
    program evaluation. Journal of Applied Econometrics, 27(5), 705-740.

    Shi, Z., & Wang, Y. (2024). L2-relaxation for Economic Prediction.

    Li, K. T., & Bell, D. R. (2017). Estimation of average treatment effects
    with panel data. Journal of Econometrics, 197(1), 65-75.

    Shi, Z., & Huang, J. (2023). Forward-selected panel data approach for
    program evaluation. Journal of Econometrics, 234(2), 512-535.
    """

    def __init__(self, config: Union[PDAConfig, dict]) -> None:
        if isinstance(config, dict):
            config = PDAConfig(**config)
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

    def fit(self) -> PDAResults:
        """Run the requested PDA variant(s) and return typed results.

        Returns
        -------
        PDAResults
            Container of per-variant
            :class:`~mlsynth.utils.pda_helpers.structures.PDAMethodFit` objects
            (coefficients, counterfactual, gap, ATE, HAC standard error, CI,
            p-value, donor weights), with convenience aliases (``att``,
            ``att_se``, ``counterfactual``, ``donor_weights``) forwarding to the
            primary variant.
        """
        balance(self.df, self.unitid, self.time)
        inputs = prepare_pda_inputs(
            self.df, unitid=self.unitid, time=self.time,
            outcome=self.outcome, treat=self.treat,
        )
        methods = resolve_methods(self.config.method, self.config.methods)
        fits = run_pda(inputs, methods, tau=self.config.tau, alpha=self.config.alpha)
        results = assemble_pda_results(inputs, fits, selected_variant=methods[0])

        if self.display_graphs:
            plot_pda(
                results, outcome=self.outcome, time=self.time,
                treated_color=self.treated_color,
                counterfactual_color=self.counterfactual_color, save=self.save,
            )
        return results
