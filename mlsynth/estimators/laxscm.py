"""Relaxed/penalized synthetic control (RESCM).

A thin, NumPy-first orchestration over :mod:`mlsynth.utils.laxscm_helpers`.
RESCM is a single convex synthetic-control program that nests a family of
estimators as corner cases, selected by *name*:

* penalized branch -- classic ``SC``, ``LASSO``, ``RIDGE``, ``ENET`` and the
  L-infinity-norm SCM (``LINF`` / ``L1LINF``) of Wang, Xing & Ye (2025):
  ``min ||y0 - mu - Y omega||^2 + P(omega)``.
* relaxation branch -- ``RELAX_L2`` / ``RELAX_ENTROPY`` / ``RELAX_EL``, the
  SCM-relaxation of Liao, Shi & Zheng (2026): keep the simplex and relax the
  exact balance first-order condition to an L-infinity tolerance, then minimise
  an information-theoretic divergence.

Pick estimators with ``methods``; the first one drives the convenience aliases
on the returned :class:`RESCMResults`.
"""

from __future__ import annotations

from typing import List, Union

import pandas as pd

from ..config_models import RESCMConfig
from ..utils.datautils import balance
from ..utils.laxscm_helpers import (
    RESCMResults,
    assemble_rescm_results,
    plot_rescm,
    prepare_rescm_inputs,
    run_rescm,
)


class RESCM:
    """Relaxed/penalized synthetic-control estimator.

    Parameters
    ----------
    config : RESCMConfig or dict
        Validated configuration. Beyond the common fields, RESCM reads
        ``methods`` (which named corner cases to fit), ``alpha`` (CI level),
        ``tau`` / ``n_splits`` / ``n_taus`` (relaxation-branch CV controls), and
        ``solver``.

    References
    ----------
    Liao, C., Shi, Z., & Zheng, Y. (2026). A Relaxation Approach to Synthetic
    Control. arXiv:2508.01793.

    Wang, L., Xing, X., & Ye, Y. (2025). An L-infinity Norm Synthetic Control
    Approach. arXiv:2510.26053.

    Abadie, A., Diamond, A., & Hainmueller, J. (2010). Synthetic control methods
    for comparative case studies. JASA, 105(490), 493-505.

    Doudchenko, N., & Imbens, G. W. (2017). Balancing, Regression,
    Difference-in-Differences and Synthetic Control Methods: A Synthesis.
    arXiv:1610.07748.
    """

    def __init__(self, config: Union[RESCMConfig, dict]) -> None:
        if isinstance(config, dict):
            config = RESCMConfig(**config)
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

    def fit(self) -> RESCMResults:
        """Fit the requested RESCM corner case(s) and return typed results.

        Returns
        -------
        RESCMResults
            Container of per-method
            :class:`~mlsynth.utils.laxscm_helpers.structures.RESCMMethodFit`
            objects (donor weights, counterfactual, gap, ATE, HAC standard
            error, CI, p-value), with convenience aliases (``att``, ``att_se``,
            ``counterfactual``, ``donor_weights``) forwarding to the first
            requested method.
        """
        balance(self.df, self.unitid, self.time)
        inputs = prepare_rescm_inputs(
            self.df, unitid=self.unitid, time=self.time,
            outcome=self.outcome, treat=self.treat,
        )
        fits = run_rescm(
            inputs, self.config.methods,
            tau=self.config.tau, n_splits=self.config.n_splits,
            n_taus=self.config.n_taus, solver=self.config.solver,
            alpha=self.config.alpha,
        )
        results = assemble_rescm_results(
            inputs, fits, selected_variant=self.config.methods[0],
        )

        if self.display_graphs:
            plot_rescm(
                results, outcome=self.outcome, time=self.time,
                treated_color=self.treated_color,
                counterfactual_color=self.counterfactual_color, save=self.save,
            )
        return results
