"""Covariate-adjusted panel data approach (CPDA) for program evaluation.

A thin orchestration over :mod:`mlsynth.utils.cpda_helpers`. CPDA is Hsiao and
Zhou (2019) Section 3, the semiparametric middle of their three constructions.

The parametric route models the factor structure and needs both ``N`` and ``T``
large to estimate it. The panel data approach models nothing and regresses the
treated unit on donor outcomes. CPDA removes the covariate part of the outcome
first, then runs the donor regression on what is left, so the donors have to
carry only the factor part and the number of unobserved factors never has to be
chosen.

Four steps, Equations 11 to 15:

1. estimate the covariate slope ``beta`` on the pre-period, by Pesaran's (2006)
   common correlated effects or Bai's (2009) interactive fixed effects;
2. residualise, ``v_t = y_t - X_t beta``, for the treated unit and every donor;
3. choose an intercept and donor weights minimising the squared pre-period
   error on a subset of those residuals;
4. predict ``y^0_1t = x'_1t beta + w' v*_t + mu`` and difference against the
   observed series.

The subset in step 3 is a choice the paper leaves open, and the estimate moves
with it, so ``selector`` is explicit and ``sensitivity=True`` reports the
spread across every rule available. See :class:`~mlsynth.config_models.CPDAConfig`.
"""

from __future__ import annotations

from typing import List, Union

import pandas as pd

from ..config_models import CPDAConfig
from ..utils.cpda_helpers import (
    CPDAResults,
    plot_cpda,
    prepare_cpda_inputs,
    run_cpda,
)
from ..utils.datautils import balance


class CPDA:
    """Covariate-adjusted panel data approach estimator.

    Parameters
    ----------
    config : CPDAConfig or dict
        Validated configuration. Beyond the common fields, CPDA reads
        ``covariates`` (required), ``beta_method`` and ``r`` (the slope step),
        ``selector`` and ``standardize_selection`` (the donor subset),
        ``sensitivity`` (report the spread across selectors), ``alpha``,
        ``seed`` and ``lrvar_lag``.

    Examples
    --------
    >>> from mlsynth import CPDA                              # doctest: +SKIP
    >>> res = CPDA({                                          # doctest: +SKIP
    ...     "df": panel, "outcome": "y", "treat": "D",
    ...     "unitid": "unit", "time": "year",
    ...     "covariates": ["lnincome", "poverty"],
    ...     "sensitivity": True,
    ... }).fit()
    >>> res.effects.att                                       # doctest: +SKIP

    References
    ----------
    Hsiao, C., & Zhou, Q. (2019). Panel parametric, semiparametric, and
    nonparametric construction of counterfactuals. Journal of Applied
    Econometrics, 34(4), 463-481.

    Pesaran, M. H. (2006). Estimation and inference in large heterogeneous
    panels with a multifactor error structure. Econometrica, 74(4), 967-1012.

    Bai, J. (2009). Panel data models with interactive fixed effects.
    Econometrica, 77(4), 1229-1279.
    """

    def __init__(self, config: Union[CPDAConfig, dict]) -> None:
        if isinstance(config, dict):
            config = CPDAConfig(**config)
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

    def fit(self) -> CPDAResults:
        """Run CPDA and return typed results.

        Returns
        -------
        CPDAResults
            An ``EffectResult`` whose standardized sub-models carry the ATT,
            its HAC interval, the counterfactual and gap paths, the donor
            coefficients and the pre-period fit, with the slope, the selector,
            the kept donors and any sensitivity sweep on ``fit``.
        """
        balance(self.df, self.unitid, self.time)
        inputs = prepare_cpda_inputs(
            self.df, unitid=self.unitid, time=self.time, outcome=self.outcome,
            treat=self.treat, covariates=self.config.covariates,
        )
        fit = run_cpda(
            inputs,
            beta_method=self.config.beta_method,
            r=self.config.r,
            selector=self.config.selector,
            standardize_selection=self.config.standardize_selection,
            alpha=self.config.alpha,
            seed=self.config.seed,
            lrvar_lag=self.config.lrvar_lag,
            sensitivity=self.config.sensitivity,
        )
        results = CPDAResults(inputs=inputs, fit=fit)

        if self.display_graphs:
            plot_cpda(results, outcome=self.outcome, time=self.time,
                      treated_color=self.treated_color,
                      counterfactual_color=self.counterfactual_color,
                      save=self.save)
        return results
