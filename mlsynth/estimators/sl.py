"""Synthetic learner (SL) for program evaluation.

A thin orchestration over :mod:`mlsynth.utils.sl_helpers`. SL is Viviano and
Bradic (2023): instead of committing to one way of predicting the treated unit's
counterfactual, it keeps a library of predictors and combines them by exponential
weights fit on a slice of the pre-period the predictors never saw.

Four steps:

1. split the pre-treatment window into an expert-training part and a weighting
   part (Algorithm 1);
2. fit every expert on the first part and predict over all periods;
3. weight the experts by ``exp(-eta * cumulative squared loss)`` on the second
   part, which is out of sample for them (Equations 11-12);
4. difference the weighted prediction against the observed path, remove the gap
   the ensemble leaves in sample (Equation 10), and test the no-effect null with
   a moving-block bootstrap (Algorithm 2, size controlled by Theorem 3.1).

What SL supplies is a test and a point estimate. It supplies no standard error
and no confidence interval, and neither does the paper, so ``att_std_err`` is
``None`` and the bootstrap quantiles are reported as critical values, which is
what they are. An interval for SL's estimate would have to come from inverting
SL's own test over a grid of candidate effects. mlsynth's
``conformal_att_interval`` will not supply it: that function refits a ridge on a
donor design, so what it returns is an interval for a different estimator's point
estimate. The inversion is separate work.

Three diagnostics the paper does not report are on the result, because measuring
them changes what an SL number means. ``effective_k`` says whether the weighting
selected an expert or averaged the library -- at the paper's own ``eta`` it
averages, 3.88 of 4. ``error_participation_ratio`` says whether the members err
independently, which is the condition for averaging to help -- measured at 1.03
to 1.08 of 4 on two panels, they do not. ``degenerate_experts`` names a member
that is constant on the weighting window, or fitting an order of magnitude worse
than the best while still carrying weight.
"""

from __future__ import annotations

from typing import Union

import pandas as pd

from ..config_models import SLConfig
from ..utils.datautils import balance
from ..utils.sl_helpers import SLResults, plot_sl, prepare_sl_inputs, run_sl


class SL:
    """Synthetic learner estimator.

    Parameters
    ----------
    config : SLConfig or dict
        Validated configuration. Beyond the common fields, SL reads ``experts``
        (the library), ``covariates`` (read by the forest expert), ``eta`` and
        ``train_periods`` (the weighting), ``post_skip`` (which post window to
        measure), ``n_boot`` / ``block`` / ``seed`` (the bootstrap) and
        ``alpha``.

    Examples
    --------
    >>> from mlsynth import SL                                 # doctest: +SKIP
    >>> res = SL({                                             # doctest: +SKIP
    ...     "df": panel, "outcome": "y", "treat": "D",
    ...     "unitid": "unit", "time": "time",
    ... }).fit()
    >>> res.fit.effective_k                                    # doctest: +SKIP
    3.88

    References
    ----------
    Viviano, D., & Bradic, J. (2023). Synthetic learner: model evaluation with
    limited overlap. Journal of Econometrics, 234(2), 691-713.
    """

    def __init__(self, config: Union[SLConfig, dict]) -> None:
        if isinstance(config, dict):
            config = SLConfig(**config)
        self.config = config

    def fit(self) -> SLResults:
        """Run SL and return the standardized result.

        Returns
        -------
        SLResults

        Raises
        ------
        MlsynthDataError
            If the panel is unbalanced, has no donors, has fewer than two
            pre-treatment periods, or names a covariate it does not carry.
        MlsynthConfigError
            If the split leaves no weighting window, or ``post_skip`` consumes
            the whole post window.
        """
        c = self.config
        balance(c.df, c.unitid, c.time)
        inputs = prepare_sl_inputs(
            c.df, unitid=c.unitid, time=c.time, outcome=c.outcome,
            treat=c.treat, covariates=c.covariates)
        fit = run_sl(
            inputs, experts=tuple(c.experts), train_periods=c.train_periods,
            eta=c.eta, post_skip=c.post_skip, n_boot=c.n_boot, block=c.block,
            alpha=c.alpha, seed=c.seed,
            lasso_folds=c.lasso_folds, factor_rank=c.factor_rank,
            forest_trees=c.forest_trees,
            forest_max_leaf_nodes=c.forest_max_leaf_nodes)
        results = SLResults(inputs=inputs, fit=fit)
        if c.display_graphs:
            plot_sl(results, outcome=c.outcome, time=c.time,
                    treated_color=c.treated_color,
                    counterfactual_color=c.counterfactual_color)
        return results
