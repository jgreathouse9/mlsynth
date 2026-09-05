"""Distributionally Robust Synthetic Control (DROSC).

Koo, T. & Guo, Z. (2026). "Distributionally Robust Synthetic Control: Ensuring
Robustness Against Highly Correlated Controls and Weight Shifts." arXiv:2511.02632.

Classical synthetic control needs two things its usual argument leaves implicit:
(E1) the pre-treatment weights are unique, and (E2) the weighting that describes
the treated unit before the intervention still describes it after. (E1) fails
when the controls are highly correlated, so many weightings fit the pre-period
indistinguishably. (E2) fails when the intervention alters the treated-control
relationship, and also when the distribution of the controls shifts, since the
weights are projections onto them. Either way the post-treatment weighting is
known only to lie in a set.

DROSC keeps the set. ``Omega(lambda)`` holds every simplex weighting satisfying
the pre-treatment moment condition to within ``robustness_lambda``, and the
reported effect is the max-min of a reward function over it: the analyst names an
effect, nature picks the plausible weighting that makes the claim look worst.
That game solves (Theorem 1) to the weighting driving the effect closest to zero,
so ``tau*`` is the projection of the origin onto the interval of effects the set
admits (Theorem 2) -- the conservative endpoint of a sensitivity analysis, chosen
instead of being left as an interval.

Reading the number. When the post-treatment weight lies in the set,
``|tau*| <= |taubar|`` and ``tau*`` cannot take the opposite sign to the true
effect (Theorem 3), so the magnitude is a lower bound and the direction is
trustworthy. Under (E1) and (E2) at ``robustness_lambda = 0`` the two coincide.
Report the effect as a function of the radius: where it first reaches zero is the
weight shift at which the compatible effects first include no effect at all.

``tau*`` is unique even where the weighting is not, because the objective sees
the weights only through one scalar. Inference is non-regular for the same reason
the estimand is interesting -- the optimum sits on a boundary whose active set
moves with sampling noise -- so the confidence set is a union of perturbation
intervals, possibly disjoint.

See :doc:`../drosc` for the full development.
"""
from __future__ import annotations

from typing import Union

import pandas as pd

from ..config_models import BaseEstimatorResults, DROSCConfig
from ..utils.drosc_helpers import plot_drosc, run_drosc


class DROSC:
    """Distributionally Robust Synthetic Control estimator.

    ``fit()`` returns the weight-robust effect at the configured radius: the
    smallest-magnitude effect consistent with the weightings the pre-treatment
    moments cannot rule out. It is a lower bound on the true effect's magnitude
    with a trustworthy sign, not an estimate of the quantity classical synthetic
    control targets, and it is read across a sweep of ``robustness_lambda``
    instead of at one value.

    Parameters
    ----------
    config : DROSCConfig or dict
        Configuration object. See :class:`mlsynth.config_models.DROSCConfig`.
    """

    def __init__(self, config: Union[DROSCConfig, dict]) -> None:
        if isinstance(config, dict):
            config = DROSCConfig(**config)
        self.config = config
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.treat: str = config.treat
        self.unitid: str = config.unitid
        self.time: str = config.time
        self.display_graphs: bool = config.display_graphs
        self.save = config.save
        self.counterfactual_color = config.counterfactual_color
        self.treated_color = config.treated_color

    def fit(self) -> BaseEstimatorResults:
        """Estimate the DROSC effect (and, if requested, its union CI)."""
        results = run_drosc(self.config)
        if self.display_graphs:
            plot_drosc(
                results,
                outcome=self.outcome,
                time=self.time,
                treated_color=self.treated_color,
                counterfactual_color=self.counterfactual_color,
                save=self.save,
            )
        return results
