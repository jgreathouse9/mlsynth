"""COMPSC: compositional synthetic controls in the Aitchison geometry.

A thin orchestration over :mod:`mlsynth.utils.compsc_helpers`. When the outcome
is a composition -- a vector of shares that sum to a whole, such as an
electricity generation mix, a budget split, or brand share within a category --
running one synthetic control per share gives every share a different donor mix,
so the fitted shares need not sum to one and no single counterfactual unit
exists. Fitting on the raw shares with common weights fixes that but measures
distance in the wrong geometry: a Euclidean objective weights each category's
proportional error by its squared share, so a category holding 2 percent of the
total is almost invisible to the optimizer next to one holding 50.

COMPSC implements Boussim (2026): map the shares to log-ratios, where a
multinomial-logit choice model makes them the relative systematic utilities
driving choice, impose the usual interactive fixed effects there, and fit a
single weight vector by stacking the log-ratio equations across categories and
pre-periods. The counterfactual is the Frechet barycenter of the donor
compositions under the Aitchison metric -- the closure of their weighted
geometric mean -- so it is a valid composition by construction and the share
effects sum to zero.

Two geometries are available via ``geometry``. ``clr`` (the default) uses the
centered log-ratio map, which is the isometry for the Aitchison metric: the fit
minimizes pre-treatment Aitchison distance and is invariant to relabelling the
categories. ``alr`` reproduces the paper's Algorithm 1 exactly, using the
additive log-ratio against a nominated ``baseline``; it is baseline-dependent,
and ``baseline_sensitivity`` on the result reports how much that choice moves
the weights.
"""

from __future__ import annotations

from typing import List, Union

import pandas as pd
from pydantic import ValidationError

from ..config_models import COMPSCConfig
from ..exceptions import MlsynthConfigError
from ..utils.compsc_helpers import (
    COMPSCResults,
    assemble_compsc_results,
    plot_compsc,
    prepare_compsc_inputs,
)


class COMPSC:
    """Compositional synthetic control estimator (Aitchison geometry).

    Parameters
    ----------
    config : COMPSCConfig or dict
        Validated configuration. Beyond the common fields (``df``, ``treat``,
        ``unitid``, ``time``, ``display_graphs``, ``save``, colors), COMPSC
        reads ``outcomes`` (the ``K`` share columns), ``geometry``
        (``clr``/``alr``), ``baseline`` (the ``alr`` reference category),
        ``target`` (which share drives the flat accessors), ``inference``,
        ``placebo_screen`` and ``zero_replacement``.

    References
    ----------
    Boussim, O. (2026). Compositional Synthetic Controls. arXiv:2607.16991.

    Aitchison, J. (1982). The Statistical Analysis of Compositional Data.
    *Journal of the Royal Statistical Society: Series B*, 44(2), 139-160.

    Sun, L., Ben-Michael, E., & Feller, A. (2025). Using Multiple Outcomes to
    Improve the Synthetic Control Method. *Review of Economics and Statistics*.
    """

    def __init__(self, config: Union[COMPSCConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = COMPSCConfig(**config)
            except ValidationError as exc:
                raise MlsynthConfigError(
                    f"Invalid COMPSC configuration: {exc}"
                ) from exc
        self.config = config
        self.df: pd.DataFrame = config.df
        self.outcomes: List[str] = list(config.outcomes)
        self.treat: str = config.treat
        self.unitid: str = config.unitid
        self.time: str = config.time
        self.target: str = config.target or self.outcomes[0]
        self.baseline: str = config.baseline or self.outcomes[-1]
        self.geometry: str = config.geometry
        self.inference: str = config.inference
        self.placebo_screen: float = config.placebo_screen
        self.zero_replacement: float = config.zero_replacement
        self.display_graphs: bool = config.display_graphs
        self.save: Union[bool, str, dict] = config.save
        self.counterfactual_color = config.counterfactual_color
        self.treated_color = config.treated_color

    def fit(self) -> COMPSCResults:
        """Assemble the panel, fit the compositional weights, and return results.

        Returns
        -------
        COMPSCResults
            Observational report. The flat accessors (``att`` /
            ``counterfactual`` / ``gap`` / ``donor_weights`` / ``pre_rmse``)
            resolve to the ``target`` share; the whole composition is in
            ``counterfactual_composition`` / ``share_effects`` /
            ``log_ratio_effects`` / ``aitchison_distance``.
        """
        inputs = prepare_compsc_inputs(
            self.df, outcomes=self.outcomes, treat=self.treat,
            unitid=self.unitid, time=self.time, target=self.target,
            baseline=self.baseline, zero_replacement=self.zero_replacement,
        )
        results = assemble_compsc_results(
            inputs, self.geometry, self.inference, self.placebo_screen,
        )

        pc = self.config.resolved_plot()
        object.__setattr__(results, "plot_config", pc)
        if self.display_graphs:
            plot_compsc(
                results, time_labels=inputs.time_labels,
                treated_color=self.treated_color,
                counterfactual_color=self.counterfactual_color,
                intervention_index=inputs.T0, save=self.save,
            )
        return results
