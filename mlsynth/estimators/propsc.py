"""PROPSC: treatment effects on proportions with common-weights synthetic controls.

A thin orchestration over :mod:`mlsynth.utils.propsc_helpers`. Political-science
(and marketing/policy) outcomes are frequently *compositional* -- a vector of
``K`` proportions that sum to a whole (vote shares, budget shares,
turnout-vs-abstention). Fitting a separate synthetic control per component uses
a different donor mix for each, so the estimated effects need not sum to zero --
incoherent for compositional data. PROPSC implements the "constant control
comparison" of Bogatyrev & Stoetzer (2026): fit a *single* set of unit weights
(and, for ``method="sdid"``, time weights) jointly across all ``K`` outcomes,
then read off each component's ATT via the synthetic-DID double difference.
Common weights make the ``K`` ATTs sum to zero by construction.

Three schemes are available via ``method``: ``sdid`` (unit + time weights and an
intercept shift, the paper's default), ``sc`` (classic synthetic control -- unit
weights only, no intercept), and ``did`` (uniform weights). The implementation
reproduces the authors' R package ``propsdid`` cell-by-cell (see
``benchmarks/cases/propsc_spain.py``).
"""

from __future__ import annotations

from typing import List, Union

import pandas as pd
from pydantic import ValidationError

from ..config_models import PROPSCConfig
from ..exceptions import MlsynthConfigError
from ..utils.propsc_helpers import (
    PROPSCResults,
    assemble_propsc_results,
    plot_propsc,
    prepare_propsc_inputs,
)


class PROPSC:
    """Compositional common-weights SC/SDID estimator.

    Parameters
    ----------
    config : PROPSCConfig or dict
        Validated configuration. Beyond the common fields (``df``, ``treat``,
        ``unitid``, ``time``, ``display_graphs``, ``save``, colors), PROPSC
        reads ``outcomes`` (the ``K`` proportion columns), ``method``
        (``sdid``/``sc``/``did``), ``target`` (which proportion drives the flat
        accessors), and ``inference``.

    References
    ----------
    Bogatyrev, K., & Stoetzer, L. F. (2026). Estimating Treatment Effects on
    Proportions with Synthetic Controls. *Political Analysis*.
    """

    def __init__(self, config: Union[PROPSCConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = PROPSCConfig(**config)
            except ValidationError as exc:
                raise MlsynthConfigError(
                    f"Invalid PROPSC configuration: {exc}"
                ) from exc
        self.config = config
        self.df: pd.DataFrame = config.df
        self.outcomes: List[str] = list(config.outcomes)
        self.treat: str = config.treat
        self.unitid: str = config.unitid
        self.time: str = config.time
        self.target: str = config.target or self.outcomes[0]
        self.method: str = config.method
        self.inference: str = config.inference
        self.display_graphs: bool = config.display_graphs
        self.save: Union[bool, str, dict] = config.save
        self.counterfactual_color = config.counterfactual_color
        self.treated_color = config.treated_color

    def fit(self) -> PROPSCResults:
        """Assemble the panel, fit the common weights, and return results.

        Returns
        -------
        PROPSCResults
            Observational report. The flat accessors (``att`` / ``att_ci`` /
            ``counterfactual`` / ``gap`` / ``donor_weights`` / ``pre_rmse``)
            resolve to the ``target`` proportion; the full compositional output
            is in ``proportions`` / ``att_vector`` / ``se_vector`` /
            ``sum_constraint``.
        """
        inputs = prepare_propsc_inputs(
            self.df, outcomes=self.outcomes, treat=self.treat,
            unitid=self.unitid, time=self.time, target=self.target,
        )
        results = assemble_propsc_results(inputs, self.method, self.inference)

        pc = self.config.resolved_plot()
        object.__setattr__(results, "plot_config", pc)
        if self.display_graphs:
            plot_propsc(
                results, time_labels=inputs.time_labels,
                treated_color=self.treated_color,
                counterfactual_color=self.counterfactual_color,
                intervention_index=inputs.T0, save=self.save,
            )
        return results
