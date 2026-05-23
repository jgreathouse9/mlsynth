"""Spatial Synthetic Difference-in-Differences (SpSyDiD) estimator.

Serenini, R., & Masek, F. (2024). *"Spatial Synthetic
Difference-in-Differences."* SSRN 4736857.

Extends Arkhangelsky-Athey-Hirshberg-Imbens-Wager (2021) SDID with a
spatial spillover term so the estimator can disentangle two estimands
that the standard SDID confounds when SUTVA is violated by geographic
spillovers:

* :math:`\\widehat \\tau` -- direct effect on the directly-treated
  units (the ATT, identical in form to standard SDID).
* :math:`\\widehat \\tau_s` -- spillover coefficient per unit of
  neighbour-treatment exposure :math:`(WD)_{it} = \\sum_j w_{ij} D_{jt}`.

The user supplies a row-standardised :math:`N \\times N` spatial weight
matrix :math:`W` (helpers in
:mod:`mlsynth.utils.spsydid_helpers.spatial` cover the standard
constructions: k-NN from coordinates, inverse distance, queen / rook
contiguity from adjacency). Donors are auto-partitioned into directly
treated, spillover-exposed, and pure controls based on
:math:`D` and :math:`W`. The SDID unit / time weights are computed on
the pure controls; the final WLS regression jointly estimates
:math:`\\tau` and :math:`\\tau_s`.

When :math:`W = 0` (no spatial structure) or no donor has any treated
neighbour, SpSyDiD numerically reduces to plain SDID with
:math:`\\widehat \\tau_s = 0`.
"""

from __future__ import annotations

from typing import Union

import pandas as pd
from pydantic import ValidationError

from ..config_models import SpSyDiDConfig
from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
)
from ..utils.spsydid_helpers.pipeline import run_spsydid
from ..utils.spsydid_helpers.plotter import plot_spsydid
from ..utils.spsydid_helpers.setup import prepare_spsydid_inputs
from ..utils.spsydid_helpers.structures import SpSyDiDResults


class SpSyDiD:
    """Spatial Synthetic Difference-in-Differences estimator.

    Parameters
    ----------
    config : SpSyDiDConfig or dict
        Configuration object. See
        :class:`mlsynth.config_models.SpSyDiDConfig`.
    """

    def __init__(self, config: Union[SpSyDiDConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = SpSyDiDConfig(**config)
            except ValidationError as exc:
                raise MlsynthConfigError(
                    f"Invalid SpSyDiD configuration: {exc}"
                ) from exc
        self.config: SpSyDiDConfig = config
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.treat: str = config.treat
        self.unitid: str = config.unitid
        self.time: str = config.time
        self.spatial_matrix = config.spatial_matrix
        self.unit_order = config.unit_order
        self.row_standardize_spatial: bool = config.row_standardize_spatial
        self.display_graphs: bool = config.display_graphs
        self.save = config.save
        self.counterfactual_color = config.counterfactual_color
        self.treated_color: str = config.treated_color

    def fit(self) -> SpSyDiDResults:
        """Run Algorithm 1 of Serenini & Masek (2024)."""
        try:
            inputs = prepare_spsydid_inputs(
                df=self.df,
                outcome=self.outcome,
                treat=self.treat,
                unitid=self.unitid,
                time=self.time,
                spatial_matrix=self.spatial_matrix,
                unit_order=self.unit_order,
                row_standardize_spatial=self.row_standardize_spatial,
            )
            results = run_spsydid(inputs)
            if self.display_graphs:
                plot_spsydid(
                    results,
                    treated_color=self.treated_color,
                    counterfactual_color=self.counterfactual_color,
                    save=self.save,
                    time_axis_label=self.time,
                    treatment_label=self.treat,
                    unit_label=self.unitid,
                    outcome_label=self.outcome,
                )
            return results
        except (MlsynthConfigError, MlsynthDataError, MlsynthEstimationError):
            raise
        except Exception as exc:
            raise MlsynthEstimationError(
                f"SpSyDiD estimation failed: {exc}"
            ) from exc
