"""SSC: Staggered Synthetic Control (Cao, Lu & Wu 2026).

Cao, J., Lu, S. & Wu, H. (2026). *"Synthetic Control Inference for Staggered
Adoption."* The Econometrics Journal.

SSC estimates heterogeneous, dynamic treatment effects when many units adopt a
policy at *different* times (staggered adoption) and the pre-treatment history
is long relative to the number of units and post-periods (large :math:`T`,
moderate :math:`N`, small :math:`S`). Two features set it apart from
difference-in-differences and other staggered synthetic-control methods:

1. **It uses every unit -- including not-yet-treated units -- as a donor.** Each
   unit's untreated outcome is modelled as an intercept plus a *simplex*
   synthetic control on all other units,

   .. math::

      y_{i,t}(\\infty) = a_i + Y_t(\\infty)' b_i + u_{i,t},
      \\qquad b_i \\ge 0,\\ \\textstyle\\sum_j b_{ij} = 1,\\ b_{ii} = 0 ,

   so it does **not** require a pool of never-treated units and does **not**
   rely on parallel trends.

2. **It delivers valid inference for policy-relevant aggregates.** All
   individual unit :math:`\\times` time effects :math:`\\tau` are estimated
   jointly by GLS; the target is any linear map :math:`\\gamma = L\\tau`
   (event-time ATT, overall ATT, or a contrast between policies). Inference uses
   Andrews' (2003) end-of-sample stability test, whose reference distribution is
   built from pre-treatment residual windows -- valid for both *sharp* and
   *non-sharp* nulls under a large-:math:`T` stationarity assumption.

This estimator targets the staggered causal setting; it returns the
event-study path of effects with confidence bands and the overall ATT.
"""

from __future__ import annotations

from typing import Union

import pandas as pd
from pydantic import ValidationError

from ..config_models import SSCConfig
from ..exceptions import (
    MlsynthConfigError,
    MlsynthDataError,
    MlsynthEstimationError,
)
from ..utils.ssc_helpers.pipeline import run_ssc
from ..utils.ssc_helpers.plotter import plot_ssc
from ..utils.ssc_helpers.setup import prepare_ssc_inputs
from ..utils.ssc_helpers.structures import SSCResults


class SSC:
    """Staggered Synthetic Control estimator.

    Parameters
    ----------
    config : SSCConfig or dict
        Configuration object. See :class:`mlsynth.config_models.SSCConfig`.
    """

    def __init__(self, config: Union[SSCConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = SSCConfig(**config)
            except ValidationError as exc:
                raise MlsynthConfigError(
                    f"Invalid SSC configuration: {exc}"
                ) from exc
        self.config: SSCConfig = config
        self.df: pd.DataFrame = config.df
        self.outcome: str = config.outcome
        self.treat: str = config.treat
        self.unitid: str = config.unitid
        self.time: str = config.time
        self.inference: bool = config.inference
        self.alpha: float = config.alpha
        self.display_graphs: bool = config.display_graphs
        self.save = config.save
        self.counterfactual_color = config.counterfactual_color
        self.treated_color: str = config.treated_color

    def fit(self) -> SSCResults:
        """Run SSC and return :class:`SSCResults`."""
        try:
            inputs = prepare_ssc_inputs(
                df=self.df, outcome=self.outcome, treat=self.treat,
                unitid=self.unitid, time=self.time,
            )
            results = run_ssc(inputs, inference=self.inference, alpha=self.alpha)
            if self.display_graphs:
                plot_ssc(
                    results,
                    treated_color=self.treated_color,
                    counterfactual_color=self.counterfactual_color,
                    save=self.save,
                    time_axis_label="Event time",
                    treatment_label=self.treat,
                    unit_label=self.unitid,
                    outcome_label=self.outcome,
                )
            return results
        except (MlsynthConfigError, MlsynthDataError, MlsynthEstimationError):
            raise
        except Exception as exc:
            raise MlsynthEstimationError(
                f"SSC estimation failed: {exc}"
            ) from exc
