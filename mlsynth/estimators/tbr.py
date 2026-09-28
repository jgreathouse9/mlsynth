"""TBR: Time-Based Regression for geo experiments.

TBR measures what an intervention did to a market when only a handful of
geographic units are available, which is where geo-based regression, needing
replication across many geos, stops working. It aggregates the geos into a
treatment group and a control group, regresses one group's series on the other
over the pre-intervention period, and projects that relation forward to say what
the treatment group would have done untreated. The estimand is the cumulative
effect at the end of the experiment, and its posterior is a shifted, scaled
t-distribution in closed form, so no resampling is involved.

Supplying a spend column adds the incremental return on ad spend, the ratio of
the cumulative effect on the response to the cumulative effect on cost.

The assumption is strong and cannot be checked after the intervention begins:
the linear relation between the two group aggregates has to hold through the
test period, and only its pretest half is observable. Au (2018) exists because
of this, and designing the groups to satisfy it is a separate step from
estimating with them.

References
----------
Kerman, J., Wang, P. and Vaver, J. (2017). Estimating Ad Effectiveness using Geo
Experiments in a Time-Based Regression Framework. Google.

Verification: ``benchmarks/studies/tbr_geo``, which cross-validates these
formulas against ``google/matched_markets`` on the reference's own panel and
reproduces the paper's section 5.2 coverage grid.
"""

from __future__ import annotations

from typing import Union

from pydantic import ValidationError

from ..exceptions import MlsynthConfigError
from ..utils.tbr_helpers.config import TBRConfig
from ..utils.tbr_helpers.pipeline import run
from ..utils.tbr_helpers.structures import TBRResults


class TBR:
    """Time-Based Regression.

    Parameters
    ----------
    config : TBRConfig
        Panel, the group design, and the reporting level. See
        :class:`mlsynth.config_models.TBRConfig`.

    Returns
    -------
    TBRResults
        An :class:`~mlsynth.config_models.EffectResult`. The flat accessors
        (``att``, ``att_ci``, ``counterfactual``, ``gap``) resolve over the
        treatment-group aggregate; the cumulative posterior, the pretest fit and
        the iROAS live in the typed ``cumulative`` / ``tbr_fit`` / ``iroas``
        fields.

    Notes
    -----
    - ``att`` is the cumulative effect divided by the number of post periods,
      the mean per-period effect. The cumulative effect the method is built
      around is ``cumulative.estimate[-1]``, also on
      ``effects.additional_effects``.
    - TBR has no donor weights, so the standardized ``weights`` slot carries a
      method note instead of a weight vector.

    Examples
    --------
    >>> import pandas as pd
    >>> from mlsynth import TBR
    >>> from mlsynth.config_models import TBRConfig
    >>> res = TBR(TBRConfig(                      # doctest: +SKIP
    ...     df=panel, unitid="geo", time="date", outcome="sales", treat="D",
    ...     control_col="is_control", cost_col="cost",
    ... )).fit()
    >>> res.cumulative.estimate[-1]               # doctest: +SKIP
    143028.62
    """

    def __init__(self, config: Union[TBRConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = TBRConfig(**config)
            except ValidationError as exc:
                raise MlsynthConfigError(
                    f"Invalid TBRConfig configuration: {exc}") from exc
        elif not isinstance(config, TBRConfig):
            raise MlsynthConfigError(
                f"TBR expects a TBRConfig or a dict of its fields; got "
                f"{type(config).__name__}")
        self.config: TBRConfig = config

    def fit(self) -> TBRResults:
        """Fit the pretest relation and report the cumulative effect."""
        return run(self.config)
