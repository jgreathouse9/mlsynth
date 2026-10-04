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
test period, and only its pretest half is observable. Designing the groups to
satisfy it is the other half of the method, and it is the searched mode of this
same estimator: given ``max_treatment_size`` in place of a named split, TBR
climbs the partitions on pretest data alone and returns one recommended design
per treatment size, scoring each on the minimum effect the interval above could
detect. That is Au (2018). A geo's weight in the regression is its membership,
so the split is the only place the method has to buy precision, and it has to be
chosen before the intervention runs.

References
----------
Kerman, J., Wang, P. and Vaver, J. (2017). Estimating Ad Effectiveness using Geo
Experiments in a Time-Based Regression Framework. Google.

Au, T. C. (2018). A Time-Based Regression Matched Markets Approach for Designing
Geo Experiments. Technical report, Google LLC.

Verification: ``benchmarks/studies/tbr_geo`` cross-validates these formulas
against ``google/matched_markets`` on the reference's own panel and reproduces
the paper's section 5.2 coverage grid; ``benchmarks/studies/tbrmm_match`` does
the same for the search, which selects the same geos as the reference at every
treatment size.
"""

from __future__ import annotations

from typing import Union

from pydantic import ValidationError

from ..exceptions import MlsynthConfigError
from ..utils.tbr_helpers.config import TBRConfig
from ..utils.tbr_helpers.pipeline import run
from ..utils.tbr_helpers.design.structures import TBRResults


class TBR:
    """Time-Based Regression.

    Parameters
    ----------
    config : TBRConfig
        Panel, the group design or the search budget, and the reporting level.
        See :class:`mlsynth.config_models.TBRConfig`.

    Returns
    -------
    TBRResults
        A :class:`~mlsynth.config_models.DesignResult`, in both modes.
        ``report`` is the estimate: its flat accessors (``att``, ``att_ci``,
        ``counterfactual``, ``gap``) resolve over the treatment-group aggregate,
        and the cumulative posterior, the pretest fit and the iROAS live in its
        typed ``cumulative`` / ``tbr_fit`` / ``iroas`` fields. A named split has
        nothing to choose between, so ``designs`` is empty and ``recommended``
        is ``None``. A searched one fills both, one design per treatment size,
        and ``report`` carries the recommendation measured on the realized
        periods when ``post_col`` marks any.

    Notes
    -----
    - ``report.effects.att`` is the cumulative effect divided by the number of
      post periods, the mean per-period effect. The cumulative effect the
      method is built around is ``report.cumulative.estimate[-1]``, also on
      ``report.effects.additional_effects``.
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
    >>> res.report.cumulative.estimate[-1]        # doctest: +SKIP
    143028.62

    The same class, asked to choose the groups instead:

    >>> res = TBR(TBRConfig(                      # doctest: +SKIP
    ...     df=history, unitid="geo", time="date", outcome="sales",
    ...     max_treatment_size=5, n_test=28,
    ... )).fit()
    >>> res.recommended.treatment_units           # doctest: +SKIP
    ['chicago', 'portland']
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
