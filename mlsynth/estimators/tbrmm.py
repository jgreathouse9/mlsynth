"""TBRMM: matched markets for a time-based-regression geo experiment.

TBR aggregates geos into a treatment group and a control group and regresses one
aggregate on the other, so a geo's weight in that regression is its membership.
There is no weight vector to solve for, which means the choice of who goes where
is not a preliminary to the estimate -- it is the only place the method has to
buy precision, and it has to be made before the intervention runs. TBRMM makes
that choice.

The score a candidate split earns is TBR's own posterior read backwards. Fix a
significance level and a power target, take the scale of the cumulative effect's
posterior, and solve for the smallest effect that would clear the threshold: the
result is the split's minimum detectable effect, computable from pretest data
alone. Ahead of it sit four tests of the assumptions that interval needs --
parameter stability through the pretest, residual autocorrelation, the
Durbin-Watson band, and an A/A test on held-out pretest periods -- and a split is
ranked on power only once it has passed them.

One design is reported per treatment size, because a larger treatment group is
not automatically better: a geo added to the treatment group brings its volume to
the treatment aggregate and takes it out of the pool available to the control
aggregate. The search itself is a hill climb over a space of 3^n labellings and
carries no optimality guarantee; the recommended split's detectable effect is the
best of many estimates from one pretest, so read it as a selected maximum.

References
----------
Au, T. C. (2018). Robust Design and Analysis of Geo Experiments with Matched
Markets. Google.

Kerman, J., Wang, P. and Vaver, J. (2017). Estimating Ad Effectiveness using Geo
Experiments in a Time-Based Regression Framework. Google.
"""

from __future__ import annotations

from typing import Union

from pydantic import ValidationError

from ..exceptions import MlsynthConfigError
from ..utils.tbrmm_helpers.config import TBRMMConfig
from ..utils.tbrmm_helpers.pipeline import run
from ..utils.tbrmm_helpers.structures import TBRMMResults


class TBRMM:
    """Matched markets for a TBR geo experiment.

    Parameters
    ----------
    config : TBRMMConfig
        Panel, the largest treatment group allowed, the planned test length, the
        objective, and the optional eligibility columns. See
        :class:`mlsynth.config_models.TBRMMConfig`.

    Returns
    -------
    TBRMMResults
        A :class:`~mlsynth.config_models.DesignResult`. ``designs`` holds one
        recommended partition per treatment size and ``recommended`` the best of
        them across sizes. ``report`` is empty until the experiment runs and TBR
        analyses it.

    Examples
    --------
    >>> import pandas as pd
    >>> from mlsynth import TBRMM
    >>> from mlsynth.config_models import TBRMMConfig
    >>> res = TBRMM(TBRMMConfig(                    # doctest: +SKIP
    ...     df=panel, unitid="geo", time="date", outcome="sales",
    ...     max_treatment_size=5, n_test=28,
    ... )).fit()
    >>> res.recommended.treatment_units             # doctest: +SKIP
    ['chicago', 'portland']
    """

    def __init__(self, config: Union[TBRMMConfig, dict]) -> None:
        if isinstance(config, dict):
            try:
                config = TBRMMConfig(**config)
            except ValidationError as exc:
                raise MlsynthConfigError(
                    f"Invalid TBRMMConfig configuration: {exc}") from exc
        elif not isinstance(config, TBRMMConfig):
            raise MlsynthConfigError(
                f"TBRMM expects a TBRMMConfig or a dict of its fields; got "
                f"{type(config).__name__}")
        self.config: TBRMMConfig = config

    def fit(self) -> TBRMMResults:
        """Search the partitions and return one recommended design per size."""
        return run(self.config)
