"""STACKEDSC -- stacked-in-event-time bias-corrected synthetic control.

Wiltshire, J. C. (2023). *"Walmart Supercenters and Monopsony Power: How a
Large, Low-Wage Employer Impacts Local Labor Markets."* Section 4.2.

One synthetic control per treated unit against a common never-treated donor
pool, each fitted on series indexed to 100 at that unit's own final
pre-treatment period, then averaged on a shared event clock under weights the
caller supplies.

Why the indexing is not cosmetic
--------------------------------

Rescaling the treated unit and every donor to their own base-period value and
requiring the weights to sum to one is algebraically the same as fitting on
levels under the constraint that the synthetic control reproduce the treated
unit's base-period level exactly. So the gap at ``e = -1`` is identically zero,
and the estimand is a percent change rather than a level difference. Turning
``normalize`` off gives a level-scale stacked SCM: a different estimator, not a
different presentation.

Why cohorts rather than units
-----------------------------

The base period belongs to the adoption time, so units adopting together share
one normalised donor block. On the paper's panel that is six blocks for 566
treated counties, which is what makes each cohort a single
multiple-right-hand-side program instead of hundreds of separate solves.

When to reach for it
--------------------

Many treated units adopting at different times, a donor pool that is small and
deliberately constructed, outcomes whose levels differ by orders of magnitude
across units, and a percent effect as the object of interest. The level
heterogeneity is the part that matters: matching such units in raw levels lets
the largest ones dominate the fit.

See :doc:`../stackedsc` for the full treatment.
"""

from __future__ import annotations

from typing import Any, Dict, Union

from ..utils.stackedsc_helpers.config import STACKEDSCConfig
from ..utils.stackedsc_helpers.pipeline import run_stackedsc
from ..utils.stackedsc_helpers.structures import STACKEDSCResults


class STACKEDSC:
    """Stacked-in-event-time synthetic control for staggered adoption.

    Parameters
    ----------
    config : STACKEDSCConfig or dict
        Configuration. See
        :class:`mlsynth.utils.stackedsc_helpers.config.STACKEDSCConfig`.

    Returns
    -------
    STACKEDSCResults
        Event-time ATT under the supplied aggregation weights, the per-treated
        unit fits it averages, and a design block recording what the fit did.
    """

    def __init__(self, config: Union[STACKEDSCConfig, Dict[str, Any]]) -> None:
        if isinstance(config, dict):
            config = STACKEDSCConfig(**config)
        self.config = config

    def fit(self) -> STACKEDSCResults:
        """Fit every treated unit and aggregate on the event clock."""
        return run_stackedsc(self.config)
