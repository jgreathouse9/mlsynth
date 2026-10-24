"""The result VanillaSC returns for a single treated unit."""

from __future__ import annotations

from typing import Optional

from pydantic import ConfigDict, Field

from ...config_models import BaseEstimatorResults
from ..bilevel.exact.structures import ExactDetails


class VanillaSCResults(BaseEstimatorResults):
    """The observational report of :meth:`mlsynth.VanillaSC.fit` for one treated unit.

    An :class:`~mlsynth.config_models.EffectResult`: the standardized sub-models
    (``effects``, ``time_series``, ``weights``, ``inference``, ``fit_diagnostics``,
    ``method_details``) and the flat accessors, plus the exact backend's typed details.

    Parameters
    ----------
    exact : ExactDetails, optional
        With ``backend="exact"``: the case the search took (exterior or interior), the
        diagonal and positive definite predictor weights that select the donor weights
        (log-det maximisers, analytic centres and, when asked for, identified ranges),
        the visible facets holding a positive definite optimum, and the face counts when
        asked for. ``None`` for every other backend.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    exact: Optional[ExactDetails] = Field(
        default=None,
        description="The exact backend's details (backend='exact'); None otherwise.",
    )
