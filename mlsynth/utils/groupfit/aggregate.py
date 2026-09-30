"""A group of units to one series.

Which of the two an estimator wants is a matter of whose notation it follows,
not of what it computes: the sum and the mean differ by the group size, so the
regression on one is the regression on the other with the slope rescaled. See
:func:`~mlsynth.utils.groupfit.fit.fit_two_group`.
"""
from __future__ import annotations

from typing import Callable, Dict, Sequence

import numpy as np

from ...exceptions import MlsynthDataError

#: The two aggregations, by name. Kerman, Wang and Vaver's eqn 1 sums the geos in
#: each group; Li and Van den Bulte's eqn 2.4 averages the control units.
_AGGREGATORS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "sum": lambda block: block.sum(axis=1),
    "mean": lambda block: block.mean(axis=1),
}

AGGREGATIONS = tuple(_AGGREGATORS)


def aggregate_group(panel: np.ndarray, columns: Sequence[int], *,
                    how: str = "sum") -> np.ndarray:
    """One series from the units in ``columns``.

    ``panel`` is periods by units. The result is one value per period, so the
    returned length is ``panel.shape[0]`` whatever the group size.
    """
    panel = np.asarray(panel, dtype=float)
    if panel.ndim != 2:
        raise MlsynthDataError(
            f"the panel has to be periods by units; got shape {panel.shape}.")
    take = list(columns)
    if not take:
        raise MlsynthDataError(
            "a group needs at least one unit to aggregate; got none.")
    if how not in _AGGREGATORS:
        raise MlsynthDataError(
            f"how is {how!r}; the aggregations are {list(AGGREGATIONS)}.")
    out_of_range = [j for j in take if not 0 <= j < panel.shape[1]]
    if out_of_range:
        raise MlsynthDataError(
            f"unit index(es) {out_of_range} are outside the panel's "
            f"{panel.shape[1]} column(s).")
    return _AGGREGATORS[how](panel[:, take])
