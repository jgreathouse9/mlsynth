"""Arrange the designs' in-space placebos as the arrays the inference functions take.

A design fit with ``inference="placebo"`` carries its in-space placebo in
``inference.details``: every unit's gap path (``unit_gaps``) and donor weights
(``unit_weights``) under VanillaSC's string labels, and the units whose refit
raised (``skipped_placebos``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ...exceptions import MlsynthConfigError


@dataclass(frozen=True)
class PlaceboStack:
    """One or more designs' placebo fits over a shared unit set.

    ``units`` starts with the treated unit (``position`` 0); ``outcomes`` is
    ``(T, N)``, ``gaps`` ``(D, T, N)`` and ``weights`` ``(D, N, N)``, missing
    where a unit's refit raised.
    """

    designs: list
    units: list
    times: list
    outcomes: np.ndarray
    gaps: np.ndarray
    weights: np.ndarray
    position: int = 0

    def mask(self, periods: Sequence[Any]) -> np.ndarray:
        periods = list(periods)
        unknown = set(periods) - set(self.times)
        if unknown:
            raise MlsynthConfigError(f"Periods {sorted(unknown)} are not on the study's time axis.")
        return np.isin(np.asarray(self.times, dtype=object), np.asarray(periods, dtype=object))


def placebo_details(fit: Any) -> Optional[dict]:
    """The in-space placebo a native fit carries, or None when it carries none."""
    details = getattr(fit.inference, "details", None) or {}
    return details if "unit_gaps" in details else None


def design_units(details: Mapping[str, Any]) -> set:
    """Every unit a design's placebo covered: the treated unit, the refit and the skipped."""
    return set(details["unit_gaps"]) | set(details["skipped_placebos"])


def stack(fits: Mapping[str, Any], panel: pd.DataFrame, unitid: str, time: str,
          outcome: str, treated: Any, designs: Sequence[str]) -> PlaceboStack:
    """Arrays for ``designs``, which must carry placebo details and share one unit set."""
    designs = list(designs)
    if not designs:
        raise MlsynthConfigError("No designs with placebo runs were selected.")
    details = [placebo_details(fits[spec]) for spec in designs]
    sets = [design_units(d) for d in details]
    if any(units != sets[0] for units in sets[1:]):
        raise MlsynthConfigError(
            "Designs combined in one test must share one unit set; designs that "
            "restrict the donor pool have their own placebo distribution.")
    labels = {str(u): u for u in panel[unitid].unique()}
    names = [str(treated), *sorted(sets[0] - {str(treated)})]
    units = [labels[n] for n in names]
    index = {n: j for j, n in enumerate(names)}
    times = list(fits[designs[0]].time_series.time_periods)
    Y = (panel.pivot(index=time, columns=unitid, values=outcome)
         .reindex(index=times, columns=units).to_numpy(float))
    G = np.full((len(designs), len(times), len(units)), np.nan)
    W = np.full((len(designs), len(units), len(units)), np.nan)
    for d, det in enumerate(details):
        for name, gaps in det["unit_gaps"].items():
            j = index[name]
            G[d, :, j] = gaps
            W[d, j] = 0.0
            for donor, weight in det["unit_weights"][name].items():
                W[d, j, index[donor]] = weight
    return PlaceboStack(designs, units, times, Y, G, W)
