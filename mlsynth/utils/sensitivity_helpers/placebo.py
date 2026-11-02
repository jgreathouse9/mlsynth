"""Arrange a study's placebo runs as the arrays the inference functions take."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ...exceptions import MlsynthConfigError

PATH_COLUMNS = ["spec_id", "unit", "time", "observed", "synthetic", "gap"]
WEIGHT_COLUMNS = ["spec_id", "unit", "donor", "weight"]
FAILURE_COLUMNS = ["spec_id", "unit", "exception_type", "message"]
WARNING_COLUMNS = ["spec_id", "unit", "warning_type", "message"]


@dataclass(frozen=True)
class PlaceboStack:
    """One or more designs' placebo runs over a shared unit set.

    ``units`` starts with the treated unit (``position`` 0); ``outcomes`` is
    ``(T, N)``, ``gaps`` ``(D, T, N)`` and ``weights`` ``(D, N, N)``, missing
    where a unit's fit failed.
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


def design_units(paths: pd.DataFrame, failures: pd.DataFrame, spec: str) -> set:
    """Every unit a design was fit for: fitted placebos, failed placebos and the treated unit."""
    return (set(paths.loc[paths.spec_id == spec, "unit"])
            | set(failures.loc[failures.spec_id == spec, "unit"]))


def stack(paths: pd.DataFrame, weights: pd.DataFrame, failures: pd.DataFrame,
          panel: pd.DataFrame, unitid: str, time: str, outcome: str,
          treated: Any, designs: Sequence[str]) -> PlaceboStack:
    """Arrays for ``designs``, which must share one unit set."""
    designs = list(designs)
    if not designs:
        raise MlsynthConfigError("No designs with placebo runs were selected.")
    sets = [design_units(paths, failures, spec) for spec in designs]
    if any(units != sets[0] for units in sets[1:]):
        raise MlsynthConfigError(
            "Designs combined in one test must share one unit set; designs that "
            "restrict the donor pool have their own placebo distribution.")
    units = [treated, *sorted(sets[0] - {treated}, key=str)]
    times = sorted(paths.time.unique())
    index = {str(u): j for j, u in enumerate(units)}
    Y = (panel.pivot(index=time, columns=unitid, values=outcome)
         .reindex(index=times, columns=units).to_numpy(float))
    G = np.full((len(designs), len(times), len(units)), np.nan)
    W = np.full((len(designs), len(units), len(units)), np.nan)
    for d, spec in enumerate(designs):
        p = paths[paths.spec_id == spec]
        w = weights[weights.spec_id == spec]
        for j, unit in enumerate(units):
            rows = p[p.unit == unit]
            if rows.empty:
                continue
            G[d, :, j] = rows.set_index("time").gap.reindex(times).to_numpy(float)
            W[d, j] = 0.0
            for donor, weight in zip(w.loc[w.unit == unit, "donor"], w.loc[w.unit == unit, "weight"]):
                W[d, j, index[str(donor)]] = weight
    return PlaceboStack(designs, units, times, Y, G, W)
