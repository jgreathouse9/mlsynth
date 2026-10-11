"""Long DataFrame to arrays for EASC (the only pandas touchpoint).

The outcome goes through the canonical :func:`mlsynth.utils.datautils.dataprep`
contract. The exposure index and the observed shock are read from their own
columns: the index is each unit's mean over a pre-treatment window, and the
shock must take one value per period.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from ..datautils import dataprep
from .structures import EASCInputs


def _position(labels: List[Any], label: Any, what: str) -> int:
    try:
        return labels.index(label)
    except ValueError:
        raise MlsynthDataError(f"EASC: {what} {label!r} is not a period of the panel.") from None


def _shock_path(df: pd.DataFrame, time: str, shock: str, labels: Sequence) -> np.ndarray:
    values = df[shock]
    if values.isna().any():
        raise MlsynthDataError(f"EASC: the shock column {shock!r} has missing values.")
    by_period = df.groupby(time)[shock]
    varying = by_period.nunique()
    if (varying > 1).any():
        bad = list(varying[varying > 1].index[:3])
        raise MlsynthDataError(
            f"EASC: the shock column {shock!r} must be the same for every unit in each period; "
            f"it varies in period(s) {bad}.")
    return by_period.first().reindex(labels).to_numpy(dtype=float)


def _exposure_index(df: pd.DataFrame, unitid: str, time: str, exposure: str, window_labels: Sequence,
                    units: Sequence) -> np.ndarray:
    inside = df[df[time].isin(list(window_labels))]
    s = inside.groupby(unitid)[exposure].mean().reindex(list(units))
    missing = [str(u) for u in s.index[s.isna()]]
    if missing:
        raise MlsynthDataError(
            f"EASC: no {exposure!r} value in the exposure window for unit(s) {missing}. Fill or drop them; "
            "the paper sets non-producers' oil rents to zero.")
    return s.to_numpy(dtype=float)


def prepare_easc_inputs(df: pd.DataFrame, *, unitid: str, time: str, outcome: str, treat: str, exposure: str,
                        shock: str, exposure_window: Optional[Tuple[Any, Any]] = None,
                        base_period: Optional[Any] = None) -> EASCInputs:
    """Ingest the panel and assemble :class:`EASCInputs`.

    Raises
    ------
    MlsynthDataError
        If there is not exactly one treated unit, a label is not a period of
        the panel, the exposure window reaches past treatment, the base period
        is not a pre-treatment period, a unit has no exposure value, or the
        shock is missing or varies across units within a period.
    """
    prep = dataprep(df, unitid, time, outcome, treat)
    if "cohorts" in prep:
        raise MlsynthDataError("EASC needs exactly one treated unit; the panel has several.")
    labels = list(prep["time_labels"])
    T0 = int(prep["pre_periods"])
    donors = list(prep["donor_names"])
    treated = prep["treated_unit_name"]

    if exposure_window is None:
        first, last = 0, T0 - 1
    else:
        first = _position(labels, exposure_window[0], "exposure_window start")
        last = _position(labels, exposure_window[1], "exposure_window end")
        if last >= T0:
            raise MlsynthDataError(
                "EASC: the exposure index must be pre-determined; exposure_window ends at "
                f"{exposure_window[1]!r}, after treatment begins at {labels[T0]!r}.")
    s = _exposure_index(df, unitid, time, exposure, labels[first:last + 1], [treated, *donors])

    if base_period is None:
        base = T0 - 1
    else:
        base = _position(labels, base_period, "base_period")
        if base >= T0:
            raise MlsynthDataError(f"EASC: base_period {base_period!r} must be a pre-treatment period.")

    return EASCInputs(
        treated_name=treated,
        donor_names=donors,
        y=np.asarray(prep["y"], dtype=float).ravel(),
        donor_matrix=np.asarray(prep["donor_matrix"], dtype=float),
        T0=T0,
        time_labels=labels,
        exposure_treated=float(s[0]),
        exposure_donors=s[1:],
        shock=_shock_path(df, time, shock, labels),
        base_index=base,
        metadata={"outcome": outcome, "exposure": exposure, "shock": shock,
                  "exposure_window": (labels[first], labels[last])},
    )
