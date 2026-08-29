"""Input preparation for TWSF.

TWSF reads a panel the other way round from the rest of the library. ``treat``
flags the *donors*, who adopt partway through; the focal unit named by
``target`` never adopts. That gives two windows:

* the unit side runs up to the first adoption date, when every unit is still
  under control, and is where the cross-unit relationship is learned;
* the time side runs from the last adoption date onward, when every donor is
  treated, and is where the treated regime's dynamics are learned.

Under the paper's common-adoption assumption the two dates coincide. When they
do not the panel is staggered, which the theory does not cover; the windows
above are the approximate mapping the paper's own case study uses, and the
caller is warned.
"""

from __future__ import annotations

import warnings
from typing import List, Optional

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from ..datautils import dataprep
from .structures import TWSFInputs


def prepare_twsf_inputs(
    df: pd.DataFrame,
    outcome: str,
    unitid: str,
    time: str,
    treat: str,
    target: str,
    horizon: int,
    donors: Optional[List[str]] = None,
) -> TWSFInputs:
    """Split a long panel into TWSF's unit-side and time-side blocks."""
    prepped = dataprep(df, unitid, time, outcome, treat, allow_no_donors=True)
    Ywide = prepped["Ywide"]
    labels = list(Ywide.index.to_numpy())

    if target not in Ywide.columns:
        raise MlsynthDataError(
            f"target {target!r} is not a unit in column {unitid!r}. Units "
            f"present: {sorted(map(str, Ywide.columns))[:8]}..."
        )

    # dataprep has already refused a panel with no treated observation, so by
    # here at least one unit adopts.
    adopted = df.loc[df[treat] == 1, [unitid, time]]
    if target in set(adopted[unitid]):
        raise MlsynthDataError(
            f"target {target!r} is flagged as treated by {treat!r}. TWSF "
            "forecasts a unit that has *not* had the intervention; a unit that "
            "already adopted is a donor, not a target."
        )

    first_adopt = adopted.groupby(unitid)[time].min()
    if donors is not None:
        missing = [d for d in donors if d not in first_adopt.index]
        if missing:
            raise MlsynthDataError(
                f"donors {missing} are never flagged by {treat!r}, so they "
                "carry no treated history to learn dynamics from."
            )
        first_adopt = first_adopt.loc[donors]
    donor_names = [str(u) for u in first_adopt.index]

    earliest, latest = first_adopt.min(), first_adopt.max()
    staggered = bool(earliest != latest)
    if staggered:
        warnings.warn(
            f"TWSF's theory assumes a common adoption date, but adoption here is "
            f"staggered: the {len(donor_names)} donors adopt between {earliest} and "
            f"{latest}. Proceeding with the "
            "approximate mapping the paper's own application uses: the unit "
            f"side ends at {earliest} and the time side starts at {latest}, so "
            "every donor is treated across the whole Page window. Treat the "
            "result as approximate and prefer a common-date donor pool where "
            "one is available.",
            UserWarning, stacklevel=3,
        )

    idx = {lab: i for i, lab in enumerate(labels)}
    unit_end, time_start = idx[earliest], idx[latest]
    if unit_end < 2:
        raise MlsynthDataError(
            f"only {unit_end} pre-adoption periods before {earliest}; the "
            "unit-side regression needs a control window to learn the "
            "cross-unit relationship from."
        )

    Yd = Ywide[list(first_adopt.index)].to_numpy(dtype=float).T
    inputs = TWSFInputs(
        y_target_pre=Ywide[target].to_numpy(dtype=float)[:unit_end],
        Y_donors_pre=Yd[:, :unit_end],
        Y_donors_post=Yd[:, time_start:],
        donor_names=donor_names,
        target_name=str(target),
        unit_side_end=earliest,
        time_side_start=latest,
        forecast_labels=_extend(labels, horizon),
        staggered=staggered,
    )
    return inputs


def _extend(labels: list, horizon: int) -> list:
    """Labels for the ``horizon`` dates after the panel ends.

    Datetime and numeric indices are extended by their own final step; anything
    else falls back to positional labels, since the forecast dates are past the
    end of the data and have no observed labels to borrow.
    """
    if len(labels) < 2:
        return list(range(1, horizon + 1))
    last, step = labels[-1], None
    try:
        step = labels[-1] - labels[-2]
        return [last + step * (h + 1) for h in range(horizon)]
    except Exception:
        return [f"+{h + 1}" for h in range(horizon)]
