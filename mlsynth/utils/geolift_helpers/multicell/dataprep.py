"""Data preparation for the multi-cell GeoLift analysis.

Reads a unit-level **cell-membership** column (cell labels ``"A"``, ``"B"``, ...
for treated geos; blank / a ``control_label`` for the shared donor pool) plus a
``post_col`` marking the treatment window, and resolves the wide panel, the
per-cell market lists, the shared control pool, and the pre/post split.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError
from mlsynth.utils.datautils import geoex_dataprep


def _is_control(value: Any, control_label: Optional[str]) -> bool:
    """A unit is control if its cell value is blank/NaN or the ``control_label``."""
    if pd.isna(value):
        return True
    s = str(value).strip()
    if s == "":
        return True
    if control_label is not None and s == str(control_label).strip():
        return True
    return False


def multicell_dataprep(
    df: pd.DataFrame,
    unit_id_column_name: str,
    time_period_column_name: str,
    outcome_column_name: str,
    *,
    cell_column_name: str,
    post_col: str,
    control_label: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve a multi-cell geo panel from a cell-membership column.

    Returns a dict with ``Ywide`` (full ``time x unit`` panel), ``cell_map``
    (``{label: [units]}`` for each treatment cell), ``control_units`` (the shared
    donor pool), ``pre_periods`` (the pre/post split from ``post_col``) and
    ``time_labels``.

    Raises
    ------
    MlsynthConfigError
        If ``cell_column_name`` or ``post_col`` is absent.
    MlsynthDataError
        If the cell label varies within a unit, or there are no treatment cells,
        or no control units.
    """
    for col in (cell_column_name, post_col):
        if col not in df.columns:
            raise MlsynthConfigError(f"column {col!r} not found in df.")

    full = geoex_dataprep(df, unit_id_column_name, time_period_column_name,
                          outcome_column_name)
    pre_periods = geoex_dataprep(
        df, unit_id_column_name, time_period_column_name, outcome_column_name,
        post_col=post_col)["pre_periods"]
    Ywide = full["Ywide"]

    # Cell membership is a per-unit attribute: it must be constant over time.
    grp = df.groupby(unit_id_column_name)[cell_column_name]
    varying = grp.nunique(dropna=False)
    if (varying > 1).any():
        bad = list(varying[varying > 1].index)[:5]
        raise MlsynthDataError(
            f"cell column {cell_column_name!r} must be constant per unit; it "
            f"varies for units {bad}."
        )
    unit_cell = grp.first()

    cell_map: Dict[str, List] = {}
    control_units: List = []
    for unit in Ywide.columns:
        value = unit_cell.get(unit)
        if _is_control(value, control_label):
            control_units.append(unit)
        else:
            cell_map.setdefault(str(value).strip(), []).append(unit)

    if not cell_map:
        raise MlsynthDataError("no treatment cells found in the cell column.")
    if not control_units:
        raise MlsynthDataError(
            "no control units found — every unit is assigned to a cell, leaving "
            "no shared donor pool."
        )

    return {
        "Ywide": Ywide,
        "cell_map": cell_map,
        "control_units": control_units,
        "pre_periods": pre_periods,
        "time_labels": full["time_labels"],
    }
