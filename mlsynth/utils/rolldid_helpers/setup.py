"""Ingestion for ROLLDID: long panel -> per-unit series + cohort structure.

The rolling-DiD transformation is unit-level, so we need each unit's full
outcome series, its treatment **cohort** (the first period in which its
treatment indicator turns on, ``g``), and the set of never-treated units. The
design is *common timing* when every treated unit shares one cohort, and
*staggered* otherwise.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from mlsynth.exceptions import MlsynthConfigError, MlsynthDataError


def rolldid_setup(
    df: pd.DataFrame,
    unit_id: str,
    time_id: str,
    outcome: str,
    treat: str,
) -> Dict[str, Any]:
    """Resolve the panel into the inputs the rolling-DiD pipeline needs.

    Returns a dict with ``Ywide`` (``time x unit``), ``cohort_of`` (``{unit: g}``
    for treated units), ``never`` (never-treated unit labels), ``treated`` (all
    eventually-treated labels), ``times`` (sorted time labels), and ``design``
    (``"common"`` / ``"staggered"``).

    Raises
    ------
    MlsynthConfigError
        If a required column is missing.
    MlsynthDataError
        If the panel is unbalanced, the treatment indicator is not 0/1, there
        are no treated units, no never-treated controls, or treatment switches
        off (the indicator must be absorbing once on).
    """
    for col in (unit_id, time_id, outcome, treat):
        if col not in df.columns:
            raise MlsynthConfigError(f"column {col!r} not found in df.")

    work = df[[unit_id, time_id, outcome, treat]].copy()
    w = work[treat]
    if not np.isin(w.dropna().unique(), [0, 1]).all():
        raise MlsynthDataError(f"treatment indicator {treat!r} must be 0/1.")

    Ywide = work.pivot(index=time_id, columns=unit_id, values=outcome).sort_index()
    if Ywide.isna().any().any():
        raise MlsynthDataError("panel is unbalanced (missing unit x time cells).")
    times = Ywide.index.to_numpy()

    # Per-unit cohort g = first period the indicator is on; never-treated have none.
    cohort_of: Dict[Any, Any] = {}
    for unit, sub in work.groupby(unit_id):
        sub = sub.sort_values(time_id)
        on = sub[sub[treat] == 1]
        if on.empty:
            continue
        g = on[time_id].iloc[0]
        # the indicator must be absorbing: once on, never off
        from_g = sub[sub[time_id] >= g][treat]
        if not (from_g == 1).all():
            raise MlsynthDataError(
                f"treatment for unit {unit!r} switches off after {g!r}; the "
                "indicator must be absorbing (DiD adoption design).")
        cohort_of[unit] = g

    treated = sorted(cohort_of, key=lambda u: str(u))
    never = [u for u in Ywide.columns if u not in cohort_of]
    if not treated:
        raise MlsynthDataError("no treated units found in the treatment indicator.")
    if not never:
        raise MlsynthDataError(
            "no never-treated control units found; ROLLDID needs a never-treated "
            "group as the comparison.")

    cohorts = sorted(set(cohort_of.values()))
    design = "common" if len(cohorts) == 1 else "staggered"

    return {
        "Ywide": Ywide,
        "cohort_of": cohort_of,
        "never": never,
        "treated": treated,
        "times": times,
        "cohorts": cohorts,
        "design": design,
    }
