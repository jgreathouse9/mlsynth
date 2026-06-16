"""Data setup for ORTHSC: dataprep -> the treated / control / instrument split.

dataprep gives the treated unit and the full donor pool; ORTHSC then partitions
the donors into the synthetic-control pool and the instrument set (units excluded
from the controls, used as instruments for the weights).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ...exceptions import MlsynthDataError
from ..datautils import dataprep


def build_orthsc_inputs(config) -> Dict[str, Any]:
    """Prepare the ORTHSC arrays from the long panel.

    Returns a dict with the treated pre/post series, the control and instrument
    matrices (units x time), the pre/post split, time labels, and the control /
    instrument / treated labels.
    """
    prep = dataprep(
        df=config.df,
        unit_id_column_name=config.unitid,
        time_period_column_name=config.time,
        outcome_column_name=config.outcome,
        treatment_indicator_column_name=config.treat,
    )
    if "y" not in prep or "donor_matrix" not in prep:
        raise MlsynthDataError(
            "ORTHSC requires a single treated unit (dataprep returned a "
            "multi-cohort structure).")

    y = np.asarray(prep["y"], dtype=float).ravel()
    Y0 = np.asarray(prep["donor_matrix"], dtype=float)            # (T, J)
    pre = int(prep["pre_periods"])
    time_labels = np.asarray(prep["time_labels"])
    donor_names = [str(d) for d in prep["donor_names"]]
    col = {name: j for j, name in enumerate(donor_names)}

    instruments: List[str] = [str(u) for u in config.instruments]
    missing = [u for u in instruments if u not in col]
    if missing:
        raise MlsynthDataError(
            f"instrument unit(s) not in the donor pool: {missing}; "
            f"donors are {donor_names}.")

    if config.controls is not None:
        controls = [str(u) for u in config.controls]
        missing_c = [u for u in controls if u not in col]
        if missing_c:
            raise MlsynthDataError(
                f"control unit(s) not in the donor pool: {missing_c}.")
    else:
        instr_set = set(instruments)
        controls = [d for d in donor_names if d not in instr_set]
    if not controls:
        raise MlsynthDataError("ORTHSC needs at least one control unit.")

    YJ = Y0[:, [col[c] for c in controls]].T                      # (Jc, T)
    Zc = Y0[:, [col[z] for z in instruments]].T                   # (Qz, T)
    return {
        "pre_y0": y[:pre], "post_y0": y[pre:],
        "pre_yj": YJ[:, :pre], "post_yj": YJ[:, pre:],
        "Z": Zc[:, :pre],
        "y": y, "YJ": YJ, "pre": pre, "time_labels": time_labels,
        "controls": controls, "instruments": instruments,
        "treated_name": str(prep.get("treated_unit_name", "treated")),
    }
