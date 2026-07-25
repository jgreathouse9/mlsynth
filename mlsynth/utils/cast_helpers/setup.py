"""Data preparation for the CAST estimator."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from ..datautils import dataprep
from .structures import CASTInputs


def prepare_cast_inputs(
    df: pd.DataFrame,
    outcome: str,
    treat: str,
    unitid: str,
    time: str,
) -> CASTInputs:
    """Pivot a long staggered panel into :class:`CASTInputs`.

    Rides :func:`mlsynth.utils.datautils.dataprep`, which returns the wide
    outcome matrix plus a per-cohort map of adoption times; both the
    single-cohort and multi-cohort (staggered) layouts are accepted, since a
    block design is just a staggered design with one adoption time.

    Raises
    ------
    MlsynthDataError
        If the panel is unbalanced, has no never-treated unit, or has no
        treated cells to impute.
    """
    prepared = dataprep(df, unitid, time, outcome, treat)

    if "cohorts" in prepared:                     # staggered / multi-cohort
        Ywide = prepared["Ywide"]
        time_labels = np.asarray(prepared["time_labels"])
        unit_names = np.asarray(list(Ywide.columns), dtype=object)
        Y = Ywide.to_numpy(dtype=float).T         # Ywide is time x unit
        adoption = {}
        for start_time, cohort in prepared["cohorts"].items():
            idx = int(np.where(time_labels == start_time)[0][0])
            for u in cohort["treated_units"]:
                adoption[u] = idx
    else:                                         # single treated unit
        y = np.asarray(prepared["y"], dtype=float).ravel()
        Y0 = np.asarray(prepared["donor_matrix"], dtype=float)
        time_labels = np.asarray(prepared.get(
            "time_labels", np.arange(int(prepared["total_periods"]))))
        unit_names = np.concatenate([
            np.array([prepared["treated_unit_name"]], dtype=object),
            np.asarray(prepared["donor_names"], dtype=object),
        ])
        Y = np.vstack([y[None, :], Y0.T])
        adoption = {prepared["treated_unit_name"]: int(prepared["pre_periods"])}

    if np.isnan(Y).any():
        raise MlsynthDataError(
            "CAST does not support missing outcomes; impute or drop them first."
        )
    N, T = Y.shape
    adoption_index = np.array([adoption.get(u, T) for u in unit_names], dtype=int)

    if (adoption_index >= T).all():                # pragma: no cover - dataprep rejects this first
        raise MlsynthDataError("CAST found no treated unit-periods.")
    if not (adoption_index >= T).any():
        raise MlsynthDataError(
            "CAST needs at least one never-treated unit to anchor the staircase."
        )
    if (adoption_index == 0).any():
        raise MlsynthDataError(
            "CAST requires at least one pre-treatment period for every unit; "
            "some unit is treated in the first period."
        )
    if N < 3:
        raise MlsynthDataError(f"CAST needs at least 3 units; got {N}.")

    A = np.ones((N, T), dtype=int)
    for i, t0 in enumerate(adoption_index):
        if t0 < T:
            A[i, t0:] = 0

    return CASTInputs(
        Y=Y, A=A, unit_names=unit_names, time_labels=time_labels,
        adoption_index=adoption_index,
    )
