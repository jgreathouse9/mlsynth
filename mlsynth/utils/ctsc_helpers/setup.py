"""Panel ingestion for the CTSC estimator.

Unlike a binary-treatment estimator, CTSC takes one or more
treatment / explanatory columns (continuous or discrete) that vary
over units and time, and reads them into a ``(n, T, K)`` array
alongside the ``(n, T)`` outcome.

:func:`~mlsynth.utils.datautils.dataprep` refuses a non-binary treatment,
which is what CTSC's treatments are, so the panel is read through
:func:`~mlsynth.utils.datautils.geoex_dataprep`, the treatment-agnostic
reader: once for the outcome, and once for each treatment variable read as
that call's outcome.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ...exceptions import MlsynthConfigError, MlsynthDataError
from ..datautils import geoex_dataprep
from .structures import CTSCInputs


def prepare_ctsc_inputs(
    df: pd.DataFrame,
    outcome: str,
    treatment_vars: Sequence[str],
    unitid: str,
    time: str,
    population_col: Optional[str] = None,
) -> CTSCInputs:
    """Read a long panel into :class:`CTSCInputs` through ``geoex_dataprep``.

    ``geoex_dataprep`` refuses a duplicate or absent unit-period, a row with
    a missing unit or period, and a missing or non-finite outcome. Each
    treatment variable is read the same way, and a missing or non-finite
    treatment value is refused under the variable's name. CTSC's own checks
    (at least one treatment variable, at least 3 units, valid population
    weights) stay on top.

    Parameters
    ----------
    df : pd.DataFrame
        Balanced long panel; one row per ``(unit, time)``. It is not
        modified.
    outcome : str
        Outcome column.
    treatment_vars : sequence of str
        The ``K >= 1`` treatment / explanatory columns (continuous or
        discrete). CTSC estimates a marginal effect for each.
    unitid, time : str
        Unit-id and time column names.
    population_col : str, optional
        Time-invariant per-unit weight column for the average effect
        (e.g. population). Defaults to uniform weights.
    """
    treatment_vars = list(treatment_vars)
    if len(treatment_vars) == 0:
        raise MlsynthConfigError("CTSC requires at least one treatment variable.")
    for col in (outcome, unitid, time, *treatment_vars):
        if col not in df.columns:
            raise MlsynthDataError(f"Required column {col!r} missing.")
    # geoex_dataprep runs balance() first, so a duplicate or absent
    # unit-period and a missing key are refused before the matrix is built,
    # and a missing or non-finite outcome after it.
    panel = geoex_dataprep(df, unitid, time, outcome)
    wide = panel["Ywide"]                                # (T, n), both axes sorted
    # Iterating the axes' arrays gives the scalars sorted(df[col].unique())
    # gave the old setup (numpy integers, Timestamps, str), so the labels keep
    # their types and time_labels its dtype.
    time_labels = np.array(list(wide.index.array))
    T = int(time_labels.size)
    unit_names = list(wide.columns.array)
    n = len(unit_names)
    if n < 3:
        raise MlsynthDataError(
            "CTSC needs at least 3 units (a synthetic control is built for "
            "every unit from the others)."
        )

    # CTSC's Y is unit x time; the C-order copy keeps the row-major layout
    # the old pivot produced.
    Y = np.ascontiguousarray(panel["Y"].T)

    K = len(treatment_vars)
    D = np.empty((n, T, K), dtype=float)
    for k, col in enumerate(treatment_vars):
        # The same panel, so the same sorted axes as the outcome's matrix.
        try:
            dose = geoex_dataprep(df, unitid, time, col)
        except MlsynthDataError as exc:
            raise MlsynthDataError(
                f"Treatment variable {col!r} is incomplete; CTSC needs every "
                f"treatment variable observed and finite in every unit-period. "
                f"{exc}"
            ) from exc
        D[:, :, k] = dose["Y"].T

    if population_col is not None:
        if population_col not in df.columns:
            raise MlsynthDataError(f"Population column {population_col!r} missing.")
        pop = df.groupby(unitid, observed=True)[population_col].first().loc[unit_names].to_numpy(float)
        if (pop < 0).any() or pop.sum() <= 0:
            raise MlsynthDataError("Population weights must be non-negative and sum > 0.")
        population_weights = pop / pop.sum()
    else:
        population_weights = np.full(n, 1.0 / n)

    return CTSCInputs(
        Y=Y,
        D=D,
        unit_names=list(unit_names),
        time_labels=time_labels,
        treatment_names=treatment_vars,
        population_weights=population_weights,
    )
