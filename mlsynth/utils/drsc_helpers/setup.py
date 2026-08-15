"""Micro-panel ingestion for DRSC.

This does not go through :func:`mlsynth.utils.datautils.dataprep`, and the
reason is structural rather than stylistic: ``dataprep`` pivots to one outcome
per ``(unit, time)`` and raises ``Index contains duplicate entries`` on a panel
whose cells hold many individual records. DRSC needs the individuals -- each
cell contributes a design matrix and an outcome vector to a distribution
regression -- so it follows the precedent :mod:`mlsynth.utils.dsc_helpers.setup`
already set for micro-panel estimators.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from .structures import DRSCCell, DRSCInputs


def prepare_drsc_inputs(
    df: pd.DataFrame,
    outcome: str,
    treat: str,
    unitid: str,
    time: str,
    covariates: Sequence[str],
) -> DRSCInputs:
    """Pivot a long micro-panel into per-cell design matrices.

    Parameters
    ----------
    df : pandas.DataFrame
        One row per (unit, time, individual). ``treat`` is constant within a
        cell.
    outcome, treat, unitid, time : str
        Column names.
    covariates : sequence of str
        Design columns, in order. An intercept is prepended here.

    Returns
    -------
    DRSCInputs

    Raises
    ------
    MlsynthDataError
        A column is missing or non-numeric, there is no treated unit or no
        donor, there are no pre-treatment periods, or a cell is too small to
        identify the distribution regression.
    """
    covariates = list(covariates)
    missing = [c for c in [outcome, treat, unitid, time] + covariates
               if c not in df.columns]
    if missing:
        raise MlsynthDataError(
            f"column(s) {missing} not in the panel; available: "
            f"{list(df.columns)}")

    for c in [outcome] + covariates:
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise MlsynthDataError(
                f"column {c!r} is not numeric; DRSC needs a numeric outcome "
                "and numeric covariates (encode categoricals as indicators "
                "yourself).")

    treated_rows = df[df[treat] == 1]
    if treated_rows.empty:
        raise MlsynthDataError(
            "no treated rows found; DRSC needs exactly one treated unit with "
            f"at least one post-treatment period (column {treat!r} == 1).")
    treated_units = list(pd.unique(treated_rows[unitid]))
    if len(treated_units) != 1:
        raise MlsynthDataError(
            f"DRSC supports a single treated unit; found {len(treated_units)}: "
            f"{treated_units[:5]}. Aggregate them or fit separately.")
    treated = treated_units[0]

    times = list(pd.unique(df[time].sort_values()))
    first_treated = treated_rows[time].min()
    pre = [t for t in times if t < first_treated]
    post = [t for t in times if t >= first_treated]
    if not pre:
        raise MlsynthDataError(
            "no pre-treatment periods: treatment is in force from the first "
            f"period ({first_treated!r}). DRSC needs at least one pre-period "
            "to estimate the weights.")
    if not post:  # pragma: no cover - implied by a non-empty treated_rows
        raise MlsynthDataError("no post-treatment periods.")

    units = [treated] + [u for u in pd.unique(df[unitid]) if u != treated]
    if len(units) < 2:
        raise MlsynthDataError(
            "no donor units: DRSC needs at least one untreated unit to form "
            "the counterfactual.")

    k = len(covariates) + 1
    cells: Dict[Tuple[Any, Any], DRSCCell] = {}
    for (u, t), g in df.groupby([unitid, time], sort=False, observed=True):
        if u not in set(units) or t not in set(times):  # pragma: no cover
            continue
        X = np.empty((len(g), k), dtype=np.float64)
        X[:, 0] = 1.0
        for j, c in enumerate(covariates):
            X[:, j + 1] = g[c].to_numpy(dtype=np.float64)
        y = g[outcome].to_numpy(dtype=np.float64)
        if len(g) <= k:
            raise MlsynthDataError(
                f"cell ({u!r}, {t!r}) has {len(g)} observation(s) for {k} "
                "parameters; the distribution regression is not identified "
                "there. DRSC's asymptotics are in the within-cell sample "
                "size, so every cell needs to be comfortably larger than the "
                "design.")
        if not (np.all(np.isfinite(X)) and np.all(np.isfinite(y))):
            raise MlsynthDataError(
                f"cell ({u!r}, {t!r}) carries NaN/inf in the outcome or "
                "covariates; DRSC needs complete cases.")
        cells[(u, t)] = DRSCCell(X=X, y=y)

    expected = {(u, t) for u in units for t in times}
    absent = sorted(expected - set(cells), key=lambda p: (str(p[0]), str(p[1])))
    if absent:
        raise MlsynthDataError(
            f"the panel is unbalanced: {len(absent)} (unit, time) cell(s) have "
            f"no rows, e.g. {absent[:3]}. DRSC needs every unit observed in "
            "every period.")

    return DRSCInputs(cells=cells, unit_names=units, covariates=covariates,
                      pre_periods=pre, post_periods=post,
                      treated_unit_name=treated, outcome=outcome)
