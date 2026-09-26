"""Long-DataFrame to NumPy boundary for CPDA (the only pandas touchpoint)."""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from ..fast_scm_helpers.structure import IndexSet
from .structures import CPDAInputs


def derive_treatment(df: pd.DataFrame, unitid: str, time: str,
                     treat: str) -> Tuple[Any, Any]:
    """Read the single treated unit and its first treated period."""
    treated_rows = df[df[treat] == 1]
    if treated_rows.empty:
        raise MlsynthDataError(
            f"No treated rows (treat == 1) found in '{treat}'. CPDA needs one "
            f"treated unit with a post-treatment window.")
    treated_units = pd.unique(treated_rows[unitid])
    if len(treated_units) != 1:
        raise MlsynthDataError(
            f"CPDA expects exactly one treated unit; found "
            f"{len(treated_units)}: {list(treated_units)[:5]}.")
    treated_unit = treated_units[0]
    intervention_time = treated_rows.loc[
        treated_rows[unitid] == treated_unit, time].min()
    return treated_unit, intervention_time


def _cube(df: pd.DataFrame, cols: Sequence[str], unitid: str, time: str,
          times: np.ndarray, order: List[Any]) -> np.ndarray:
    """Stack covariates into ``(T, len(order), k)``, aligned to ``times``."""
    if not cols:  # pragma: no cover - CPDAConfig refuses an empty covariate list
        return np.empty((len(times), len(order), 0))
    planes = []
    for c in cols:
        wide = df.pivot(index=time, columns=unitid, values=c).reindex(times)
        if wide.isna().any().any():
            raise MlsynthDataError(
                f"Covariate '{c}' is incomplete after pivoting; CPDA needs a "
                f"balanced panel with no gaps.")
        planes.append(wide[order].to_numpy(dtype=float))
    return np.stack(planes, axis=2)


def prepare_cpda_inputs(df: pd.DataFrame, *, unitid: str, time: str,
                        outcome: str, treat: str,
                        covariates: Sequence[str]) -> CPDAInputs:
    """Pivot the panel to NumPy, build ``IndexSet``\\ s, split pre and post.

    Parameters
    ----------
    df : pd.DataFrame
        Long panel.
    unitid, time, outcome, treat : str
        Column names.
    covariates : sequence of str
        The ``x_it`` of Equation 2. CPDA is the covariate-adjusted panel data
        approach, so at least one is required; the config enforces that.

    Returns
    -------
    CPDAInputs
        Pure-NumPy container.

    Raises
    ------
    MlsynthDataError
        If a named column is missing, the panel has gaps, there is not exactly
        one treated unit, or there are fewer than two pre-treatment periods.
    """
    missing = [c for c in [outcome, treat, unitid, time, *covariates]
               if c not in df.columns]
    if missing:
        raise MlsynthDataError(
            f"Columns not found in the panel: {missing}. CPDA reads the "
            f"outcome, the treatment flag, the unit and time keys, and every "
            f"named covariate.")

    treated_unit, intervention_time = derive_treatment(df, unitid, time, treat)

    times = np.sort(pd.unique(df[time]))
    time_index = IndexSet.from_labels(times)

    wide = df.pivot(index=time, columns=unitid, values=outcome).reindex(times)
    if wide.isna().any().any():
        raise MlsynthDataError(
            "CPDA requires a complete outcome panel after pivoting; some "
            "unit-period cells are missing.")

    donors = [u for u in wide.columns if u != treated_unit]
    if not donors:
        raise MlsynthDataError("CPDA needs at least one donor unit.")
    unit_index = IndexSet.from_labels(donors)

    cov = list(covariates)
    treated_cube = _cube(df, cov, unitid, time, times, [treated_unit])
    donor_cube = _cube(df, cov, unitid, time, times, donors)

    T0 = int(np.sum(times < intervention_time))
    if T0 < 2:
        raise MlsynthDataError(
            f"CPDA needs at least two pre-treatment periods; found {T0}.")

    return CPDAInputs(
        unit_index=unit_index,
        time_index=time_index,
        y=wide[treated_unit].to_numpy(dtype=float),
        x=treated_cube[:, 0, :],
        Yco=wide[donors].to_numpy(dtype=float),
        Xco=donor_cube,
        T0=T0,
        treated_label=treated_unit,
        covariate_names=tuple(cov),
        metadata={"outcome": outcome, "intervention_time": intervention_time},
    )
