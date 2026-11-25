"""Long-DataFrame to NumPy boundary for CPDA (the only pandas touchpoint)."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from ..datautils import PanelKeys, balance, dataprep
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


def _covariate_wides(df: pd.DataFrame, cols: Sequence[str], unitid: str,
                     time: str, treat: str,
                     keys: PanelKeys) -> Dict[str, pd.DataFrame]:
    """Each covariate's ``(T, units)`` matrix, read through ``dataprep``.

    ``dataprep`` reads the named column as its outcome, so a covariate gets the
    outcome's refusal: a missing or non-finite value in any unit-period. The
    refusal is raised again under the covariate's name.
    """
    wides = {}
    for c in cols:
        try:
            prep = dataprep(df, unitid, time, c, treat, allow_no_donors=True,
                            keys=keys)
        except MlsynthDataError as exc:
            raise MlsynthDataError(
                f"Covariate '{c}' is incomplete; CPDA needs every covariate "
                f"observed and finite in every unit-period. {exc}") from exc
        wides[c] = prep.Ywide
    return wides


def _cube(wides: Dict[str, pd.DataFrame], cols: Sequence[str],
          times: np.ndarray, order: List[Any]) -> np.ndarray:
    """Stack covariates into ``(T, len(order), k)``, aligned to ``times``."""
    if not cols:  # pragma: no cover - CPDAConfig refuses an empty covariate list
        return np.empty((len(times), len(order), 0))
    planes = [wides[c][order].to_numpy(dtype=float) for c in cols]
    return np.stack(planes, axis=2)


def prepare_cpda_inputs(df: pd.DataFrame, *, unitid: str, time: str,
                        outcome: str, treat: str,
                        covariates: Sequence[str]) -> CPDAInputs:
    """Read the panel through ``balance`` and ``dataprep``, build ``IndexSet``\\ s, split pre and post.

    ``balance`` refuses a duplicate or absent unit-period. ``dataprep`` supplies
    the outcome matrix, the time labels, the donor order and the pre-period
    count, and refuses a missing or non-finite outcome anywhere in the panel, a
    non-binary treatment and a treatment that switches off after it starts.
    Each covariate's matrix is read through ``dataprep`` on the same key scan,
    so a missing or non-finite covariate value is refused too.

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
        If a named column is missing, the panel is unbalanced or has a
        duplicate unit-period, an outcome or covariate value is missing or not
        finite, the treatment is not binary or switches off, there is not
        exactly one treated unit, there is no donor, or there are fewer than
        two pre-treatment periods.
    """
    missing = [c for c in [outcome, treat, unitid, time, *covariates]
               if c not in df.columns]
    if missing:
        raise MlsynthDataError(
            f"Columns not found in the panel: {missing}. CPDA reads the "
            f"outcome, the treatment flag, the unit and time keys, and every "
            f"named covariate.")

    # balance() refuses a duplicate or absent unit-period, and its key scan is
    # shared by every dataprep call below instead of being taken again.
    keys = balance(df, unitid, time)
    treated_unit, intervention_time = derive_treatment(df, unitid, time, treat)
    prep = dataprep(df, unitid, time, outcome, treat, allow_no_donors=True,
                    keys=keys)

    times = np.asarray(prep.time_labels)
    time_index = IndexSet.from_labels(times)
    wide = prep.Ywide

    donors = list(prep.donor_names)
    if not donors:
        raise MlsynthDataError("CPDA needs at least one donor unit.")
    unit_index = IndexSet.from_labels(donors)

    cov = list(covariates)
    wides = _covariate_wides(df, cov, unitid, time, treat, keys)
    treated_cube = _cube(wides, cov, times, [treated_unit])
    donor_cube = _cube(wides, cov, times, donors)

    T0 = int(prep.pre_periods)
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
