"""Long-DataFrame -> NumPy boundary for MTGP (wraps ``dataprep``)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...exceptions import MlsynthConfigError, MlsynthDataError
from ..datautils import dataprep
from .structures import MTGPInputs


def prepare_mtgp_inputs(
    df: pd.DataFrame, outcome: str, unitid: str, time: str, treat: str,
    population: str | None = None,
) -> MTGPInputs:
    """Pivot a long panel into MTGP's ``(T, D)`` matrix (treated column 0).

    When ``population`` is given, build the per-cell inverse-population noise
    scaling (``pop.mean() / pop``); otherwise the noise is homoskedastic.
    """
    prepared = dataprep(df, unitid, time, outcome, treat)
    if "cohorts" in prepared:
        raise MlsynthDataError(
            "MTGP supports a single treated unit; the panel appears to "
            "contain multiple treatment cohorts."
        )
    y_target = np.asarray(prepared["y"], dtype=float)           # (T,)
    donors = np.asarray(prepared["donor_matrix"], dtype=float)   # (T, N)
    if donors.ndim != 2 or donors.shape[1] < 1:  # pragma: no cover - dataprep guards
        raise MlsynthDataError("MTGP requires a 2D donor matrix with >= 1 donor.")
    T = int(prepared["total_periods"])
    T0 = int(prepared["pre_periods"])
    if T0 < 2:
        raise MlsynthDataError(f"MTGP needs at least 2 pre-treatment periods; got {T0}.")
    if T - T0 < 1:  # pragma: no cover - a treated period forces >= 1 post
        raise MlsynthDataError("MTGP needs at least 1 post-treatment period.")
    Y = np.column_stack([y_target, donors])                     # (T, D), treated col 0
    if not np.isfinite(Y).all():
        raise MlsynthDataError("MTGP requires a balanced panel with no missing outcomes.")

    treated_name = str(prepared.get("treated_unit_name", "treated"))
    donor_names = list(prepared.get("donor_names", range(donors.shape[1])))
    time_labels = np.asarray(prepared["time_labels"])

    if population is None:
        inv_pop = np.ones_like(Y)
    else:
        if population not in df.columns:
            raise MlsynthConfigError(f"population column '{population}' not in the panel.")
        wide = df.pivot(index=time, columns=unitid, values=population)
        try:
            pop = wide.reindex(index=pd.Index(time_labels),
                               columns=[treated_name, *donor_names]).to_numpy(dtype=float)
        except Exception as exc:  # pragma: no cover - defensive
            raise MlsynthDataError(f"could not align population column: {exc}") from exc
        if not np.isfinite(pop).all() or np.any(pop <= 0):
            raise MlsynthDataError("MTGP population must be positive and fully observed.")
        inv_pop = float(np.mean(pop)) / pop                     # dimensionless, median ~1

    return MTGPInputs(
        Y=Y, y_target=y_target, inv_pop=inv_pop, T0=T0, T=T, D=Y.shape[1],
        treated_unit_name=treated_name, donor_names=donor_names,
        time_labels=time_labels,
    )
