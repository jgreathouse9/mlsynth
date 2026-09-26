"""Long-DataFrame to NumPy boundary for SL (the only pandas touchpoint).

Ingestion goes through :func:`mlsynth.utils.datautils.dataprep`, which already
resolves the treated unit, the donor pool and the pre/post split. Nothing here
re-derives any of that. The one thing ``dataprep`` does not supply is a
time-varying covariate block -- its ``covariates=`` argument aggregates to a
per-unit pre-treatment mean, which is the right object for predictor-weight
balancing and the wrong one for the forest expert, which needs the paths. So the
covariates are pivoted here, on the unit order ``dataprep`` returned, and the two
stay aligned by construction.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from ..datautils import dataprep
from ..fast_scm_helpers.structure import IndexSet
from .structures import SLInputs


def _covariate_block(df: pd.DataFrame, cols: Sequence[str], *, unitid: str,
                     time: str, times: np.ndarray,
                     order: Sequence) -> np.ndarray:
    """Every unit's path for each named covariate, as ``(T, len(cols) * n_units)``.

    This mirrors the authors' ``external_covariates``, which is the full state
    matrix of the covariate, not the treated unit's column alone.
    """
    planes = []
    for c in cols:
        wide = df.pivot(index=time, columns=unitid, values=c).reindex(times)
        missing = [u for u in order if u not in wide.columns]
        if missing:  # pragma: no cover - dataprep's pivot would have raised first
            raise MlsynthDataError(
                f"Covariate '{c}' is missing units {missing[:5]} after pivoting.")
        block = wide[list(order)]
        if block.isna().any().any():
            raise MlsynthDataError(
                f"Covariate '{c}' is incomplete after pivoting; SL needs a "
                f"balanced panel with no gaps.")
        planes.append(block.to_numpy(dtype=float))
    return np.column_stack(planes)


def prepare_sl_inputs(df: pd.DataFrame, *, unitid: str, time: str, outcome: str,
                      treat: str, covariates: Sequence[str] = ()) -> SLInputs:
    """Pivot the panel to NumPy via ``dataprep`` and attach the covariate block.

    Parameters
    ----------
    df : pd.DataFrame
        Long panel.
    unitid, time, outcome, treat : str
        Column names.
    covariates : sequence of str
        Time-varying covariates for the forest expert. Optional; the other three
        experts read donor outcomes alone.

    Returns
    -------
    SLInputs

    Raises
    ------
    MlsynthDataError
        If a named covariate is missing or incomplete, or if there are fewer
        than two pre-treatment periods to split.
    """
    cov = list(covariates)
    missing = [c for c in cov if c not in df.columns]
    if missing:
        raise MlsynthDataError(
            f"Covariates not found in the panel: {missing}. SL reads them for "
            f"the 'forest' expert.")

    prepped = dataprep(df, unitid, time, outcome, treat)
    if "y" not in prepped:
        raise MlsynthDataError(
            "SL expects exactly one treated unit; dataprep returned a "
            "multi-cohort panel. Subset to one treated unit, or use a "
            "staggered-adoption estimator.")

    times = np.asarray(prepped["time_labels"])
    donors = list(prepped["donor_names"])
    treated = prepped["treated_unit_name"]
    T0 = int(prepped["pre_periods"])
    if T0 < 2:
        raise MlsynthDataError(
            f"SL needs at least two pre-treatment periods, since Algorithm 1 "
            f"splits them into an expert-training window and a weighting "
            f"window; found {T0}.")

    block = (_covariate_block(df, cov, unitid=unitid, time=time, times=times,
                              order=[treated, *donors]) if cov else None)

    return SLInputs(
        unit_index=IndexSet.from_labels(donors),
        time_index=IndexSet.from_labels(times),
        y=np.asarray(prepped["y"], dtype=float).ravel(),
        Yco=np.asarray(prepped["donor_matrix"], dtype=float),
        T0=T0,
        treated_label=treated,
        covariates=block,
        covariate_names=tuple(cov),
        metadata={"outcome": outcome,
                  "post_periods": int(prepped["post_periods"]),
                  "total_periods": int(prepped["total_periods"])},
    )
