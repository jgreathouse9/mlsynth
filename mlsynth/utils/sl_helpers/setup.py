"""Long-DataFrame to NumPy boundary for SL (the only pandas touchpoint).

Every pivot here is :func:`mlsynth.utils.datautils.dataprep`. It resolves the
treated unit, the donor pool and the pre/post split for the outcome, and it is
indifferent to which column it is handed -- its job is to organise and sort a
panel -- so the forest expert's time-varying covariates come from calling it once
per covariate column with that column in the outcome slot, then reindexing onto
the unit and period order the outcome call returned. That is the pattern
``compsc_helpers.setup`` uses for its multiple outcomes, and it keeps the
covariate block aligned with ``Yco`` by construction instead of by a second,
separately written pivot.

``dataprep``'s own ``covariates=`` argument is a different object: it aggregates
each column to a per-unit pre-treatment mean, which is what predictor-weight
balancing wants and not what a forecaster fit on the paths wants.

That pivot is also the limit of what panel covariates can say: the block it
returns is one column per panel unit, so a series for a unit outside the donor
pool has nowhere to go. The authors' own application passes ``generate_experts``
employment for 50 states against a donor pool of six, which
``external_covariates`` is for. It is aligned by period label and appended after
the panel-derived planes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from ...exceptions import MlsynthDataError
from ..datautils import dataprep
from ..fast_scm_helpers.structure import IndexSet
from .structures import SLInputs


def _covariate_block(df: pd.DataFrame, cols: Sequence[str], *, unitid: str,
                     time: str, treat: str, times: np.ndarray,
                     order: List[Any]) -> np.ndarray:
    """Every unit's path for each named covariate, as ``(T, len(cols) * n_units)``.

    One ``dataprep`` call per column, reindexed onto the outcome's own unit and
    period order. This mirrors the authors' ``external_covariates``, which is the
    full state matrix of the covariate, not the treated unit's column alone.

    ``dataprep`` returns a gap as a missing cell instead of raising, so
    completeness is checked here and the message names the column.
    """
    planes = []
    for c in cols:
        wide = dataprep(df, unitid, time, c, treat)["Ywide"]
        block = wide.reindex(index=times, columns=order)
        if block.isna().any().any():
            raise MlsynthDataError(
                f"Covariate '{c}' is incomplete after pivoting; SL needs a "
                f"balanced panel with no gaps.")
        planes.append(block.to_numpy(dtype=float))
    return np.column_stack(planes)


def _external_block(ext: pd.DataFrame, *, time: str,
                    times: np.ndarray) -> "tuple[np.ndarray, tuple]":
    """The external frame reindexed onto the panel's periods.

    Aligning by label and not by row order is the whole point: a positionally
    stacked block passes every shape check and pairs each period with another
    period's covariates.
    """
    names = tuple(c for c in ext.columns if c != time)
    frame = ext.set_index(time)[list(names)]
    bad = [c for c in names if not pd.api.types.is_numeric_dtype(frame[c])]
    if bad:
        raise MlsynthDataError(
            f"External covariate columns {bad} are not numeric; the forest's "
            f"design has to be a float matrix.")
    block = frame.reindex(index=times)
    if block.isna().any().any():
        gaps = block.index[block.isna().any(axis=1)].tolist()
        raise MlsynthDataError(
            f"external_covariates is missing {len(gaps)} of the panel's "
            f"periods, the first being {gaps[:5]}. SL needs a value at every "
            f"period, since the forest predicts over all of them.")
    return block.to_numpy(dtype=float), names


def prepare_sl_inputs(df: pd.DataFrame, *, unitid: str, time: str, outcome: str,
                      treat: str, covariates: Sequence[str] = (),
                      external_covariates: "pd.DataFrame | None" = None) -> SLInputs:
    """Pivot the panel to NumPy through ``dataprep``, outcome and covariates alike.

    Parameters
    ----------
    df : pd.DataFrame
        Long panel.
    unitid, time, outcome, treat : str
        Column names.
    covariates : sequence of str
        Time-varying covariates for the forest expert, pivoted per panel unit.
        Optional; the other three experts read donor outcomes alone.
    external_covariates : pd.DataFrame, optional
        Series for units outside the panel, carrying the ``time`` column plus
        one column each. Appended to the block after the panel-derived planes.

    Returns
    -------
    SLInputs

    Raises
    ------
    MlsynthDataError
        If a named covariate is missing or incomplete, if
        ``external_covariates`` is non-numeric or does not cover every period,
        if the panel is multi-cohort, or if there are fewer than two
        pre-treatment periods to split.
    """
    cov = list(covariates)
    missing = [c for c in cov if c not in df.columns]
    if missing:
        raise MlsynthDataError(
            f"Covariates not found in the panel: {missing}. SL reads them for "
            f"the 'forest' expert.")

    prepped: Dict[str, Any] = dataprep(df, unitid, time, outcome, treat)
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

    planes = []
    if cov:
        planes.append(_covariate_block(df, cov, unitid=unitid, time=time,
                                       treat=treat, times=times,
                                       order=[treated, *donors]))
    ext_names: tuple = ()
    if external_covariates is not None:
        ext_block, ext_names = _external_block(external_covariates, time=time,
                                               times=times)
        planes.append(ext_block)
    block = np.column_stack(planes) if planes else None

    return SLInputs(
        unit_index=IndexSet.from_labels(donors),
        time_index=IndexSet.from_labels(times),
        y=np.asarray(prepped["y"], dtype=float).ravel(),
        Yco=np.asarray(prepped["donor_matrix"], dtype=float),
        T0=T0,
        treated_label=treated,
        covariates=block,
        covariate_names=tuple(cov),
        external_covariate_names=ext_names,
        metadata={"outcome": outcome,
                  "post_periods": int(prepped["post_periods"]),
                  "total_periods": int(prepped["total_periods"])},
    )
