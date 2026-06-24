"""Data ingestion for CMBSTS.

Wraps :func:`mlsynth.utils.datautils.dataprep` for the treated unit's pre/post
split and the wide outcome panel, then assembles the multivariate outcome group
``Y`` and the regression block ``X`` (control-unit paths plus exogenous
covariate columns). Optionally screens controls by dynamic time warping.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from ...exceptions import MlsynthConfigError, MlsynthDataError
from ..datautils import dataprep
from .structures import CMBSTSInputs


def _dtw_controls(
    Ywide: pd.DataFrame, treated: Any, pool: List[Any], T0: int, k: int
) -> List[Any]:
    """Keep the ``k`` pool units closest to the treated series by DTW (pre-period)."""
    try:
        from fastdtw import fastdtw
    except ImportError as exc:  # pragma: no cover - exercised only without fastdtw
        raise MlsynthConfigError(
            "CMBSTS: control_selection='dtw' needs the optional 'fastdtw' package "
            "(pip install fastdtw). Use control_selection='explicit' to avoid it."
        ) from exc
    target = Ywide[treated].to_numpy()[:T0]
    dists = []
    for u in pool:
        d, _ = fastdtw(target, Ywide[u].to_numpy()[:T0])
        dists.append((float(d), u))
    dists.sort(key=lambda r: r[0])
    return [u for _, u in dists[:k]]


def _covariate_matrix(
    df: pd.DataFrame, unitid: str, time: str, treated: Any,
    covariates: List[str], time_labels: np.ndarray,
) -> np.ndarray:
    """Covariate columns read from the treated unit's rows, aligned to time."""
    sub = df[df[unitid] == treated]
    missing = [c for c in covariates if c not in df.columns]
    if missing:
        raise MlsynthDataError(f"CMBSTS: covariate columns not found in df: {missing}.")
    sub = sub.drop_duplicates(subset=[time]).set_index(time)
    try:
        block = sub.reindex(time_labels)[covariates]
    except KeyError as exc:  # pragma: no cover - defensive, reindex tolerates misses
        raise MlsynthDataError(f"CMBSTS: covariate alignment failed: {exc}.") from exc
    arr = block.to_numpy(dtype=float)
    if not np.all(np.isfinite(arr)):
        raise MlsynthDataError(
            "CMBSTS: covariate values contain NaN/inf after aligning to the treated "
            "unit's time index (are the covariate columns populated on its rows?)."
        )
    return arr


def prepare_cmbsts_inputs(config: Any) -> CMBSTSInputs:
    """Build :class:`CMBSTSInputs` from the long DataFrame on the config."""
    prep = dataprep(
        df=config.df,
        unit_id_column_name=config.unitid,
        time_period_column_name=config.time,
        outcome_column_name=config.outcome,
        treatment_indicator_column_name=config.treat,
        allow_no_donors=True,
    )
    if "cohorts" in prep:
        raise MlsynthDataError(
            "CMBSTS: exactly one treated unit is expected (the others go in "
            "'group_units', not the treatment indicator). Found multiple treated "
            "units / cohorts."
        )
    treated = prep["treated_unit_name"]
    Ywide: pd.DataFrame = prep["Ywide"]
    T0 = int(prep["pre_periods"])
    T = int(prep["total_periods"])
    time_labels = np.asarray(prep["time_labels"])
    if T0 < 2:
        raise MlsynthDataError("CMBSTS: need at least 2 pre-intervention periods.")
    if T - T0 < 1:  # pragma: no cover - dataprep requires a sustained post window
        raise MlsynthDataError("CMBSTS: need at least 1 post-intervention period.")

    available = list(Ywide.columns)
    group_units = list(config.group_units or [])
    for u in group_units:
        if u not in available:
            raise MlsynthDataError(f"CMBSTS: group unit '{u}' not found in the panel.")
    if treated in group_units:
        raise MlsynthDataError("CMBSTS: the treated unit must not appear in 'group_units'.")
    series_names: List[Any] = [treated] + group_units
    Y = Ywide[series_names].to_numpy(dtype=float)

    # Control units (regressor series): explicit or DTW-screened.
    if config.control_selection == "dtw":
        pool = list(config.control_pool) if config.control_pool else [
            u for u in available if u not in series_names
        ]
        pool = [u for u in pool if u in available and u not in series_names]
        if len(pool) < config.n_controls:
            raise MlsynthDataError(
                f"CMBSTS: DTW pool has {len(pool)} eligible units, "
                f"fewer than n_controls={config.n_controls}."
            )
        control_units = _dtw_controls(Ywide, treated, pool, T0, int(config.n_controls))
    else:
        control_units = list(config.control_units or [])
        for u in control_units:
            if u not in available:
                raise MlsynthDataError(f"CMBSTS: control unit '{u}' not found in the panel.")
            if u in series_names:
                raise MlsynthDataError(
                    f"CMBSTS: control unit '{u}' is also a treated/group series."
                )

    blocks: List[np.ndarray] = []
    covariate_names = list(config.covariates or [])
    if covariate_names:
        blocks.append(_covariate_matrix(
            config.df, config.unitid, config.time, treated, covariate_names, time_labels))
    if control_units:
        blocks.append(Ywide[control_units].to_numpy(dtype=float))
    X = np.hstack(blocks) if blocks else None

    excl_post: Optional[np.ndarray] = None
    if config.excl_dates is not None:
        if config.excl_dates not in config.df.columns:
            raise MlsynthDataError(f"CMBSTS: excl_dates column '{config.excl_dates}' not in df.")
        sub = (config.df[config.df[config.unitid] == treated]
               .drop_duplicates(subset=[config.time]).set_index(config.time))
        flag = sub.reindex(time_labels)[config.excl_dates].to_numpy(dtype=float)
        excl_post = np.nan_to_num(flag[T0:], nan=0.0).astype(int)

    intervention_time = time_labels[T0] if T0 < T else time_labels[-1]
    return CMBSTSInputs(
        Y=Y, X=X, T0=T0, T=T, d=Y.shape[1],
        series_names=series_names, control_names=list(control_units),
        covariate_names=covariate_names, time_labels=time_labels,
        intervention_time=intervention_time, excl_post=excl_post,
    )
