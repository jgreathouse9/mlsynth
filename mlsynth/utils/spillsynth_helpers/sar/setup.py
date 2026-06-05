"""Long-format panel ingestion for the SAR spillover SCM (``method='sar'``).

Turns the user's long DataFrame plus a spatial-weight specification into a
:class:`SARInputs`: the treated/control outcome split, a row-normalised
control-to-control weight matrix ``W``, a treated-to-control weight vector
``w``, and (optionally) a time-varying covariate cube. The spatial weights may
be supplied as labelled ``pandas`` objects (aligned by unit label) or as bare
NumPy arrays (assumed already in control-label order).
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from ....exceptions import MlsynthDataError
from .sampler import row_normalize
from .structures import SARInputs


def _align_W(spatial_W, control_labels) -> np.ndarray:
    """Return an ``(N, N)`` control-to-control matrix in ``control_labels`` order."""
    labels = list(control_labels)
    N = len(labels)
    if isinstance(spatial_W, pd.DataFrame):
        cols = list(spatial_W.columns)
        # Drop a leading label column if present (e.g. exported "state" index).
        if spatial_W.shape[1] == N + 1 and spatial_W.index.name is None:
            first = cols[0]
            spatial_W = spatial_W.set_index(first)
        missing = [u for u in labels if u not in spatial_W.index or u not in spatial_W.columns]
        if missing:
            raise MlsynthDataError(
                f"SPILLSYNTH/sar: spatial_W missing rows/cols for units {missing[:5]}."
            )
        W = spatial_W.loc[labels, labels].to_numpy(dtype=float)
    else:
        W = np.asarray(spatial_W, dtype=float)
        if W.shape != (N, N):
            raise MlsynthDataError(
                f"SPILLSYNTH/sar: spatial_W has shape {W.shape}, expected ({N}, {N}). "
                "Pass a labelled DataFrame to align by unit, or an N x N array."
            )
    return W


def _align_w(spatial_w, control_labels) -> np.ndarray:
    """Return an ``(N,)`` treated-to-control weight vector in label order."""
    labels = list(control_labels)
    N = len(labels)
    if isinstance(spatial_w, pd.DataFrame):
        # Use the last numeric column as the weight, indexed by the first column.
        if spatial_w.shape[1] >= 2:
            spatial_w = spatial_w.set_index(spatial_w.columns[0]).iloc[:, -1]
        else:
            spatial_w = spatial_w.iloc[:, 0]
    if isinstance(spatial_w, pd.Series):
        missing = [u for u in labels if u not in spatial_w.index]
        if missing:
            raise MlsynthDataError(
                f"SPILLSYNTH/sar: spatial_w missing units {missing[:5]}."
            )
        w = spatial_w.loc[labels].to_numpy(dtype=float)
    elif isinstance(spatial_w, dict):
        w = np.array([float(spatial_w.get(u, 0.0)) for u in labels], dtype=float)
    else:
        w = np.asarray(spatial_w, dtype=float).ravel()
        if w.shape[0] != N:
            raise MlsynthDataError(
                f"SPILLSYNTH/sar: spatial_w has length {w.shape[0]}, expected {N}."
            )
    if not np.any(w > 0):
        raise MlsynthDataError(
            "SPILLSYNTH/sar: spatial_w has no positive entries (treated unit "
            "is linked to no controls)."
        )
    return w


def prepare_sar_inputs(
    df: pd.DataFrame,
    outcome: str,
    treat: str,
    unitid: str,
    time: str,
    spatial_W,
    spatial_w,
    covariates: Optional[Sequence[str]] = None,
) -> SARInputs:
    """Build :class:`SARInputs` from a long panel and a spatial-weight spec.

    The treated unit is the one whose ``treat`` indicator is ever 1; the first
    period at which it switches on defines ``T0``. ``spatial_W`` /
    ``spatial_w`` are aligned to the control-unit order. Covariate columns, if
    given, are assembled into a ``(T0, N, K)`` cube (controls only).
    """
    for col in (outcome, treat, unitid, time):
        if col not in df.columns:
            raise MlsynthDataError(f"SPILLSYNTH/sar: required column {col!r} missing.")
    if df[outcome].isna().any():
        raise MlsynthDataError("SPILLSYNTH/sar: outcome column contains NaN.")

    time_labels = np.array(sorted(df[time].unique()))
    T = int(time_labels.size)
    units = sorted(df[unitid].unique())

    treated_mask = df.groupby(unitid)[treat].max()
    treated_units = [u for u in units if treated_mask.get(u, 0) == 1]
    if len(treated_units) != 1:
        raise MlsynthDataError(
            f"SPILLSYNTH/sar: needs exactly one treated unit, found {len(treated_units)}."
        )
    treated = treated_units[0]
    controls = [u for u in units if u != treated]
    N = len(controls)
    if N < 3:
        raise MlsynthDataError("SPILLSYNTH/sar: needs at least 3 control units.")

    ywide = df.pivot(index=time, columns=unitid, values=outcome)
    if ywide.isna().any().any():
        raise MlsynthDataError("SPILLSYNTH/sar: panel is unbalanced (missing cells).")
    ywide = ywide.loc[time_labels]
    Y0 = ywide[treated].to_numpy(dtype=float)
    Yc = ywide[controls].to_numpy(dtype=float)

    # First treated period -> T0.
    twide = df.pivot(index=time, columns=unitid, values=treat).loc[time_labels, treated]
    on = np.where(twide.to_numpy() != 0)[0]
    if on.size == 0:
        raise MlsynthDataError("SPILLSYNTH/sar: treated unit never switches on.")
    T0 = int(on[0])
    if T0 < 2:
        raise MlsynthDataError("SPILLSYNTH/sar: needs >=2 pre-treatment periods.")
    if T - T0 < 1:
        raise MlsynthDataError("SPILLSYNTH/sar: needs >=1 post-treatment period.")

    W = _align_W(spatial_W, controls)
    w = _align_w(spatial_w, controls)
    Wn = row_normalize(W)
    wn = w / w.sum()

    X = None
    cov_names: tuple = ()
    if covariates:
        cov_names = tuple(covariates)
        cubes = []
        for c in cov_names:
            if c not in df.columns:
                raise MlsynthDataError(f"SPILLSYNTH/sar: covariate {c!r} missing.")
            cw = df.pivot(index=time, columns=unitid, values=c).loc[time_labels, controls]
            if cw.isna().any().any():
                raise MlsynthDataError(
                    f"SPILLSYNTH/sar: covariate {c!r} has missing control cells."
                )
            cubes.append(cw.to_numpy(dtype=float))
        X = np.stack(cubes, axis=2)            # (T, N, K)

    Y = np.vstack([Y0[None, :], Yc.T])         # (N+1, T)
    return SARInputs(
        Y=Y, Y0=Y0, Yc=Yc, Wn=Wn, wn=wn, T0=T0,
        treated_label=treated, control_labels=tuple(controls),
        time_labels=time_labels, X=X, covariate_names=cov_names,
    )
