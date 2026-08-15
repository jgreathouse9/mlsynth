"""Ingestion for DMLFM: long panel to the sampler's design blocks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ...exceptions import MlsynthConfigError, MlsynthDataError
from ..datautils import dataprep


@dataclass
class DMLFMInputs:
    """Everything :func:`~mlsynth.utils.dmlfm_helpers.sampler.run_gibbs` needs.

    Estimation uses the control observations -- every ``(i, t)`` with the
    treatment indicator at zero, which includes the treated unit's own
    pre-treatment rows. Prediction covers every row of the treated unit, so the
    pre-treatment fit is reported alongside the post-treatment gap.
    """

    y: np.ndarray
    X: np.ndarray
    Z: np.ndarray
    A: np.ndarray
    X_tr: np.ndarray
    Z_tr: np.ndarray
    A_tr: np.ndarray
    unit_index: np.ndarray
    time_index: np.ndarray
    unit_index_tr: np.ndarray
    time_index_tr: np.ndarray
    unit_starts: np.ndarray
    time_starts: np.ndarray
    order_by_time: np.ndarray
    n_units: int
    n_periods: int
    r: int
    niter: int
    burn: int
    ar1: bool
    xlasso: bool
    zlasso: bool
    alasso: bool
    flasso: bool
    a1: float
    a2: float
    b1: float
    b2: float
    c1: float
    c2: float
    p1: float
    p2: float
    e1: float
    e2: float
    time_labels: np.ndarray
    pre_periods: int
    treated_name: str


def _starts(codes: np.ndarray) -> np.ndarray:
    """First row of each contiguous group -- ``breakID`` (blasso.cpp:119)."""
    change = np.flatnonzero(np.diff(codes)) + 1
    return np.concatenate([[0], change])


def prepare_dmlfm_inputs(cfg) -> DMLFMInputs:
    df = cfg.df.copy()
    for col in (cfg.outcome, cfg.treat, cfg.unitid, cfg.time):
        if col not in df.columns:
            raise MlsynthDataError(f"column {col!r} is not in the panel")

    covs = list(cfg.covariates or [])
    missing = [c for c in covs if c not in df.columns]
    if missing:
        raise MlsynthDataError(f"covariates not in the panel: {missing}")

    # dataprep validates balance, a single treatment date, and the donor pool,
    # and gives the canonical pre/post split and time labels.
    prep = dataprep(df, cfg.unitid, cfg.time, cfg.outcome, cfg.treat)
    n_periods = int(prep["total_periods"])
    pre_periods = int(prep["pre_periods"])
    if pre_periods < 1:
        raise MlsynthDataError("DMLFM needs at least one pre-treatment period")

    if df[[cfg.outcome] + covs].isna().any().any():
        raise MlsynthDataError(
            "DMLFM needs a complete panel; the outcome or a covariate has "
            "missing values")

    # The sampler itself tolerates ragged groups, and so does the pblasso
    # reference, but mlsynth ingestion has no way to carry an observation mask,
    # so an unbalanced panel is refused instead of silently fitting a panel the
    # result contract cannot describe.
    counts = df.groupby(cfg.unitid, observed=True)[cfg.time].nunique()
    if counts.nunique() != 1 or int(counts.iloc[0]) != n_periods:
        raise MlsynthDataError(
            "DMLFM needs a balanced panel: units span "
            f"{int(counts.min())} to {int(counts.max())} of {n_periods} periods")

    df = df.sort_values([cfg.unitid, cfg.time]).reset_index(drop=True)
    units = list(pd.unique(df[cfg.unitid]))
    times = list(pd.unique(df[cfg.time].sort_values()))
    n_units = len(units)
    if cfg.r > n_units:
        raise MlsynthConfigError(
            f"r={cfg.r} exceeds the {n_units} units in the panel")

    ucode = df[cfg.unitid].map({u: i for i, u in enumerate(units)}).to_numpy()
    tcode = df[cfg.time].map({t: i for i, t in enumerate(times)}).to_numpy()
    d = df[cfg.treat].to_numpy()

    treated_units = sorted({u for u in df.loc[d == 1, cfg.unitid]})
    if len(treated_units) != 1:
        raise MlsynthDataError(
            f"DMLFM expects one treated unit, found {len(treated_units)}")
    treated_name = str(treated_units[0])
    tr_rows = np.flatnonzero(df[cfg.unitid].to_numpy() == treated_units[0])
    fit_rows = np.flatnonzero(d == 0)

    # blasso_default.R:88-91 divides every covariate by its pooled standard
    # deviation before fitting -- no centring, denominator n-1. The scaling is
    # what keeps the time-varying coefficient block from dominating the factor
    # term when a covariate is large in level: on this panel ``pgdp`` runs to
    # five figures, and unscaled it absorbs the common structure the factors
    # are meant to carry.
    scales = np.ones(len(covs))
    if covs and cfg.scale_covariates:
        scales = df[covs].to_numpy(float).std(axis=0, ddof=1)
        if np.any(scales == 0):
            zero = [c for c, s in zip(covs, scales) if s == 0]
            raise MlsynthDataError(f"covariates with zero variance: {zero}")

    def blocks(rows):
        cov = (df.loc[df.index[rows], covs].to_numpy(float) / scales if covs
               else np.zeros((len(rows), 0)))
        ones = np.ones((len(rows), 1))
        X = np.hstack([ones, cov])
        Z = (np.hstack([ones, cov]) if cfg.re in ("unit", "both")
             else np.zeros((len(rows), 0)))
        A = (np.hstack([ones, cov]) if cfg.re in ("time", "both")
             else np.zeros((len(rows), 0)))
        return X, Z, A

    X, Z, A = blocks(fit_rows)
    X_tr, Z_tr, A_tr = blocks(tr_rows)

    unit_index = ucode[fit_rows]
    time_index = tcode[fit_rows]
    order_by_time = np.lexsort((unit_index, time_index))

    return DMLFMInputs(
        y=df[cfg.outcome].to_numpy(float)[fit_rows],
        X=X, Z=Z, A=A, X_tr=X_tr, Z_tr=Z_tr, A_tr=A_tr,
        unit_index=unit_index, time_index=time_index,
        unit_index_tr=ucode[tr_rows], time_index_tr=tcode[tr_rows],
        unit_starts=_starts(unit_index),
        time_starts=_starts(time_index[order_by_time]),
        order_by_time=order_by_time,
        n_units=n_units, n_periods=n_periods, r=int(cfg.r),
        niter=int(cfg.niter), burn=int(cfg.burn), ar1=bool(cfg.ar1),
        xlasso=bool(cfg.xlasso), zlasso=bool(cfg.zlasso),
        alasso=bool(cfg.alasso), flasso=bool(cfg.flasso),
        a1=cfg.a1, a2=cfg.a2, b1=cfg.b1, b2=cfg.b2,
        c1=cfg.c1, c2=cfg.c2, p1=cfg.p1, p2=cfg.p2,
        e1=cfg.e1, e2=cfg.e2,
        time_labels=np.asarray(prep["time_labels"]),
        pre_periods=pre_periods, treated_name=treated_name)
