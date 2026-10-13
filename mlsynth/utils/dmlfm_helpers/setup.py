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
    treatment indicator at zero, which includes each treated unit's own
    pre-adoption rows. Prediction covers every row of every treated unit, so
    the pre-adoption fit is reported alongside the post-adoption gap.

    This is the estimation set of Pang, Liu & Xu (2022) Eq. (A.5) whether one
    unit is treated or several, and whether they adopt together or apart, so
    staggered adoption needs nothing of the sampler. ``X_tr`` and its
    companions stack the treated units in panel order, each unit's rows in
    time order, giving ``n_treated * n_periods`` prediction rows.
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
    prior: str
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
    treated_names: list
    adoption_index: np.ndarray
    n_treated: int


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

    # dataprep validates balance and the donor pool and gives the canonical
    # time labels. Under staggered adoption it returns a ``cohorts`` mapping
    # keyed by adoption label instead of one pre/post split.
    prep = dataprep(df, cfg.unitid, cfg.time, cfg.outcome, cfg.treat)
    cohorts = prep.get("cohorts")
    if cohorts:
        n_periods = int(next(iter(cohorts.values()))["total_periods"])
        pre_periods = int(min(c["pre_periods"] for c in cohorts.values()))
    else:
        n_periods = int(prep["total_periods"])
        pre_periods = int(prep["pre_periods"])
    if pre_periods < 1:
        raise MlsynthDataError(
            "DMLFM needs at least one pre-treatment period; a treated unit "
            "adopts in the first period of the panel")

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

    # Each treated unit's adoption period, read off its own time-ordered
    # indicator. dataprep has already rejected a panel with no treated unit and
    # one whose treatment is not absorbing, so the first treated entry is the
    # adoption period and every later entry is treated.
    adoption = np.full(n_units, -1, dtype=int)
    for i in range(n_units):
        own = d[ucode == i]
        if own.any():
            adoption[i] = int(np.argmax(own == 1))
    treated_idx = np.flatnonzero(adoption >= 0)

    treated_names = [str(units[i]) for i in treated_idx]
    treated_name = treated_names[0]
    adoption_index = adoption[treated_idx]
    tr_rows = np.flatnonzero(np.isin(ucode, treated_idx))

    # The estimation set is cell-level, not unit-level: every (i, t) with the
    # indicator at zero, including a treated unit's own pre-adoption rows. A
    # panel with no unit reserved as a donor is therefore still estimable, which
    # is what makes staggered adoption work without a donor pool set aside.
    fit_rows = np.flatnonzero(d == 0)

    # The per-period blocks are identified only by untreated observations in
    # that period. A period in which every unit is already treated leaves its
    # time-varying coefficient and its factor drawn from the prior alone, so
    # the counterfactual there carries no information from the panel.
    if cfg.re in ("time", "both") or cfg.r > 0:
        covered = np.zeros(n_periods, dtype=bool)
        covered[tcode[fit_rows]] = True
        if not covered.all():
            bare = [str(times[t]) for t in np.flatnonzero(~covered)]
            raise MlsynthDataError(
                "DMLFM needs an untreated observation in every period to "
                "identify the time-varying coefficients and the factors; "
                f"none is left in {', '.join(bare[:5])}"
                + (f" and {len(bare) - 5} more" if len(bare) > 5 else ""))

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
        prior=str(cfg.prior),
        xlasso=bool(cfg.xlasso), zlasso=bool(cfg.zlasso),
        alasso=bool(cfg.alasso), flasso=bool(cfg.flasso),
        a1=cfg.a1, a2=cfg.a2, b1=cfg.b1, b2=cfg.b2,
        c1=cfg.c1, c2=cfg.c2, p1=cfg.p1, p2=cfg.p2,
        e1=cfg.e1, e2=cfg.e2,
        time_labels=np.asarray(prep["time_labels"]),
        pre_periods=pre_periods, treated_name=treated_name,
        treated_names=treated_names, adoption_index=adoption_index,
        n_treated=int(treated_idx.size))
