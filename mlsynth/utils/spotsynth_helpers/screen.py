"""Spillover-detection screen (O'Riordan & Gilligan-Lee 2025, Algorithm 1).

The central object is :func:`spillover_screen`, which implements the donor
forecast test underpinning Theorem 3.1: under invariant causal mechanisms and
the proxy-completeness condition, a *valid* donor's post-intervention value is
forecastable from pre-intervention donor data. A donor whose realised
post-intervention value departs from that forecast has either been hit by a
spillover or seen its latent distribution shift -- in either case it is unsafe
to keep in the donor pool.

Two forecast anchors are provided. Both normalise each donor to zero mean and
unit standard deviation over the pre-intervention window (Algorithm 1, step 1),
optionally on time-averaged ("bucketed") data (Section 3.2.1; Figure 3).

* ``"loo"`` (default) -- a leave-one-out anchor. Predict each donor's *whole*
  post-intervention trajectory from the *other* donors' common factors and rank
  by the mean absolute deviation. It differences out the common factor and
  accumulates evidence over the post-period, so it detects contamination
  regardless of how *gradually* the spillover arrives -- but it needs a valid
  *majority* (the other donors form the reference). This is the right anchor for
  applied work (a few suspect donors in a mostly-valid pool) and is onset-robust.
* ``"lag"`` -- the paper's Algorithm 1. Fit the forecast on *lagged* donor data
  over pre-intervention transitions, then predict the *first* post-intervention
  point from the last (clean) pre-intervention cross-section. Anchored to clean
  pre-data, it survives a *mostly-invalid* pool -- but it assumes a *sharp*
  spillover (present at full size by the first post-period) and is blind to
  gradual onsets. Use it only in the paper's mostly-invalid / sharp regime, where
  ``loo`` inverts. See ``run_forecast_power_analysis`` for the power comparison.
"""

from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np
from scipy.stats import norm

from ...exceptions import MlsynthDataError
from .structures import SpilloverScreen


def _bucketize(Z: np.ndarray, idx: np.ndarray, k: int) -> np.ndarray:
    """Average rows of ``Z`` selected by ``idx`` into consecutive ``k``-buckets."""
    groups = [idx[i:i + k] for i in range(0, len(idx), k) if len(idx[i:i + k]) == k]
    if not groups:
        groups = [idx]
    return np.array([Z[g].mean(axis=0) for g in groups])


def _factor_design(L: np.ndarray, n_factors: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centre ``L`` and return (centre, loadings ``V_r``, score design with intercept).

    The leading ``n_factors`` right singular vectors regularise the lagged-donor
    regression so the forecast is well posed when ``n_donors`` exceeds the number
    of pre-intervention observations.
    """
    centre = L.mean(axis=0)
    Lc = L - centre
    _, _, Vt = np.linalg.svd(Lc, full_matrices=False)
    r = int(min(n_factors, Vt.shape[0]))
    Vr = Vt[:r].T
    F = Lc @ Vr
    design = np.column_stack([np.ones(len(F)), F])
    return centre, Vr, design


def _forecast_loo(Z: np.ndarray, pre: np.ndarray, post: np.ndarray, *,
                  n_factors: int) -> Tuple[np.ndarray, np.ndarray]:
    """Leave-one-out variant: mean absolute post-period deviation from the
    common-factor forecast built on the *other* donors."""
    T, n = Z.shape
    A = np.zeros(n)
    resid_sd = np.zeros(n)
    for i in range(n):
        oth = [j for j in range(n) if j != i]
        Xo = Z[:, oth]
        centre = Xo[pre].mean(axis=0)
        _, _, Vt = np.linalg.svd(Xo[pre] - centre, full_matrices=False)
        r = int(min(n_factors, Vt.shape[0]))
        Vr = Vt[:r].T
        F = (Xo - centre) @ Vr
        Fd = np.column_stack([np.ones(T), F])
        beta, *_ = np.linalg.lstsq(Fd[pre], Z[pre, i], rcond=None)
        resid = Z[:, i] - Fd @ beta
        A[i] = float(np.mean(np.abs(resid[post])))
        resid_sd[i] = resid[pre].std(ddof=1) + 1e-9
    return A, resid_sd


def spillover_screen(
    D: np.ndarray,
    T0: int,
    donor_names,
    *,
    selection: str = "S1",
    forecast: str = "loo",
    n_donors=None,
    ppi: float = 0.8,
    n_factors: int = 5,
    time_average=None,
) -> SpilloverScreen:
    """Run the Algorithm 1 spillover screen and select valid donors.

    Parameters
    ----------
    D : np.ndarray
        Donor-pool outcomes, shape ``(T, n_donors)``.
    T0 : int
        Number of pre-intervention periods.
    donor_names : sequence
        Donor names (length ``n_donors``).
    selection : {"S1", "S2", "all"}
        ``S1`` keeps the ``n_donors`` donors with the smallest forecast error;
        ``S2`` keeps the donors whose realised value lies inside the ``ppi``
        posterior predictive interval; ``all`` keeps every donor (the baseline).
    forecast : {"loo", "lag"}
        Forecast anchor (default ``"loo"``; see module docstring). Use ``"lag"``
        only for the mostly-invalid / sharp-spillover regime.
    n_donors : int, optional
        Number of donors to retain under ``S1`` (default: half the pool,
        minimum 2).
    ppi : float
        Posterior-predictive-interval level for ``S2`` (default 0.8).
    n_factors : int
        Number of donor factors used to regularise the forecast.
    time_average : int, optional
        Bucket width for time-averaging the data before screening
        (``"lag"`` only; Section 3.2.1).

    Returns
    -------
    SpilloverScreen
    """
    D = np.asarray(D, dtype=float)
    if D.ndim != 2:
        raise MlsynthDataError(
            f"Donor matrix D must be 2-D (T, n_donors); got shape {D.shape}.")
    T, n = D.shape
    if n < 1:
        raise MlsynthDataError("Donor matrix D has no donor columns.")
    if not (0 < T0 < T):
        raise MlsynthDataError(
            f"T0 must satisfy 0 < T0 < T (T={T}); got T0={T0}.")
    if len(donor_names) != n:
        raise MlsynthDataError(
            f"donor_names has length {len(donor_names)} but D has {n} columns.")
    if not np.all(np.isfinite(D)):
        raise MlsynthDataError("Donor matrix D contains non-finite values.")
    if selection not in ("S1", "S2", "all"):
        raise ValueError(
            f"Unknown selection {selection!r} (use 'S1', 'S2', or 'all').")

    post = np.arange(T) >= T0
    pre = ~post
    bucket = int(time_average) if time_average else 1

    mu = D[pre].mean(axis=0)
    sd_raw = D[pre].std(axis=0)
    if np.any(sd_raw <= 1e-12):
        warnings.warn(
            "One or more donors are (near-)constant over the pre-intervention "
            "window; their standardised forecast errors are unreliable.",
            RuntimeWarning, stacklevel=2,
        )
    sd = sd_raw + 1e-12
    Z = (D - mu) / sd

    if forecast == "lag":
        A, sr = _lag_errors(Z, pre, post, n_factors=n_factors, bucket=bucket)
    elif forecast == "loo":
        A, sr = _forecast_loo(Z, pre, post, n_factors=n_factors)
    else:
        raise ValueError(f"Unknown forecast anchor {forecast!r} (use 'lag' or 'loo').")

    # S2: inside the (1 - alpha) PPI of the forecast error.
    z = float(norm.ppf(0.5 + ppi / 2.0))
    inside = A <= z * sr

    order = np.argsort(A)  # ascending error: most-valid first
    if selection == "all":
        selected = np.arange(n)
    elif selection == "S2":
        selected = np.where(inside)[0]
        if selected.size < 2:                      # fall back to S1 if PPI empties the pool
            warnings.warn(
                "S2 posterior-predictive screen retained fewer than 2 donors; "
                "falling back to the S1 (smallest-error) rule. Consider raising "
                "`ppi` or using selection='S1'.",
                RuntimeWarning, stacklevel=2,
            )
            k = _default_keep(n_donors, n)
            selected = order[:k]
    else:  # S1
        k = _default_keep(n_donors, n)
        selected = order[:k]

    selected = np.sort(selected)
    excluded = np.array([i for i in range(n) if i not in set(selected.tolist())], dtype=int)

    meta = {
        "n_donors": int(n), "n_selected": int(selected.size),
        "n_excluded": int(excluded.size), "ppi": float(ppi),
        "n_factors": int(n_factors), "time_average": int(bucket),
    }
    return SpilloverScreen(
        donor_names=list(donor_names), forecast_error=A, inside_ppi=inside,
        selected_idx=selected, excluded_idx=excluded, selection=selection,
        forecast=forecast, metadata=meta,
    )


def _default_keep(n_donors, n) -> int:
    if n_donors is None:
        return max(2, n // 2)
    return int(max(1, min(int(n_donors), n)))


def _lag_errors(Z, pre, post, *, n_factors, bucket):
    """First-post-bucket forecast error and per-donor residual sd (lag anchor)."""
    pre_i = np.where(pre)[0]
    post_i = np.where(post)[0]
    Zpre = _bucketize(Z, pre_i, bucket)
    Zpost = _bucketize(Z, post_i, bucket)
    if Zpre.shape[0] < 3 or Zpost.shape[0] < 1:
        raise ValueError("Too few (bucketed) pre/post periods for the lag forecast.")
    Xlag, Yfit = Zpre[:-1], Zpre[1:]
    x_test_lag, x_test = Zpre[-1], Zpost[0]
    n = Z.shape[1]
    A = np.zeros(n)
    sr = np.zeros(n)
    centre, Vr, Fd = _factor_design(Xlag, n_factors)
    Ft = np.concatenate([[1.0], (x_test_lag - centre) @ Vr])
    for i in range(n):
        beta, *_ = np.linalg.lstsq(Fd, Yfit[:, i], rcond=None)
        resid = Yfit[:, i] - Fd @ beta
        sr[i] = resid.std(ddof=1) + 1e-9
        A[i] = abs(x_test[i] - float(Ft @ beta))
    return A, sr
