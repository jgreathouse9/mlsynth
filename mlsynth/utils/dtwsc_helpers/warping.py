"""The TFDTW warping engine of Cao & Chadefaux (2025).

Two-phase dynamic time warping that learns how fast each donor moves relative
to the treated unit, then resamples the donor onto the treated unit's clock.

Phase one aligns the pre-treatment windows and reads a per-period *speed* off
the warping path: how many donor periods the alignment spends on each treated
period. Phase two extends those speeds past the intervention without letting
the post-treatment data influence them -- it slides a short window along the
donor's post-period, finds the pre-period stretch it most resembles, and
inherits that stretch's speed. Speed differences that the donor already had
before the intervention are therefore removed, while any change in speed caused
by the intervention survives into the counterfactual.

Reference: ``R/{TFDTW,misc}.R`` of github.com/conflictlab/dsc (MIT), pinned at
``b1cd241`` for the cross-check in ``benchmarks/reference/dtwsc_basque/``. This
is a clean-room re-implementation; nothing is vendored.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ...exceptions import MlsynthEstimationError
from .dtw import (
    ASYMMETRIC_P2,
    SYMMETRIC_P1,
    dtw,
    ref_too_short,
    warp,
)


# ------------------------------------------------------------------ scalars
def t_normalize(data, reference=None) -> np.ndarray:
    """Z-score ``data``, optionally against another series' moments.

    Uses the sample standard deviation (``ddof=1``), matching R's ``sd``. The
    population denominator would rescale every speed by ``sqrt(n/(n-1))``.
    """
    x = np.asarray(data, dtype=float)
    ref = x if reference is None else np.asarray(reference, dtype=float)
    scale = np.std(ref, ddof=1)
    if not np.isfinite(scale) or scale == 0.0:
        return np.zeros_like(x)   # a flat window carries no speed information
    return (x - np.mean(ref)) / scale


def quantile_r7(x, p: float) -> float:
    """R ``stats::quantile`` type 7, evaluated the way R evaluates it.

    R computes ``(1 - h) * x[lo] + h * x[hi]``; numpy's ``quantile`` takes a
    different lerp branch when ``h > 0.5``. They differ by about 2 ULP, and
    :func:`remove_outliers` compares against a bound that can sit within 3 ULP
    of the data -- the warping weights are small-denominator rationals like 2/3
    and 4/3, so they land on that boundary routinely.
    """
    v = np.sort(np.asarray(x, dtype=float)[~np.isnan(np.asarray(x, float))])
    if v.size == 0:
        return float("nan")
    h = (v.size - 1) * float(p)
    lo = int(np.floor(h))
    hi = min(lo + 1, v.size - 1)
    h -= lo
    return float((1 - h) * v[lo] + h * v[hi])


def remove_outliers(col, n_iqr: float = 3.0) -> np.ndarray:
    """Blank the entries of ``col`` outside ``[Q1 - k*IQR, Q3 + k*IQR]``."""
    q1, q3 = quantile_r7(col, 0.25), quantile_r7(col, 0.75)
    iqr = q3 - q1
    out = np.array(col, dtype=float, copy=True)
    with np.errstate(invalid="ignore"):
        out[(out > q3 + n_iqr * iqr) | (out < q1 - n_iqr * iqr)] = np.nan
    return out


# ------------------------------------------------------- path <-> speeds
def warp2weight(W) -> np.ndarray:
    """Turn a warping-path matrix into per-donor-period speeds.

    ``W[j, i] = 1`` when donor period ``j`` is aligned to treated period ``i``.
    Each treated period contributes a total of one unit of "time" shared out
    over the donor periods it touches, so the row sums say how much treated
    time each donor period accounts for -- its speed.
    """
    W = np.asarray(W, dtype=float)
    denom = W.sum(axis=0, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, W / denom, 0.0).sum(axis=1)


def warp_with_weight(ts, weight) -> np.ndarray:
    """Resample ``ts`` so that period ``i`` occupies ``weight[i]`` of the clock.

    Speeds above one stretch a period out; below one compress it. The output is
    shorter than the input when the speeds average below one, which is why the
    caller pads to the panel length.
    """
    ts = np.asarray(ts, dtype=float)
    weight = np.asarray(weight, dtype=float)
    if ts.size != weight.size:
        raise MlsynthEstimationError(
            f"warp_with_weight: series has {ts.size} periods but the speed "
            f"vector has {weight.size}."
        )
    if ts.size == 0:
        raise MlsynthEstimationError("warp_with_weight: empty series.")
    n = weight.size
    W = np.zeros((n, 2 * n + 1))
    count, residual = 0, 0.0
    for i in range(n):
        z = float(weight[i])
        if z <= 0:
            raise MlsynthEstimationError(
                "warp_with_weight: speeds must be strictly positive."
            )
        while z > 0:
            if z + residual >= 1:
                now = 1 - residual
                residual = 0.0
                z -= now
                W[i, count] = now
                count += 1
            else:
                residual += z
                W[i, count] = z
                z = 0.0
    occupied = np.nonzero(W.sum(axis=0))[0]
    ind_max = int(occupied.max())
    W = W[:, : ind_max + 1]
    W[n - 1, ind_max] = 1 - W[: n - 1, ind_max].sum()
    return W.T @ ts


# ---------------------------------------------------------------- phase one
def first_dtw(
    x,
    y,
    t_treat: int,
    buffer: int = 0,
    step_pattern: np.ndarray = SYMMETRIC_P1,
    match_method: str = "fixed",
) -> Dict[str, Any]:
    """Align the donor's pre-period to the treated unit's and read the speeds.

    Returns the donor period the treated unit's last pre-period maps onto
    (``cutoff`` -- where the donor's own pre-period ends on its own clock) and
    the trimmed path matrix ``W_a`` those speeds come from.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if t_treat < 2:
        raise MlsynthEstimationError(
            "DTWSC: need at least two pre-treatment periods to learn a warp."
        )
    yq = t_normalize(y[:t_treat])
    xr = t_normalize(x[: t_treat + buffer], x[:t_treat])
    if match_method == "open.end":
        b = buffer
        while ref_too_short(yq, xr, step_pattern) and (t_treat + b) < x.size:
            b += 1
            xr = t_normalize(x[: t_treat + b], x[:t_treat])
        alignment = dtw(yq, xr, step_pattern=step_pattern, open_end=True)
    else:
        alignment = dtw(yq, xr, step_pattern=step_pattern, open_end=False)

    wr = warp(alignment, index_reference=True) + 1.0        # R's 1-based grid
    cutoff = int(np.round(wr[t_treat - 1]))
    cutoff = int(np.clip(cutoff, 2, x.size))
    W = np.zeros((int(alignment.index2.max()) + 1,
                  int(alignment.index1.max()) + 1))
    W[alignment.index2, alignment.index1] = 1.0
    return {"cutoff": cutoff, "W_a": W[:cutoff, :t_treat]}


def _moving_average(x, width: int) -> np.ndarray:
    """Centred moving average, ``NaN`` where the window overhangs an end."""
    x = np.asarray(x, dtype=float)
    n = x.size
    out = np.full(n, np.nan)
    off = width // 2
    for i in range(n):
        lo, hi = i - (width - 1 - off), i + off
        if lo >= 0 and hi < n:
            out[i] = x[lo: hi + 1].mean()
    return out


def _forward_fill(x) -> np.ndarray:
    out = np.array(x, dtype=float, copy=True)
    last = np.nan
    for i in range(out.size):
        if np.isnan(out[i]):
            out[i] = last
        else:
            last = out[i]
    return out


# ---------------------------------------------------------------- phase two
def second_dtw(
    x_post,
    x_pre,
    k: int,
    weight_a,
    default_margin: int = 3,
    n_q: int = 1,
    n_r: int = 1,
    step_pattern: np.ndarray = ASYMMETRIC_P2,
    dist_quant: float = 1.0,
    n_iqr: float = 3.0,
) -> np.ndarray:
    """Carry the pre-period speeds into the post-period.

    Each length-``k`` window of the donor's post-period is matched against
    every window of its pre-period; the cheapest match donates its speeds. The
    post-treatment outcome therefore never sets the speed -- only the shape of
    the donor's own history does.
    """
    x_post = np.asarray(x_post, dtype=float)
    x_pre = np.asarray(x_pre, dtype=float)
    weight_a = np.asarray(weight_a, dtype=float)
    n_pre, n_post = x_pre.size, x_post.size
    if k < 2:
        raise MlsynthEstimationError("DTWSC: window length k must be >= 2.")
    if n_post < k:
        raise MlsynthEstimationError(
            f"DTWSC: post-treatment window (k={k}) exceeds the {n_post} "
            "periods available after the cutoff."
        )

    stacked, distances = [], []
    i = 1
    while i <= n_post - k + 1:
        query = t_normalize(x_post[i - 1: i + k - 1])
        costs = []
        j, margin = 1, default_margin
        while j <= n_pre - k + 2:
            window = t_normalize(x_pre[j - 1: min(j + k + margin - 1, n_pre)])
            if ref_too_short(query, window, step_pattern):
                if j < n_pre - k - margin + 1:  # pragma: no cover - the window
                    # is k + margin long against a k-long query, so it can only
                    # fail to span in degenerate shapes: a random search over
                    # 4000 panels x both step patterns produced zero growths.
                    # Kept because the reference has it and a caller may set
                    # default_margin=0.
                    margin += 1
                    continue
                margin = default_margin
                j += n_r
                continue
            try:
                cost = dtw(query, window, step_pattern=step_pattern,
                           open_end=True).distance
            except MlsynthEstimationError:      # pragma: no cover - ref_too_short
                cost = np.inf                   # already screens these out
            costs.append((cost, j, margin))
            j += n_r
            margin = default_margin

        if not costs:  # pragma: no cover - reachable only if EVERY pre-period
            # window is rejected, which the k <= n_post guard above and the
            # margin growth make unreachable on a validated panel; kept so the
            # failure is named rather than surfacing as an empty argmin.
            raise MlsynthEstimationError(
                "DTWSC: no usable pre-treatment window to match the "
                f"post-treatment window at position {i}; try a smaller k."
            )
        best = int(np.argmin([c[0] for c in costs]))
        _, j_opt, margin_opt = costs[best]
        ref = t_normalize(x_pre[j_opt - 1: min(j_opt + k + margin_opt - 1, n_pre)])
        alignment = dtw(query, ref, step_pattern=step_pattern, open_end=True)
        W = np.zeros((int(alignment.index1.max()) + 1,
                      int(alignment.index2.max()) + 1))
        W[alignment.index1, alignment.index2] = 1.0
        speeds_here = weight_a[j_opt - 1: j_opt - 1 + W.shape[1]]
        if speeds_here.size < W.shape[1]:       # pragma: no cover - clipped ref
            speeds_here = np.pad(speeds_here,
                                 (0, W.shape[1] - speeds_here.size),
                                 constant_values=1.0)
        row = np.full(n_post, np.nan)
        row[i - 1: i + k - 1] = (W @ speeds_here) / W.sum(axis=1)
        stacked.append(row)
        distances.append(alignment.distance)
        i += n_q

    stacked = np.asarray(stacked)
    distances = np.asarray(distances)
    if dist_quant < 1.0:
        keep = ~(distances > quantile_r7(distances, dist_quant))
        stacked = stacked[keep]
    stacked = np.column_stack(
        [remove_outliers(stacked[:, c], n_iqr) for c in range(stacked.shape[1])]
    )
    with np.errstate(invalid="ignore"):
        avg = np.nanmean(stacked, axis=0)
    avg[np.isnan(avg)] = 1.0
    return avg


def tfdtw(
    x,
    y,
    k: int,
    t_treat: int,
    buffer: int = 0,
    step_pattern1: np.ndarray = SYMMETRIC_P1,
    step_pattern2: np.ndarray = ASYMMETRIC_P2,
    n_burn: int = 3,
    ma: int = 3,
    match_method: str = "fixed",
    default_margin: int = 3,
    n_q: int = 1,
    n_r: int = 1,
    dist_quant: float = 1.0,
    n_iqr: float = 3.0,
) -> Dict[str, Any]:
    """Run both warping phases for one donor.

    Returns ``cutoff`` (donor period the treated pre-period ends on),
    ``weight_a`` (pre-period speeds, length ``cutoff``) and ``avg_weight``
    (post-period speeds, from the cutoff onward).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    phase1 = first_dtw(x, y, t_treat, buffer, step_pattern1, match_method)
    cutoff = phase1["cutoff"]

    raw_speeds = warp2weight(phase1["W_a"])
    smoothed = _forward_fill(_moving_average(raw_speeds, ma))
    gaps = np.isnan(smoothed)
    smoothed[gaps] = raw_speeds[gaps]          # leading NaNs keep the raw speed

    start = max(cutoff - n_burn - 1, 0)
    avg = second_dtw(
        x[start:], x[:cutoff], k, smoothed,
        default_margin=default_margin, n_q=n_q, n_r=n_r,
        step_pattern=step_pattern2, dist_quant=dist_quant, n_iqr=n_iqr,
    )
    burn = min(n_burn, max(avg.size - 1, 0))
    return {"cutoff": cutoff, "weight_a": smoothed, "avg_weight": avg[burn:]}


def warp_series(
    x_raw,
    cutoff: int,
    weight_a,
    avg_weight,
    t_treat: int,
    n_out: int,
) -> np.ndarray:
    """Stitch the pre- and post-period warps into one donor series.

    Padded with ``NaN`` when the warp compresses the donor past the end of the
    panel -- the reference does the same, and the missing tail is real rather
    than something to extrapolate over.
    """
    x_raw = np.asarray(x_raw, dtype=float)
    pre = warp_with_weight(x_raw[:cutoff], weight_a)[:t_treat]
    post = warp_with_weight(x_raw[cutoff - 1:], avg_weight)[1:]
    out = np.concatenate([pre, post])
    if out.size < n_out:
        out = np.concatenate([out, np.full(n_out - out.size, np.nan)])
    return out[:n_out]


def savgol_second_derivative(
    Y,
    filter_width: int = 5,
    n_poly: int = 3,
) -> np.ndarray:
    """Smooth each column of ``Y`` and take its second derivative.

    The alignment runs on curvature rather than level: two units at different
    income levels but with the same turning points should be judged similar,
    and the second derivative is what makes that true. The reference pads each
    series with an ARIMA forecast before filtering; here the series is
    edge-padded instead, which differs only over the ``(filter_width - 1) / 2``
    points at each end and is not worth an ARIMA dependency.
    """
    from scipy.signal import savgol_filter

    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
        squeeze = True
    else:
        squeeze = False
    if filter_width % 2 == 0:
        raise MlsynthEstimationError("DTWSC: filter_width must be odd.")
    if filter_width <= n_poly:
        raise MlsynthEstimationError(
            "DTWSC: filter_width must exceed the polynomial order."
        )
    n_buffer = (filter_width - 1) // 2
    out = np.empty_like(Y)
    for c in range(Y.shape[1]):
        col = t_normalize(Y[:, c])
        padded = np.concatenate([np.repeat(col[0], n_buffer), col,
                                 np.repeat(col[-1], n_buffer)])
        deriv = savgol_filter(padded, filter_width, n_poly, deriv=2)
        out[:, c] = deriv[n_buffer: n_buffer + col.size]
    return out[:, 0] if squeeze else out
