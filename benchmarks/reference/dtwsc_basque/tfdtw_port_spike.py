"""Python port of the TFDTW warping engine of Cao & Chadefaux (2025).

Reference: https://github.com/conflictlab/dsc, R/{TFDTW,misc}.R (MIT).
Clean-room re-implementation for validation; nothing is vendored.

The single non-obvious point is ``warp_R``: R's ``dtw::warp`` averages the tied
indices of the many-to-one alignment map, so it returns fractional positions
(e.g. 7.5) that ``round()`` can lift to the next integer. ``dtw-python``'s
``warp`` does not. That difference -- not ``open.end`` -- is what previously
blocked the port.
"""
from __future__ import annotations

import numpy as np
import dtw as D

SYMMETRIC_P1 = D.stepPattern.symmetricP1
SYMMETRIC_P2 = D.stepPattern.symmetricP2
ASYMMETRIC_P2 = D.stepPattern.asymmetricP2


# ---------------------------------------------------------------- primitives
def t_normalize(data, reference=None):
    """R misc.R::t.normalize -- z-score with R's sample sd (ddof=1)."""
    ref = data if reference is None else reference
    return (np.asarray(data, float) - np.mean(ref)) / np.std(ref, ddof=1)


def warp_R(alignment, index_reference=False):
    """R ``dtw::warp`` semantics: mean of the tied indices, 0-based output."""
    if index_reference:
        src, dst = alignment.index1, alignment.index2
    else:
        src, dst = alignment.index2, alignment.index1
    return np.array([dst[src == j].mean() for j in range(int(src.max()) + 1)])


def ref_too_short(query, reference, step_pattern):
    """R misc.R::RefTooShort. Note R calls dtw(reference, query)."""
    try:
        a = D.dtw(reference, query, step_pattern=step_pattern,
                  open_end=True, keep_internals=True)
    except Exception:
        return False
    if len(np.unique(a.index2)) == 1:
        return True
    wq = warp_R(a, index_reference=False) + 1.0          # R is 1-based
    return bool(np.round(np.max(wq)) < len(query))


def warp2weight(W):
    """R misc.R::warp2weight -- row sums of the column-normalised path matrix."""
    W = np.asarray(W, float)
    return (W / W.sum(axis=0, keepdims=True)).sum(axis=1)


def warp_with_weight(ts, weight):
    """R misc.R::warpWITHweight -- resample ``ts`` at the speeds in ``weight``."""
    ts = np.asarray(ts, float)
    weight = np.asarray(weight, float)
    if ts.size != weight.size:
        raise ValueError("Lengths of ts and weight are different.")
    n = weight.size
    W = np.zeros((n, 2 * n))
    count, residual = 0, 0.0
    for i in range(n):
        z = weight[i]
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
    ind_max = int(np.max(np.nonzero(W.sum(axis=0))[0]))
    W = W[:, : ind_max + 1]
    W[n - 1, ind_max] = 1 - W[: n - 1, ind_max].sum()
    return W.T @ ts


def quantile_R7(x, p):
    """R ``stats::quantile`` type 7, evaluated the way R evaluates it.

    R computes ``(1 - h) * x[lo] + h * x[hi]``; numpy's ``quantile`` uses a
    different lerp branch for ``h > 0.5``. The two disagree by ~2 ULP, which is
    enough to flip ``RemoveOutliers`` when a weight sits on the Q1 - 3*IQR
    boundary -- and small-denominator rationals like 2/3 and 4/3 land there
    often, so this is not a theoretical concern.
    """
    v = np.sort(np.asarray(x, float)[~np.isnan(x)])
    if v.size == 0:
        return np.nan
    h = (v.size - 1) * p
    lo = int(np.floor(h))
    hi = min(lo + 1, v.size - 1)
    h -= lo
    return (1 - h) * v[lo] + h * v[hi]


def remove_outliers(col, n_IQR=3):
    """R misc.R::RemoveOutliers."""
    q1, q3 = quantile_R7(col, 0.25), quantile_R7(col, 0.75)
    iqr = q3 - q1
    out = np.asarray(col, float).copy()
    out[(out > q3 + n_IQR * iqr) | (out < q1 - n_IQR * iqr)] = np.nan
    return out


def _ma_filter(x, ma):
    """R stats::filter(x, rep(1/ma, ma)) -- centred, NA at both ends."""
    x = np.asarray(x, float)
    n, out = x.size, np.full(x.size, np.nan)
    off = ma // 2                       # R's offset for sides=2
    for i in range(n):
        lo, hi = i - (ma - 1 - off), i + off
        if lo >= 0 and hi < n:
            out[i] = x[lo: hi + 1].mean()
    return out


def _na_locf(x):
    out = np.asarray(x, float).copy()
    last = np.nan
    for i in range(out.size):
        if np.isnan(out[i]):
            out[i] = last
        else:
            last = out[i]
    return out


# ------------------------------------------------------------------- phase 1
def first_dtw(x, y, t_treat, buffer=0, step_pattern=SYMMETRIC_P1,
              match_method="fixed"):
    """R TFDTW.R::first.dtw."""
    yq = t_normalize(y[:t_treat])
    xr = t_normalize(x[: t_treat + buffer], x[:t_treat])
    if match_method == "fixed":
        a = D.dtw(yq, xr, step_pattern=step_pattern, open_end=False,
                  keep_internals=True)
    else:
        b = buffer
        while ref_too_short(yq, xr, step_pattern) and (t_treat + b) < len(x):
            b += 1
            xr = t_normalize(x[: t_treat + b], x[:t_treat])
        a = D.dtw(yq, xr, step_pattern=step_pattern, open_end=True,
                  keep_internals=True)
    wr = warp_R(a, index_reference=True) + 1.0            # R 1-based
    cutoff = int(np.round(wr[t_treat - 1]))
    W = np.zeros((int(a.index2.max()) + 1, int(a.index1.max()) + 1))
    W[a.index2, a.index1] = 1.0
    return {"cutoff": cutoff, "W_a": W[:cutoff, :t_treat]}


# ------------------------------------------------------------------- phase 2
def second_dtw(x_post, x_pre, k, weight_a, default_margin=3, n_q=1, n_r=1,
               step_pattern=ASYMMETRIC_P2, dist_quant=1.0, n_IQR=3):
    """R TFDTW.R::second.dtw -- double sliding window over the post period."""
    n_pre, n_post = len(x_pre), len(x_post)
    stacked, distance, i = [], [], 1
    while i <= n_post - k + 1:
        Q = t_normalize(x_post[i - 1: i + k - 1])
        costs, j, margin = [], 1, default_margin
        while True:
            if j > n_pre - k + 3:
                break
            R = t_normalize(x_pre[j - 1: min(j + k + margin - 1, n_pre)])
            if ref_too_short(Q, R, step_pattern):
                if j < n_pre - k - margin + 1:
                    margin += 1
                    continue
                margin = default_margin
                j += n_r
                continue
            a = D.dtw(Q, R, step_pattern=step_pattern, open_end=True,
                      distance_only=True)
            costs.append((a.distance, j, margin))
            j += n_r
            margin = default_margin
        cost_arr = np.array([c[0] for c in costs])
        best = int(np.argmin(cost_arr))                   # R: which(...)[1]
        _, j_opt, margin_opt = costs[best]
        Rs = t_normalize(x_pre[j_opt - 1: min(j_opt + k + margin_opt - 1, n_pre)])
        a = D.dtw(Q, Rs, step_pattern=step_pattern, open_end=True,
                  keep_internals=True)
        Wpp = np.zeros((int(a.index1.max()) + 1, int(a.index2.max()) + 1))
        Wpp[a.index1, a.index2] = 1.0
        w_a_Rs = weight_a[j_opt - 1: j_opt - 1 + Wpp.shape[1]]
        weight_b = (Wpp @ w_a_Rs) / Wpp.sum(axis=1)
        row = np.full(n_post, np.nan)
        row[i - 1: i + k - 1] = weight_b
        stacked.append(row)
        distance.append(a.distance)
        i += n_q
    stacked = np.array(stacked)
    distance = np.array(distance)
    keep = ~(distance > np.nanquantile(distance, dist_quant))
    stacked = stacked[keep]
    stacked = np.column_stack([remove_outliers(stacked[:, c], n_IQR)
                               for c in range(stacked.shape[1])])
    with np.errstate(invalid="ignore"):
        avg = np.nanmean(stacked, axis=0)
    avg[np.isnan(avg)] = 1.0
    return avg


def TFDTW(x, y, k, t_treat, buffer=0, step_pattern1=SYMMETRIC_P1,
          step_pattern2=ASYMMETRIC_P2, n_burn=3, ma=3, **kw):
    """R TFDTW.R::TFDTW -- returns cutoff, weight.a, avg.weight."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    r1 = first_dtw(x, y, t_treat, buffer, step_pattern1)
    cutoff = r1["cutoff"]
    x_pre, x_post = x[:cutoff], x[cutoff - n_burn - 1:]
    w_a_o = warp2weight(r1["W_a"])
    w_a = _na_locf(_ma_filter(w_a_o, ma))
    w_a[np.isnan(w_a)] = w_a_o[np.isnan(w_a)]             # ma.na = "original"
    avg = second_dtw(x_post, x_pre, k, w_a, step_pattern=step_pattern2, **kw)
    return {"cutoff": cutoff, "weight_a": w_a, "avg_weight": avg[n_burn:]}


def warped_series(x_raw, cutoff, weight_a, avg_weight, t_treat, n_out):
    """R dsc.R lines 192-195: stitch the pre- and post-period warps."""
    pre = warp_with_weight(x_raw[:cutoff], weight_a)[:t_treat]
    post = warp_with_weight(x_raw[cutoff - 1:], avg_weight)[1:]
    out = np.concatenate([pre, post])
    if out.size < n_out:
        out = np.concatenate([out, np.full(n_out - out.size, np.nan)])
    return out[:n_out]
