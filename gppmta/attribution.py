"""Graphical attribution: direct and total removal effects (Sec. 3).

For a path ``D`` and a customer-initiated event ``i`` write the contribution
of an earlier event ``l`` to its intensity as
``C[l, i] = alpha[e_l, e_i] psi_{e_l}(t_i - t_l)`` and its share as
``W[l, i] = C[l, i] / lambda_{e_i}(t_i | H^D)``. Then

* the direct removal effect (DRE, eq. 4) of a removal set ``R`` on the
  conversion at ``c`` is ``sum_{l in R} W[l, c]``;
* the total removal effect (TRE, Sec. 3.2) is ``sum_l W[l, c] P_l`` with
  ``P_l = P(l in R°)`` the probability that the thinning of Algorithm 1
  removes ``l``. Thinning removes a later customer-initiated event ``i`` with
  probability ``sum_{l in R°} W[l, i]``, which is linear in the indicators of
  earlier removals, so ``P_i = sum_{l < i} W[l, i] P_l`` with ``P = 1`` on
  ``R``: one triangular solve gives the exact expectation. Algorithm 2 (score
  backpropagation) computes the adjoint of the same linear system; both are
  here, together with the Monte Carlo of Algorithm 1, and the tests hold all
  three to one another.
"""

from __future__ import annotations

import warnings
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular

from .fit import GPPParams
from .paths import Path


def path_shares(params: GPPParams, path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Intensities at each customer-initiated event and the share matrix ``W``.

    Returns ``lam`` (``nan`` at firm-initiated events) and ``W`` with
    ``W[l, i]`` the share of ``lambda_{e_i}(t_i)`` contributed by event ``l``.
    A column whose intensity is zero (possible only under fitted parameters)
    is ``nan``.
    """
    t, e = path.times, path.types
    dt = t[None, :] - t[:, None]  # dt[l, i] = t_i - t_l
    after = dt > 0
    beta = params.decay[e][:, None]
    psi = np.where(after, beta * np.exp(-beta * np.where(after, dt, 0.0)), 0.0)
    C = params.alpha_full[np.ix_(e, e)] * psi
    cust = e < params.q
    lam = np.full(len(t), np.nan)
    lam[cust] = params.mu[e[cust]] + C[:, cust].sum(axis=0)
    W = np.zeros_like(C)
    with np.errstate(divide="ignore", invalid="ignore"):
        W[:, cust] = C[:, cust] / lam[cust]
    W[:, cust & (lam == 0)] = np.nan
    return lam, W


def _removal_mask(path: Path, c: int, removal: np.ndarray | Sequence[int]) -> np.ndarray:
    R = np.asarray(removal)
    if R.dtype == bool:
        if R.shape != (c,):
            raise ValueError("a boolean removal set must cover the events before the conversion")
        return R
    mask = np.zeros(c, dtype=bool)
    if R.size and (R.min() < 0 or R.max() >= c):
        raise ValueError("removal set must index events strictly before the conversion")
    mask[R.astype(int)] = True
    return mask


def _check_conversion(path: Path, c: int) -> None:
    if not 0 <= c < len(path) or path.types[c] != 0:
        raise ValueError(f"event {c} is not a conversion")


def direct_removal(params: GPPParams, path: Path, c: int, removal, W=None) -> float:
    """DRE of ``removal`` (indices or a mask over events ``0..c-1``) on conversion ``c``."""
    _check_conversion(path, c)
    R = _removal_mask(path, c, removal)
    if W is None:
        _, W = path_shares(params, path)
    return float(W[:c, c][R].sum())


def removal_probabilities(params: GPPParams, path: Path, c: int, removal, W=None) -> np.ndarray:
    """``P(l in R°)`` for each event before ``c`` under the thinning of Algorithm 1."""
    _check_conversion(path, c)
    R = _removal_mask(path, c, removal)
    if W is None:
        _, W = path_shares(params, path)
    if not R.any():
        return np.zeros(c)
    imin = int(np.argmax(R))
    cust = path.types[:c] < params.q
    thinned = cust & ~R & (np.arange(c) > imin)
    M = np.zeros((c, c))
    M[thinned] = W[:c, :c].T[thinned]
    return solve_triangular(np.eye(c) - M, R.astype(float), lower=True, unit_diagonal=True)


def total_removal(params: GPPParams, path: Path, c: int, removal, W=None) -> float:
    """TRE of ``removal`` on conversion ``c``: exact expectation over thinning."""
    if W is None:
        _, W = path_shares(params, path)
    P = removal_probabilities(params, path, c, removal, W)
    return float(W[:c, c] @ P)


def total_removal_backprop(params: GPPParams, path: Path, c: int, removal) -> float:
    """Algorithm 2 (score backpropagation), transcribed loop for loop."""
    _check_conversion(path, c)
    R = _removal_mask(path, c, removal)
    if not R.any():
        return 0.0
    _, W = path_shares(params, path)
    A = params.alpha_full
    e = path.types
    imin = int(np.argmax(R))
    y = np.zeros(c)
    y[imin:c] = W[imin:c, c]
    for i in range(c - 1, imin, -1):  # i > i_min(R), descending
        if R[i] or e[i] >= params.q:  # only F^{E_c}(D \ R) is thinned
            continue
        for ip in range(i):
            if A[e[ip], e[i]] > 0:  # (e_{i'} -> e_i) in the graph
                y[ip] += y[i] * W[ip, i]
    return float(y[R].sum())


def total_removal_thinning(
    params: GPPParams, path: Path, c: int, removal, n_draws: int = 1000, seed=None
) -> float:
    """Algorithm 1 (Monte Carlo thinning)."""
    _check_conversion(path, c)
    R = _removal_mask(path, c, removal)
    if not R.any():
        return 0.0
    rng = np.random.default_rng(seed)
    _, W = path_shares(params, path)
    cust = path.types[:c] < params.q
    imin = int(np.argmax(R))
    draws = np.empty(n_draws)
    for d in range(n_draws):
        Rc = R.copy()
        for i in range(imin + 1, c):
            if Rc[i] or not cust[i]:
                continue
            if rng.random() < W[:i, i][Rc[:i]].sum():
                Rc[i] = True
        draws[d] = W[:c, c][Rc].sum()
    return float(draws.mean())


def attribute(
    params: GPPParams,
    paths: Sequence[Path],
    channels: Mapping[str, Sequence[int]],
    path_ids: Sequence | None = None,
) -> pd.DataFrame:
    """Channel-level DRE and TRE for every conversion in ``paths``.

    One row per conversion: ``baseline`` is ``mu_1 / lambda_1``, the share of the
    conversion intensity that no touchpoint explains; ``dre_<ch>`` and
    ``tre_<ch>`` remove all of the channel's touchpoints before the conversion
    (the TRE of a channel is not the sum of its touchpoints' TREs, Sec. 3.3).
    """
    chan = {ch: np.asarray(list(types), dtype=int) for ch, types in channels.items()}
    for ch, types in chan.items():
        if np.any(types < 1) or np.any(types >= params.p):
            raise ValueError(f"channel {ch!r} must list touchpoint types in 1..{params.p - 1}")
    ids = range(len(paths)) if path_ids is None else path_ids
    rows = []
    undefined = 0
    for pid, path in zip(ids, paths):
        conv = np.flatnonzero(path.types == 0)
        if conv.size == 0:
            continue
        lam, W = path_shares(params, path)
        for c in conv:
            row = {"path": pid, "event": int(c), "time": path.times[c], "intensity": lam[c]}
            if not lam[c] > 0:
                undefined += 1
            row["baseline"] = params.mu[0] / lam[c] if lam[c] > 0 else np.nan
            for ch, types in chan.items():
                R = np.isin(path.types[:c], types)
                row[f"dre_{ch}"] = float(W[:c, c][R].sum())
                row[f"tre_{ch}"] = total_removal(params, path, int(c), R, W) if R.any() else 0.0
            rows.append(row)
    if undefined:
        warnings.warn(
            f"{undefined} conversion(s) have zero fitted intensity; their scores are NaN",
            RuntimeWarning,
            stacklevel=2,
        )
    cols = ["path", "event", "time", "intensity", "baseline"]
    cols += [f"{k}_{ch}" for ch in chan for k in ("dre", "tre")]
    return pd.DataFrame(rows, columns=cols)


def touchpoint_scores(params: GPPParams, path: Path, c: int) -> pd.DataFrame:
    """DRE and TRE of each single touchpoint before conversion ``c``."""
    _check_conversion(path, c)
    _, W = path_shares(params, path)
    out = []
    for i in range(c):
        if path.types[i] == 0:
            continue
        out.append({
            "event": i,
            "time": path.times[i],
            "type": int(path.types[i]),
            "dre": float(W[i, c]),
            "tre": total_removal(params, path, c, [i], W),
        })
    return pd.DataFrame(out, columns=["event", "time", "type", "dre", "tre"])


def channel_shares(scores: pd.DataFrame, kind: str = "tre") -> pd.Series:
    """Proportion of the channel-level aggregated score (CAS) per channel."""
    if kind not in ("dre", "tre"):
        raise ValueError("kind must be 'dre' or 'tre'")
    cols = [col for col in scores.columns if col.startswith(f"{kind}_")]
    cas = scores[cols].sum()
    cas.index = [col[len(kind) + 1:] for col in cols]
    return cas / cas.sum()


def kl_divergence(p, p_hat) -> float:
    p, p_hat = np.asarray(p, float), np.asarray(p_hat, float)
    m = p > 0
    return float(np.sum(p[m] * np.log(p[m] / p_hat[m])))


def hellinger(p, p_hat) -> float:
    p, p_hat = np.asarray(p, float), np.asarray(p_hat, float)
    return float(np.sqrt(0.5 * np.sum((np.sqrt(p_hat) - np.sqrt(p)) ** 2)))
