"""CAST estimation pipeline (Xia, Yan & Wainwright 2025).

Selects the factor rank, imputes every counterfactual cell through the
staircase of four-block sub-problems, attaches the entrywise variances, and
aggregates them into per-period effects with standard errors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from scipy.stats import norm

from ...exceptions import MlsynthEstimationError
from .four_block import entrywise_variance, impute_missing, staircase_blocks, svd_r


@dataclass
class CASTFit:
    """Point-estimate artifacts of a CAST fit."""
    rank: int
    counterfactual: np.ndarray        # N x T, observed values kept where control
    effects: np.ndarray               # N x T, nan off the treated cells
    std_errors: np.ndarray            # N x T, nan off the treated cells
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    period_effects: Dict[Any, Any]
    significance: Dict[Any, Any]
    extras: Dict[str, Any]


def _control_block(Y: np.ndarray, A: np.ndarray) -> np.ndarray:
    """The largest fully observed sub-block, used for rank selection.

    Follows the reference's ``check_svd``: among the staircase blocks, take the
    one minimising ``sqrt(1/N1 + 1/T1)`` -- the most informative rectangle.
    """
    _, _, N_all, T_all, k = staircase_blocks(A)
    crit = np.sqrt(1.0 / np.flip(N_all) + 1.0 / T_all)
    idx = int(np.argmin(crit))
    order, _, _, _, _ = staircase_blocks(A)
    return np.asarray(Y, dtype=float)[order][: N_all[k - 1 - idx], : T_all[idx]]


def select_rank(Y: np.ndarray, A: np.ndarray, method: str = "broken_stick",
                max_rank: int = 10) -> int:
    """Choose the factor rank from the spectrum of the observed control block.

    ``broken_stick`` reproduces the reference package's default; ``svht`` is
    the Gavish-Donoho optimal hard threshold for a square-ish noisy matrix;
    ``ratio`` takes the largest consecutive singular-value gap.
    """
    block = _control_block(Y, A)
    s = np.linalg.svd(block, compute_uv=False)
    s = s[s > 0]
    cap = int(min(max_rank, s.size))
    if cap < 1:                                   # pragma: no cover - defensive
        raise MlsynthEstimationError("CAST: control block has no spectrum.")

    if method == "svht":
        n, m = block.shape
        beta = min(n, m) / max(n, m)
        lam = np.sqrt(2 * (beta + 1) + 8 * beta / ((beta + 1) + np.sqrt(beta ** 2 + 14 * beta + 1)))
        thresh = lam * np.median(s)
        r = int(max(1, (s > thresh).sum()))
    elif method == "ratio":
        if s.size < 2:                            # pragma: no cover - tiny panel
            r = 1
        else:
            r = int(np.argmax(s[:-1] / s[1:]) + 1)
    else:                                         # broken_stick (reference default)
        share = s / s.sum()
        r = 1
        for j in range(1, s.size):
            stick = np.sum(1.0 / np.arange(j + 1, s.size + 1)) / s.size
            if share[j] < stick:
                r = j
                break
            r = j + 1
    return int(min(max(r, 1), cap))


def aggregate_effects(effects: np.ndarray, var: np.ndarray, treated: np.ndarray,
                      time_labels: np.ndarray, weights: Optional[np.ndarray],
                      renormalize: bool) -> Dict[Any, Any]:
    """Per-period aggregate effect and its standard error.

    The aggregate is the bilinear functional ``tau_w,t = sum_i w_i tau_i,t`` of
    paper eq. (4); its variance follows from the entrywise variances, treating
    the per-unit imputation errors within a period as independent. ``weights``
    may be a length-N vector or an N x T matrix (a period-varying weight).
    """
    out: Dict[Any, Any] = {}
    W = None if weights is None else np.asarray(weights, dtype=float)
    for j, label in enumerate(time_labels):
        rows = np.where(treated[:, j])[0]
        if rows.size == 0:
            continue
        if W is None:
            w = np.ones(rows.size)
        else:
            w = W[rows, j] if W.ndim == 2 else W[rows]
        if renormalize:
            total = w.sum()
            if not np.isfinite(total) or total == 0:   # pragma: no cover - defensive
                continue
            w = w / total
        eff = float(np.dot(w, effects[rows, j]))
        se = float(np.sqrt(np.dot(w ** 2, var[rows, j])))
        out[label] = (eff, se)
    return out


def summarize_significance(effects: np.ndarray, se: np.ndarray, treated: np.ndarray,
                           time_labels: np.ndarray, alpha: float) -> Dict[Any, Any]:
    """Per-period counts of significantly positive / negative / null effects."""
    z = norm.ppf(1 - alpha / 2)
    out: Dict[Any, Any] = {}
    for j, label in enumerate(time_labels):
        rows = np.where(treated[:, j])[0]
        if rows.size == 0:
            continue
        e, s = effects[rows, j], se[rows, j]
        sig = np.abs(e) > z * s
        out[label] = {
            "positive": int(np.sum(sig & (e > 0))),
            "negative": int(np.sum(sig & (e < 0))),
            "null": int(np.sum(~sig)),
            "n_treated": int(rows.size),
        }
    return out


def run_cast(inputs, config) -> CASTFit:
    """Run CAST end to end on prepared inputs."""
    Y, A = inputs.Y, inputs.A
    treated = inputs.treated_mask

    r = config.rank
    if r is None:
        r = select_rank(Y, A, method=config.rank_method, max_rank=config.max_rank)
    if r >= min(inputs.N, inputs.T):
        raise MlsynthEstimationError(
            f"CAST: rank r={r} must be smaller than min(N, T)="
            f"{min(inputs.N, inputs.T)}."
        )

    try:
        M_est = impute_missing(Y, A, r)
        var = entrywise_variance(Y, A, r)
    except (np.linalg.LinAlgError, ValueError) as exc:   # pragma: no cover
        raise MlsynthEstimationError(f"CAST estimation failed ({exc}).") from exc

    counterfactual = np.where(treated, M_est, Y)
    se_full = np.sqrt(np.maximum(var, 0.0))
    z = norm.ppf(1 - config.alpha / 2)

    nan = np.full_like(Y, np.nan, dtype=float)
    effects = np.where(treated, Y - counterfactual, nan)
    se = np.where(treated, se_full, nan)
    lo = np.where(treated, effects - z * se_full, nan)
    hi = np.where(treated, effects + z * se_full, nan)

    weights = None if config.weights is None else np.asarray(config.weights, dtype=float)
    if weights is not None:
        expected = f"a length-{inputs.N} vector or an {inputs.N} x {inputs.T} matrix"
        if weights.ndim == 1 and weights.shape[0] != inputs.N:
            raise MlsynthEstimationError(
                f"CAST: weights has length {weights.shape[0]}, expected {expected}."
            )
        if weights.ndim == 2 and weights.shape != (inputs.N, inputs.T):
            raise MlsynthEstimationError(
                f"CAST: weights has shape {weights.shape}, expected {expected}."
            )

    period = aggregate_effects(np.where(treated, Y - counterfactual, 0.0), var,
                               treated, inputs.time_labels, weights,
                               config.renormalize_weights)
    signif = summarize_significance(np.where(treated, Y - counterfactual, 0.0),
                                    se_full, treated, inputs.time_labels, config.alpha)

    return CASTFit(
        rank=int(r), counterfactual=counterfactual, effects=effects,
        std_errors=se, ci_lower=lo, ci_upper=hi,
        period_effects=period, significance=signif,
        extras={"rank_method": config.rank_method if config.rank is None else "fixed",
                "weighted": weights is not None,
                "renormalize_weights": config.renormalize_weights},
    )
