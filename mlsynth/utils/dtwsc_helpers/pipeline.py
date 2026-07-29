"""Orchestration for DTWSC: warp the donors, then fit a synthetic control."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ...config_models import WeightsResults
from ...exceptions import MlsynthEstimationError
from ..results_helpers import build_effect_submodels
from .dtw import ASYMMETRIC_P2, SYMMETRIC_P1
from .structures import DTWSCInputs, DTWSCResults
from .warping import savgol_second_derivative, tfdtw, warp_series

_PATTERNS = {"symmetricP1": SYMMETRIC_P1, "asymmetricP2": ASYMMETRIC_P2}


def _simplex_weights(target: np.ndarray, donors: np.ndarray) -> np.ndarray:
    """Non-negative weights summing to one, minimising pre-period squared error.

    Solved as a small QP over the simplex. Rows with a missing donor value are
    dropped rather than imputed, because a compressed warp leaves genuinely
    unobserved cells.
    """
    import cvxpy as cp

    ok = np.isfinite(donors).all(axis=1) & np.isfinite(target)
    A, b = donors[ok], target[ok]
    if A.shape[0] == 0:  # pragma: no cover - warp_series only ever pads the
        # TAIL, so the pre-treatment block always has finite rows; kept as a
        # guard in case a future warp variant can blank a pre-period.
        raise MlsynthEstimationError(
            "DTWSC: no pre-treatment period survives the warp for every donor."
        )
    J = A.shape[1]
    if J == 1:
        return np.ones(1)
    w = cp.Variable(J, nonneg=True)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(A @ w - b)),
                         [cp.sum(w) == 1])
    try:
        problem.solve(solver=cp.ECOS, abstol=1e-9, reltol=1e-9)
    except Exception:                       # pragma: no cover - solver fallback
        problem.solve()
    if w.value is None:                     # pragma: no cover - infeasible
        raise MlsynthEstimationError("DTWSC: the donor weight solve failed.")
    weights = np.clip(np.asarray(w.value, dtype=float).ravel(), 0.0, None)
    total = weights.sum()
    return weights / total if total > 0 else np.full(J, 1.0 / J)


def run_dtwsc(inputs: DTWSCInputs, *, k: int, warp: bool, smooth: bool,
              filter_width: int, poly_order: int, buffer: int, n_burn: int,
              ma: int, default_margin: int, n_q: int, n_r: int,
              dist_quant: float, n_iqr: float, step_pattern1: str,
              step_pattern2: str, match_method: str) -> DTWSCResults:
    """Warp every donor onto the treated unit's clock, then fit the control."""
    y = inputs.y
    donors = inputs.donor_matrix
    t_treat = inputs.n_pre

    warped = np.array(donors, dtype=float, copy=True)
    cutoffs: Dict[Any, int] = {}
    pre_speeds: Dict[Any, Any] = {}
    post_speeds: Dict[Any, Any] = {}

    if warp:
        # Alignment sees curvature; the warp is applied to the raw outcome.
        if smooth:
            aligned_y = savgol_second_derivative(y, filter_width, poly_order)
            aligned_donors = savgol_second_derivative(donors, filter_width,
                                                      poly_order)
        else:
            aligned_y, aligned_donors = y, donors
        for j, name in enumerate(inputs.donor_names):
            res = tfdtw(
                aligned_donors[:, j], aligned_y, k=k, t_treat=t_treat,
                buffer=buffer, step_pattern1=_PATTERNS[step_pattern1],
                step_pattern2=_PATTERNS[step_pattern2], n_burn=n_burn, ma=ma,
                match_method=match_method, default_margin=default_margin,
                n_q=n_q, n_r=n_r, dist_quant=dist_quant, n_iqr=n_iqr,
            )
            warped[:, j] = warp_series(donors[:, j], res["cutoff"],
                                       res["weight_a"], res["avg_weight"],
                                       t_treat, inputs.T)
            cutoffs[name] = int(res["cutoff"])
            pre_speeds[name] = res["weight_a"]
            post_speeds[name] = res["avg_weight"]

    pre = slice(0, t_treat)
    weights = _simplex_weights(y[pre], warped[pre])
    counterfactual = np.full(inputs.T, np.nan)
    for t in range(inputs.T):
        row = warped[t]
        ok = np.isfinite(row)
        if ok.any():
            # Renormalise over the donors still observed at t, so a compressed
            # tail shrinks the pool rather than silently zeroing a donor.
            mass = weights[ok].sum()
            if mass > 0:
                counterfactual[t] = float(row[ok] @ (weights[ok] / mass))

    # A warp that compresses every donor past the end of the panel leaves the
    # counterfactual undefined at those periods. The reference implementation
    # returns NA there and its users average over the rest; we do the same, but
    # report how many periods were dropped rather than letting a NaN silently
    # poison the ATT or silently vanish.
    gap = y - counterfactual
    pre_gap, post_gap = gap[:t_treat], gap[t_treat:]
    n_post_defined = int(np.isfinite(post_gap).sum())
    if n_post_defined == 0:  # pragma: no cover - would need EVERY donor's
        # warp to compress past the treatment date, which the cutoff clamp in
        # first_dtw prevents; kept so the failure is reported, not a NaN ATT.
        raise MlsynthEstimationError(
            "DTWSC: the warp leaves no post-treatment period with an observed "
            "donor, so no effect is identified. Try a smaller k, or "
            "warp=False."
        )
    with np.errstate(invalid="ignore"):
        att = float(np.nanmean(post_gap))
        rmse_pre = float(np.sqrt(np.nanmean(pre_gap ** 2)))
        rmse_post = float(np.sqrt(np.nanmean(post_gap ** 2)))
        base = np.nanmean(np.abs(y[t_treat:][np.isfinite(post_gap)]))
    att_percent = float(100.0 * att / base) if base else float("nan")

    submodels = build_effect_submodels(
        observed_outcome=y,
        counterfactual_outcome=counterfactual,
        n_pre_periods=int(t_treat),
        n_post_periods=int(inputs.n_post),
        time_periods=inputs.time_labels,
        effects_overrides={"att": att, "att_percent": att_percent},
        fit_overrides={"rmse_pre": rmse_pre, "rmse_post": rmse_post},
        weights=WeightsResults(
            donor_weights={str(n): float(w)
                           for n, w in zip(inputs.donor_names, weights)}
        ),
        method_name="DTWSC",
        intervention_time=(inputs.time_labels[t_treat]
                           if t_treat < inputs.T else inputs.time_labels[-1]),
    )
    metadata = {
        "T": inputs.T, "n_pre": int(t_treat), "n_post": int(inputs.n_post),
        "n_donors": inputs.J, "k": int(k), "warp": bool(warp),
        "smooth": bool(smooth), "treated_unit": inputs.treated_name,
        "n_unwarped_tail_cells": int((~np.isfinite(warped)).sum()),
        "n_post_periods_undefined": int(inputs.n_post - n_post_defined),
    }
    return DTWSCResults(
        **submodels, inputs=inputs, warped_donor_matrix=warped,
        cutoffs=cutoffs, pre_period_speeds=pre_speeds,
        post_period_speeds=post_speeds, warp_applied=bool(warp),
        metadata=metadata,
    )
