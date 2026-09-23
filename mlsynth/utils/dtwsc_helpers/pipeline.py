"""Orchestration for DTWSC: warp the donors, then fit a synthetic control."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ...config_models import InferenceResults, WeightsResults
from ...exceptions import MlsynthEstimationError
from ..results_helpers import build_effect_submodels
from .dtw import PATTERNS
from .inference import (
    icc_corrected_ttest,
    placebo_band,
    placebo_donor_pools,
)
from .structures import DTWSCInputs, DTWSCResults
from .warping import savgol_second_derivative, tfdtw, warp_series

_PATTERNS = PATTERNS


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


def _sc_via_vanillasc(y, warped, ctx, backend, covariates,
                      covariate_windows, fit_window, donor_names):
    """Fit the synthetic control through :class:`mlsynth.VanillaSC`.

    The paper's synthetic-control half is ``Synth`` with a V-matrix over
    predictors, which mlsynth already replicates. Rather than reimplement it,
    rebuild a long panel whose outcome is the WARPED donor series (the treated
    unit keeps its own, unwarped, path) and hand it over.
    """
    from ...estimators.vanillasc import VanillaSC

    times = np.asarray(ctx["time_labels"])
    n_pre = int(ctx["n_pre"])
    rows = []
    treated = str(ctx["treated_label"])
    for t_i, t in enumerate(times):
        rows.append({"__unit": treated, "__time": t, "__y": float(y[t_i]),
                     "__treat": int(t_i >= n_pre)})
    for j, name in enumerate(donor_names):
        for t_i, t in enumerate(times):
            rows.append({"__unit": str(name), "__time": t,
                         "__y": float(warped[t_i, j]), "__treat": 0})
    panel = pd.DataFrame(rows)
    # A warp that compresses a donor past the panel end leaves NaN; VanillaSC
    # needs a balanced frame, so carry the donor's last observed value forward
    # rather than dropping the period for every unit.
    panel["__y"] = panel.groupby("__unit", observed=True)["__y"].ffill()
    if panel["__y"].isna().any():
        panel["__y"] = panel.groupby("__unit", observed=True)["__y"].bfill()

    if covariates:
        src = ctx.get("covariate_frame")
        if src is None:
            raise MlsynthEstimationError(
                "DTWSC: covariates were requested but the input frame did not "
                "carry them; pass them as columns of `df`."
            )
        cov = src.copy()
        cov["__unit"] = cov["__unit"].astype(str)
        panel = panel.merge(cov, on=["__unit", "__time"], how="left")
        missing = [c for c in covariates if c not in panel.columns]
        if missing:
            raise MlsynthEstimationError(
                f"DTWSC: covariate column(s) {missing} not found in `df`."
            )

    cfg = {"df": panel, "outcome": "__y", "treat": "__treat",
           "unitid": "__unit", "time": "__time", "display_graphs": False,
           "backend": backend, "inference": False}
    if covariates:
        cfg["covariates"] = list(covariates)
    if covariate_windows:
        cfg["covariate_windows"] = covariate_windows
    if fit_window:
        cfg["fit_window"] = fit_window
    res = VanillaSC(cfg).fit()
    cf = np.asarray(res.counterfactual, dtype=float).ravel()
    w = np.array([float(res.donor_weights.get(str(n), 0.0))
                  for n in donor_names])
    total = w.sum()
    if total > 0:
        w = w / total
    return cf, w


def _fit_one(y, donors, t_treat, *, warp, smooth, filter_width, poly_order,
             k, buffer, n_burn, ma, default_margin, n_q, n_r, dist_quant,
             n_iqr, step_pattern1, step_pattern2, match_method,
             sc_backend="simplex", covariates=None, covariate_windows=None,
             fit_window=None, ctx=None, donor_names=None):
    """Warp a donor block onto ``y``'s clock and fit the simplex control.

    Returns ``(counterfactual, warped, cutoffs, pre_speeds, post_speeds,
    weights)``. Shared by the real fit and every placebo run so the two can
    never drift apart.
    """
    warped = np.array(donors, dtype=float, copy=True)
    cutoffs, pre_speeds, post_speeds = {}, {}, {}
    if warp:
        if smooth:
            aligned_y = savgol_second_derivative(y, filter_width, poly_order)
            aligned_donors = savgol_second_derivative(donors, filter_width,
                                                      poly_order)
        else:
            aligned_y, aligned_donors = y, donors
        for j in range(donors.shape[1]):
            res = tfdtw(
                aligned_donors[:, j], aligned_y, k=k, t_treat=t_treat,
                buffer=buffer, step_pattern1=_PATTERNS[step_pattern1],
                step_pattern2=_PATTERNS[step_pattern2], n_burn=n_burn, ma=ma,
                match_method=match_method, default_margin=default_margin,
                n_q=n_q, n_r=n_r, dist_quant=dist_quant, n_iqr=n_iqr,
            )
            warped[:, j] = warp_series(donors[:, j], res["cutoff"],
                                       res["weight_a"], res["avg_weight"],
                                       t_treat, y.size)
            cutoffs[j] = int(res["cutoff"])
            pre_speeds[j] = res["weight_a"]
            post_speeds[j] = res["avg_weight"]

    if sc_backend == "simplex":
        weights = _simplex_weights(y[:t_treat], warped[:t_treat])
        counterfactual = np.full(y.size, np.nan)
        for t in range(y.size):
            row = warped[t]
            ok = np.isfinite(row)
            if ok.any():
                mass = weights[ok].sum()
                if mass > 0:
                    counterfactual[t] = float(row[ok] @ (weights[ok] / mass))
    else:
        counterfactual, weights = _sc_via_vanillasc(
            y, warped, ctx, sc_backend, covariates, covariate_windows,
            fit_window, donor_names)
    return counterfactual, warped, cutoffs, pre_speeds, post_speeds, weights


def _mse(gap, lo, hi):
    seg = gap[lo:hi]
    with np.errstate(invalid="ignore"):
        return float(np.nanmean(seg ** 2)) if seg.size else float("nan")


def _run_placebos(inputs, opts, alpha, placebo_pairs, mse_window):
    """Re-analyse every donor as if treated, over perturbed donor pools.

    Follows ``Figure_5_*.R``: the treated unit is excluded from every placebo
    pool, so no placebo counterfactual can borrow from it.
    """
    y_all = inputs.donor_matrix
    t_treat, T = inputs.n_pre, inputs.T
    n_donors = inputs.J
    pools = placebo_donor_pools(n_donors + 1, treated=n_donors,
                                n_pairs=placebo_pairs)

    gaps_w, gaps_u, rows = [], [], []
    per_unit_w = {}
    hi = T if mse_window is None else min(T, t_treat + int(mse_window))
    for data_id, pool in enumerate(pools):
        for target in pool:
            others = [u for u in pool if u != target]
            if len(others) < 2:  # pragma: no cover - placebo_donor_pools
                # already refuses to emit a pool that cannot spare a target and
                # still leave two donors; kept as a local invariant.
                continue
            y = y_all[:, target]
            block = y_all[:, others]
            names = [inputs.donor_names[u] for u in others]
            ctx = {"time_labels": inputs.time_labels, "n_pre": t_treat,
                   "treated_label": inputs.donor_names[target],
                   "covariate_frame": inputs.covariate_frame}
            cf_w, *_ = _fit_one(y, block, t_treat, warp=True, ctx=ctx,
                                donor_names=names, **opts)
            cf_u, *_ = _fit_one(y, block, t_treat, warp=False, ctx=ctx,
                                donor_names=names, **opts)
            gw, gu = y - cf_w, y - cf_u
            gaps_w.append(gw)
            gaps_u.append(gu)
            name = inputs.donor_names[target]
            per_unit_w.setdefault(str(name), gw)
            mse_w, mse_u = _mse(gw, t_treat, hi), _mse(gu, t_treat, hi)
            if np.isfinite(mse_w) and np.isfinite(mse_u) and mse_u > 0:
                rows.append((np.log(mse_w / mse_u), data_id, target))

    if not gaps_w:  # pragma: no cover - unreachable while at least one pool
        # survives, which placebo_donor_pools guarantees for >= 3 units; kept
        # so an empty placebo set is named rather than crashing in nanquantile.
        raise MlsynthEstimationError(
            "DTWSC: no placebo run produced a usable fit."
        )
    band = placebo_band(np.asarray(gaps_w), alpha)
    band_u = placebo_band(np.asarray(gaps_u), alpha)
    band["lower_unwarped"] = band_u["lower"]
    band["upper_unwarped"] = band_u["upper"]
    eff = icc_corrected_ttest(
        np.array([r[0] for r in rows]),
        np.array([r[1] for r in rows]),
        np.array([r[2] for r in rows]),
    )
    eff["n_pools"] = len(pools)
    return band, per_unit_w, eff


def run_dtwsc(inputs: DTWSCInputs, *, k: int, warp: bool, smooth: bool,
              filter_width: int, poly_order: int, buffer: int, n_burn: int,
              ma: int, default_margin: int, n_q: int, n_r: int,
              dist_quant: float, n_iqr: float, step_pattern1: str,
              step_pattern2: str, match_method: str,
              inference: str = "none", placebo_pairs: int = 0,
              alpha: float = 0.05, mse_window: Optional[int] = 10,
              sc_backend: str = "simplex", covariates=None,
              covariate_windows=None, fit_window=None,
              rescale_units: bool = False) -> DTWSCResults:
    """Warp every donor onto the treated unit's clock, then fit the control."""
    y = inputs.y
    donors = inputs.donor_matrix
    t_treat = inputs.n_pre

    opts = dict(smooth=smooth, filter_width=filter_width,
                poly_order=poly_order, k=k, buffer=buffer, n_burn=n_burn,
                ma=ma, default_margin=default_margin, n_q=n_q, n_r=n_r,
                dist_quant=dist_quant, n_iqr=n_iqr,
                step_pattern1=step_pattern1, step_pattern2=step_pattern2,
                match_method=match_method, sc_backend=sc_backend,
                covariates=covariates, covariate_windows=covariate_windows,
                fit_window=fit_window)
    main_ctx = {"time_labels": inputs.time_labels, "n_pre": t_treat,
                "treated_label": inputs.treated_name,
                "covariate_frame": inputs.covariate_frame}
    counterfactual, warped, cut_ix, pre_ix, post_ix, weights = _fit_one(
        y, donors, t_treat, warp=warp, ctx=main_ctx,
        donor_names=inputs.donor_names, **opts)
    names = inputs.donor_names
    cutoffs: Dict[Any, int] = {names[j]: v for j, v in cut_ix.items()}
    pre_speeds: Dict[Any, Any] = {names[j]: v for j, v in pre_ix.items()}
    post_speeds: Dict[Any, Any] = {names[j]: v for j, v in post_ix.items()}

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

    band = per_unit = eff = None
    std_inference = None
    if inference == "placebo":
        band, per_unit, eff = _run_placebos(inputs, opts, alpha,
                                            placebo_pairs, mse_window)
        # The reported interval is the placebo band averaged over the post
        # period -- the scalar summary of a pointwise object. Read
        # `placebo_band` for the path, which is what the paper plots.
        with np.errstate(invalid="ignore"):
            lo = float(np.nanmean(band["lower"][t_treat:]))
            hi = float(np.nanmean(band["upper"][t_treat:]))
        std_inference = InferenceResults(
            method="placebo",
            ci_lower=lo,
            ci_upper=hi,
            confidence_level=float(1.0 - alpha),
            p_value=float(eff["p"]),
            details={
                "n_placebo_runs": int(eff["n_runs"]),
                "n_donor_pools": int(eff["n_pools"]),
                "efficiency_t": float(eff["t"]),
                "efficiency_df": float(eff["df"]),
                "icc": float(eff["icc"]),
                "vif": float(eff["vif"]),
                "note": (
                    "Pointwise placebo band (Cao & Chadefaux 2025). ci_lower / "
                    "ci_upper are the post-period means of the band, not a "
                    "sampling interval for the ATT; p_value is the efficiency "
                    "test on log(MSE_DSC/MSE_SC), which asks whether warping "
                    "helps, not whether the treatment had an effect."
                ),
            },
        )

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
        inference=std_inference,
        intervention_time=(inputs.time_labels[t_treat]
                           if t_treat < inputs.T else inputs.time_labels[-1]),
    )
    metadata = {
        "T": inputs.T, "n_pre": int(t_treat), "n_post": int(inputs.n_post),
        "n_donors": inputs.J, "k": int(k), "warp": bool(warp),
        "smooth": bool(smooth), "treated_unit": inputs.treated_name,
        "sc_backend": str(sc_backend),
        "rescale_units": bool(rescale_units),
        "covariates": list(covariates) if covariates else None,
        "n_unwarped_tail_cells": int((~np.isfinite(warped)).sum()),
        "n_post_periods_undefined": int(inputs.n_post - n_post_defined),
    }
    return DTWSCResults(
        **submodels, inputs=inputs, warped_donor_matrix=warped,
        cutoffs=cutoffs, pre_period_speeds=pre_speeds,
        post_period_speeds=post_speeds, warp_applied=bool(warp),
        placebo_band=band, placebo_gaps=per_unit, efficiency_test=eff,
        metadata=metadata,
    )
