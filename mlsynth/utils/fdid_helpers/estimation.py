"""Forward-selection and difference-in-differences estimation for FDID.

This module holds the heavy numerical core of the Forward
Difference-in-Differences estimator of Li (2023):

* :func:`forward_did_select` -- the vectorised forward-selection loop that
  greedily adds the donor most improving pre-treatment R^2, tracks the
  R^2 path, and returns the optimal donor subset alongside the textbook
  all-donor difference-in-differences benchmark.
* :func:`did_from_mean` -- the difference-in-differences estimate for a
  given donor average (ATT, fit, analytical inference, and vectors).

Both previously lived in the shared ``selector_helpers`` grab-bag and the
legacy ``estutils`` module; they are FDID-specific and now live with the
rest of the FDID pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...exceptions import MlsynthEstimationError
from ..groupfit import MIN_WINDOW, fit_two_group
from .inference import adid_inference, did_inference, hac_lag


def did_from_mean(
    treated: np.ndarray,
    mean_ctrl: np.ndarray,
    pre_periods: int,
    inference: str = "analytic",
    lrvar_lag: Optional[int] = None,
) -> Dict[str, Any]:
    """Difference-in-differences estimate from a pre-computed donor average.

    Parameters
    ----------
    treated : np.ndarray
        Treated-unit outcome vector, shape ``(T,)``.
    mean_ctrl : np.ndarray
        Average outcome of the selected donor pool, shape ``(T,)``.
    pre_periods : int
        Number of pre-treatment periods ``T0``.
    inference : {"analytic", "hac"}, default "analytic"
        Standard error to report; see
        :func:`~mlsynth.utils.fdid_helpers.inference.did_inference`.
    lrvar_lag : int, optional
        Truncation lag for ``inference="hac"``.

    Returns
    -------
    dict
        Structured result with ``Effects``, ``Fit``, ``Inference``, and
        ``Vectors`` blocks.
    """
    T = len(treated)
    T0 = pre_periods
    T1 = T - T0
    treated_pre, treated_post = treated[:T0], treated[T0:]
    ctrl_pre, ctrl_post = mean_ctrl[:T0], mean_ctrl[T0:]

    intercept = (treated_pre - ctrl_pre).mean()
    counterfactual = intercept + mean_ctrl

    att = (treated_post.mean() - treated_pre.mean()) - (
        ctrl_post.mean() - ctrl_pre.mean()
    )

    resid_pre = treated_pre - counterfactual[:T0]
    rmse = np.sqrt(np.mean(resid_pre ** 2)) if T0 > 0 else np.nan
    ss_tot = np.sum((treated_pre - treated_pre.mean()) ** 2)
    ss_res = np.sum(resid_pre ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan

    # Resolve the truncation lag here so the reported "Lag" is the one the
    # standard error actually used, and not a second guess at the default.
    used_lag = (
        (hac_lag(T0, T1) if lrvar_lag is None else int(lrvar_lag))
        if inference == "hac"
        else None
    )
    se, ci, pval, satt = did_inference(
        att, resid_pre, T0, T1, method=inference, lrvar_lag=used_lag
    )
    post_cf_mean = counterfactual[T0:].mean()

    # Nothing below is rounded. These values are read straight into
    # ``FDIDMethodFit`` by ``results_assembly``, and from there into the
    # standardized ``time_series`` contract, so a display convention applied here
    # becomes the number every caller gets. It used to be: the counterfactual and
    # the observed series went out at 3 decimals and the scalars at 4, which on a
    # proportion-scale outcome cost a fifth of the effect -- an ATT of -0.000166
    # reported as -0.0002. Li's own ``Fun_FDID.R`` returns raw doubles and her
    # readme prints eight significant figures, so the quantization was never the
    # method's either. Formatting belongs to whoever displays the result.
    return {
        "Effects": {
            "ATT": float(att),
            "Percent ATT": (100.0 * att / post_cf_mean
                            if post_cf_mean != 0 else np.nan),
            "SATT": float(satt),
        },
        "Fit": {
            "T0 RMSE": float(rmse),
            "R-Squared": float(r2),
            "Pre-Periods": T0,
        },
        "Inference": {
            "P-Value": float(pval),
            "95% CI": (float(ci[0]), float(ci[1])),
            "SE": float(se),
            "Intercept": float(intercept),
            "Method": inference,
            "Lag": used_lag,
        },
        "Vectors": {
            "Observed": np.asarray(treated, dtype=float),
            "Counterfactual": np.asarray(counterfactual, dtype=float),
            "Gap": np.column_stack(
                (treated - counterfactual, np.arange(T) - T0 + 1)),
        },
    }


def _record_verbose_step(
    intermediary_results: list,
    it: int,
    best_idx: int,
    best_r2: float,
    r2_cand: np.ndarray,
    selected: List[int],
    donor_names: List[Any],
    current_mean_pre: np.ndarray,
    k: int,
) -> None:
    """Append one forward-selection step to the verbose diagnostics log."""
    intermediary_results.append(
        {
            "iteration": it + 1,
            "selected_idx": best_idx,
            "selected_name": donor_names[best_idx],
            "selected_names": [donor_names[i] for i in selected],
            "n_selected": k + 1,
            "best_R2_this_step": best_r2,
            "R2_all_candidates": r2_cand.copy(),
            "running_mean_pre": current_mean_pre.copy(),
        }
    )


def _choose_optimal_subset(
    selected: List[int], R2_path: np.ndarray
) -> Tuple[List[int], np.ndarray]:
    """Keep the donor prefix up to (and including) the R^2-maximising step."""
    if len(selected) == 0:
        return [], []
    best_iter = int(np.argmax(R2_path))
    return selected[: best_iter + 1], R2_path[: best_iter + 1]


def _compute_fdid_result(
    treated_outcome: np.ndarray,
    control_outcomes: np.ndarray,
    optimal_idxs: List[int],
    pre_periods: int,
    R2_path: np.ndarray,
    donor_names: List[Any],
    inference: str = "analytic",
    lrvar_lag: Optional[int] = None,
) -> Dict[str, Any]:
    """Difference-in-differences result for the selected donor subset."""
    optimal_mean = control_outcomes[:, optimal_idxs].mean(axis=1)
    result = did_from_mean(
        treated_outcome, optimal_mean, pre_periods,
        inference=inference, lrvar_lag=lrvar_lag,
    )
    result.update(
        {
            "R2_at_each_step": R2_path,
            "selected_controls": optimal_idxs,
            "selected_names": [donor_names[i] for i in optimal_idxs],
        }
    )
    return result


def adid_from_mean(
    treated: np.ndarray,
    mean_ctrl: np.ndarray,
    pre_periods: int,
    inference: str = "analytic",
    lrvar_lag: Optional[int] = None,
) -> Dict[str, Any]:
    """Augmented difference-in-differences from a pre-computed donor average.

    Li and Van den Bulte (2022), Equation 2.4: regress the treated outcome on a
    constant and the control average over the pre-period,

        y_1t = delta_1 + delta_2 * ybar_co,t + e_1t,   t = 1, ..., T1,

    and predict the counterfactual as ``delta_1 + delta_2 * ybar_co,t``.
    :func:`did_from_mean` is the same construction with ``delta_2`` held at
    one, which is why the two live side by side: ADID buys one parameter, and
    what it buys is the treated unit being allowed to run at a different scale
    to the control average instead of merely a different level.

    The donor average is the caller's. The paper averages the whole control
    group, so :func:`forward_did_select` passes the unselected mean here; a
    forward-selected average with a free slope is a combination neither paper
    studies and this function makes no claim about it.

    Parameters
    ----------
    treated : np.ndarray
        Treated-unit outcome vector, shape ``(T,)``.
    mean_ctrl : np.ndarray
        Average outcome of the donor pool, shape ``(T,)``.
    pre_periods : int
        Number of pre-treatment periods ``T1``.
    inference : {"analytic", "hac"}, default "analytic"
        Which branch of their Appendix A.1 variance to report; see
        :func:`~mlsynth.utils.fdid_helpers.inference.adid_inference`.
    lrvar_lag : int, optional
        Truncation lag for ``inference="hac"``.

    Returns
    -------
    dict
        ``Effects``, ``Fit``, ``Inference`` and ``Vectors`` blocks, matching
        :func:`did_from_mean` so both reach the same result assembly.
        ``Inference["Slope"]`` is ``delta_2``, which no other fit reports.

    Raises
    ------
    MlsynthEstimationError
        If there are fewer pre-treatment periods than the two parameters need,
        or the control average is constant over them, which leaves the slope
        unidentified.
    """
    treated = np.asarray(treated, dtype=float).ravel()
    mean_ctrl = np.asarray(mean_ctrl, dtype=float).ravel()
    T = treated.size
    T0 = int(pre_periods)
    T1 = T - T0

    if T0 < MIN_WINDOW:
        raise MlsynthEstimationError(
            f"ADID fits an intercept and a slope, so it needs at least "
            f"{MIN_WINDOW} pre-treatment periods to leave a residual; found "
            f"{T0}. Use the DID fit, which estimates one parameter."
        )

    # ADID's own identification rule, and stricter than the shared helper's. The
    # helper asks for S_xx > 0; a donor average moving in its thirteenth digit
    # satisfies that and still leaves the slope decided by rounding, so the
    # threshold here is relative to the series' own magnitude. Which designs to
    # refuse is the estimator's, so it is tested here and not in the helper.
    spread = float(np.ptp(mean_ctrl[:T0]))
    scale = max(float(np.max(np.abs(mean_ctrl[:T0]))), 1.0)
    if spread <= 1e-12 * scale:
        raise MlsynthEstimationError(
            "The donor average is constant over the pre-treatment window, so "
            "ADID's slope is not identified: every slope fits the data equally "
            "well and the counterfactual would be arbitrary. Use the DID fit, "
            "which holds the slope at one."
        )

    fit = fit_two_group(treated[:T0], mean_ctrl[:T0])
    counterfactual = fit.predict(mean_ctrl)
    resid_pre = fit.resid

    att = float(np.mean(treated[T0:] - counterfactual[T0:])) if T1 > 0 else np.nan
    rmse = float(np.sqrt(np.mean(resid_pre ** 2)))
    ss_tot = float(np.sum((treated[:T0] - treated[:T0].mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid_pre ** 2)) / ss_tot if ss_tot > 1e-12 else np.nan

    used_lag = (
        (hac_lag(T0, T1) if lrvar_lag is None else int(lrvar_lag))
        if inference == "hac"
        else None
    )
    se, ci, pval, satt = adid_inference(
        att, resid_pre, fit.sums, mean_ctrl[:T0],
        float(np.mean(mean_ctrl[T0:])) if T1 > 0 else 0.0,
        T0, T1, method=inference, lrvar_lag=used_lag,
    )
    post_cf_mean = float(np.mean(counterfactual[T0:])) if T1 > 0 else np.nan

    return {
        "Effects": {
            "ATT": float(att),
            "Percent ATT": (100.0 * att / post_cf_mean
                            if post_cf_mean not in (0.0, np.nan) else np.nan),
            "SATT": float(satt),
        },
        "Fit": {
            "T0 RMSE": float(rmse),
            "R-Squared": float(r2),
            "Pre-Periods": T0,
        },
        "Inference": {
            "P-Value": float(pval),
            "95% CI": (float(ci[0]), float(ci[1])),
            "SE": float(se),
            "Intercept": float(fit.alpha),
            "Slope": float(fit.beta),
            "Method": inference,
            "Lag": used_lag,
        },
        "Vectors": {
            "Observed": np.asarray(treated, dtype=float),
            "Counterfactual": np.asarray(counterfactual, dtype=float),
            "Gap": np.column_stack(
                (treated - counterfactual, np.arange(T) - T0 + 1)),
        },
    }


def forward_did_select(
    treated_outcome: np.ndarray,
    control_outcomes: np.ndarray,
    pre_periods: int,
    donor_names: List[Any],
    verbose: bool = False,
    inference: str = "analytic",
    lrvar_lag: Optional[int] = None,
) -> Dict[str, Any]:
    """Run Li (2023) forward-selected difference-in-differences.

    Sequentially adds the control unit that most improves pre-treatment
    fit (R^2) with the treated unit, tracks the path of R^2 values, and
    returns both the textbook all-donor DID and the optimal FDID estimate.

    Parameters
    ----------
    treated_outcome : np.ndarray
        Treated-unit outcome vector, shape ``(T,)``.
    control_outcomes : np.ndarray
        Outcome matrix for all potential control units, shape ``(T, N)``.
    pre_periods : int
        Number of pre-treatment periods ``T0``.
    donor_names : list
        Donor labels; length must equal ``N``.
    verbose : bool, default False
        If True, attach per-step diagnostics under ``"intermediary"``.
    inference : {"analytic", "hac"}, default "analytic"
        Standard error to report. Inference is downstream of selection: the
        forward search maximises pre-treatment R^2 and never consults a
        standard error, so this changes the reported interval and nothing
        else.
    lrvar_lag : int, optional
        Truncation lag for ``inference="hac"``.

    Returns
    -------
    dict
        ``{"DID": <all-donor result>, "FDID": <forward-selected result>}``.

    References
    ----------
    Li, K. T. (2023). Frontiers: A Simple Forward Difference-in-Differences
    Method. Marketing Science, 43(2), 267-279.
    https://doi.org/10.1287/mksc.2022.0212
    """
    if len(donor_names) != control_outcomes.shape[1]:
        raise ValueError("donor_names length must match number of control units")

    T, N = control_outcomes.shape
    T0 = pre_periods
    treated_pre = treated_outcome[:T0]

    y_c = treated_pre - treated_pre.mean()
    ss_tot = np.sum(y_c ** 2)
    if ss_tot <= 1e-12:
        ss_tot = 1e-12

    X_pre = control_outcomes[:T0]

    mean_all = control_outcomes.mean(axis=1)
    did_all = did_from_mean(
        treated_outcome, mean_all, T0, inference=inference, lrvar_lag=lrvar_lag
    )
    # The same donor average, with the slope fitted instead of held at one.
    # It is refused on a panel that cannot identify a slope, and the selection
    # below does not depend on it, so the refusal is recorded and not raised.
    try:
        adid_all = adid_from_mean(
            treated_outcome, mean_all, T0, inference=inference,
            lrvar_lag=lrvar_lag
        )
    except MlsynthEstimationError as exc:
        adid_all, adid_unavailable = None, str(exc)
    else:
        adid_unavailable = None

    # --- constants precomputed once (independent of the selection step) ---
    # Adding donor j to k already-selected donors gives the candidate average
    # (S + x_j) / (k + 1). The DID R^2 depends only on the *centred* donors, so
    # centre once and cache each donor's squared norm ``q`` and its cross-product
    # with the treated ``p``. Then a step costs one matvec against the centred
    # running sum ``S_c`` -- O(N.T0) per step, O(N^2.T0) overall (versus the
    # reference's O(N^3.T0)), with identical selections.
    x_c = X_pre - X_pre.mean(axis=0)              # (T0, N) centred donors
    q = np.einsum("tj,tj->j", x_c, x_c)           # ||x_c_j||^2
    p = x_c.T @ y_c                               # y_c . x_c_j

    S_c = np.zeros(T0, dtype=float)               # centred running sum of selected
    Sc2 = 0.0                                     # ||S_c||^2
    ySc = 0.0                                     # y_c . S_c
    run_sum_pre = np.zeros(T0, dtype=float)       # uncentred running sum (verbose)

    selected: List[int] = []
    R2_path = np.empty(N, dtype=float)
    remaining_mask = np.ones(N, dtype=bool)
    intermediary_results = [] if verbose else None

    for it in range(N):
        k = len(selected)
        c = 1.0 / (k + 1)
        dots = x_c.T @ S_c                        # (N,) one matvec
        ss_X = c * c * (Sc2 + 2.0 * dots + q)
        cross = c * (ySc + p)
        r2_all = 1.0 - (ss_tot + ss_X - 2.0 * cross) / ss_tot
        # never re-select an already-chosen donor (first-tie argmax over the rest)
        r2_all[~remaining_mask] = -np.inf

        best_idx = int(np.nanargmax(r2_all))
        best_r2 = float(r2_all[best_idx])

        if verbose:
            remaining_idx = np.where(remaining_mask)[0]
            _record_verbose_step(
                intermediary_results=intermediary_results,
                it=it, best_idx=best_idx, best_r2=best_r2,
                r2_cand=r2_all[remaining_idx],
                selected=selected + [best_idx], donor_names=donor_names,
                current_mean_pre=(run_sum_pre + X_pre[:, best_idx]) / (k + 1), k=k,
            )

        selected.append(best_idx)
        R2_path[it] = best_r2
        remaining_mask[best_idx] = False
        # fold the selected donor into the running sums (O(T0))
        Sc2 += 2.0 * dots[best_idx] + q[best_idx]
        ySc += p[best_idx]
        S_c = S_c + x_c[:, best_idx]
        run_sum_pre = run_sum_pre + X_pre[:, best_idx]

    optimal_idxs, R2_path = _choose_optimal_subset(selected, R2_path)

    fdid_result = _compute_fdid_result(
        treated_outcome=treated_outcome,
        control_outcomes=control_outcomes,
        optimal_idxs=optimal_idxs,
        pre_periods=T0,
        R2_path=R2_path,
        donor_names=donor_names,
        inference=inference,
        lrvar_lag=lrvar_lag,
    )

    if verbose:
        fdid_result["intermediary"] = intermediary_results

    out = {"DID": did_all, "FDID": fdid_result}
    if adid_all is not None:
        out["ADID"] = adid_all
    else:
        out["ADID_unavailable"] = adid_unavailable
    return out
