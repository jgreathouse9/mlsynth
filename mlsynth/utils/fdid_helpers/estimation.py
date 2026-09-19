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

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .inference import did_inference, hac_lag


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

    return {
        "Effects": {
            "ATT": round(float(att), 4),
            "Percent ATT": round(100 * att / post_cf_mean, 3)
            if post_cf_mean != 0 else np.nan,
            "SATT": round(float(satt), 3) if not np.isnan(satt) else np.nan,
        },
        "Fit": {
            "T0 RMSE": round(float(rmse), 4),
            "R-Squared": round(float(r2), 4) if not np.isnan(r2) else np.nan,
            "Pre-Periods": T0,
        },
        "Inference": {
            "P-Value": round(float(pval), 4) if not np.isnan(pval) else np.nan,
            "95% CI": (round(float(ci[0]), 4), round(float(ci[1]), 4))
            if not np.isnan(ci[0]) else (np.nan, np.nan),
            "SE": round(float(se), 4) if not np.isnan(se) else np.nan,
            "Intercept": round(float(intercept), 4),
            "Method": inference,
            "Lag": used_lag,
        },
        "Vectors": {
            "Observed": np.round(treated, 3),
            "Counterfactual": np.round(counterfactual, 3),
            "Gap": np.round(
                np.column_stack(
                    (treated - counterfactual, np.arange(T) - T0 + 1)
                ),
                3,
            ),
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


def forward_selection_path(
    donors_pre: np.ndarray,
    targets_pre: Sequence[np.ndarray],
    weights: Optional[Sequence[float]] = None,
    verbose_hook: Optional[Any] = None,
) -> Tuple[List[int], np.ndarray]:
    r"""Greedy forward-selection order and its criterion path.

    Adding donor ``j`` to ``k`` already-selected donors gives the candidate
    average ``(S + x_j) / (k + 1)``. The difference-in-differences :math:`R^2`
    depends only on the *centred* donors, so centre once and cache each donor's
    squared norm and its cross-product with each target. A step then costs one
    matvec against the centred running sum -- :math:`O(N T_0)` per step,
    :math:`O(N^2 T_0)` overall, against the reference implementation's
    :math:`O(N^3 T_0)`, with identical selections.

    Several targets are supported so the criterion can be a convex combination
    of per-unit and pooled pre-treatment fit. The donor-side quantities are
    shared across targets; only the cross-products are per target.

    Parameters
    ----------
    donors_pre : np.ndarray
        Donor outcomes over the pre-window, shape ``(T0, N)``.
    targets_pre : sequence of np.ndarray
        One or more length-``T0`` target series to fit.
    weights : sequence of float, optional
        Convex weights over ``targets_pre``; defaults to equal weights. A
        single target needs none.
    verbose_hook : callable, optional
        Called as ``hook(it, best_idx, best_crit, crit_remaining, selected, k)``
        after each step, for the caller's diagnostics log.

    Returns
    -------
    order : list of int
        All ``N`` donors in the order the greedy search added them.
    path : np.ndarray
        Criterion value after each step, shape ``(N,)``. The caller truncates
        at the maximum.

    Raises
    ------
    ValueError
        If ``weights`` does not match ``targets_pre``, or a target's length
        does not match the pre-window.
    """
    X_pre = np.asarray(donors_pre, dtype=float)
    T0, N = X_pre.shape
    targets = [np.asarray(t, dtype=float).ravel() for t in targets_pre]
    if not targets:
        raise ValueError("forward selection needs at least one target series.")
    for t in targets:
        if t.size != T0:
            raise ValueError(
                f"target length {t.size} does not match the pre-window {T0}."
            )
    if weights is None:
        w = np.full(len(targets), 1.0 / len(targets))
    else:
        w = np.asarray(weights, dtype=float)
        if w.size != len(targets):
            raise ValueError("weights must have one entry per target series.")

    x_c = X_pre - X_pre.mean(axis=0)              # (T0, N) centred donors
    q = np.einsum("tj,tj->j", x_c, x_c)           # ||x_c_j||^2

    y_c = [t - t.mean() for t in targets]
    # A degenerate target has no variation to explain; floor its total sum of
    # squares so the ratio stays finite and the donor ranking is unaffected.
    ss_tot = np.array([max(float(y @ y), 1e-12) for y in y_c])
    p = [x_c.T @ y for y in y_c]                  # per-target cross-products

    S_c = np.zeros(T0, dtype=float)
    Sc2 = 0.0
    ySc = np.zeros(len(targets), dtype=float)

    order: List[int] = []
    path = np.empty(N, dtype=float)
    remaining = np.ones(N, dtype=bool)

    for it in range(N):
        k = len(order)
        c = 1.0 / (k + 1)
        dots = x_c.T @ S_c                        # (N,) one matvec
        ss_X = c * c * (Sc2 + 2.0 * dots + q)
        crit = np.zeros(N, dtype=float)
        for m in range(len(targets)):
            cross = c * (ySc[m] + p[m])
            crit += w[m] * (
                1.0 - (ss_tot[m] + ss_X - 2.0 * cross) / ss_tot[m]
            )
        crit[~remaining] = -np.inf

        best_idx = int(np.nanargmax(crit))
        best_crit = float(crit[best_idx])
        if verbose_hook is not None:
            verbose_hook(it, best_idx, best_crit, crit[remaining],
                         order + [best_idx], k)

        order.append(best_idx)
        path[it] = best_crit
        remaining[best_idx] = False
        Sc2 += 2.0 * dots[best_idx] + q[best_idx]
        for m in range(len(targets)):
            ySc[m] += p[m][best_idx]
        S_c = S_c + x_c[:, best_idx]

    return order, path


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

    T0 = pre_periods
    X_pre = control_outcomes[:T0]

    mean_all = control_outcomes.mean(axis=1)
    did_all = did_from_mean(
        treated_outcome, mean_all, T0, inference=inference, lrvar_lag=lrvar_lag
    )

    intermediary_results = [] if verbose else None

    def _hook(it, best_idx, best_r2, r2_remaining, selected, k):
        _record_verbose_step(
            intermediary_results=intermediary_results,
            it=it, best_idx=best_idx, best_r2=best_r2,
            r2_cand=r2_remaining,
            selected=selected, donor_names=donor_names,
            current_mean_pre=X_pre[:, selected].mean(axis=1), k=k,
        )

    selected, R2_path = forward_selection_path(
        X_pre, [treated_outcome[:T0]],
        verbose_hook=_hook if verbose else None,
    )

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

    return {"DID": did_all, "FDID": fdid_result}
