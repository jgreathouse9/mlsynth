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

from typing import Any, Dict, List, Tuple

import numpy as np

from .inference import did_inference


def _r2_batch(y_c: np.ndarray, ss_tot: float, X_pre: np.ndarray) -> np.ndarray:
    """Pre-treatment R^2 of each candidate donor average vs the treated unit.

    Parameters
    ----------
    y_c : np.ndarray
        Centred treated pre-treatment vector (``y - mean(y)``).
    ss_tot : float
        Total sum of squares of ``y_c``.
    X_pre : np.ndarray
        Candidate pre-treatment vectors, shape ``(T0, N)``.

    Returns
    -------
    np.ndarray
        R^2 for each candidate, shape ``(N,)``.
    """
    X_mean = X_pre.mean(axis=0)
    ss_X = np.sum((X_pre - X_mean) ** 2, axis=0)
    cross = np.dot(y_c, X_pre - X_mean)
    ss_res = ss_tot + ss_X - 2.0 * cross
    return 1.0 - ss_res / ss_tot


def did_from_mean(
    treated: np.ndarray, mean_ctrl: np.ndarray, pre_periods: int
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

    se, ci, pval, satt = did_inference(att, resid_pre, T0, T1)
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


def _select_best_donor(
    X_pre: np.ndarray,
    current_mean_pre: np.ndarray,
    k: int,
    remaining_idx: np.ndarray,
    y_c: np.ndarray,
    ss_tot: float,
) -> Tuple[int, float, np.ndarray]:
    """Pick the remaining donor whose addition maximises pre-period R^2."""
    candidates = X_pre[:, remaining_idx]
    new_means = (current_mean_pre[:, None] * k + candidates) / (k + 1)
    r2_cand = _r2_batch(y_c, ss_tot, new_means)
    best_pos = int(np.nanargmax(r2_cand))
    best_idx = int(remaining_idx[best_pos])
    best_r2 = float(r2_cand[best_pos])
    return best_idx, best_r2, r2_cand


def _update_synthetic_control(
    current_mean: np.ndarray, control_outcomes: np.ndarray, best_idx: int, k: int
) -> np.ndarray:
    """Incrementally fold a newly selected donor into the running average."""
    diff = control_outcomes[:, best_idx] - current_mean
    return current_mean + diff / (k + 1)


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
) -> Dict[str, Any]:
    """Difference-in-differences result for the selected donor subset."""
    optimal_mean = control_outcomes[:, optimal_idxs].mean(axis=1)
    result = did_from_mean(treated_outcome, optimal_mean, pre_periods)
    result.update(
        {
            "R2_at_each_step": R2_path,
            "selected_controls": optimal_idxs,
            "selected_names": [donor_names[i] for i in optimal_idxs],
        }
    )
    return result


def forward_did_select(
    treated_outcome: np.ndarray,
    control_outcomes: np.ndarray,
    pre_periods: int,
    donor_names: List[Any],
    verbose: bool = False,
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
    did_all = did_from_mean(treated_outcome, mean_all, T0)

    current_mean = np.zeros(T, dtype=float)
    remaining_mask = np.ones(N, dtype=bool)
    selected: List[int] = []
    R2_path = np.empty(N, dtype=float)
    intermediary_results = [] if verbose else None

    for it in range(N):
        k = len(selected)
        remaining_idx = np.where(remaining_mask)[0]
        if len(remaining_idx) == 0:
            break

        best_idx, best_r2, r2_cand = _select_best_donor(
            X_pre=X_pre,
            current_mean_pre=current_mean[:T0],
            k=k,
            remaining_idx=remaining_idx,
            y_c=y_c,
            ss_tot=ss_tot,
        )

        current_mean = _update_synthetic_control(
            current_mean=current_mean,
            control_outcomes=control_outcomes,
            best_idx=best_idx,
            k=k,
        )

        selected.append(best_idx)
        R2_path[it] = best_r2
        remaining_mask[best_idx] = False

        if verbose:
            _record_verbose_step(
                intermediary_results=intermediary_results,
                it=it,
                best_idx=best_idx,
                best_r2=best_r2,
                r2_cand=r2_cand,
                selected=selected,
                donor_names=donor_names,
                current_mean_pre=current_mean[:T0],
                k=k,
            )

    optimal_idxs, R2_path = _choose_optimal_subset(selected, R2_path)

    fdid_result = _compute_fdid_result(
        treated_outcome=treated_outcome,
        control_outcomes=control_outcomes,
        optimal_idxs=optimal_idxs,
        pre_periods=T0,
        R2_path=R2_path,
        donor_names=donor_names,
    )

    if verbose:
        fdid_result["intermediary"] = intermediary_results

    return {"DID": did_all, "FDID": fdid_result}
