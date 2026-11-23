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
    ss_tot = np.sum(_centred(treated_pre) ** 2)
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


_U = 2.0 ** -53                       # the unit roundoff of IEEE double precision


def _gamma(k: int) -> float:
    """gamma_k = k u / (1 - k u), inflated past its own rounding (Rump, "Verification methods: rigorous results using
    floating-point arithmetic", Acta Numerica 19 (2010) 287-449, Theorem 2.1)."""
    return k * _U / (1.0 - k * _U) * (1.0 + 4.0 * _U)


def _ss_bounds(ss, e_r, rbar, g_s, g_mu):
    """Rigorous lower and upper bounds on the exact residual sums of squares behind the computed ``ss``.

    ``e_r`` bounds the 2-norm of each residual's error before its level is removed: the error of a sum of k terms,
    of the scaling by 1 / (k + 1) and of the centred data, at most gamma_{k+8} (||y_c|| + c (sum ||x_p|| + ||x_j||))
    (Rump 2010, Theorem 2.1). Removing the level is an orthogonal projection, which does not lengthen that error; the
    computed mean and the sum of squares carry gamma_{T0+2} and gamma_{T0+6}. ``rbar`` bounds each computed residual's
    norm. The bounds assume no underflow, the theorem's condition."""
    q_up = np.sqrt(ss / (1.0 - g_s))
    q_lo = np.sqrt(np.maximum(ss / (1.0 + g_s) - (g_mu * rbar) ** 2, 0.0))
    lo = np.maximum(q_lo - e_r, 0.0) ** 2 * (1.0 - 8.0 * _U)
    up = (q_up + e_r) ** 2 * (1.0 + 8.0 * _U)
    return lo, up


class _ExactPool:
    """The pre-period residual sum of squares of a donor set in exact rational arithmetic on the panel's doubles, for
    the few candidates whose rounding bounds overlap. Converted to rationals on first use."""

    def __init__(self, treated_pre: np.ndarray, X_pre: np.ndarray):
        self._y, self._X, self._cols = treated_pre, X_pre, None

    def ss(self, cols: List[int]):
        from fractions import Fraction
        if self._cols is None:
            T0 = self._y.size
            yv = [Fraction(float(v)) for v in self._y]
            ym = sum(yv) / T0
            self._yc = [v - ym for v in yv]
            self._cols = [[Fraction(float(self._X[t, j])) for t in range(T0)] for j in range(self._X.shape[1])]
        T0 = len(self._yc)
        m = [sum(self._cols[j][t] for j in cols) / len(cols) for t in range(T0)]
        mm = sum(m) / T0
        return sum((self._yc[t] - (m[t] - mm)) ** 2 for t in range(T0))

    def argmin(self, sets: List[List[int]], labels) -> Tuple[int, float]:
        """The label of the set of least exact residual, the first on ties, and that residual as a double."""
        vals = [self.ss(cols) for cols in sets]
        i = min(range(len(vals)), key=lambda q: (vals[q], q))
        return int(labels[i]), float(vals[i])


def _centred(v: np.ndarray) -> np.ndarray:
    """``v`` less its mean, by the corrected two-pass algorithm: the mean subtracted, then the mean of what is left.
    A sum of squares about a computed mean exceeds the exact one by T times the mean's error squared, about 1e-6
    relative at levels of 1e9 over noise of 1e-4; the second pass makes that error second order (Chan, Golub and
    LeVeque, "Algorithms for computing the sample variance", Amer. Statist. 37 (1983) 241-247)."""
    c = v - v.mean()
    return c - c.mean()


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
    selected: List[int], R2_path: np.ndarray, path_lo: Optional[np.ndarray] = None,
    path_up: Optional[np.ndarray] = None, exact: Optional["_ExactPool"] = None,
) -> Tuple[List[int], np.ndarray]:
    """Keep the donor prefix up to (and including) the R^2-maximising step, the first on ties.

    With ``path_lo`` and ``path_up``, rigorous bounds on each step's residual sum of squares, and ``exact``, the steps
    whose bounds overlap the best one are decided in exact arithmetic, so the prefix kept is the one exact arithmetic
    keeps."""
    if len(selected) == 0:
        return [], []
    best_iter = int(np.argmax(R2_path))
    if path_lo is not None:
        tied = np.flatnonzero(path_lo <= path_up[best_iter])
        if tied.size > 1:
            best_iter, _ = exact.argmin([selected[:i + 1] for i in tied], tied)
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
    ss_tot = float(np.sum(_centred(treated[:T0]) ** 2))
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

    Each step's residual sum of squares is formed directly from the centred
    donors, not expanded into ss_tot + ss_X - 2 cross (CLAUDE.md invariant
    9), and carries rigorous rounding bounds (:func:`_ss_bounds`). Candidates
    whose bounds overlap the leader's are decided in exact arithmetic
    (:class:`_ExactPool`), ties to the first index as Li's Fun_FDID.R breaks
    them with which.max, so every pick and the number of donors kept are those
    of exact arithmetic on the panel's doubles. A donor with a non-finite
    pre-period is never a candidate; a non-finite treated pre-period raises.

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

    y_c = _centred(treated_pre)
    ss_tot = np.sum(y_c ** 2)
    if ss_tot <= 1e-12:
        ss_tot = 1e-12

    X_pre = control_outcomes[:T0]
    # A donor with a non-finite pre-period has no comparable fit and is never a candidate, as the search always left
    # it; a non-finite treated pre-period leaves no donor comparable, and raises as the search always did.
    finite_donor = np.isfinite(X_pre).all(axis=0)
    if not (np.isfinite(treated_pre).all() and finite_donor.any()):
        raise ValueError("All-NaN slice encountered: the pre-period holds non-finite outcomes, so no donor's fit "
                         "can be compared.")
    n_steps = int(finite_donor.sum())

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

    # --- the forward search (module docstring of _ExactPool) ---
    # Adding donor j to k already-selected donors gives the candidate average (S + x_j) / (k + 1), and the step's
    # cost is the residual sum of squares of the DID fit, ||P(y - (S + x_j) / (k + 1))||^2 with P the centring
    # projection. It is formed directly from the centred data: the residual matrix of every candidate, its level
    # removed again, its squares summed. That is O(N T0) per step, the order of the one matrix-vector product the
    # expanded form ss_tot + ss_X - 2 cross needed, without its cancellation (CLAUDE.md invariant 9).
    x_c = np.where(finite_donor, X_pre - X_pre.mean(axis=0), 0.0)   # (T0, N) centred donors: the two-pass algorithm
    nx = np.sqrt(np.einsum("tj,tj->j", x_c, x_c)) * (1.0 + _gamma(T0 + 3))
    ny = float(np.sqrt(y_c @ y_c)) * (1.0 + _gamma(T0 + 3))
    g_s, g_mu = _gamma(T0 + 6), _gamma(T0 + 2)
    exact = _ExactPool(treated_pre, X_pre)

    S_c = np.zeros(T0, dtype=float)               # centred running sum of selected
    s_norm = 0.0                                  # sum of the selected donors' norm bounds
    run_sum_pre = np.zeros(T0, dtype=float)       # uncentred running sum (verbose)

    selected: List[int] = []
    path_ss = np.empty(n_steps, dtype=float)
    path_lo = np.empty(n_steps, dtype=float)
    path_up = np.empty(n_steps, dtype=float)
    remaining_mask = finite_donor.copy()
    intermediary_results = [] if verbose else None

    for it in range(n_steps):
        k = len(selected)
        c = 1.0 / (k + 1)
        E = (y_c - c * S_c)[:, None] - c * x_c      # (T0, N) every candidate's residual
        E -= E.mean(axis=0)                         # the fitted level removed again
        ss = np.einsum("tj,tj->j", E, E)
        lo, up = _ss_bounds(ss, _gamma(k + 8) * (ny + c * (s_norm + nx)), ny + c * (s_norm + nx), g_s, g_mu)
        ss[~remaining_mask] = np.inf
        lo[~remaining_mask] = np.inf
        best_idx = int(np.argmin(ss))               # first-tie argmin = first-tie argmax of R^2
        tied = np.flatnonzero(lo <= up[best_idx])
        if tied.size > 1:
            # candidates the rounding cannot separate: decided in exact arithmetic, ties to the first index
            best_idx, exact_ss = exact.argmin([selected + [int(j)] for j in tied], tied)
            lo[best_idx] = up[best_idx] = ss[best_idx] = exact_ss
        r2_all = 1.0 - ss / ss_tot
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
        path_ss[it], path_lo[it], path_up[it] = ss[best_idx], lo[best_idx], up[best_idx]
        remaining_mask[best_idx] = False
        S_c = S_c + x_c[:, best_idx]
        s_norm += float(nx[best_idx])
        run_sum_pre = run_sum_pre + X_pre[:, best_idx]

    R2_path = 1.0 - path_ss / ss_tot
    optimal_idxs, R2_path = _choose_optimal_subset(selected, R2_path, path_lo, path_up, exact)

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
