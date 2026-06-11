"""NumPy-first FSCM engine: forward selection + rolling-origin CV.

Implements Cerulli (2024) with two matching modes:

* **Trajectory mode** (no covariates and no special predictors): the SCM
  weights match the treated unit's pre-treatment *outcome trajectory*.
* **Predictor mode** (covariates and/or ``match_periods`` given): the weights
  match Abadie's *predictor* specification, with the predictor-weight matrix
  ``V`` and donor weights jointly determined by the **bilevel optimization**
  of Malo et al. (2024) -- ``V`` is solved once on the full donor pool and then
  reused through forward selection (see :mod:`.bilevel`).

In both modes the donor pool is grown greedily and the donor *count* is chosen
by rolling-origin cross-validation; the final weights are refit on the full
pre-period over the selected donors. All optimization is self-contained -- the
simplex problems are solved by the FISTA primitive in :mod:`.bilevel.simplex`,
not by ``Opt.SCopt``.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np

from ...config_models import (
    EffectsResults,
    FitDiagnosticsResults,
    MethodDetailsResults,
    TimeSeriesResults,
    WeightsResults,
)
from ..bilevel import BilevelProblem, lower_level_weights, simplex_lstsq, solve_bilevel
from .structures import FSCMInputs, FSCMResults, FSCMSelectionPath

_EPS = 1e-12
_LOWER_EPS = 1e-6   # non-Archimedean tie-break for the lower-level problem


# --------------------------------------------------------------------------- #
# Predictor block (predictor mode)
# --------------------------------------------------------------------------- #
def _predictor_block(inputs: FSCMInputs) -> Tuple[np.ndarray, np.ndarray]:
    """Standardized ADH predictor block over the full donor pool.

    Returns ``Pt`` (P,) for the treated unit and ``Pd`` (P, N) for all donors,
    z-scored row-wise across donors so the optimized ``V`` is scale-free.
    """
    rows_t, rows_d = [], []
    if inputs.has_covariates:
        rows_t.append(inputs.cov_treated)            # (P_c,)
        rows_d.append(inputs.cov_donors.T)           # (P_c, N)
    if inputs.has_match_periods:
        rows_t.append(inputs.y[inputs.match_idx])    # (P_l,)
        rows_d.append(inputs.Y[inputs.match_idx, :]) # (P_l, N)
    Praw_t = np.concatenate(rows_t)
    Praw_d = np.vstack(rows_d)
    center = Praw_d.mean(axis=1)
    scale = Praw_d.std(axis=1) + _EPS
    Pt = (Praw_t - center) / scale
    Pd = (Praw_d - center[:, None]) / scale[:, None]
    return Pt, Pd


def _predictor_names(inputs: FSCMInputs) -> List[Any]:
    return list(inputs.covariate_names) + [f"y[{p}]" for p in inputs.match_periods]


# --------------------------------------------------------------------------- #
# Weight fitting and scoring
# --------------------------------------------------------------------------- #
def _fit_weights(
    inputs: FSCMInputs,
    idx: List[int],
    fit_slice: slice,
    *,
    Pt: Optional[np.ndarray],
    Pd: Optional[np.ndarray],
    v: Optional[np.ndarray],
) -> np.ndarray:
    """Solve donor weights for subset ``idx``.

    Predictor mode solves the bilevel lower-level problem for the fixed global
    ``V`` over the full pre-period; trajectory mode matches the outcome over
    ``fit_slice``. Both use the self-contained FISTA simplex solver.
    """
    if Pt is not None:
        subprob = BilevelProblem(
            y1_pre=inputs.y[: inputs.T0],
            Y0_pre=inputs.Y[: inputs.T0][:, idx],
            X1=Pt, X0=Pd[:, idx],
        )
        return lower_level_weights(subprob, v, _LOWER_EPS)
    return simplex_lstsq(inputs.Y[fit_slice][:, idx], inputs.y[fit_slice])


def _outcome_rmspe(
    inputs: FSCMInputs, idx: List[int], weights: np.ndarray, eval_slice: slice
) -> float:
    """RMSPE of the synthetic outcome over ``eval_slice``."""
    resid = inputs.y[eval_slice] - inputs.Y[eval_slice][:, idx] @ weights
    return float(np.sqrt(np.mean(resid ** 2)))


def _rolling_origin_rmspe(
    inputs: FSCMInputs,
    idx: List[int],
    origins: np.ndarray,
    *,
    Pt: Optional[np.ndarray],
    Pd: Optional[np.ndarray],
    v: Optional[np.ndarray],
) -> float:
    """Expanding-window, one-step-ahead forecast RMSPE over ``origins``.

    In predictor mode the predictor block is fixed, so the weights do not
    depend on the origin and are reused; in trajectory mode the weights are
    refit on ``[0, t)`` at each origin ``t``.
    """
    if Pt is not None:
        w = _fit_weights(inputs, idx, slice(0, inputs.T0), Pt=Pt, Pd=Pd, v=v)
        errs = [(inputs.y[t] - inputs.Y[t, idx] @ w) ** 2 for t in origins]
        return float(np.sqrt(np.mean(errs)))
    errs = []
    for t in origins:
        w = simplex_lstsq(inputs.Y[:t][:, idx], inputs.y[:t])
        errs.append((inputs.y[t] - inputs.Y[t, idx] @ w) ** 2)
    return float(np.sqrt(np.mean(errs)))


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _forward_select(
    inputs: FSCMInputs,
    origins: np.ndarray,
    cap: int,
    *,
    Pt, Pd, v,
) -> Tuple[List[int], FSCMSelectionPath]:
    """Greedy forward selection scored on in-sample fit, sized by rolling CV."""
    full_pre = slice(0, inputs.T0)
    remaining = list(range(inputs.n_donors))
    selected: List[int] = []
    order: List[Any] = []
    train_rmspe: List[float] = []
    test_rmspe: List[float] = []

    for _ in range(cap):
        best_j, best_score, best_idx_list = None, np.inf, None
        for j in remaining:
            cand = selected + [j]
            w = _fit_weights(inputs, cand, full_pre, Pt=Pt, Pd=Pd, v=v)
            score = _outcome_rmspe(inputs, cand, w, full_pre)
            if score < best_score:
                best_j, best_score, best_idx_list = j, score, cand
        selected.append(best_j)
        remaining.remove(best_j)
        order.append(inputs.donor_labels[best_j])
        train_rmspe.append(best_score)
        test_rmspe.append(
            _rolling_origin_rmspe(inputs, best_idx_list, origins, Pt=Pt, Pd=Pd, v=v)
        )

    test_arr = np.asarray(test_rmspe)
    optimal_size = int(np.argmin(test_arr)) + 1
    path = FSCMSelectionPath(
        sizes=np.arange(1, cap + 1),
        order=order,
        train_rmspe=np.asarray(train_rmspe),
        test_rmspe=test_arr,
        optimal_size=optimal_size,
    )
    return selected[:optimal_size], path


def run_fscm(
    inputs: FSCMInputs,
    *,
    forward_selection: bool = True,
    cv_split: float = 0.5,
    max_donors: Optional[int] = None,
) -> FSCMResults:
    """Synthetic control via the Malo et al. (2024) bilevel solver.

    Parameters
    ----------
    inputs : FSCMInputs
        Prepared NumPy panel.
    forward_selection : bool, default True
        If True, run Cerulli's forward selection with rolling-origin OOS
        validation to choose a donor subset (each candidate fit by the bilevel
        solver). If False, take the full bilevel solve over all donors with no
        selection.
    cv_split : float, default 0.5
        Sets the first rolling origin (forward selection only): forecasting
        begins at period ``round(T0 * cv_split)`` and sweeps to the pre-period
        end.
    max_donors : int, optional
        Cap on the number of forward-selection steps (default: all donors).
    """
    T0, N = inputs.T0, inputs.n_donors
    full_pre = slice(0, T0)

    start = int(round(T0 * cv_split))
    start = min(max(start, 2), T0 - 1)
    origins = np.arange(start, T0)

    # Predictor mode: solve the bilevel problem once on the full pool for V.
    bilevel_sol = None
    if inputs.has_predictors:
        Pt, Pd = _predictor_block(inputs)
        prob = BilevelProblem(
            y1_pre=inputs.y[:T0], Y0_pre=inputs.Y[:T0],
            X1=Pt, X0=Pd, predictor_names=_predictor_names(inputs),
        )
        bilevel_sol = solve_bilevel(prob)
        v = bilevel_sol.V
    else:
        Pt = Pd = v = None

    if forward_selection:
        cap = N if max_donors is None else min(int(max_donors), N)
        sel_idx, path = _forward_select(inputs, origins, cap, Pt=Pt, Pd=Pd, v=v)
        w_sel = _fit_weights(inputs, sel_idx, full_pre, Pt=Pt, Pd=Pd, v=v)
        active_idx, active_w = sel_idx, w_sel
    else:
        # Full bilevel solve over all donors; report the weight-bearing ones.
        path = None
        all_idx = list(range(N))
        w_full = _fit_weights(inputs, all_idx, full_pre, Pt=Pt, Pd=Pd, v=v)
        sel_idx, w_sel = all_idx, w_full
        nz = np.where(w_full > 1e-4)[0]
        active_idx = nz.tolist()
        active_w = w_full[nz]

    counterfactual = inputs.Y[:, sel_idx] @ w_sel
    gap = inputs.y - counterfactual
    att = float(np.mean(gap[T0:])) if inputs.T2 > 0 else float("nan")

    donor_weights = {label: 0.0 for label in inputs.donor_labels}
    for i, w in zip(sel_idx, w_sel):
        donor_weights[inputs.donor_labels[i]] = float(w)

    pre_resid = gap[:T0]
    ss_tot = np.sum((inputs.y[:T0] - inputs.y[:T0].mean()) ** 2)
    diagnostics = {
        "pre_rmse": float(np.sqrt(np.mean(pre_resid ** 2))),
        "pre_r_squared": float(1.0 - np.sum(pre_resid ** 2) / (ss_tot + _EPS)),
        "n_donors_available": int(N),
    }
    if path is not None:
        diagnostics["cv_rmspe_at_optimum"] = float(path.test_rmspe[path.optimal_size - 1])
        diagnostics["cv_rmspe_full_pool"] = float(path.test_rmspe[-1])
        diagnostics["n_cv_origins"] = int(origins.size)

    metadata = {
        "forward_selection": forward_selection,
        "matching_mode": "predictor" if inputs.has_predictors else "trajectory",
        "solver": "bilevel" if inputs.has_predictors else "simplex_lstsq",
        "covariates": list(inputs.covariate_names),
        "match_periods": list(inputs.match_periods),
    }
    if forward_selection:
        metadata["cv_method"] = "rolling_origin"
        metadata["cv_origins"] = origins.tolist()
    if bilevel_sol is not None:
        metadata["V_weights"] = dict(zip(_predictor_names(inputs), np.round(v, 4).tolist()))
        metadata["bilevel_stage"] = bilevel_sol.stage
        diagnostics["bilevel_upper_loss"] = bilevel_sol.upper_loss
        diagnostics["bilevel_lower_bound"] = bilevel_sol.lower_bound
        diagnostics["bilevel_gap"] = bilevel_sol.gap

    labels = np.asarray(inputs.time_index.labels)
    T = inputs.T
    return FSCMResults(
        inputs=inputs,
        selected_donors=[inputs.donor_labels[i] for i in active_idx],
        weights_vector=np.asarray(active_w),
        selection_path=path,
        diagnostics=diagnostics,
        metadata=metadata,
        effects=EffectsResults(att=att),
        time_series=TimeSeriesResults(
            observed_outcome=np.asarray(inputs.y, dtype=float),
            counterfactual_outcome=np.asarray(counterfactual, dtype=float),
            estimated_gap=np.asarray(gap, dtype=float),
            time_periods=labels,
            intervention_time=(labels[T0] if T0 < T else None),
        ),
        weights=WeightsResults(
            donor_weights={str(k): float(v) for k, v in donor_weights.items()},
            summary_stats={"constraint": "simplex (non-negative, sum to 1)"},
        ),
        fit_diagnostics=FitDiagnosticsResults(
            rmse_pre=diagnostics["pre_rmse"],
            r_squared_pre=diagnostics["pre_r_squared"],
            additional_metrics=diagnostics,
        ),
        method_details=MethodDetailsResults(method_name="FSCM", is_recommended=True),
    )
