"""Rolling-origin cross-validation to tune ``(m, phi)`` for MASC.

Direct port of ``cv_masc`` from the R reference (lines 214-332 of
``masc/R/crossvalidation.R``).

The CV grid loops over candidate ``m`` (one CV pass per ``m``); for
each ``m``, the analytic ``phi`` from Kellogg et al. (2021) eq. (15)
gives the CV-optimal weighting between match and SC at that ``m`` in
closed form. The chosen ``(m_hat, phi_hat)`` minimises the resulting
CV criterion across the grid.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from .estimation import (
    _standardize_predictors,
    analytic_phi,
    nearest_neighbor_weights,
    sc_simplex_weights,
)


def _aggregate_covariates(
    cov_treated_panel: Optional[np.ndarray],
    cov_donors_panel: Optional[np.ndarray],
    covariate_names: Sequence[Any],
    time_index: Optional[np.ndarray],
    pre_end_period: int,
    covariate_windows: Optional[Dict[Any, Tuple[Any, Any]]] = None,
) -> "tuple[Optional[np.ndarray], Optional[np.ndarray]]":
    """Average each covariate over a pre-window (per-fold capable).

    For each covariate ``c``:

    * If ``covariate_windows[c]`` is ``(start, end)``, average over
      ``time_index`` rows where ``start <= t <= end``.
    * Otherwise average over the first ``pre_end_period`` rows (i.e.
      ``time_index[0..pre_end_period - 1]``).

    Mirrors the R reference's per-fold aggregation in ``masc/R/
    estimator.R`` lines 25-29.
    """
    if cov_treated_panel is None or cov_donors_panel is None:
        return None, None

    P = cov_treated_panel.shape[1]
    X_treated = np.zeros(P)
    X_donors = np.zeros((P, cov_donors_panel.shape[1]))
    for p in range(P):
        cname = covariate_names[p] if p < len(covariate_names) else None
        win = (covariate_windows or {}).get(cname)
        if win is not None and time_index is not None:
            start, end = win
            mask = (time_index >= start) & (time_index <= end)
        else:
            mask = np.zeros(cov_treated_panel.shape[0], dtype=bool)
            mask[:pre_end_period] = True
        if not mask.any():
            raise ValueError(
                f"Covariate window for {cname!r} selects no periods."
            )
        X_treated[p] = float(cov_treated_panel[mask, p].mean())
        X_donors[p] = cov_donors_panel[mask, :, p].mean(axis=0)
    return X_treated, X_donors


def _build_sc_cache(
    *,
    Y_treated: np.ndarray,
    Y_donors: np.ndarray,
    treatment_period: int,
    set_f: Sequence[int],
    fold_weights: np.ndarray,
    forecast_minlength: int,
    forecast_maxlength: int,
    solver: Optional[str],
    sc_backend: str = "mscmt",
    match_on: str = "outcomes",
    cov_treated_panel: Optional[np.ndarray] = None,
    cov_donors_panel: Optional[np.ndarray] = None,
    covariate_names: Sequence[Any] = (),
    time_index: Optional[np.ndarray] = None,
    covariate_windows: Optional[Dict[Any, Tuple[Any, Any]]] = None,
) -> "list":
    """Pre-compute per-fold SC weights and forecast vectors.

    Returns a list of ``(Y0_pre, YJ_pre, Y0_post, YJ_post, Y_sc_post, obj_w)``
    tuples, one per *usable* fold (folds whose forecast window is empty are
    dropped).  Running this once -- rather than once per ``m`` -- is the
    key CV speedup when ``cov_*_panel`` is set and ``sc_simplex_weights``
    delegates to the bilevel V solver.
    """
    cache = []
    for fold_idx, f in enumerate(set_f):
        post_start = f + forecast_minlength
        post_end = min(f + forecast_maxlength, treatment_period - 1)
        if post_start > post_end:
            continue
        idx_pre = slice(0, f)
        idx_post = slice(post_start - 1, post_end)
        Y0_pre = Y_treated[idx_pre]
        YJ_pre = Y_donors[idx_pre]
        X_treated, X_donors = _aggregate_covariates(
            cov_treated_panel, cov_donors_panel, covariate_names,
            time_index, pre_end_period=f,
            covariate_windows=covariate_windows,
        )
        w_sc = sc_simplex_weights(
            Y0_pre, YJ_pre,
            X_treated=X_treated, X_donors=X_donors,
            solver=solver, sc_backend=sc_backend,
        )
        # Matching features: the outcome path (default) or the
        # row-standardised covariate block (the reference's solve.covmatch).
        if match_on == "covariates":
            if X_treated is None:
                raise ValueError(
                    "match_on='covariates' requires covariates."
                )
            M0, MJ = _standardize_predictors(X_treated, X_donors)
        else:
            M0, MJ = Y0_pre, YJ_pre
        Y0_post = Y_treated[idx_post]
        YJ_post = Y_donors[idx_post]
        Y_sc_fold = YJ_post @ w_sc
        n_per = Y0_post.shape[0]
        obj_w = np.full(n_per, fold_weights[fold_idx] / n_per)
        cache.append((M0, MJ, Y0_post, YJ_post, Y_sc_fold, obj_w))
    return cache


def _cv_for_m(
    Y_treated: np.ndarray,
    Y_donors: np.ndarray,
    treatment_period: int,
    m: int,
    set_f: Sequence[int],
    fold_weights: np.ndarray,
    forecast_minlength: int,
    forecast_maxlength: int,
    solver: Optional[str],
    sc_cache: Sequence[
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ],
) -> Tuple[float, float, np.ndarray]:
    """Run rolling-origin CV at a single ``m`` and return ``(cv, phi, by_fold)``.

    For each fold the cached tuple is
    ``(Y0_pre, YJ_pre, Y0_post, YJ_post, Y_sc_post, obj_w)``: the pre-period
    blocks the match needs, the post-period blocks the forecast needs, the
    fold's SC forecast vector (independent of ``m``) and its objective
    weights. Pre-computing this once per fold in ``cross_validate`` is the
    whole point of the cache.
    """
    Y_sc_stack = []
    Y_match_stack = []
    Y_treated_stack = []
    obj_weight_stack = []
    fold_records = []

    for (Y0_pre, YJ_pre, Y0_post, YJ_post, Y_sc_fold, obj_w) in sc_cache:
        w_match = nearest_neighbor_weights(Y0_pre, YJ_pre, m)
        Y_match_fold = YJ_post @ w_match

        Y_sc_stack.append(Y_sc_fold)
        Y_match_stack.append(Y_match_fold)
        Y_treated_stack.append(Y0_post)
        obj_weight_stack.append(obj_w)
        fold_records.append((Y0_post, Y_match_fold, Y_sc_fold))

    Y_sc_v = np.concatenate(Y_sc_stack)
    Y_match_v = np.concatenate(Y_match_stack)
    Y_treated_v = np.concatenate(Y_treated_stack)
    obj_w_v = np.concatenate(obj_weight_stack)

    phi = analytic_phi(Y_treated_v, Y_match_v, Y_sc_v, obj_w_v)

    cv_error = float(
        np.sum(
            obj_w_v
            * (Y_treated_v - phi * Y_match_v - (1.0 - phi) * Y_sc_v) ** 2
        )
    )
    by_fold = np.array([
        float(
            np.mean(
                (yt - phi * ym - (1.0 - phi) * ysc) ** 2
            )
        )
        for (yt, ym, ysc) in fold_records
    ])
    return cv_error, phi, by_fold


def cross_validate(
    Y_treated: np.ndarray,
    Y_donors: np.ndarray,
    treatment_period: int,
    *,
    m_grid: Optional[Sequence[int]] = None,
    min_preperiods: Optional[int] = None,
    set_f: Optional[Sequence[int]] = None,
    fold_weights: Optional[np.ndarray] = None,
    forecast_minlength: int = 1,
    forecast_maxlength: int = 1,
    solver: Optional[str] = None,
    sc_backend: str = "mscmt",
    match_on: str = "outcomes",
    cov_treated_panel: Optional[np.ndarray] = None,
    cov_donors_panel: Optional[np.ndarray] = None,
    covariate_names: Sequence[Any] = (),
    time_index: Optional[np.ndarray] = None,
    covariate_windows: Optional[Dict[Any, Tuple[Any, Any]]] = None,
) -> Tuple[int, float, float, np.ndarray, np.ndarray]:
    """Pick ``(m_hat, phi_hat)`` by rolling-origin CV across the grid.

    Returns
    -------
    m_hat : int
    phi_hat : float
    cv_error_min : float
    cv_grid : np.ndarray
        Shape ``(len(m_grid), 3)`` with columns ``[m, phi, cv_error]``.
    by_fold_at_min : np.ndarray
        CV error by fold at ``(m_hat, phi_hat)``.
    """
    J = Y_donors.shape[1]
    if m_grid is None:
        m_grid = list(range(1, J + 1))
    if min_preperiods is None and set_f is None:
        min_preperiods = max(int(np.ceil(treatment_period / 2)), 1)
    if set_f is None:
        set_f = list(range(int(min_preperiods), treatment_period - 1))
    set_f = list(set_f)
    if len(set_f) == 0:
        raise ValueError(
            "MASC CV: empty fold set. Increase pre-period length or "
            "lower min_preperiods."
        )
    if fold_weights is None:
        fold_weights = np.full(len(set_f), 1.0 / len(set_f))
    else:
        fold_weights = np.asarray(fold_weights, dtype=float)
        if fold_weights.shape[0] != len(set_f):
            raise ValueError(
                "fold_weights length must match set_f length."
            )
        fold_weights = fold_weights / fold_weights.sum()

    # Pre-build the per-fold SC cache once. The SC fit (and per-fold
    # covariate aggregation) depends only on the fold cut ``f``, not on
    # ``m``; computing it inside the ``m_grid`` loop was an O(|m_grid|)
    # waste — the dominant cost when covariates trigger the bilevel V
    # solver.
    sc_cache = _build_sc_cache(
        Y_treated=Y_treated,
        Y_donors=Y_donors,
        treatment_period=treatment_period,
        set_f=set_f,
        fold_weights=fold_weights,
        forecast_minlength=forecast_minlength,
        forecast_maxlength=forecast_maxlength,
        solver=solver,
        sc_backend=sc_backend,
        match_on=match_on,
        cov_treated_panel=cov_treated_panel,
        cov_donors_panel=cov_donors_panel,
        covariate_names=covariate_names,
        time_index=time_index,
        covariate_windows=covariate_windows,
    )

    cv_grid = np.zeros((len(m_grid), 3))
    by_fold_per_m = []
    for idx, m in enumerate(m_grid):
        cv, phi, by_fold = _cv_for_m(
            Y_treated, Y_donors, treatment_period, m, set_f,
            fold_weights, forecast_minlength, forecast_maxlength, solver,
            sc_cache=sc_cache,
        )
        cv_grid[idx] = [m, phi, cv]
        by_fold_per_m.append(by_fold)

    best_idx = int(np.argmin(cv_grid[:, 2]))
    return (
        int(cv_grid[best_idx, 0]),
        float(cv_grid[best_idx, 1]),
        float(cv_grid[best_idx, 2]),
        cv_grid,
        by_fold_per_m[best_idx],
    )
