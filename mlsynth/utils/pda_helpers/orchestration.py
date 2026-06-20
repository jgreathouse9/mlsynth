"""Run the requested PDA variant(s) and assemble typed per-method fits."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .structures import FS, HCW, L2, LASSO, PDAInputs, PDAMethodFit
from .l2 import fit_l2, l2_ate_inference
from .lasso import fit_lasso, lasso_ate_inference, lasso_cv_alpha
from .fs import forward_select, fs_ate_inference
from .hcw import fit_hcw, hcw_ate_inference
from ..inferutils import pda_prediction_intervals

# Map config method strings to internal keys.
_NORMALIZE = {"l2": L2, "L2": L2, "LASSO": LASSO, "lasso": LASSO, "fs": FS,
              "FS": FS, "hcw": HCW, "HCW": HCW}


def _build_refit(method, X, T0, *, tau, l2_standardize, fs_intercept, lasso_alpha,
                 hcw_criterion="AICc", hcw_nvmax=None):
    """A bootstrap refit callback for the engine: ``y_boot -> (cf, support_idx)``.

    Each variant refits on the bootstrap pre-period at *fixed* tuning parameters
    (the L2 penalty ``tau``, the LASSO penalty ``lasso_alpha``; forward selection
    re-runs its deterministic greedy search), per Jiang et al. (2025).
    """
    if method == L2:
        def refit(y_boot):
            beta, _, cf, _ = fit_l2(y_boot, X, T0, tau=tau, standardize=l2_standardize)
            return cf, np.where(np.abs(beta) > 1e-8)[0]
    elif method == LASSO:
        def refit(y_boot):
            beta, _, cf, support = fit_lasso(y_boot, X, T0, alpha=lasso_alpha)
            return cf, np.where(support)[0]
    elif method == FS:
        def refit(y_boot):
            sel_idx, _, _, cf = forward_select(y_boot, X, T0, intercept=fs_intercept)
            return cf, np.asarray(sel_idx, dtype=int)
    elif method == HCW:
        def refit(y_boot):
            sel_idx, _, _, cf = fit_hcw(y_boot, X, T0, criterion=hcw_criterion, nvmax=hcw_nvmax)
            return cf, np.asarray(sel_idx, dtype=int)
    else:  # pragma: no cover - guarded by resolve_methods
        raise ValueError(f"Unknown PDA method: {method!r}")
    return refit


def resolve_methods(method: str, methods: Optional[List[str]]) -> List[str]:
    """Internal method keys to run (explicit ``methods`` win over ``method``)."""
    chosen = methods if methods else [method]
    return [_NORMALIZE.get(m, m) for m in chosen]


def _weights_dict(beta: np.ndarray, labels: np.ndarray) -> Dict[Any, float]:
    return {labels[i]: float(round(beta[i], 4)) for i in range(len(beta)) if abs(beta[i]) > 1e-8}


def run_pda(
    inputs: PDAInputs, methods: List[str], tau: Optional[float], alpha: float,
    fs_intercept: bool = False, lrvar_lag: Optional[int] = None,
    l2_standardize: bool = True, l2_tau_grid: Optional[Sequence[float]] = None,
    hcw_criterion: str = "AICc", hcw_nvmax: Optional[int] = None,
    prediction_intervals: bool = False, pi_n_boot: int = 999,
    pi_seed: Optional[int] = 0,
) -> Dict[str, PDAMethodFit]:
    """Fit each requested PDA variant with its own paper's inference.

    When ``prediction_intervals`` is set, Jiang et al. (2025) bootstrap
    prediction intervals (per-period treatment effect and counterfactual) are
    attached to each variant via :func:`mlsynth.utils.inferutils.pda_prediction_intervals`.
    """
    y, X, T0 = inputs.y, inputs.X, inputs.T0
    labels = inputs.donor_labels
    fits: Dict[str, PDAMethodFit] = {}

    for m in methods:
        meta: Dict[str, Any] = {}
        selected = None
        lasso_alpha = None
        if m == L2:
            beta, intercept, cf, tau_used = fit_l2(
                y, X, T0, tau=tau, standardize=l2_standardize,
                tau_grid=l2_tau_grid)
            att, se, ci, p = l2_ate_inference(y, cf, T0, alpha=alpha)
            meta["tau"] = tau_used
            support_idx = np.where(np.abs(beta) > 1e-8)[0]
        elif m == LASSO:
            beta, intercept, cf, support = fit_lasso(y, X, T0)
            att, se, ci, p = lasso_ate_inference(y, X, cf, support, T0, alpha=alpha)
            support_idx = np.where(support)[0]
            selected = [labels[i] for i in support_idx]
        elif m == FS:
            sel_idx, beta, intercept, cf = forward_select(y, X, T0, intercept=fs_intercept)
            att, se, ci, p = fs_ate_inference(y, cf, T0, alpha=alpha, lrvar_lag=lrvar_lag)
            support_idx = np.asarray(sel_idx, dtype=int)
            selected = [labels[i] for i in sel_idx]
        elif m == HCW:
            sel_idx, beta, intercept, cf = fit_hcw(
                y, X, T0, criterion=hcw_criterion, nvmax=hcw_nvmax)
            att, se, ci, p = hcw_ate_inference(y, cf, T0, alpha=alpha, lrvar_lag=lrvar_lag)
            support_idx = np.asarray(sel_idx, dtype=int)
            selected = [labels[i] for i in sel_idx]
            meta["criterion"] = hcw_criterion
        else:
            raise ValueError(f"Unknown PDA method: {m!r}")

        pis = None
        if prediction_intervals:
            if m == LASSO:
                lasso_alpha = lasso_cv_alpha(y, X, T0)
            refit = _build_refit(
                m, X, T0, tau=meta.get("tau", tau),
                l2_standardize=l2_standardize, fs_intercept=fs_intercept,
                lasso_alpha=lasso_alpha, hcw_criterion=hcw_criterion,
                hcw_nvmax=hcw_nvmax)
            pis = pda_prediction_intervals(
                y, X, T0, counterfactual=cf, support=support_idx, refit=refit,
                alpha=alpha, n_boot=pi_n_boot, seed=pi_seed)

        fits[m] = PDAMethodFit(
            name=m, beta=beta, intercept=intercept, counterfactual=cf,
            gap=y - cf, att=att, att_se=se, ci=ci, p_value=p,
            donor_weights=_weights_dict(beta, labels),
            selected_donors=selected, prediction_intervals=pis, metadata=meta,
        )
    return fits
