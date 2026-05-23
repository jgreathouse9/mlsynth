"""Orchestration for the MC-NNM estimator (Athey et al. 2021)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .completion import mcnnm_cv, mcnnm_fit
from .structures import MCNNMInference, MCNNMInputs, MCNNMResults

_EPS = 1e-12


def _att_from_fit(Y, D, completed, T0, time_labels):
    treated = D > 0
    effects = np.full_like(Y, np.nan)
    effects[treated] = Y[treated] - completed[treated]
    att = float(np.nanmean(effects[treated])) if treated.any() else np.nan
    att_by_period = {}
    for t in range(T0, Y.shape[1]):
        col = treated[:, t]
        if col.any():
            att_by_period[time_labels[t]] = float(np.nanmean(effects[col, t]))
    return att, effects, att_by_period


def run_mcnnm(
    inputs: MCNNMInputs,
    *,
    est_u: bool = True,
    est_v: bool = True,
    n_lam: int = 40,
    n_folds: int = 5,
    max_iter: int = 400,
    tol: float = 1e-5,
    inference: bool = False,
    alpha_level: float = 0.05,
    random_state: int = 0,
) -> MCNNMResults:
    """Run MC-NNM (CV over the threshold) and assemble :class:`MCNNMResults`.

    Parameters
    ----------
    inputs : MCNNMInputs
    est_u, est_v : bool
        Estimate unit / time fixed effects (recommended; default True).
    n_lam : int
        Number of candidate thresholds in the CV grid.
    n_folds : int
        Cross-validation folds over observed cells.
    inference : bool
        If True, run a leave-one-control jackknife (at the CV-selected
        threshold) for the ATT SE / CI.
    """
    Y, mask, D, T0 = inputs.Y, inputs.mask, inputs.D, inputs.T0

    fit = mcnnm_cv(Y, mask, est_u=est_u, est_v=est_v, n_lam=n_lam,
                   n_folds=n_folds, max_iter=max_iter, tol=tol,
                   random_state=random_state)
    completed = fit["completed"]
    att, effects, att_by_period = _att_from_fit(
        Y, D, completed, T0, inputs.time_labels
    )
    s = np.linalg.svd(fit["L"], compute_uv=False)
    rank = int((s > 1e-6 * (s[0] if s.size else 1.0)).sum())

    inf = None
    if inference:
        inf = _jackknife(inputs, fit["best_lambda"], est_u, est_v,
                         max_iter, tol, alpha_level)

    metadata = {
        "N": inputs.N, "T": inputs.T, "T0": T0,
        "n_treated": int(inputs.treated_idx.size),
        "n_control": int(inputs.N - inputs.treated_idx.size),
        "estimate_unit_fe": est_u, "estimate_time_fe": est_v,
        "n_missing": int((1.0 - mask).sum()),
    }
    return MCNNMResults(
        inputs=inputs, att=att, counterfactual=completed, effects=effects,
        att_by_period=att_by_period, L=fit["L"], gamma=fit["gamma"],
        delta=fit["delta"], best_lambda=float(fit["best_lambda"]), rank=rank,
        inference=inf, metadata=metadata,
    )


def _jackknife(inputs, thr, est_u, est_v, max_iter, tol, alpha_level):
    """Leave-one-control-out jackknife ATT SE at the fixed CV threshold."""
    Y, mask, D = inputs.Y, inputs.mask, inputs.D
    control_idx = np.array([i for i in range(inputs.N)
                            if i not in set(inputs.treated_idx.tolist())])
    atts = []
    for c in control_idx:
        keep = np.array([i for i in range(inputs.N) if i != c])
        fit = mcnnm_fit(Y[keep], mask[keep], thr, est_u=est_u, est_v=est_v,
                        max_iter=max_iter, tol=tol)
        Dk = D[keep]
        tp = Dk > 0
        if tp.any():
            atts.append(float((Y[keep] - fit["completed"])[tp].mean()))
    atts = np.asarray(atts)
    if atts.size < 2:
        return MCNNMInference("jackknife", float("nan"),
                              (float("nan"), float("nan")),
                              float(alpha_level), int(atts.size))
    q = atts.size
    se = float(np.sqrt((q - 1) / q * np.sum((atts - atts.mean()) ** 2)))
    from scipy.stats import norm
    z = float(norm.ppf(1.0 - alpha_level / 2.0))
    point = float(atts.mean())
    return MCNNMInference("jackknife", se, (point - z * se, point + z * se),
                          float(alpha_level), int(q))
