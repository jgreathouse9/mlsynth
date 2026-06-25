"""SCTA estimation engine: temporal aggregation, fixed-V solve, frontier.

All functions are pure NumPy. The matching vector stacks the temporal
aggregates (block means of ``K`` consecutive pre-periods) on top of the
disaggregated pre-period outcomes, demeaned by each unit's disaggregated
pre-treatment mean (the Doudchenko-Imbens / fixed-effects intercept shift the
paper adopts). A fixed diagonal ``V = diag(K*nu on aggregate rows, 1 on
disaggregated rows)`` weights the joint fit; the simplex is solved at the
*true* optimum (mlsynth's active-set QP), optionally ridge-augmented.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..bilevel.ridge_augment import ridge_augment_weights, simplex_qp
from ..scmo_helpers.inference import conformal_inference
from .structures import SCTAFit, SCTAInputs


def _block_means(pre: np.ndarray, K: int) -> np.ndarray:
    """Average consecutive blocks of ``K`` pre-periods.

    ``pre`` is ``(T0,)`` or ``(T0, J)``; returns ``(n_blocks,)`` / ``(n_blocks, J)``.
    """
    T0 = pre.shape[0]
    n_blocks = T0 // K
    head = pre[: n_blocks * K]
    if pre.ndim == 1:
        return head.reshape(n_blocks, K).mean(axis=1)
    return head.reshape(n_blocks, K, pre.shape[1]).mean(axis=1)


def _stacked(treated_pre: np.ndarray, donor_pre: np.ndarray, K: int
             ) -> Tuple[np.ndarray, np.ndarray]:
    """Build the stacked matching design ``[aggregate ; disaggregated]``.

    Returns ``A`` ``(n_blocks + T0,)`` and ``B`` ``(n_blocks + T0, J)``.
    """
    a_agg = _block_means(treated_pre, K)               # (n_blocks,)
    b_agg = _block_means(donor_pre, K)                 # (n_blocks, J)
    A = np.concatenate([a_agg, treated_pre])           # (n_blocks + T0,)
    B = np.vstack([b_agg, donor_pre])                  # (n_blocks + T0, J)
    return A, B


def _v_sqrt(n_blocks: int, T0: int, K: int, nu: float) -> np.ndarray:
    """Row scaling sqrt(diag(V)): K*nu on the aggregate rows, 1 elsewhere."""
    return np.sqrt(np.concatenate([
        np.full(n_blocks, K * nu, dtype=float),
        np.full(T0, 1.0, dtype=float),
    ]))


def _weights(A: np.ndarray, B: np.ndarray, sV: np.ndarray,
             augment: Optional[str], ridge_lambda: Optional[float]) -> np.ndarray:
    """Solve the (sqrt-V scaled) simplex SC, optionally ridge-augmented."""
    As, Bs = sV * A, sV[:, None] * B
    if augment == "ridge":
        ra = ridge_augment_weights(As, Bs, lambda_=ridge_lambda)
        return np.asarray(ra.W, dtype=float)
    return np.asarray(simplex_qp(Bs, As), dtype=float)


def _rmses(y: np.ndarray, cf: np.ndarray, T0: int, K: int) -> Tuple[float, float]:
    """Disaggregated and aggregated pre-treatment imbalance RMSEs."""
    g = y[:T0] - cf[:T0]
    rmse_dis = float(np.sqrt(np.mean(g ** 2)))
    g_agg = _block_means(y[:T0], K) - _block_means(cf[:T0], K)
    rmse_agg = float(np.sqrt(np.mean(g_agg ** 2)))
    return rmse_dis, rmse_agg


def fit_one(inputs: SCTAInputs, nu: float, *, augment: Optional[str] = None,
            ridge_lambda: Optional[float] = None, demean: bool = True) -> SCTAFit:
    """Fit SCTA at a single ``nu`` and assemble an :class:`SCTAFit`."""
    y = inputs.y
    Yd = inputs.donor_matrix
    T0, K = inputs.T0, inputs.block_length
    n_blocks = inputs.n_blocks

    mu_t = float(y[:T0].mean()) if demean else 0.0
    mu_d = Yd[:T0].mean(axis=0) if demean else np.zeros(inputs.n_donors)

    treated_pre = y[:T0] - mu_t
    donor_pre = Yd[:T0] - mu_d
    A, B = _stacked(treated_pre, donor_pre, K)
    sV = _v_sqrt(n_blocks, T0, K, nu)
    w = _weights(A, B, sV, augment, ridge_lambda)

    cf = mu_t + (Yd - mu_d) @ w if demean else Yd @ w
    gap = y - cf
    att = float(np.mean(gap[T0:])) if gap.shape[0] > T0 else float("nan")
    rmse_dis, rmse_agg = _rmses(y, cf, T0, K)

    donor_weights = {inputs.donor_names[i]: float(round(w[i], 6))
                     for i in range(len(w))}
    return SCTAFit(
        nu=float(nu), weights=w, counterfactual=cf, gap=gap, att=att,
        pre_rmse=rmse_dis, rmse_dis=rmse_dis, rmse_agg=rmse_agg,
        donor_weights=donor_weights, metadata={"augment": augment, "demean": demean},
    )


def run_scta(inputs: SCTAInputs, *, nu: float, augment: Optional[str] = None,
             ridge_lambda: Optional[float] = None, demean: bool = True,
             conformal_alpha: float = 0.1,
             frontier: Optional[List[float]] = None
             ) -> Tuple[SCTAFit, Optional[List[Dict[str, float]]]]:
    """Headline fit at ``nu`` plus an optional imbalance-frontier sweep.

    Attaches a CWZ conformal ATT / p-value / interval to the headline fit.
    """
    fit = fit_one(inputs, nu, augment=augment, ridge_lambda=ridge_lambda,
                  demean=demean)
    _, p_value, ci = conformal_inference(
        inputs.y, fit.counterfactual, inputs.T0, alpha=conformal_alpha)
    fit = replace(fit, p_value=p_value, ci=ci,
                  metadata={**fit.metadata, "inference_method": "conformal"})
    front: Optional[List[Dict[str, float]]] = None
    if frontier is not None:
        front = []
        for v in frontier:
            f = fit_one(inputs, v, augment=augment, ridge_lambda=ridge_lambda,
                        demean=demean)
            front.append({"nu": float(v), "rmse_dis": f.rmse_dis,
                          "rmse_agg": f.rmse_agg, "att": f.att})
    return fit, front
