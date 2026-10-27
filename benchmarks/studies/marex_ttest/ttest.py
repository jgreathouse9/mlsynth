"""Chernozhukov, Wuthrich and Zhu's debiased t-test on a MAREX design.

The test (JPE 2026, Section II) estimates the average post-period effect of a
treated series against a synthetic control and removes the control's bias by
cross-fitting. The pre-period's last ``K r`` periods, ``r = min(span // K, T1)``,
form ``K`` consecutive blocks. For block ``k`` the control's weights are fitted
on the other pre-periods, and

    tau_k = mean_post(y - Y0 w_k) - mean_{block k}(y - Y0 w_k),

the post gap less the bias the fit shows on the block it did not see. The
estimate is the average of the ``tau_k``, its standard error
``sqrt(1 + K r / T1) sd(tau_k) / sqrt(K)``, and the self-normalised statistic is
asymptotically t with ``K - 1`` degrees of freedom. It needs no long-run
variance, and the paper proves it valid under nonstationarity shared by all
units (Assumption 4), and under deviations from it when the synthetic control
is correctly specified and ``T1`` is small next to ``T0`` (Assumptions 5-6).

``span`` is the number of pre-periods the blocks may occupy, counted back from
``T0``. ``span = T0`` is the library's placement
(:func:`mlsynth.utils.inferutils.debiased_sc_ttest`, a port of ``scinference``).
On a MAREX design the treated set was chosen on the fitting window, so a block
there shows the in-sample fit; ``span = blank_periods`` keeps every block in the
blank window the design never saw. ``weights`` fixes the control instead of
refitting it on each fold, which isolates what the refit contributes.

The arms on a MAREX design test the treated composite ``Y @ w`` against the
units the design left untreated, the pool MAREX's inversion band refits on.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy.stats import t as _t

from mlsynth.utils.solvers.active_set import solve_simplex_qp

#: refit or fixed control, blocks over the pre-period or the blank window, K.
ARMS = ("refit_pre_3", "refit_pre_4", "refit_blank_3", "refit_blank_4",
        "fixed_blank_3", "fixed_blank_4")


@dataclass(frozen=True)
class TTest:
    att: float
    se: float
    tstat: float
    tau_k: np.ndarray
    r: int
    K: int


def blocks(T0: int, T1: int, K: int, span: int) -> np.ndarray:
    """The ``K`` held-out blocks, ``(K, r)`` period indices ending at ``T0 - 1``."""
    r = min(span // K, T1)
    if r < 1 or span > T0:
        raise ValueError(f"a span of {span} pre-periods (of {T0}) holds no {K} blocks "
                         f"of at least one period with T1 = {T1}")
    return T0 - K * r + np.arange(K * r).reshape(K, r)


def debiased_ttest(y, Y0, T0: int, T1: int, K: int, *, span: int,
                   weights: Optional[np.ndarray] = None) -> TTest:
    """The debiased t-test with its blocks in the last ``span`` pre-periods.

    ``weights`` fixes the control; otherwise each fold refits the simplex
    synthetic control of ``y`` on ``Y0`` over the pre-periods outside its block.
    """
    y, Y0 = np.asarray(y, dtype=float).ravel(), np.asarray(Y0, dtype=float)
    if y.size != T0 + T1 or Y0.shape[0] != T0 + T1:
        raise ValueError(f"y and Y0 must cover T0 + T1 = {T0 + T1} periods")
    if weights is not None and np.shape(weights) != (Y0.shape[1],):
        raise ValueError(f"weights must have one entry per column of Y0 ({Y0.shape[1]})")
    B = blocks(T0, T1, K, span)
    r = B.shape[1]
    tau = np.empty(K)
    for k in range(K):
        keep = np.setdiff1d(np.arange(T0), B[k])
        w = weights if weights is not None else solve_simplex_qp(Y0[keep], y[keep])
        gap = y - Y0 @ w
        tau[k] = gap[T0:].mean() - gap[B[k]].mean()
    att = float(tau.mean())
    se = float(np.sqrt(1.0 + K * r / T1) * tau.std(ddof=1) / np.sqrt(K))
    return TTest(att=att, se=se, tstat=_tstat(att, se), tau_k=tau, r=r, K=K)


def _tstat(att: float, se: float) -> float:
    if se > 0:
        return att / se
    return 0.0 if att == 0 else float(np.inf * np.sign(att))


def pvalue(tstat: float, K: int) -> float:
    """Two-sided p-value against t with ``K - 1`` degrees of freedom."""
    return float(min(1.0, 2.0 * _t.sf(abs(tstat), K - 1)))


def shifted_pvalue(res: TTest, tau: float) -> float:
    """The p-value had ``tau`` been added to every post period of ``y``.

    The fold weights read only the pre-period, so each ``tau_k`` moves by
    ``tau`` and the standard error does not move.
    """
    return pvalue(_tstat(res.att + tau, res.se), res.K)


def arms(Y: np.ndarray, d, H: int) -> Dict[str, TTest]:
    """Every arm on the design ``d`` of panel ``Y`` (periods by units)."""
    untreated = d.w <= 1e-8
    y, Y0 = Y[:d.T0 + H] @ d.w, Y[:d.T0 + H, untreated]
    v = d.v[untreated]
    out = {}
    for K in (3, 4):
        out[f"refit_pre_{K}"] = debiased_ttest(y, Y0, d.T0, H, K, span=d.T0)
        out[f"refit_blank_{K}"] = debiased_ttest(y, Y0, d.T0, H, K, span=d.blank)
        out[f"fixed_blank_{K}"] = debiased_ttest(y, Y0, d.T0, H, K, span=d.blank, weights=v)
    return out
