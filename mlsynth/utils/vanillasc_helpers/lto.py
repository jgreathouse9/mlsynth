"""Leave-Two-Out (LTO) refined placebo test for the synthetic control.

Lei & Sudijono (2025), "Inference for Synthetic Controls via Refined Placebo
Tests" (arXiv:2401.07152). The ordinary placebo / permutation test builds its
null distribution from only :math:`N` reference estimates, so its p-value lives
on the coarse grid :math:`\\{1/N, 2/N, \\dots, 1\\}` and has *zero size* when
:math:`\\alpha < 1/N`. The LTO test bypasses this by leaving **two** control
units out at a time, producing :math:`O(N^2)` reference comparisons while
retaining the same finite-sample Type-I error guarantee under uniform
assignment.

Procedure (naive LTO, eqs. 5-7)
-------------------------------
Let :math:`I` be the treated unit and :math:`[N]\\setminus\\{I\\}` the controls
(:math:`N = J + 1` with :math:`J` donors). For every unordered pair of distinct
controls :math:`\\{i, j\\}`:

1. Build the synthetic control for each :math:`k \\in \\{i, j, I\\}` using the
   donor pool :math:`[N]\\setminus\\{i, j, I\\}` (all controls except
   :math:`i, j`), and form the residual
   :math:`R_{i,j,I;k} = \\lvert S(Y_k, \\hat Y_k)\\rvert` with :math:`S` the
   post/pre RMSPE-ratio statistic.
2. Let :math:`R^{\\mathrm{LTO}}_{i,j} = \\max(R_{i,j,I;i}, R_{i,j,I;j})`; the
   treated unit "wins" the triple when :math:`R_{i,j,I;I} >
   R^{\\mathrm{LTO}}_{i,j}`.

The naive LTO p-value counts the fraction of pairs the treated unit does *not*
win,

.. math::

   p_{\\mathrm{naive\\text{-}LTO}}
     = \\frac{1}{(N-1)(N-2)} \\sum_{i \\neq j}
       \\mathbf{1}\\{R_{i,j,I;I} \\le R^{\\mathrm{LTO}}_{i,j}\\},

which (Theorem 2.2) satisfies
:math:`\\mathbb{P}_{H_0}(p_{\\mathrm{naive\\text{-}LTO}} \\le \\alpha) \\le
\\lfloor N f(N, \\alpha)\\rfloor / N`.

Powered LTO (Theorem 2.3)
-------------------------
For testing at a *fixed* level :math:`\\alpha`, the powered p-value
:math:`p_{\\mathrm{powered\\text{-}LTO}}(\\alpha) =
p_{\\mathrm{naive\\text{-}LTO}} - c(N, \\alpha) + \\delta` shifts the naive value
down by the largest amount that leaves the discrete Type-I bound unchanged,
strictly increasing power. It is only valid for the :math:`\\alpha` it was
computed at (reject when it is :math:`\\le \\alpha`).
"""

from __future__ import annotations

from math import floor, sqrt
from typing import Any, Dict, Optional

import numpy as np

_EPS = 1e-12
_DELTA = 1e-10


def lto_f(N: int, alpha: float) -> float:
    """Type-I error rate function :math:`f(N, \\alpha)` (Lei-Sudijono eq. 9)."""
    a = 1.0 - 1.0 / N
    inner = 9.0 * a ** 2 - 12.0 * (
        -4.0 / (3.0 * N ** 2) + 1.0 / N + alpha * a * (1.0 - 2.0 / N)
    )
    if inner < 0.0:  # outside the valid alpha range; clamp
        inner = 0.0
    return (3.0 - 3.0 / N - sqrt(inner)) / 2.0


def lto_type_i_bound(N: int, alpha: float) -> float:
    """Discrete Type-I error upper bound :math:`\\lfloor N f(N,\\alpha)\\rfloor/N`."""
    return floor(N * lto_f(N, alpha)) / N


def lto_powered_offset(N: int, alpha: float) -> float:
    """``c(N, alpha)``: largest shift leaving the discrete Type-I bound fixed.

    Defined (Theorem 2.3) as the smallest ``c`` with
    ``f(N, alpha + c) = (floor(N f(N, alpha)) + 1) / N``. Found by bisection on
    the monotone increasing ``f``. Reproduces the paper's values
    (``c(39, 0.05) = 0.002``, ``c(17, 0.05) = 0.0125``).
    """
    target = (floor(N * lto_f(N, alpha)) + 1.0) / N
    lo, hi = 0.0, 1.0 - alpha
    # f is increasing in its second argument; target > f(N, alpha) by construction
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if lto_f(N, alpha + mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _rmspe_ratio_resid(y_k: np.ndarray, cf: np.ndarray, pre: int) -> float:
    """``|post/pre RMSPE ratio|`` residual statistic (ADH15 / eq. 6)."""
    gap = y_k - cf
    pre_r = float(np.sqrt(np.mean(gap[:pre] ** 2)))
    post_r = float(np.sqrt(np.mean(gap[pre:] ** 2))) if gap[pre:].size else float("nan")
    if pre_r <= _EPS:
        return float("inf")
    return abs(post_r / pre_r)


def lto_placebo_test(
    engine: Any,
    y: np.ndarray,
    Y0: np.ndarray,
    pre: int,
    *,
    X1: Optional[np.ndarray] = None,
    X0: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    max_pairs: Optional[int] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    """Run the Lei-Sudijono (2025) LTO refined placebo test.

    Parameters
    ----------
    engine : BilevelSCM
        Fitted-config synthetic-control engine; ``engine.fit(...)`` is re-run
        for each leave-two-out subproblem (any backend works, but the cost is
        :math:`O(J^2)` fits, so fast backends are recommended).
    y : np.ndarray
        Treated outcome over all periods, shape ``(T,)``.
    Y0 : np.ndarray
        Donor outcomes, shape ``(T, J)``.
    pre : int
        Number of pre-treatment periods.
    X1, X0 : np.ndarray, optional
        Treated predictor vector ``(P,)`` and donor predictor matrix ``(P, J)``
        (already windowed and scaled). ``None`` for outcome-only matching.
    alpha : float
        Level at which the powered LTO p-value and Type-I bound are reported.
    max_pairs : int, optional
        Cap on the number of donor pairs evaluated (deterministic subsample,
        for expensive backends). ``None`` -> all :math:`\\binom{J}{2}` pairs.
    seed : int
        RNG seed for the pair subsample when ``max_pairs`` is set.

    Returns
    -------
    dict
        ``p_value`` (naive LTO), ``p_powered`` (valid only at ``alpha``),
        ``c`` (powered offset), ``type_i_bound``, ``n_pairs``, ``treated_losses``,
        ``N``, ``alpha``, ``reject`` (powered decision at ``alpha``), and
        ``subsampled``.
    """
    Y0 = np.asarray(Y0, float)
    y = np.asarray(y, float).ravel()
    T, J = Y0.shape
    if J < 3:
        raise ValueError(
            "LTO placebo test needs at least 3 donor units (to leave two out "
            "and retain a non-empty control pool)."
        )
    N = J + 1

    pairs = [(a, b) for a in range(J) for b in range(a + 1, J)]
    subsampled = False
    if max_pairs is not None and len(pairs) > max_pairs:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[i] for i in sorted(idx)]
        subsampled = True

    def _resid(y_k, pool, x1):
        x0p = X0[:, pool] if X0 is not None else None
        rk = engine.fit(y_k[:pre], Y0[:, pool][:pre], X1=x1, X0=x0p)
        return _rmspe_ratio_resid(y_k, rk.counterfactual(Y0[:, pool]), pre)

    losses = 0
    n_pairs = 0
    for a, b in pairs:
        pool = [k for k in range(J) if k != a and k != b]
        if not pool:  # pragma: no cover - pool size = J-2 >= 1 when J >= 3
            continue
        try:
            R_I = _resid(y, pool, X1)
            R_a = _resid(Y0[:, a], pool, (X0[:, a] if X0 is not None else None))
            R_b = _resid(Y0[:, b], pool, (X0[:, b] if X0 is not None else None))
        except Exception:  # pragma: no cover - defensive donor-refit guard
            continue
        if not np.isfinite(R_I):  # pragma: no cover - donor with zero pre-error
            R_I = np.finfo(float).max
        n_pairs += 1
        if not (R_I > max(R_a, R_b)):     # treated unit did not win the triple
            losses += 1

    if n_pairs == 0:  # pragma: no cover - unreachable when J >= 3
        raise ValueError("LTO placebo test: no leave-two-out subproblem could be fit.")

    p_naive = losses / n_pairs
    c = lto_powered_offset(N, alpha)
    p_powered = max(p_naive - c + _DELTA, 0.0)
    return {
        "p_value": float(p_naive),
        "p_powered": float(p_powered),
        "c": float(c),
        "type_i_bound": float(lto_type_i_bound(N, alpha)),
        "n_pairs": int(n_pairs),
        "treated_losses": int(losses),
        "N": int(N),
        "alpha": float(alpha),
        "reject": bool(p_powered <= alpha),
        "subsampled": subsampled,
    }
