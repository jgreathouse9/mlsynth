"""Information-criterion (IC) design selection for SYNDES.

The holdout selector (:mod:`mlsynth.utils.syndes_helpers.holdout`) spends part of
the pre-period on a validation tail. When the pre-period is short -- the regime
SYNDES is built for -- that split is noisy, and an information criterion that
uses the *whole* pre-window is the more reliable model-selection device
(Pouliot, Xie & Liu 2024, who show cross-validation/holdout underperforms an IC
in short-``T0`` synthetic-control settings).

For a candidate design with pre-period contrast series ``g = Y_pre @ c`` the IC is

    IC(d) = SSR_pre(d) + 2 * sigma^2 * df(d),

mirroring Pouliot et al.'s ``E||Y - Yhat||^2 + 2 sigma^2 df``:

* ``SSR_pre`` -- in-sample contrast sum of squares (the design's pre-period fit);
* ``df`` -- the synthetic-control degrees of freedom. Pouliot et al.'s
  closed form for the unpenalised SCM is ``df = |A| - 1`` where ``A`` is the set
  of active donors (positive weight), and "model selection is free" -- the search
  over which donors to use costs no extra df. We take ``A`` to be the active
  *control* donors (the synthetic control that reconstructs the treated
  aggregate), so ``df = max(|active control donors| - 1, 0)``;
* ``sigma^2`` -- a Mallows-``Cp``-style noise estimate: the smallest per-period
  contrast variance in the candidate pool (the best-fitting / least-biased
  design's residual variance). This needs no data split, only the pool already
  solved on the full pre-period.

The candidate with the smallest IC wins: it penalises designs that buy a tighter
in-sample fit by activating more donors. ``df`` is deliberately a single, simple,
paper-aligned choice computed in one place (:func:`design_df`) so it can be
refined without touching the ranking logic.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple

import numpy as np


def _control_weights(design: Any) -> Optional[np.ndarray]:
    """Aggregate ``(N,)`` control-side weights for any SYNDES design mode.

    Mirrors the contrast bookkeeping in the estimator: the two-way / equal-weight
    / annealed modes store the control simplex in ``control_weights`` directly,
    whereas ``per_unit`` leaves it ``None`` and keeps the per-treated-unit
    synthetic controls in the ``(K, N)`` ``treated_weights`` matrix, whose column
    sum is the aggregate control side.
    """
    tw = getattr(design, "treated_weights", None)
    cw = getattr(design, "control_weights", None)
    if tw is not None and np.asarray(tw).ndim == 2:
        return np.asarray(tw, dtype=float).sum(axis=0)
    if cw is not None:
        return np.asarray(cw, dtype=float).reshape(-1)
    return None


def design_df(design: Any, tol: float = 1e-8) -> int:
    """Synthetic-control degrees of freedom of a design: active controls minus one.

    ``df = max(|A| - 1, 0)`` with ``A`` the active control donors (Pouliot et al.
    2024, ``df = |A| - 1`` for the unpenalised SCM; the ``-1`` is the simplex
    sum-to-one constraint). Returns ``0`` when no control side is defined.
    """
    cw = _control_weights(design)
    if cw is None:
        return 0
    n_active = int(np.sum(np.abs(cw) > tol))
    return max(n_active - 1, 0)


def select_by_ic(
    designs: List[Any],
    Y_pre: np.ndarray,
    *,
    df_fn: Callable[[Any], int] = design_df,
) -> Tuple[List[Any], List[float], List[int], float]:
    """Rank a SYNDES candidate pool by the information criterion.

    Parameters
    ----------
    designs : list
        Candidate designs (each exposing ``contrast_weights`` and the weight
        fields :func:`design_df` reads). Typically the pool solved on the full
        pre-period.
    Y_pre : np.ndarray
        Pre-treatment outcome matrix, shape ``(T_pre, N)``.
    df_fn : callable, optional
        Degrees-of-freedom function (defaults to :func:`design_df`); injected for
        testability.

    Returns
    -------
    (ranked_designs, ic_values, df_values, sigma2) : tuple
        Designs ordered by ascending IC; the matching IC values and degrees of
        freedom (same order); and the Mallows-``Cp`` noise estimate ``sigma^2``.
        Ties keep the input order (stable sort).
    """
    if not designs:
        return [], [], [], 0.0
    Y = np.asarray(Y_pre, dtype=float)
    T = Y.shape[0]
    ssr: List[float] = []
    for d in designs:
        c = getattr(d, "contrast_weights", None)
        if c is None:
            ssr.append(float("inf"))
            continue
        g = Y @ np.asarray(c, dtype=float).reshape(-1)
        ssr.append(float(np.sum(g ** 2)))
    finite = [s for s in ssr if np.isfinite(s)]
    sigma2 = (min(finite) / T) if (finite and T > 0) else 0.0
    dfs = [int(df_fn(d)) for d in designs]
    ic = [ssr[i] + 2.0 * sigma2 * dfs[i] for i in range(len(designs))]
    order = sorted(range(len(designs)), key=lambda i: (ic[i], i))   # stable
    ranked = [designs[i] for i in order]
    return ranked, [ic[i] for i in order], [dfs[i] for i in order], sigma2
