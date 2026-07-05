"""Simplex synthetic-control weight solver for SPOTSYNTH.

After the spillover screen selects the valid donor subset, SPOTSYNTH fits the
canonical Abadie-Diamond-Hainmueller program on those donors: non-negative
weights summing to one, matching the treated unit's pre-intervention path
(O'Riordan & Gilligan-Lee 2025, equation (4) with the simplex restriction of
Abadie et al.).
"""

from __future__ import annotations

from typing import Tuple

import cvxpy as cp
import numpy as np

from ...exceptions import MlsynthEstimationError


def simplex_weights(y: np.ndarray, D: np.ndarray, T0: int) -> Tuple[np.ndarray, np.ndarray]:
    """Fit simplex synthetic-control weights on the pre-intervention window.

    Parameters
    ----------
    y : np.ndarray
        Treated-unit series, length ``T``.
    D : np.ndarray
        Donor matrix, shape ``(T, n_donors)``.
    T0 : int
        Number of pre-intervention periods.

    Returns
    -------
    (weights, counterfactual) : tuple of np.ndarray
        ``weights`` has length ``n_donors``; ``counterfactual`` is ``D @ weights``
        over all ``T`` periods.
    """
    if D.shape[1] == 0:
        raise MlsynthEstimationError("No donors available to build the synthetic control.")
    pre = slice(0, T0)
    n = D.shape[1]
    # Common rescaling for conditioning: the simplex argmin of
    # ``||y - D w||^2`` is invariant to a shared positive scaling of ``(y, D)``
    # (the objective just scales by ``c^2``), so solving on a unit-magnitude
    # version keeps CLARABEL well-conditioned on large-magnitude outcomes
    # (e.g. streaming counts ~1e7, which otherwise yield "no solution"). This
    # realises the paper's Algorithm 1 scale-normalisation for the frequentist
    # solve; the counterfactual is returned on the raw scale via ``D @ weights``.
    scale = float(np.sqrt(np.mean(np.square(D[pre]))))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    w = cp.Variable(n, nonneg=True)
    objective = cp.Minimize(cp.sum_squares(y[pre] / scale - (D[pre] / scale) @ w))
    problem = cp.Problem(objective, [cp.sum(w) == 1])
    try:
        problem.solve(solver=cp.CLARABEL)
    except Exception as exc:  # pragma: no cover - solver fallback
        try:
            problem.solve(solver=cp.SCS)
        except Exception as exc2:
            raise MlsynthEstimationError(f"SC solver failed: {exc2}") from exc
    if w.value is None:
        raise MlsynthEstimationError("SC solver returned no solution.")
    weights = np.asarray(w.value, dtype=float).ravel()
    weights[weights < 0] = 0.0
    s = weights.sum()
    if s > 0:
        weights = weights / s
    return weights, D @ weights

