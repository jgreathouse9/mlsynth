"""The weight program: eq. (5)-(7).

The weights minimise the pre-treatment discrepancy between the treated unit's
DR parameter function and a weighted combination of the donors', averaged over
pre-periods and integrated over the outcome grid, subject only to adding up to
one. Because the objective is quadratic and the single constraint is linear,
the solution is closed-form -- no solver, no simplex projection.

Dropping non-negativity is deliberate (Doudchenko-Imbens): it lets the
synthetic unit sit outside the donor hull, which is what makes the pre-period
fit attainable when the treated unit is extreme. The price is a Gram matrix
that can be badly conditioned, so see :func:`solve_weights` on float64.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from ...exceptions import MlsynthEstimationError


def gram_and_cross(theta: Dict[Tuple[int, object], np.ndarray], J: int,
                   pre_periods: Sequence[object]) -> Tuple[np.ndarray, np.ndarray]:
    """Time-and-grid-averaged Gram matrix ``G`` and cross-vector ``c``.

    ``theta[(i, t)]`` is the ``(m, k)`` coefficient function for unit index
    ``i`` (0 = treated) in period ``t``.
    """
    m = next(iter(theta.values())).shape[0]
    G = np.zeros((J, J), dtype=np.float64)
    c = np.zeros(J, dtype=np.float64)
    for t in pre_periods:
        for a in range(J):
            c[a] += float(np.sum(theta[(0, t)] * theta[(a + 1, t)]))
            for b in range(a, J):
                v = float(np.sum(theta[(a + 1, t)] * theta[(b + 1, t)]))
                G[a, b] += v
                if b != a:
                    G[b, a] += v
    denom = len(pre_periods) * m
    return G / denom, c / denom


def solve_weights(G: np.ndarray, c: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    """Closed-form min-norm weights under ``1'w = 1`` (eq. 7).

    The Gram matrix is ill-conditioned by construction -- donors' DR parameter
    functions are similar, which is exactly why a weighted combination can
    track the treated unit -- so the inverse amplifies input error sharply. On
    the paper's panel ``kappa(G) ~ 1e5`` and a relative input perturbation of
    1e-7 moves the weights by ~0.15. The estimand is far more stable than the
    weights, but the weights are what get reported, so the inputs must be
    float64 end to end.
    """
    J = G.shape[0]
    A = G + ridge * np.eye(J) if ridge > 0 else G
    one = np.ones(J)
    try:
        Ai = np.linalg.inv(A)
    except np.linalg.LinAlgError as exc:
        raise MlsynthEstimationError(
            "the Gram matrix is singular; donor DR parameter functions are "
            "linearly dependent. Drop duplicate donors or set `ridge` > 0."
        ) from exc
    Aic, Ai1 = Ai @ c, Ai @ one
    denom = float(one @ Ai1)
    if not np.isfinite(denom) or abs(denom) < 1e-300:  # pragma: no cover
        raise MlsynthEstimationError(
            "the adding-up constraint is degenerate for this Gram matrix.")
    w = Aic - Ai1 * ((float(one @ Aic) - 1.0) / denom)
    if not np.all(np.isfinite(w)):  # pragma: no cover - defensive
        raise MlsynthEstimationError(
            "the weight solve produced non-finite values; check for duplicate "
            "donor units or set `ridge` > 0.")
    return w


def projection_matrix(G: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    """``P = dw/dc``, the derivative of the weights in the cross-vector."""
    J = G.shape[0]
    A = G + ridge * np.eye(J) if ridge > 0 else G
    one = np.ones(J)
    Ai = np.linalg.inv(A)
    Ai1 = Ai @ one
    return Ai - np.outer(Ai1, Ai1) / float(one @ Ai1)


def counterfactual(theta: Dict[Tuple[int, object], np.ndarray], w: np.ndarray,
                   t: object) -> np.ndarray:
    """Counterfactual DR parameter, eq. (9)."""
    out = np.zeros_like(theta[(1, t)])
    for a in range(len(w)):
        out = out + w[a] * theta[(a + 1, t)]
    return out
