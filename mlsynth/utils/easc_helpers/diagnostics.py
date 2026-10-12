"""Leverage and the silent-factor diagnostic (Spruk 2026, Sections 2 and 4.1).

Pure NumPy. ``T`` is the number of periods, ``pre`` a boolean mask of the
pre-treatment periods, and a factor matrix ``G`` holds one period per row with
the constant included. None of these functions reads the treated unit's
post-treatment outcomes.
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np


def with_constant(*blocks: np.ndarray) -> np.ndarray:
    """The factor vector ``g_t = (1, blocks...)`` stacked by period."""
    T = np.asarray(blocks[0]).shape[0]
    return np.column_stack([np.ones(T), *blocks])


def donor_factors(Y0: np.ndarray, k: int, exposure: Optional[np.ndarray] = None) -> np.ndarray:
    """Latent factors estimated from the donors (Section 4.1, Online Appendix B).

    The ``T x J`` donor matrix is demeaned by unit over all periods (donors are
    untreated throughout) and then by period. When ``exposure`` is given, each
    period's cross-sectional projection on the standardised index is removed,
    so the factors carry nothing of the form ``g_t s_j``: they are orthogonal
    to the exposure channel. The factors are the first ``k`` left singular
    vectors scaled by their singular values and by ``1/sqrt(T)``.
    """
    Y0 = np.asarray(Y0, dtype=float)
    Z = Y0 - Y0.mean(axis=0, keepdims=True)
    Z = Z - Z.mean(axis=1, keepdims=True)
    if exposure is not None:
        s = np.asarray(exposure, dtype=float)
        sc = (s - s.mean()) / s.std()
        Z = Z - np.outer(Z @ sc / (sc @ sc), sc)
    U, sv, _ = np.linalg.svd(Z, full_matrices=False)
    return U[:, :k] * sv[:k] / np.sqrt(Z.shape[0])


def _pre_moment_inverse(G: np.ndarray, pre: np.ndarray) -> np.ndarray:
    """``S_0^+``, the pseudo-inverse of the pre-period second-moment matrix."""
    S0 = G[pre].T @ G[pre] / pre.sum()
    return np.linalg.pinv(S0)


def leverage(G: np.ndarray, pre: np.ndarray) -> np.ndarray:
    """``kappa_t = (g_t' S_0^{-1} g_t)^{1/2}`` for every period (equation 3).

    Proposition 1 bounds the post-treatment bias of any weighting estimator in
    period ``t`` by ``kappa_t`` times the systematic pre-treatment misfit. Over
    the pre-period ``kappa_t^2`` averages the number of factors, so values far
    above that signal that the factors have left their pre-treatment span. The
    pseudo-inverse allows a singular ``S_0``, the case the paper studies.
    """
    G = np.asarray(G, dtype=float)
    Si = _pre_moment_inverse(G, pre)
    return np.sqrt(np.maximum(np.einsum("ti,ij,tj->t", G, Si, G), 0.0))


def phase_leverage(G: np.ndarray, pre: np.ndarray, mask: np.ndarray) -> float:
    """Leverage of a phase-averaged factor vector, ``(gbar' S_0^{-1} gbar)^{1/2}``.

    The bias of a phase average is ``Delta' gbar``, so the leverage of the
    averaged vector bounds it, and it is never larger than the average of the
    per-period leverages.
    """
    G = np.asarray(G, dtype=float)
    gb = G[mask].mean(axis=0)
    return float(np.sqrt(max(gb @ _pre_moment_inverse(G, pre) @ gb, 0.0)))


def partial_leverage(x: np.ndarray, F: np.ndarray, pre: np.ndarray) -> Tuple[np.ndarray, float]:
    """Partial leverage ``ell_t`` of the observed shock and its pre-period R^2 (Corollary 1).

    The shock is regressed on ``(1, F)`` over the pre-period. ``ell_t`` is the
    residual in every period divided by the residual's pre-period root mean
    square, and the R^2 is that regression's. A pre-period R^2 near one with a
    post-period ``|ell_t|`` well above 2 is the signature of a silent factor.

    When the shock is an exact linear function of the factors before treatment
    (``S_r = 0``, the limiting case of Corollary 1) the residual has no scale:
    ``ell_t`` is returned as NaN with a warning, and the R^2 as 1.
    """
    x = np.asarray(x, dtype=float)
    X = with_constant(F)
    b = np.linalg.lstsq(X[pre], x[pre], rcond=None)[0]
    res = x - X @ b
    sr = float(np.sqrt(np.mean(res[pre] ** 2)))
    scale = max(float(np.max(np.abs(x[pre]))), 1.0)
    if sr <= 1e-10 * scale:
        warnings.warn(
            "EASC: the observed shock is an exact linear function of the latent factors "
            "before treatment (S_r = 0), so its partial leverage is undefined.",
            UserWarning, stacklevel=2)
        return np.full(x.shape, np.nan), 1.0
    var_x = float(x[pre].var())
    return res / sr, float(1.0 - res[pre].var() / var_x)


def exposure_imbalance(s1: float, s0: np.ndarray, w: np.ndarray) -> float:
    """``EI(w) = s_1 - sum_j w_j s_j`` (equation 7)."""
    return float(s1 - np.asarray(s0, dtype=float) @ np.asarray(w, dtype=float))


def min_imbalance(s1: float, s0: np.ndarray) -> float:
    """``d_min``, the distance from ``s_1`` to the donors' exposure range (Proposition 3(a))."""
    s0 = np.asarray(s0, dtype=float)
    return float(max(0.0, s1 - s0.max(), s0.min() - s1))
