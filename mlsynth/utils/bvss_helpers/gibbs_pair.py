"""Joint Gibbs update of (\\gamma_i, \\gamma_j, \\mu_i, \\mu_j) for one donor pair.

Implements the novel two-coordinate Gibbs move of Xu & Zhou (2025),
Section 3 / Lemmas S1-S2. Given the other entries of ``\\mu`` fixed,
the four-case conditional distribution of the inclusion pair
:math:`(\\gamma_i, \\gamma_j)` is:

  - (0, 0): infeasible if ``s = 1 - \\sum_{k\\neq i,j} \\mu_k > 0``.
  - (1, 0): forces ``\\mu_i = s, \\mu_j = 0``.
  - (0, 1): forces ``\\mu_i = 0, \\mu_j = s``.
  - (1, 1): draws ``\\mu_i = u`` from a truncated normal
    :math:`N_{(0, s)}(\\beta_{i,j}, (\\phi \\Lambda_{i,j})^{-1})` and sets
    ``\\mu_j = s - u``.

When ``s = 0`` the simplex constraint already pins
``\\mu_i = \\mu_j = 0`` and no draw is needed.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.stats import norm, truncnorm

from .posterior import AM, RSS, RSS2


def _compute_candidate_posteriors(
    mutemp: np.ndarray,
    i: int,
    j: int,
    X: np.ndarray,
    Y: np.ndarray,
    tau: float,
    phi: float,
    Gram: np.ndarray,
    theta: float,
    epsilon: float = 1e-12,
) -> Tuple[float, np.ndarray, Optional[float], Optional[float], np.ndarray]:
    """Compute the four-case conditional probabilities for pair (i, j).

    Returns
    -------
    s : float
        Remaining mass ``1 - \\sum_{k\\neq i,j} \\mu_k``.
    z : np.ndarray
        Residual vector ``Y - X \\mu`` with ``\\mu_i = \\mu_j = 0``.
    L : float or None
        Quadratic term ``\\Lambda_{i,j}`` from Lemma S1; ``None`` if
        ``s = 0``.
    O : float or None
        Mean of the (1, 1) truncated normal, ``\\beta_{i,j}`` from
        Eq. S3; ``None`` if ``s = 0``.
    ptotal : np.ndarray
        Length-4 vector of normalized probabilities for states
        ``(0, 0), (1, 0), (0, 1), (1, 1)``.
    """

    mutemp[[i, j]] = 0
    s = 1 - np.sum(mutemp)
    z = Y - X @ mutemp

    if abs(s) < epsilon:
        return s, z, None, None, np.array([1.0, 0.0, 0.0, 0.0])

    g00 = (mutemp != 0).astype(int)
    g11 = g01 = g10 = g00.copy()
    g10[i] = 1
    g01[j] = 1
    g11[[i, j]] = 1

    L = max(RSS(g11, tau, X[:, j] - X[:, i], X, Gram), epsilon)
    O = RSS2(g11, tau, X[:, i] - X[:, j], z - s * X[:, j], X, Gram) / L

    A11 = AM(g11, tau, theta, Gram, len(mutemp))
    A10 = AM(g10, tau, theta, Gram, len(mutemp))
    A01 = AM(g01, tau, theta, Gram, len(mutemp))

    p10 = A10 - phi * RSS(g10, tau, z - s * X[:, i], X, Gram) / 2
    p01 = A01 - phi * RSS(g01, tau, z - s * X[:, j], X, Gram) / 2
    NC = max(
        norm.cdf((s - O) * np.sqrt(phi * L)) - norm.cdf(-O * np.sqrt(phi * L)),
        epsilon,
    )
    p11 = (
        A11
        + np.log(NC)
        - phi * (RSS(g11, tau, z - s * X[:, j], X, Gram) - O ** 2 * L) / 2
    )
    p11 += np.log(np.sqrt(2 * np.pi) / np.sqrt(max(phi * L, epsilon)))

    ptemp = np.array([p10, p01, p11])
    pbar = np.max(ptemp)
    post_p = np.exp(ptemp - pbar)
    post_p /= np.sum(post_p)
    ptotal = np.array([0.0, *post_p])

    return s, z, L, O, ptotal


def _sample_pair(
    mutemp: np.ndarray,
    i: int,
    j: int,
    s: float,
    L: Optional[float],
    O: Optional[float],
    phi: float,
    ptotal: np.ndarray,
    epsilon: float = 1e-12,
    rng: Optional[np.random.Generator] = None,
) -> None:
    """Draw a state from ``ptotal`` and update ``mutemp`` in place.

    The four states are:
        0 : (\\gamma_i, \\gamma_j) = (0, 0)   ->   \\mu_i = \\mu_j = 0
        1 : (\\gamma_i, \\gamma_j) = (1, 0)   ->   \\mu_i = s, \\mu_j = 0
        2 : (\\gamma_i, \\gamma_j) = (0, 1)   ->   \\mu_i = 0, \\mu_j = s
        3 : (\\gamma_i, \\gamma_j) = (1, 1)   ->   \\mu_i = u ~ TruncNormal(0, s),
                                                  \\mu_j = s - u

    If ``rng`` is provided it is used for both the categorical draw and
    the truncated-normal draw, otherwise :mod:`numpy.random` and
    :mod:`scipy.stats.truncnorm` use their global states.
    """

    if rng is None:
        gamma_state = np.random.choice([0, 1, 2, 3], p=ptotal)
    else:
        gamma_state = rng.choice([0, 1, 2, 3], p=ptotal)

    if gamma_state == 0:
        mutemp[[i, j]] = 0
    elif gamma_state == 1:
        mutemp[i] = s
        mutemp[j] = 0
    elif gamma_state == 2:
        mutemp[i] = 0
        mutemp[j] = s
    else:
        scale = 1 / np.sqrt(max(phi * L, epsilon))
        a, b = (0 - O) / scale, (s - O) / scale
        if rng is None:
            mutemp[i] = truncnorm.rvs(a, b, loc=O, scale=scale)
        else:
            mutemp[i] = truncnorm.rvs(a, b, loc=O, scale=scale, random_state=rng)
        mutemp[j] = s - mutemp[i]
