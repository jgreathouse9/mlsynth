"""Joint Gibbs update of (\\gamma_i, \\gamma_j, \\mu_i, \\mu_j) for one donor pair.

Implements the two-coordinate Gibbs move of Xu & Zhou (2025), Section 3.2 and
Lemmas 1-2. Given the other entries of ``\\mu`` fixed, the four-case conditional
distribution of the inclusion pair :math:`(\\gamma_i, \\gamma_j)` is:

  - (0, 0): infeasible if ``s = 1 - \\sum_{k\\neq i,j} \\mu_k > 0``.
  - (1, 0): forces ``\\mu_i = s, \\mu_j = 0``.
  - (0, 1): forces ``\\mu_i = 0, \\mu_j = s``.
  - (1, 1): draws ``\\mu_i = u`` from a truncated normal
    :math:`N_{(0, s)}(\\beta_{i,j}, (\\phi \\Lambda_{i,j})^{-1})` and sets
    ``\\mu_j = s - u``.

When ``s = 0`` the simplex constraint already pins ``\\mu_i = \\mu_j = 0`` and no
draw is needed.

Each case is scored at its own inclusion vector -- Eq. (11)'s
:math:`\\gamma^i, \\gamma^j, \\gamma^{ij}`, which are three different sets of two
different cardinalities. Lemma 2 indexes both the complexity factor
:math:`A(\\gamma, \\tau)` and the projector :math:`\\Sigma_{\\gamma, \\tau}` by the
state's own vector.

Only the pair's three inclusion vectors change between the four cases, so each
of the three posterior covariances :math:`V_{\\gamma, \\tau}` is built once per
pair and shared by the complexity factor and the quadratic forms that need it.
``mlsynth/tests/test_bvss.py`` pins the result of the sharing against the
unshared primitives, draw for draw.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.linalg import det, solve
from scipy.special import factorial, ndtr
from scipy.stats import truncnorm

from .posterior import AM, RSS, RSS2  # noqa: F401  (public primitives)

_P00 = np.array([1.0, 0.0, 0.0, 0.0])
_P00.flags.writeable = False   # returned by the s = 0 path; never a draw target
_SQRT_2PI = np.sqrt(2 * np.pi)
_LOG_FACT: dict[int, np.ndarray] = {}


def _log_fact(n: int) -> np.ndarray:
    """``log (max(k - 1, 1))!`` for ``k = 0 .. n``, built once per pool size.

    The per-call ``scipy.special.factorial`` this replaces returns the same
    float for every ``k``; a test pins the table against it.
    """

    table = _LOG_FACT.get(n)
    if table is None:
        k = np.arange(n + 1)
        table = np.log(factorial(np.maximum(k - 1, 1)))
        _LOG_FACT[n] = table
    return table


def _vm(idx: np.ndarray, tau: float, Gram: np.ndarray) -> np.ndarray:
    """``V_{\\gamma, \\tau} = X_\\gamma^T X_\\gamma + \\tau^{-1} I`` for a selected index set."""

    return Gram[idx[:, None], idx] + np.eye(idx.size) / tau


def _am(V: np.ndarray, k: int, N: int, log_theta: float,
        log1m_theta: float, log_fact: np.ndarray) -> float:
    """Lemma 2's :math:`\\log A(\\gamma, \\tau)` from an already-built ``V``."""

    d = det(V, check_finite=False)
    det_log = np.log(d) if d > 0 else 0.0
    return k * log_theta + (N - k) * log1m_theta + log_fact[k] - 0.5 * det_log


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
) -> Tuple[float, Optional[np.ndarray], Optional[float], Optional[float], np.ndarray]:
    """Compute the four-case conditional probabilities for pair (i, j).

    Returns
    -------
    s : float
        Remaining mass ``1 - \\sum_{k\\neq i,j} \\mu_k``.
    z : np.ndarray or None
        Residual vector ``Y - X \\mu`` with ``\\mu_i = \\mu_j = 0``; ``None`` when
        ``s = 0``, where no case needs it.
    L : float or None
        Quadratic term ``\\Lambda_{i,j}`` from Lemma 1; ``None`` if ``s = 0``.
    O : float or None
        Mean of the (1, 1) truncated normal, ``\\beta_{i,j}``; ``None`` if ``s = 0``.
    ptotal : np.ndarray
        Length-4 vector of normalized probabilities for states
        ``(0, 0), (1, 0), (0, 1), (1, 1)``.
    """

    mutemp[[i, j]] = 0
    s = 1 - np.sum(mutemp)

    if abs(s) < epsilon:
        return s, None, None, None, _P00

    N = mutemp.shape[0]
    z = Y - X @ mutemp

    # Eq. (11)'s three inclusion vectors, as the index sets they select.
    sel = mutemp != 0
    sel10 = sel.copy()
    sel10[i] = True
    sel01 = sel.copy()
    sel01[j] = True
    sel11 = sel10.copy()
    sel11[j] = True
    idx10 = np.flatnonzero(sel10)
    idx01 = np.flatnonzero(sel01)
    idx11 = np.flatnonzero(sel11)

    log_theta = np.log(theta)
    log1m_theta = np.log(1 - theta)
    log_fact = _log_fact(N)

    # --- the (1, 1) set: one V, one factorisation's worth of solves --------
    V11 = _vm(idx11, tau, Gram)
    Xg11 = X[:, idx11]
    w = z - s * X[:, j]                      # Lemma 2's y-check(0)
    d_ji = X[:, j] - X[:, i]
    d_ij = X[:, i] - X[:, j]

    Xz_dji = Xg11.T @ d_ji
    Xz_w = Xg11.T @ w
    sol_w = solve(V11, Xz_w, check_finite=False)   # shared by O and p11

    L = max(float(d_ji @ d_ji - Xz_dji @ solve(V11, Xz_dji, check_finite=False)),
            epsilon)
    O = float(d_ij @ w - (Xg11.T @ d_ij) @ sol_w) / L
    rss11_w = float(w @ w - Xz_w @ sol_w)

    A11 = _am(V11, idx11.size, N, log_theta, log1m_theta, log_fact)

    # --- the two one-donor sets -------------------------------------------
    V10 = _vm(idx10, tau, Gram)
    z10 = z - s * X[:, i]                    # Lemma 2's y-check(s)
    Xz10 = X[:, idx10].T @ z10
    rss10 = float(z10 @ z10 - Xz10 @ solve(V10, Xz10, check_finite=False))
    A10 = _am(V10, idx10.size, N, log_theta, log1m_theta, log_fact)

    V01 = _vm(idx01, tau, Gram)
    Xz01 = X[:, idx01].T @ w
    rss01 = float(w @ w - Xz01 @ solve(V01, Xz01, check_finite=False))
    A01 = _am(V01, idx01.size, N, log_theta, log1m_theta, log_fact)

    p10 = A10 - phi * rss10 / 2
    p01 = A01 - phi * rss01 / 2

    root = np.sqrt(phi * L)
    NC = max(ndtr((s - O) * root) - ndtr(-O * root), epsilon)
    p11 = A11 + np.log(NC) - phi * (rss11_w - O ** 2 * L) / 2
    p11 += np.log(_SQRT_2PI / np.sqrt(max(phi * L, epsilon)))

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

    If ``rng`` is provided it is used for both the categorical draw and the
    truncated-normal draw, otherwise :mod:`numpy.random` and
    :mod:`scipy.stats.truncnorm` use their global states. The categorical draw
    is the inverse-CDF form of ``Generator.choice``, which draws one uniform and
    leaves the stream where ``choice`` leaves it.
    """

    if rng is None:
        gamma_state = np.random.choice([0, 1, 2, 3], p=ptotal)
    else:
        cdf = ptotal.cumsum()
        cdf /= cdf[-1]
        gamma_state = min(int(cdf.searchsorted(rng.random(), side="right")), 3)

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
