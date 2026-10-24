"""The full-form persp+N certificate this study measured against (#833).

When this study ran, MAREX's node bound (``mlsynth/utils/marex_helpers/branch.py``)
kept one block ``M = [[1, x'], [x, X]]`` per side, with multipliers on
``M_00 = 1`` (``y0``) and ``X 1 = x`` (``beta``) and the trace bound
``tr M <= 2``. #900 moved the library to the face form
(``mlsynth/utils/persp_bound.py``), which has neither multiplier and the trace
bound 1. The regularised dual of ``dual.py`` is written for the full form, so
its certificate is kept here as the library had it, unchanged, on the library's
exact pieces.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import List, Optional, Sequence, Tuple

import numpy as np

from mlsynth.utils.marex_helpers.certify import Dyadic
from mlsynth.utils.persp_bound import _from_fractions, _psd2, lambda_min_lower


@dataclass(frozen=True)
class SideDuals:
    """One side's multipliers, as a solver returned them, and what they price.

    ``U`` are the markets the side can use, ``free`` the pairs ``(i, f)`` of a
    position in ``U`` and the market's position among the free markets,
    ``control`` whether the side's indicator is ``1 - z``, and ``lin`` its
    penalty vector over the cluster. ``y0``, ``alpha``, ``beta``, ``H``, ``mu``
    and ``Lam`` multiply ``M_00 = 1``, ``1'x = 1``, ``X 1 = x``, ``X >= 0``,
    ``ind_j - x_j >= 0`` and the 2x2 perspective blocks; any values will do.
    """

    U: List[int]
    free: List[Tuple[int, int]]
    control: bool
    lin: Optional[np.ndarray]
    y0: float
    alpha: float
    beta: np.ndarray
    H: np.ndarray
    mu: List[float]
    Lam: List[np.ndarray]


def certificate(G: Dyadic, scale: Fraction, r: int, n_free: int,
                 sides: Sequence[SideDuals]) -> Optional[Fraction]:
    """A lower bound on every design of a node, from any multipliers.

    Each multiplier is first made to have its constraint's sign (``H``, ``mu``
    clipped at zero, each 2x2 block made semidefinite), so every subtracted term
    is zero or positive at a design; the Lagrangian is then linear in the lifted
    variables, ``const + <S, M> + c'z`` per side in units of ``scale``, and is
    bounded below over the designs exactly. ``None`` when a smallest eigenvalue
    could not be bounded.
    """
    const = Fraction(0)
    cz = [Fraction(0)] * n_free
    for s in sides:
        U = s.U
        k = len(U)
        H = np.clip((s.H + s.H.T) / 2.0, 0.0, None)
        H = np.triu(H) + np.triu(H, 1).T
        const += Fraction(s.y0) + Fraction(s.alpha)
        S = [[Fraction(0)] * (k + 1) for _ in range(k + 1)]
        S[0][0] = -Fraction(s.y0)
        for a in range(k):
            for b in range(k):
                S[a + 1][b + 1] = (G.frac(U[a], U[b]) / scale - Fraction(H[a, b])
                                   - (Fraction(s.beta[a]) + Fraction(s.beta[b])) / 2)
        xc = [-Fraction(s.alpha) + Fraction(s.beta[a]) for a in range(k)]
        if s.lin is not None:
            xc = [xc[a] + Fraction(float(s.lin[U[a]])) / scale for a in range(k)]
        for (i, f), mu_raw, L in zip(s.free, s.mu, s.Lam):
            mu = Fraction(max(float(mu_raw), 0.0))
            a_, b_, c_ = (Fraction(v) for v in _psd2(np.asarray(L, float)))
            xc[i] += mu - 2 * b_
            S[i + 1][i + 1] -= c_
            if s.control:                       # the indicator is 1 - z
                const -= mu + a_
                cz[f] += mu + a_
            else:                               # the indicator is z
                cz[f] -= mu + a_
        for a in range(k):
            S[0][a + 1] = S[a + 1][0] = xc[a] / 2
        lam = lambda_min_lower(_from_fractions(S))
        if lam is None:
            return None
        const += 2 * min(Fraction(0), lam)
    if n_free:
        const += sum(sorted(cz)[:r])
    return const * scale
