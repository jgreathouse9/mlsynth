"""Dynamic time warping with Sakoe-Chiba step patterns.

A self-contained implementation of the step patterns the Dynamic Synthetic
Control method needs, so that :mod:`mlsynth` gains no runtime dependency for
DTWSC. The six here are exactly the ones the authors' grid search selects:
``symmetricP1`` / ``symmetricP2`` / ``asymmetricP1`` / ``asymmetricP2``
(Sakoe & Chiba 1978), ``typeIc`` (Rabiner & Juang), and ``mori2006``.

Step patterns are encoded in the standard ``(pattern, di, dj, weight)`` table:
the row with ``weight == -1`` opens a pattern and gives the offset back to the
predecessor cell; the remaining rows of that pattern name the cells whose local
distance is accumulated, with their coefficients. The recursion is

.. math::

   D(i, j) = \\min_{p} \\Big[ D(i - d^p_{i,0},\\, j - d^p_{j,0})
             + \\sum_{r \\ge 1} w^p_r \\, d(i - d^p_{i,r},\\, j - d^p_{j,r}) \\Big].

``warp`` deliberately reproduces R's ``dtw::warp``, which *averages* the tied
indices of the many-to-one alignment map and therefore returns fractional
positions. That behaviour is load-bearing: :func:`ref_too_short` rounds the
maximum of the warp, so an averaged 7.5 rounds up to 8 where a truncated 7 does
not, and the two give different answers about whether a window is usable.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ...exceptions import MlsynthEstimationError

#: ``(pattern, di, dj, weight)`` rows; ``weight == -1`` opens a pattern.
SYMMETRIC_P1: np.ndarray = np.array(
    [
        [1, 1, 2, -1.0], [1, 0, 1, 2.0], [1, 0, 0, 1.0],
        [2, 1, 1, -1.0], [2, 0, 0, 2.0],
        [3, 2, 1, -1.0], [3, 1, 0, 2.0], [3, 0, 0, 1.0],
    ],
    dtype=float,
)

_T = 2.0 / 3.0
ASYMMETRIC_P2: np.ndarray = np.array(
    [
        [1, 2, 3, -1.0], [1, 1, 2, _T], [1, 0, 1, _T], [1, 0, 0, _T],
        [2, 1, 1, -1.0], [2, 0, 0, 1.0],
        [3, 3, 2, -1.0], [3, 2, 1, 1.0], [3, 1, 0, 1.0], [3, 0, 0, 1.0],
    ],
    dtype=float,
)

SYMMETRIC_P2: np.ndarray = np.array(
    [
        [1, 2, 3, -1.0], [1, 1, 2, 2.0], [1, 0, 1, 2.0], [1, 0, 0, 1.0],
        [2, 1, 1, -1.0], [2, 0, 0, 2.0],
        [3, 3, 2, -1.0], [3, 2, 1, 2.0], [3, 1, 0, 2.0], [3, 0, 0, 1.0],
    ],
    dtype=float,
)

_H = 0.5
ASYMMETRIC_P1: np.ndarray = np.array(
    [
        [1, 1, 2, -1.0], [1, 0, 1, _H], [1, 0, 0, _H],
        [2, 1, 1, -1.0], [2, 0, 0, 1.0],
        [3, 2, 1, -1.0], [3, 1, 0, 1.0], [3, 0, 0, 1.0],
    ],
    dtype=float,
)

#: Rabiner-Juang type Ic. Note the zero-weight row: the third pattern reaches
#: (i, j) from (i, j-2) charging only the intermediate cell, so the endpoint
#: itself is free. A weight of zero is meaningful, not a placeholder.
TYPE_IC: np.ndarray = np.array(
    [
        [1, 2, 1, -1.0], [1, 1, 0, 1.0], [1, 0, 0, 1.0],
        [2, 1, 1, -1.0], [2, 0, 0, 1.0],
        [3, 1, 2, -1.0], [3, 0, 1, 1.0], [3, 0, 0, 0.0],
    ],
    dtype=float,
)

MORI2006: np.ndarray = np.array(
    [
        [1, 2, 1, -1.0], [1, 1, 0, 2.0], [1, 0, 0, 1.0],
        [2, 1, 1, -1.0], [2, 0, 0, 3.0],
        [3, 1, 2, -1.0], [3, 0, 1, 3.0], [3, 0, 0, 3.0],
    ],
    dtype=float,
)

#: Normalisation per pattern. ``"N+M"`` divides by query plus reference length,
#: ``"N"`` by the query's, ``"M"`` by the reference's. ``mori2006`` is the only
#: one of the six normalised by M, and getting it wrong leaves the path right
#: but every normalised distance -- hence every open-end endpoint -- wrong.
_NORM: Dict[int, str] = {
    id(SYMMETRIC_P1): "N+M", id(SYMMETRIC_P2): "N+M",
    id(ASYMMETRIC_P1): "N", id(ASYMMETRIC_P2): "N",
    id(TYPE_IC): "N", id(MORI2006): "M",
}

#: Public name -> table, for config and benchmark lookup.
PATTERNS: Dict[str, np.ndarray] = {
    "symmetricP1": SYMMETRIC_P1, "symmetricP2": SYMMETRIC_P2,
    "asymmetricP1": ASYMMETRIC_P1, "asymmetricP2": ASYMMETRIC_P2,
    "typeIc": TYPE_IC, "mori2006": MORI2006,
}


def norm_mode(step_pattern: np.ndarray) -> str:
    """The pattern's normalisation mode -- ``"N+M"``, ``"N"`` or ``"M"``."""
    return _NORM.get(id(step_pattern), "N")


class DTWAlignment:
    """Result of :func:`dtw`.

    Attributes
    ----------
    index1, index2 : np.ndarray
        Aligned index pairs into the query and the reference, 0-based and in
        alignment order.
    distance : float
        Accumulated (unnormalised) cost of the chosen path.
    normalized_distance : float
        ``distance`` divided by the pattern's normalisation factor.
    jmin : int
        Reference index the path ends on -- ``len(reference) - 1`` unless the
        alignment was open-ended and stopped short.
    """

    __slots__ = ("index1", "index2", "distance", "normalized_distance", "jmin")

    def __init__(self, index1, index2, distance, normalized_distance, jmin):
        self.index1 = index1
        self.index2 = index2
        self.distance = distance
        self.normalized_distance = normalized_distance
        self.jmin = jmin


def _pattern_rows(step_pattern: np.ndarray) -> List[Tuple[Tuple[int, int], List[Tuple[int, int, float]]]]:
    """Group a step-pattern table into ``(origin_offset, [(di, dj, w), ...])``."""
    out = []
    for p in np.unique(step_pattern[:, 0]):
        rows = step_pattern[step_pattern[:, 0] == p]
        opener = rows[rows[:, 3] == -1.0][0]
        adds = [(int(r[1]), int(r[2]), float(r[3])) for r in rows if r[3] != -1.0]
        out.append(((int(opener[1]), int(opener[2])), adds))
    return out


def _norm_factor(step_pattern: np.ndarray, n: int, m: int) -> float:
    mode = norm_mode(step_pattern)
    if mode == "N+M":
        return float(n + m)
    if mode == "M":
        return float(m)
    return float(n)


def dtw(
    query: np.ndarray,
    reference: np.ndarray,
    step_pattern: np.ndarray = SYMMETRIC_P1,
    open_end: bool = False,
) -> DTWAlignment:
    """Align ``query`` to ``reference`` under ``step_pattern``.

    Parameters
    ----------
    query, reference : np.ndarray
        1-D series. Local cost is the absolute difference, matching R
        ``dtw``'s default ``dist.method = "Euclidean"`` on scalars.
    step_pattern : np.ndarray, default :data:`SYMMETRIC_P1`
    open_end : bool, default False
        If True the path may end at any reference index, and the endpoint
        minimising the *normalised* cost is taken -- R's ``open.end = TRUE``.

    Raises
    ------
    MlsynthEstimationError
        If no admissible path exists (the reference is too short for the
        pattern's slope constraint). R signals this by raising as well, and
        callers are expected to catch it.
    """
    x = np.asarray(query, dtype=float)
    y = np.asarray(reference, dtype=float)
    n, m = x.size, y.size
    if n == 0 or m == 0:
        raise MlsynthEstimationError("DTW: empty query or reference.")

    local = np.abs(x[:, None] - y[None, :])
    patterns = _pattern_rows(step_pattern)

    cost = np.full((n, m), np.inf)
    ptr = np.full((n, m), -1, dtype=int)
    cost[0, 0] = local[0, 0]

    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                continue
            best, best_p = np.inf, -1
            for p_idx, ((di0, dj0), adds) in enumerate(patterns):
                pi, pj = i - di0, j - dj0
                if pi < 0 or pj < 0:
                    continue
                base = cost[pi, pj]
                if not np.isfinite(base):
                    continue
                acc = base
                ok = True
                for di, dj, w in adds:
                    ai, aj = i - di, j - dj
                    if ai < 0 or aj < 0:  # pragma: no cover - every add-cell
                        # offset is componentwise <= the pattern's origin
                        # offset, so the (pi, pj) bounds check above already
                        # guarantees these are in range for both patterns.
                        ok = False
                        break
                    acc += w * local[ai, aj]
                if ok and acc < best:
                    best, best_p = acc, p_idx
            cost[i, j] = best
            ptr[i, j] = best_p

    last = cost[n - 1]
    if open_end:
        nf = np.array([_norm_factor(step_pattern, n, j + 1) for j in range(m)])
        with np.errstate(invalid="ignore"):
            normed = last / nf
        if not np.isfinite(normed).any():  # pragma: no cover - the closed-end
            # cell is a member of `normed`, so an all-infinite row would mean
            # the recursion never reached the last query index at all, which
            # dtw() cannot produce for a non-empty query.
            raise MlsynthEstimationError("DTW: no admissible open-end path.")
        jmin = int(np.nanargmin(np.where(np.isfinite(normed), normed, np.inf)))
    else:
        jmin = m - 1
        if not np.isfinite(last[jmin]):
            raise MlsynthEstimationError("DTW: no admissible path to the end.")

    idx1, idx2 = _backtrack(ptr, patterns, n - 1, jmin)
    distance = float(cost[n - 1, jmin])
    return DTWAlignment(
        index1=np.asarray(idx1, dtype=int),
        index2=np.asarray(idx2, dtype=int),
        distance=distance,
        normalized_distance=distance / _norm_factor(step_pattern, n, jmin + 1),
        jmin=jmin,
    )


def _backtrack(ptr, patterns, i, j):
    """Walk the pointer matrix back to the origin, emitting the aligned pairs."""
    pairs = []
    while True:
        if i == 0 and j == 0:
            pairs.append((0, 0))
            break
        p_idx = ptr[i, j]
        if p_idx < 0:                       # pragma: no cover - guarded by dtw()
            raise MlsynthEstimationError("DTW: backtracking hit a dead cell.")
        (di0, dj0), adds = patterns[p_idx]
        # We walk backwards and reverse at the end, so emit nearest-to-(i, j)
        # FIRST -- the global reverse then puts each step's cells in path order.
        for di, dj, _w in sorted(adds, key=lambda r: (r[0], r[1])):
            pairs.append((i - di, j - dj))
        i, j = i - di0, j - dj0
    pairs.reverse()
    seen, out = set(), []
    for pr in pairs:                        # the openers can re-emit a cell
        if pr not in seen:
            seen.add(pr)
            out.append(pr)
    return [p[0] for p in out], [p[1] for p in out]


def warp(alignment: DTWAlignment, index_reference: bool = False) -> np.ndarray:
    """R ``dtw::warp`` -- the many-to-one map, averaging tied indices.

    With ``index_reference=False`` the result maps each reference index to a
    (possibly fractional) query index; with ``True``, the other way round.
    Output is 0-based, so add 1 before comparing against R.
    """
    if index_reference:
        src, dst = alignment.index1, alignment.index2
    else:
        src, dst = alignment.index2, alignment.index1
    return np.array([dst[src == j].mean() for j in range(int(src.max()) + 1)])


def ref_too_short(query, reference, step_pattern=SYMMETRIC_P1) -> bool:
    """R ``misc.R::RefTooShort`` -- is ``reference`` too short to span ``query``?

    Note the transposed call: R aligns the *reference* as the query, opens the
    end, and asks whether the warp reaches the last index of ``query``.
    """
    try:
        a = dtw(reference, query, step_pattern=step_pattern, open_end=True)
    except MlsynthEstimationError:
        return False
    if np.unique(a.index2).size == 1:
        return True
    wq = warp(a, index_reference=False) + 1.0        # compare on R's 1-based grid
    return bool(np.round(np.max(wq)) < len(query))
