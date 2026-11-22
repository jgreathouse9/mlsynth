"""PANGEO's pairs (``max_supergeo_size=1``) on the exact path: a closed form where one exists, the blossom
matcher elsewhere.

With Q = 1 every supergeo is one unit, so a design of an arm is a perfect matching of its units, and the
exact-cover MIP of :mod:`.mip` is the minimum-weight perfect matching problem on the complete graph of the arm's
units (Edmonds, "Maximum matching and a polyhedron with 0,1-vertices", J. Res. NBS 69B (1965) 125-130). Two cases
follow, closed form first (CLAUDE.md invariant 8):

* The supergeo paper's objective (Chen, Doudchenko, Jiang, Stein and Ying, "Supergeo design: generalized matching
  for geographic experiments", arXiv:2301.12044, eq. 3), the squared difference of the two units' totals over the
  estimation window, with no covariate penalty. The cost of a pair is a convex function of the difference of two
  scalars, and the pairing of neighbours in sorted order of the totals is optimal: if a <= b <= c <= d, then
  (b - a)^2 + (d - c)^2 <= (c - a)^2 + (d - b)^2 and <= (d - a)^2 + (c - b)^2, so any pairing with two crossing or
  nested pairs is no cheaper after uncrossing them, and a pairing with neither is the sorted one. This is
  :func:`sorted_pairing`, O(n log n).
* Every other Q = 1 cost: the trajectory objectives (``ss_res``, ``r2``, ``weighted``) and any objective with a
  covariate penalty. The cost of a pair is not a function of one scalar per unit, there is no closed form, and the
  matching is solved by Edmonds' blossom algorithm (:func:`.matching.min_weight_matching`), which returns the dual
  certificate that proves the pairing optimal.

An arm with an odd number of units has no perfect matching; :func:`solve_pairs` raises for it.
"""
from __future__ import annotations

import time
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ...exceptions import MlsynthEstimationError
from .matching import check_certificate, min_weight_matching


def sorted_pairing(z: Sequence[float]) -> List[Tuple[int, int]]:
    """The optimal pairing of positions ``0..len(z)-1`` under the cost ``(z_a - z_b)^2``: neighbours in sorted
    order of ``z``, equal values ordered by position (the exchange argument in the module docstring).

    Returns the pairs as ``(i, j)`` with ``i < j``, in sorted order of ``z``. Raises ValueError for an odd count,
    which has no perfect pairing.
    """
    z = np.asarray(z, dtype=float)
    if len(z) % 2:
        raise ValueError(f"a perfect pairing needs an even count, not {len(z)}")
    order = np.lexsort((np.arange(len(z)), z))
    return [tuple(sorted((int(order[k]), int(order[k + 1])))) for k in range(0, len(z), 2)]


def _check_arm(n: int, min_pairs: int) -> None:
    """An odd arm has no cover by pairs, and an arm of n units gives n / 2 pairs (an empty arm none)."""
    if n % 2:
        raise MlsynthEstimationError(
            f"PANGEO: pairs (max_supergeo_size=1) need an even number of units in each arm; this arm has {n}, "
            "which has no exact cover by pairs.")
    if n // 2 < min_pairs:
        raise MlsynthEstimationError(
            f"PANGEO: an arm of {n} units gives {n // 2} pairs, fewer than min_pairs={min_pairs}.")


def solve_pairs(candidates: Optional[List[dict]], unit_indices: np.ndarray, *,
                totals: Optional[np.ndarray] = None, min_pairs: int = 1):
    """The optimal design of pairs of one arm, as ``(chosen, diagnostics)``.

    ``chosen`` is a list of ``{"members", "score", "side_a", "side_b"}`` dicts, the contract of
    :func:`.mip.solve_partition`. With ``totals`` (one per unit of ``unit_indices``, the units' totals over the
    estimation window) the pairs are :func:`sorted_pairing`'s and each score is the squared difference of the
    totals; ``candidates`` is not read. Without ``totals``, ``candidates`` must hold every pair of the arm's units
    with its score, and the pairs are the blossom matcher's, with its certificate on the diagnostics.

    Raises MlsynthEstimationError for an odd arm, an arm too small for ``min_pairs`` (an empty one included),
    and a candidate list missing a pair's score.
    """
    unit_indices = np.asarray(unit_indices)
    n = len(unit_indices)
    _check_arm(n, min_pairs)
    t0 = time.perf_counter()
    if totals is not None:
        totals = np.asarray(totals, dtype=float)
        pairs = sorted_pairing(totals)
        chosen = []
        for a, b in pairs:
            d = float(totals[a]) - float(totals[b])
            chosen.append({"members": np.array([unit_indices[a], unit_indices[b]]), "score": d * d,
                           "side_a": np.array([unit_indices[a]], dtype=int),
                           "side_b": np.array([unit_indices[b]], dtype=int)})
        extra = {"path": "closed_form_sorted_pairs", "n_candidate_pairs": 0,
                 "solver_name": "sorted pairing (closed form)"}
    else:
        pos = {int(u): k for k, u in enumerate(unit_indices)}
        W = np.full((n, n), np.inf)
        np.fill_diagonal(W, 0.0)
        by_pair = {}
        for c in candidates:
            i, j = (pos[int(u)] for u in c["members"])
            W[i, j] = W[j, i] = float(c["score"])
            by_pair[(min(i, j), max(i, j))] = c
        if not np.isfinite(W).all():
            missing = int((~np.isfinite(W)).sum() // 2)
            raise MlsynthEstimationError(
                f"PANGEO: the pairs path needs every pair's score; {missing} of the arm's pairs have none.")
        result = min_weight_matching(W)
        chosen = [by_pair[p] for p in result["pairs"]]
        extra = {"path": "blossom_matching", "n_candidate_pairs": len(candidates),
                 "solver_name": "Edmonds blossom matching",
                 "certificate_violation": float(check_certificate(W, result)),
                 "certificate_tolerance": float(result["tolerance"]),
                 "n_blossoms": len(result["blossoms"])}
    diagnostics = {**extra, "objective_value": float(sum(c["score"] for c in chosen)), "status": "optimal",
                   "proved_optimal": True, "solve_seconds": float(time.perf_counter() - t0),
                   "n_selected_pairs": len(chosen), "backend": None}
    return chosen, diagnostics
