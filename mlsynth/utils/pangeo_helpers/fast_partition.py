"""OSD-style fast partition for PANGEO (Shaw 2025 analog).

The exact PANGEO design (``parallelism.enumerate_candidate_pairs`` + the
set-partitioning MIP in ``mip.solve_partition``) considers every admissible
supergeo subset of size ``2..2Q`` -- ``O(n^{2Q})`` candidates -- then solves an
NP-hard exact-cover MIP. That is the runtime bottleneck.

This module borrows the two-stage idea of Shaw (2025), "Optimized Supergeo
Design": replace the exhaustive candidate enumeration with a clustering step,
keeping the exact per-group split.

Setup (Stage 1) -- ``group_units``
    Each unit's pre-period trajectory is *level-removed* (subtract its own
    temporal mean, leaving the shape), embedded with PCA, hierarchically
    ordered, and chunked into size-bounded groups (each splittable into two
    halves of size ``<= Q``). Level removal is what makes the grouping target
    PANGEO's parallel-trends notion -- units that move in parallel (same shape,
    any level) land together -- rather than Supergeo's level matching. The
    chunking guarantees an exact cover with every group of size ``2..2Q``.

Optimization (Stage 2) -- ``fast_partition``
    For each group, the unchanged :func:`parallelism.best_split` returns the
    optimal level-removed split (the free-:math:`\\delta` DiD gap variance
    ``min_split sum_t [(Ybar_A - Ybar_B)_t - delta]^2``). A handful of candidate
    groupings are generated (varying linkage / a small embedding perturbation,
    as in OSD) and the design with the smallest total score is returned, in the
    same ``{members, score, side_a, side_b}`` contract as the exact path -- with
    no subset enumeration and no MIP.

This is a heuristic: it restricts pairs to clustered groups (a ``(1+eps)``-style
trade under trajectory-cluster separability), so its total score is ``>=`` the
exact optimum but tight when the parallel structure is clear. The exact path
remains available as the oracle / default.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ...exceptions import MlsynthEstimationError
from .parallelism import best_split

_LINKAGES = ("ward", "average", "complete")


def _size_bounded_chunks(
    order: np.ndarray, max_size: int, *, block: Optional[int] = None
) -> List[np.ndarray]:
    """Chunk an ordering into groups of size ``2..block`` (no singletons).

    Walks the (similarity) ordering taking blocks of ``block`` (default
    ``2*max_size``); if a block would leave a lone trailing unit
    (``remaining == block + 1``), it emits two near-equal blocks instead.
    Every emitted group has size ``<= block <= 2*max_size`` and is therefore
    splittable into two halves each ``<= max_size``. A ``block < 2*max_size``
    forces more (smaller) groups when ``min_pairs`` requires it.
    """
    g = 2 * max_size if block is None else block
    n = len(order)
    chunks: List[np.ndarray] = []
    i = 0
    while i < n:
        rem = n - i
        if rem == g + 1:                       # avoid a trailing singleton
            head = (g + 1) - (g // 2)          # == Q+1 when block == 2Q
            chunks.append(order[i:i + head])
            chunks.append(order[i + head:])
            break
        take = min(g, rem)
        chunks.append(order[i:i + take])
        i += take
    return chunks


def group_units(
    Ypre: np.ndarray,
    unit_indices: np.ndarray,
    max_size: int,
    *,
    seed: int = 0,
    linkage: str = "ward",
    perturb: float = 0.0,
    embedding_dim: int = 8,
    min_groups: int = 1,
) -> List[np.ndarray]:
    """Partition ``unit_indices`` into size-bounded, parallel-shape groups.

    Returns a list of arrays of unit indices (into ``Ypre``'s rows), each of
    size ``2..2*max_size`` and together covering ``unit_indices`` exactly once.
    At least ``min_groups`` groups are produced (the chunk block size is capped
    so the cover yields enough supergeo pairs); raises if that is infeasible.
    """
    unit_indices = np.asarray(unit_indices, dtype=int)
    n = len(unit_indices)
    if n < 2:
        raise MlsynthEstimationError(
            "PANGEO fast partition needs >= 2 units in the arm."
        )
    if max_size == 1 and n % 2 == 1:
        raise MlsynthEstimationError(
            f"PANGEO fast partition: an exact cover by size-2 supergeo pairs "
            f"needs an even arm size (got n={n} with max_supergeo_size=1)."
        )
    # Effective chunk block: <= 2Q, but small enough to yield >= min_groups
    # groups (each supergeo pair needs >= 2 units, so n >= 2*min_groups).
    block = 2 * max_size
    if min_groups > 1:
        block = min(block, n // min_groups)
        if block < 2:
            raise MlsynthEstimationError(
                f"PANGEO fast partition: cannot form min_pairs={min_groups} "
                f"supergeo pairs from {n} units (need n >= 2*min_pairs)."
            )

    Y = np.asarray(Ypre, dtype=float)[unit_indices]
    # Level removal: keep the trajectory *shape* (parallel-trends notion).
    shape = Y - Y.mean(axis=1, keepdims=True)

    # PCA embedding (denoise; mirrors OSD). Skip when there is nothing to reduce.
    d = min(embedding_dim, n - 1, shape.shape[1])
    if d >= 1 and shape.shape[1] > d:
        from sklearn.decomposition import PCA

        emb = PCA(n_components=d, random_state=seed).fit_transform(shape)
    else:
        emb = shape

    if perturb > 0.0:
        rng = np.random.default_rng(seed)
        emb = emb + rng.standard_normal(emb.shape) * (perturb * emb.std(axis=0))

    from scipy.cluster.hierarchy import leaves_list, linkage as _linkage

    order = leaves_list(_linkage(emb, method=linkage))   # similar units adjacent
    chunks = _size_bounded_chunks(
        np.asarray(order, dtype=int), max_size, block=block)
    return [unit_indices[c] for c in chunks]


def fast_partition(
    unit_indices: np.ndarray,
    Ypre: np.ndarray,
    max_size: int,
    *,
    objective: str = "ss_res",
    weights: Optional[np.ndarray] = None,
    cov: Optional[np.ndarray] = None,
    cov_scales: Optional[np.ndarray] = None,
    cov_weights: Optional[np.ndarray] = None,
    unit_weights: Optional[np.ndarray] = None,
    n_candidates: int = 5,
    min_pairs: int = 1,
    seed: int = 0,
) -> List[dict]:
    """OSD-style fast supergeo-pair partition (drop-in for the enumerate+MIP path).

    Generates ``n_candidates`` size-bounded groupings (Stage 1), splits each
    group with :func:`parallelism.best_split` (Stage 2), and returns the design
    with the smallest total score -- a list of ``{members, score, side_a,
    side_b}`` dicts, the same contract as ``solve_partition``. At least
    ``min_pairs`` supergeo pairs are produced (raises if infeasible).
    """
    unit_indices = np.asarray(unit_indices, dtype=int)
    bs_kwargs = dict(objective=objective, weights=weights, cov=cov,
                     cov_scales=cov_scales, cov_weights=cov_weights,
                     unit_weights=unit_weights)

    best_design: Optional[List[dict]] = None
    best_total = np.inf
    for c in range(max(1, n_candidates)):
        groups = group_units(
            Ypre, unit_indices, max_size, seed=seed + c,
            linkage=_LINKAGES[c % len(_LINKAGES)], perturb=0.0 if c == 0 else 0.05,
            min_groups=min_pairs)
        design: List[dict] = []
        total = 0.0
        ok = True
        for g in groups:
            score, side_a, side_b = best_split(g, Ypre, max_size, **bs_kwargs)
            if not np.isfinite(score):
                ok = False
                break
            design.append({
                "members": np.asarray(g, dtype=int),
                "score": float(score),
                "side_a": np.asarray(side_a, dtype=int),
                "side_b": np.asarray(side_b, dtype=int),
            })
            total += score
        if ok and total < best_total:
            best_total, best_design = total, design

    if best_design is None:  # pragma: no cover - every group is splittable by size
        raise MlsynthEstimationError(
            "PANGEO fast partition: no feasible grouping produced a valid design."
        )
    return best_design
