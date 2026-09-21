"""Cluster-count diagnostics for fGRC: the Gap statistic and the paper's rule.

Implements the number-of-clusters half of Algorithm 1 of

    Yamamoto, M. & Hwang, H. (2017). "Dimension-Reduced Clustering of
    Functional Data via Subspace Separation." Journal of Classification
    34:294-326.

For a candidate ``K`` the method is fitted, the Gap statistic (Tibshirani,
Walther & Hastie 2001) is computed on the resulting component scores over a
wider grid of ``k``, and ``K`` is accepted when

.. math:: \\operatorname{argmax}_k \\operatorname{Gap}(k \\mid L_C, L_D, K) = K.

Among accepted candidates the largest Gap wins; when none is accepted the rule
relaxes to the ``t``-th largest rank, ``t = 2, 3, ...``.

What the rule can and cannot support
------------------------------------
This is a diagnostic, and no configuration wires it to ``fgrc_k``. A root-cause
analysis on the Basque panel established two things about it, both measured.

First, the acceptance is not independent evidence. The Gap is read on
``G A[:, :c1]``, the subspace fGRC chose *under the assumption of* ``K``
clusters by minimising within-cluster scatter in exactly those coordinates,
against reference clouds drawn inside that same fixed subspace, which never pay
the selection cost. On white noise at ``N=30`` and ``c2=1`` the fitted subspace
puts its Gap maximum at ``K`` in 10 to 11 replications out of 12; a random
two-dimensional projection of the same basis matrix does so in 0 to 2, which is
chance. The acceptance rate under the null is accordingly uncontrolled, and it
is worst in the small-``N``, ``c2=1`` regime panel data lives in: on
structureless panels at ``N=18`` with ``c2=1`` every replication accepted a
candidate.

Second, an argmax at an end of ``k_eval`` is a property of the grid. On the
17-unit Basque panel the maximum moved with the grid -- 1, 6, 8, 10, 12, 16 for
grids ``1..4`` through ``1..16`` -- so the relaxation ranks were the candidate
list sorted by distance from the edge. Candidates whose curve peaks at an edge
are therefore reported in :attr:`FGRCSelection.boundary` and are excluded from
both the accepted set and the relaxation; when that leaves nothing,
:attr:`FGRCSelection.selected_k` is ``None`` and the rule declines.

Tibshirani's one-standard-error reading of the same curves is reported in
:attr:`FGRCSelection.one_se`. It is the conservative instrument of the two:
across 18 structureless panels it answered ``k = 1`` on all 54 candidate
curves, and on the paper's planted design it answers 3.

What this does not do. Algorithm 1 is a cascade of three stages, and only the
third is here: the smoothing ``lambda`` by GCV and the penalties
``(rho1, rho2)`` by the Calinski-Harabasz pseudo-F index are taken from the
caller instead of searched.

Neither the authors' R package nor any other implementation ships this
procedure, so there is no reference to check a port against. It is validated
against the planted design of the paper's own Section 5, where the number of
clusters is known by construction, and against the null of that same design
with the separation removed, where it is known that there is nothing to find.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from ....exceptions import MlsynthConfigError, MlsynthDataError, MlsynthEstimationError
from .fgrc import basis_expand, optim_grc


@dataclass(frozen=True)
class FGRCSelection:
    """What the rule concluded, and the evidence for it.

    Attributes
    ----------
    selected_k : int or None
        The retained number of clusters, or ``None`` when every candidate's Gap
        curve peaked at an end of ``k_eval`` and the rule declined.
    confident : tuple of int
        Candidates accepted by the plain check, ``argmax_k Gap(k | K) = K``.
        Empty when the rule had to relax or declined. Read it against the
        module docstring: on a structureless panel this set is often non-empty.
    relaxation_level : int
        The ``t`` at which a non-empty rank class first appeared. ``1`` is the
        paper's primary rule, higher is a weaker conclusion, and ``0`` means
        the rule declined.
    boundary : tuple of int
        Candidates whose Gap curve attains its maximum at ``min(k_eval)`` or
        ``max(k_eval)``. These rank against the grid instead of the data, so
        they are excluded from ``confident`` and from the relaxation.
    one_se : dict of int -> int
        Tibshirani's one-standard-error choice read off each candidate's curve:
        the smallest ``k`` whose Gap is within one reference standard error of
        the next ``k``'s. The conservative reading of the same evidence.
    gaps : dict of int -> np.ndarray
        Gap curve over ``k_eval`` for each candidate ``K``.
    gap_se : dict of int -> np.ndarray
        Tibshirani's ``s_k`` for each curve.
    k_eval : tuple of int
        The wider grid the Gap statistic was evaluated on.
    losses : dict of int -> float
        fGRC objective at each candidate, for reference.
    """

    selected_k: Optional[int]
    confident: Tuple[int, ...]
    relaxation_level: int
    boundary: Tuple[int, ...]
    one_se: Dict[int, int]
    gaps: Dict[int, np.ndarray]
    gap_se: Dict[int, np.ndarray]
    k_eval: Tuple[int, ...]
    losses: Dict[int, float]


def _within_dispersion(F: np.ndarray, k: int, seed: int) -> float:
    """Pooled within-cluster sum of squares at ``k``, Tibshirani's ``W_k``."""
    if k == 1:
        return float(((F - F.mean(axis=0)) ** 2).sum())
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(F)
    return float(km.inertia_)


def gap_statistic(F: np.ndarray, k_values: Iterable[int], n_ref: int = 20,
                  seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Gap statistic of Tibshirani, Walther & Hastie (2001) over ``k_values``.

    The reference distribution is uniform over a box aligned to the principal
    components of ``F``, which is the authors' recommended variant: a box in
    the raw coordinates overstates the reference dispersion for data that lie
    on a rotated shape, and reports structure that is only orientation.

    Returns ``(gap, s)`` where ``s`` is the reference standard error inflated
    by ``sqrt(1 + 1/n_ref)``.
    """
    F = np.asarray(F, dtype=float)
    if F.ndim != 2 or F.shape[0] < 2:
        raise MlsynthConfigError(
            f"gap_statistic: need a 2-d score matrix with at least 2 rows; got {F.shape}.")
    ks = [int(k) for k in k_values]
    if not ks:
        raise MlsynthConfigError("gap_statistic: k_values is empty.")
    if min(ks) < 1:
        raise MlsynthConfigError(f"gap_statistic: k must be >= 1; got {min(ks)}.")
    if max(ks) >= F.shape[0]:
        raise MlsynthConfigError(
            f"gap_statistic: largest k ({max(ks)}) must be below the number of "
            f"units ({F.shape[0]}).")
    if not np.all(np.isfinite(F)):
        raise MlsynthDataError("gap_statistic: the score matrix has non-finite entries.")
    centred = F - F.mean(axis=0)
    if float(np.max(np.abs(centred))) <= 0.0:
        raise MlsynthDataError(
            "gap_statistic: the score matrix is constant, so no clustering is "
            "defined on it; check the fGRC subspace dimensions.")

    # PCA-aligned reference box
    _u, _s, Vt = np.linalg.svd(centred, full_matrices=False)
    Xp = centred @ Vt.T
    lo, hi = Xp.min(axis=0), Xp.max(axis=0)

    rng = np.random.default_rng(seed)
    obs = np.array([np.log(max(_within_dispersion(F, k, seed), 1e-300)) for k in ks])
    ref = np.empty((n_ref, len(ks)))
    for b in range(n_ref):
        Z = rng.uniform(lo, hi, size=Xp.shape) @ Vt + F.mean(axis=0)
        for j, k in enumerate(ks):
            ref[b, j] = np.log(max(_within_dispersion(Z, k, seed + b), 1e-300))
    gap = ref.mean(axis=0) - obs
    s = ref.std(axis=0, ddof=0) * np.sqrt(1.0 + 1.0 / n_ref)
    return gap, s


def one_se_k(gap: np.ndarray, gap_se: np.ndarray,
             k_values: Sequence[int]) -> int:
    """Tibshirani's one-standard-error choice: the smallest ``k`` whose Gap is
    within one reference standard error of the next ``k``'s.

    This is the rule the Gap statistic's authors propose in place of the plain
    argmax, which chases the tail on a cloud with a dense core and outliers.
    """
    ks = [int(k) for k in k_values]
    for j in range(len(ks) - 1):
        if gap[j] >= gap[j + 1] - gap_se[j + 1]:
            return ks[j]
    return ks[-1]


def select_fgrc_k(trajectories: np.ndarray, k_candidates: Sequence[int],
                  c1: int = 2, c2: int = 1, n_knots: Optional[int] = None,
                  order: int = 4, rho1: float = 1.0, rho2: float = 0.0,
                  k_eval: Optional[Sequence[int]] = None, n_ref: int = 20,
                  center: bool = True, n_random: int = 40, nstart: int = 40,
                  n_ite: int = 100, eps: float = 1e-5,
                  seed: int = 0) -> FGRCSelection:
    """Read the Gap-statistic evidence on the number of fGRC clusters.

    Read the module docstring before acting on ``selected_k`` or ``confident``:
    the check is not independent of the fit it is checking, and its acceptance
    rate under the null is uncontrolled.

    Parameters
    ----------
    trajectories : np.ndarray
        Panel of smooth series, shape ``(n_units, n_time)``.
    k_candidates : sequence of int
        The candidates to decide among, each ``>= 2``.
    k_eval : sequence of int, optional
        The wider grid the Gap statistic is read over, the paper's
        ``Theta-tilde(K)``. Defaults to ``1 .. max(k_candidates) + 2``. It must
        extend past the largest candidate, or the check cannot reject one.
    """
    X = np.asarray(trajectories, dtype=float)
    if X.ndim != 2:
        raise MlsynthConfigError(
            f"select_fgrc_k: trajectories must be 2-d (units x time); got {X.shape}.")
    cands = [int(k) for k in k_candidates]
    if not cands:
        raise MlsynthConfigError("select_fgrc_k: k_candidates is empty.")
    if min(cands) < 2:
        raise MlsynthConfigError(
            f"select_fgrc_k: every candidate must be >= 2; got {min(cands)}.")
    if c1 < 1:
        raise MlsynthConfigError(f"select_fgrc_k: c1 must be >= 1; got {c1}.")
    if c2 < 0:
        raise MlsynthConfigError(f"select_fgrc_k: c2 must be >= 0; got {c2}.")

    grid = (tuple(int(k) for k in k_eval) if k_eval is not None
            else tuple(range(1, max(cands) + 3)))
    grid = tuple(k for k in grid if k < X.shape[0])
    if not grid or max(grid) <= max(cands):
        raise MlsynthConfigError(
            f"select_fgrc_k: the evaluation grid must extend past the largest "
            f"candidate ({max(cands)}) and stay below the number of units "
            f"({X.shape[0]}); with {X.shape[0]} units there is no such grid, so "
            f"the check has no power to reject a candidate. Lower k_candidates "
            f"or use a larger panel.")

    if center:
        X = X - X.mean(axis=0)
    knots = int(n_knots) if n_knots is not None else max(4, X.shape[1] // 2 - 2)
    G = basis_expand(X, knots, order)
    if c1 + c2 > G.shape[1]:
        raise MlsynthConfigError(
            f"select_fgrc_k: c1+c2 ({c1 + c2}) exceeds the number of B-spline "
            f"basis functions ({G.shape[1]}); reduce c1/c2 or raise n_knots.")

    gaps: Dict[int, np.ndarray] = {}
    ses: Dict[int, np.ndarray] = {}
    losses: Dict[int, float] = {}
    for K in cands:
        try:
            _labels, A, loss = optim_grc(G, c1, c2, K, rho1, rho2,
                                         n_random, nstart, n_ite, eps, seed)
        except MlsynthEstimationError:
            continue                      # a candidate that cannot be fitted is not a candidate
        F_hat = G @ A[:, :c1]             # scores on the cluster subspace
        gaps[K], ses[K] = gap_statistic(F_hat, grid, n_ref=n_ref, seed=seed)
        losses[K] = float(loss)
    if not gaps:
        raise MlsynthEstimationError(
            "select_fgrc_k: no candidate could be fitted; reduce k_candidates or "
            "check the panel for degenerate trajectories.")

    ones = {K: one_se_k(gaps[K], ses[K], grid) for K in gaps}
    edges = {min(grid), max(grid)}
    boundary = tuple(sorted(K for K, g in gaps.items()
                            if grid[int(np.argmax(g))] in edges))
    eligible = [K for K in gaps if K not in boundary]
    common = dict(boundary=boundary, one_se=ones, gaps=gaps, gap_se=ses,
                  k_eval=grid, losses=losses)
    if not eligible:
        return FGRCSelection(selected_k=None, confident=(), relaxation_level=0,
                             **common)

    # rank of each candidate's own k within its Gap curve: t = 1 means argmax
    rank_of = {K: list(np.argsort(-gaps[K])).index(grid.index(K)) + 1
               for K in eligible}
    for t in range(1, len(grid) + 1):
        confident = tuple(sorted(K for K in eligible if rank_of[K] == t))
        if confident:
            best = max(confident, key=lambda K: float(gaps[K][grid.index(K)]))
            return FGRCSelection(selected_k=int(best),
                                 confident=confident if t == 1 else (),
                                 relaxation_level=t, **common)
    raise AssertionError(                              # pragma: no cover
        "unreachable: every eligible candidate has a rank in 1..len(k_eval)")
