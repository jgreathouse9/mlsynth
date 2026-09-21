"""Model selection for fGRC: the Gap-statistic confidence rule.

Implements the number-of-clusters half of Algorithm 1 of

    Yamamoto, M. & Hwang, H. (2017). "Dimension-Reduced Clustering of
    Functional Data via Subspace Separation." Journal of Classification
    34:294-326.

The rule is a self-consistency check and not a maximisation. For a candidate
``K`` the method is fitted, the Gap statistic (Tibshirani, Walther & Hastie
2001) is computed on the resulting component scores over a wider grid of ``k``,
and ``K`` counts as confident when

.. math:: \\operatorname{argmax}_k \\operatorname{Gap}(k \\mid L_C, L_D, K) = K,

that is, the subspace fitted under the assumption of ``K`` clusters
independently looks like it holds ``K`` clusters. Among confident candidates
the largest Gap wins. When none is confident the rule relaxes to the ``t``-th
largest argmax, ``t = 2, 3, ...``, which is what the paper prescribes for an
empty confident set.

What this does not do. Algorithm 1 is a cascade of three stages, and only the
third is here: the smoothing ``lambda`` by GCV and the penalties
``(rho1, rho2)`` by the Calinski-Harabasz pseudo-F index are taken from the
caller instead of searched. A selector over ``K`` at fixed penalties answers
the question the number of clusters poses; the penalty search is a separate
piece of the same algorithm.

Neither the authors' R package nor any other implementation ships this
procedure, so there is no reference to check a port against. It is validated
instead against the planted design of the paper's own Section 5, where the
number of clusters is known by construction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from ....exceptions import MlsynthConfigError, MlsynthDataError, MlsynthEstimationError
from .fgrc import basis_expand, optim_grc


@dataclass(frozen=True)
class FGRCSelection:
    """What the selector concluded, and the evidence for it.

    Attributes
    ----------
    selected_k : int
        The retained number of clusters.
    confident : tuple of int
        Candidates satisfying the consistency check at ``relaxation_level``.
        Empty when the rule had to relax past the plain argmax, which is
        itself the finding: no candidate's subspace looked like it held that
        many clusters.
    relaxation_level : int
        The ``t`` at which a non-empty confident set first appeared. ``1`` is
        the paper's primary rule; anything higher is a weaker conclusion.
    gaps : dict of int -> np.ndarray
        Gap curve over ``k_eval`` for each candidate ``K``.
    gap_se : dict of int -> np.ndarray
        Tibshirani's ``s_k`` for each curve, carried so a caller can apply the
        one-standard-error rule instead of the argmax if it prefers.
    k_eval : tuple of int
        The wider grid the Gap statistic was evaluated on.
    losses : dict of int -> float
        fGRC objective at each candidate, for reference.
    """

    selected_k: int
    confident: Tuple[int, ...]
    relaxation_level: int
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


def select_fgrc_k(trajectories: np.ndarray, k_candidates: Sequence[int],
                  c1: int = 2, c2: int = 1, n_knots: Optional[int] = None,
                  order: int = 4, rho1: float = 1.0, rho2: float = 0.0,
                  k_eval: Optional[Sequence[int]] = None, n_ref: int = 20,
                  center: bool = True, n_random: int = 40, nstart: int = 40,
                  n_ite: int = 100, eps: float = 1e-5,
                  seed: int = 0) -> FGRCSelection:
    """Choose the number of fGRC clusters by the Gap-statistic confidence rule.

    Parameters
    ----------
    trajectories : np.ndarray
        Panel of smooth series, shape ``(n_units, n_time)``.
    k_candidates : sequence of int
        The candidates to decide among, each ``>= 2``.
    k_eval : sequence of int, optional
        The wider grid the Gap statistic is read over, the paper's
        ``Theta-tilde(K)``. Defaults to ``1 .. max(k_candidates) + 2``, which
        must extend past the candidates for the check to have the power to
        reject one.
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
    if max(grid) >= X.shape[0]:
        grid = tuple(k for k in grid if k < X.shape[0])
    if not grid:
        raise MlsynthConfigError(
            "select_fgrc_k: no evaluation grid fits inside the panel; too few units.")

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

    # rank of each candidate's own k within its Gap curve: t = 1 means argmax
    order_of = {K: list(np.argsort(-g)) for K, g in gaps.items()}
    rank_of = {K: (order_of[K].index(grid.index(K)) + 1 if K in grid else len(grid) + 1)
               for K in gaps}
    for t in range(1, len(grid) + 2):
        confident = tuple(sorted(K for K in gaps if rank_of[K] == t))
        if confident:
            best = max(confident, key=lambda K: float(gaps[K][grid.index(K)]))
            return FGRCSelection(selected_k=int(best),
                                 confident=confident if t == 1 else (),
                                 relaxation_level=t, gaps=gaps, gap_se=ses,
                                 k_eval=grid, losses=losses)
    best = min(gaps, key=lambda K: losses[K])          # pragma: no cover - unreachable
    return FGRCSelection(int(best), (), len(grid) + 2, gaps, ses, grid, losses)
