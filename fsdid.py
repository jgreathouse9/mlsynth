"""
Supergeo_solver.py
==================
Supergeo Design: Generalized Matching for Geographic Experiments

    Chen, Doudchenko, Jiang, Stein, Ying (2023)
    arXiv:2301.12044

Every design decision in this module traces directly to a section of the
paper.  Cross-references are written inline as [§N] or [Eq. N].

Extension: demeaned-R² score
----------------------------
The paper collapses the pre-period panel to a scalar Z_g = Σ_t Y0[t,g] and
matches on that.  For panel data with many pre-periods this discards all
temporal structure and cannot distinguish two geos with the same total but
diverging trends.

This implementation replaces the paper's (Z_{G+} − Z_{G−})² score with a
demeaned-R² score that directly measures pre-trend co-movement:

    1.  For a candidate bipartition (G+, G−), aggregate the full time series
        across each side: ts+ = Σ_{g∈G+} Y0[:,g],  ts− = Σ_{g∈G−} Y0[:,g].
    2.  Demean both series by their own means, removing level differences.
        Two series separated by a constant gap become a perfect match.
    3.  Regress demeaned ts− on demeaned ts+ (no intercept — both already
        mean-zero) and compute 1 − R² as the score.

Score is bounded in [0, 1]:
    0  → perfect parallel trends (ideal pair)
    1  → orthogonal movements in deviation space (worst pair)

This score is the MIP objective.  The MIP, incidence matrix, heuristics,
power analysis, and empirical estimator are all unchanged — only the scalar
passed into the MIP objective differs.

Note: the analytical variance formula Var[θ̂] = total_loss / B² [Eq. 3]
was derived for the paper's scalar score.  Under the demeaned-R² score the
power analysis numbers are heuristic rather than exact, but the design
quality and parallel-trends justification are strictly superior for panels
with many pre-periods.

Architecture
------------

  _aa_kernel(diffs, n_sims, seed)
      Numba-jitted xorshift64 Monte Carlo kernel for the AA placebo test.
      Streams through n_sims sign-flip draws without allocating an
      (n_sims × K) matrix.  [Appendix D]

  _r2_demeaned(ts_plus, ts_minus)
      Demeaned-R² between two aggregated time series.  Score ∈ [0, 1].
      A constant level gap → score = 0.  Orthogonal movements → score = 1.

  _score_and_split(indices, Y0, rng)
      Enumerate all bipartitions of a candidate group, compute
      _r2_demeaned for each, and return the minimum-score split.
      Symmetry reduction halves enumeration; stored orientation is
      randomised to avoid systematic treatment-side bias.

  _candidates_cluster(n, Y0, min_size, max_size, n_clusters, n_runs, noise_scale, rng)
      Cluster heuristic: embed geos in PCA trajectory space, run n_runs
      perturbed KMeans instances, and generate candidates within each
      cluster.  Geos that move similarly cluster together, producing a
      much more homogeneous candidate pool than random partitioning.

  _build_incidence_matrix(n, candidates)
      Sparse N × M CSC matrix A where A[i,j] = 1 iff unit i ∈ candidate j.
      Passed directly to CVXPY so the exact-cover constraint is a single
      matrix expression rather than N separate scalar constraints.  [§3.2]

  mde / power_at_effect / power_curve
      Heuristic power analysis using Student-t with K−1 df.

  SupergeoSolver
      Main class.  generate_candidates() → solve() → summary() /
      run_aa_test() / mde() / power_at_effect() / power_curve().

  empirical_estimator
      Half-synthetic evaluation harness.  [§4]

Public API
----------
SupergeoSolver(unit_names, Y0, min_size=2, max_size=4, kappa=1, seed=42)
    .generate_candidates(heuristic, **kwargs)
    .solve(solver, budget)         → stores and returns design DataFrame
    .summary()
    .run_aa_test(n_sims, alpha)   → dict
    .mde(budget, alpha, power)    → float
    .power_at_effect(delta, ...)  → float
    .power_curve(deltas, ...)     → pd.DataFrame

empirical_estimator(design, unit_names, R_test, S_test,
                    theta_true, r_spend, n_iter, rng)  → dict
"""

from __future__ import annotations

import warnings
from itertools import combinations
from typing import Literal

import cvxpy as cp
import numpy as np
import pandas as pd
from numba import njit
from scipy.interpolate import make_interp_spline
from scipy.sparse import csc_matrix
from scipy.stats import t as t_dist
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# =============================================================================
# AA PLACEBO SIMULATION KERNEL  [Appendix D]
# =============================================================================

@njit
def _aa_kernel(diffs: np.ndarray, n_sims: int, seed: int) -> np.ndarray:
    """
    Monte Carlo sign-flip placebo test kernel.

    Under the sharp null of zero treatment effects [Appendix D], the
    treatment assignment A_k ∈ {±1} is arbitrary.  We repeatedly re-draw
    signs and accumulate Σ_k A_k · diff_k to build the null distribution.

    RNG: xorshift64 — a 64-bit non-linear shift-register generator with
    period 2^64 − 1 and full-period bit-level uniformity.  The sign of each
    flip is taken from the most-significant bit of the state, which has the
    same period as the full generator.  This is strictly better than a
    multiplicative LCG whose low-order bits cycle rapidly.

    Memory: O(K + n_sims).  No (n_sims × K) matrix is ever allocated.

    Parameters
    ----------
    diffs : (K,) float64   Z_{G_k+} − Z_{G_k−} for each pair k
    n_sims : int            Number of Monte Carlo draws
    seed : int            Non-zero seed for the xorshift64 state

    Returns
    -------
    out : (n_sims,) float64   Simulated null statistics Σ_k A_k · diff_k
    """
    K = diffs.shape[0]
    out = np.empty(n_sims, dtype=np.float64)
    state = np.uint64(seed) if seed != 0 else np.uint64(1)

    for s in range(n_sims):
        total = 0.0
        for k in range(K):
            # xorshift64 update
            state ^= state << np.uint64(13)
            state ^= state >> np.uint64(7)
            state ^= state << np.uint64(17)
            # MSB → sign: 0 = +1, 1 = −1
            sign = 1.0 if (state >> np.uint64(63)) == np.uint64(0) else -1.0
            total += sign * diffs[k]
        out[s] = total

    return out


# =============================================================================
# DEMEANED-R² SCORE  (extension — replaces paper's scalar score)
# =============================================================================

def _r2_demeaned(ts_plus: np.ndarray, ts_minus: np.ndarray) -> float:
    """
    1 − R² between two aggregated time series after removing level differences.

    Parameters
    ----------
    ts_plus : (T,) float64   Aggregated treatment-side time series
    ts_minus : (T,) float64   Aggregated control-side time series

    Returns
    -------
    float   1 − R²  (lower is better)

    Motivation
    ----------
    The paper's score (Z_{G+} − Z_{G−})² compares scalar totals.  For a
    panel with many pre-periods this discards temporal structure and treats
    two geos with the same aggregate but diverging trends as a perfect match.

    This score asks instead: do the two basket time series *move together*,
    regardless of their absolute levels?  This is precisely the pre-trend
    co-movement that parallel trends requires.

    Construction
    ------------
    1. Demean each series by its own mean.  A constant level gap between
       ts_plus and ts_minus becomes a zero gap after demeaning — constant
       differences are absorbed, not penalised.
    2. Regress demeaned ts_minus on demeaned ts_plus with no intercept
       (both are already mean-zero, so an intercept would be redundant).
    3. Return 1 − R² = SS_res / SS_tot.

    Bounds
    ------
    Score ∈ [0, 1]:
        0  →  perfect co-movement (parallel trends, ideal pair)
        1  →  orthogonal deviations (worst possible pair)
    """
    tp = ts_plus  - ts_plus.mean()
    tc = ts_minus - ts_minus.mean()

    ss_tot = float(tc @ tc)
    if ss_tot < 1e-12:
        # Constant series: perfectly predictable regardless of ts_plus
        return 0.0

    # OLS beta without intercept on mean-zero series: β = (tp'tc) / (tp'tp)
    ss_tp = float(tp @ tp)
    if ss_tp < 1e-12:
        # ts_plus is constant after demeaning — can't explain any variation
        return 1.0

    beta   = float(tp @ tc) / ss_tp
    resid  = tc - beta * tp
    ss_res = float(resid @ resid)

    return min(1.0, ss_res / ss_tot)   # clamp numerical noise above 1


# =============================================================================
# SCORE AND SPLIT  [Eq. 7 / §3.1  — extended to panel data]
# =============================================================================

def _score_and_split(
    indices: tuple[int, ...],
    Y0: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, list[int], list[int]]:
    """
    Compute the demeaned-R² score and optimal (G+, G−) split for a group G.

    For each candidate bipartition (G+, G−) of the group:
        ts+ = Σ_{g∈G+} Y0[:, g]     (T,) aggregated treatment series
        ts− = Σ_{g∈G−} Y0[:, g]     (T,) aggregated control series
        score = _r2_demeaned(ts+, ts−)

    The bipartition minimising the score is returned.  Score ∈ [0, 1],
    with 0 meaning perfect parallel trends.

    Parameters
    ----------
    indices : tuple[int]      Geo indices forming the candidate group G
    Y0 : (T, N) float64  Full pre-period panel
    rng : Generator       For the random orientation step only

    Returns
    -------
    score : float           Best (lowest) 1 − R² across all bipartitions
    g_plus : list[int]       Treatment-side geo indices
    g_minus : list[int]       Control-side geo indices

    Symmetry reduction
    ------------------
    _r2_demeaned(ts+, ts−) = _r2_demeaned(ts−, ts+) because the regression
    is symmetric up to sign of beta, and SS_res is identical under sign flip.
    We therefore anchor index 0 in the plus set, enumerating each unordered
    bipartition exactly once.  For max_size = 4 this is at most
        C(3,0) + C(3,1) + C(3,2) = 7 evaluations per candidate.

    Random orientation
    ------------------
    After finding the optimal bipartition, plus/minus labels are assigned by
    an unbiased coin flip, so neither side is systematically favoured across
    the candidate pool.
    """
    idx    = list(indices)
    n      = len(idx)
    Y_grp  = Y0[:, idx]   # (T, |G|) — local slice

    best_score       = float("inf")
    best_plus_local: list[int]  = []
    best_minus_local: list[int] = []

    # Anchor element 0 in plus; vary elements 1…n-1
    for r in range(0, n - 1):
        for extra in combinations(range(1, n), r):
            plus_local  = (0,) + extra
            minus_local = [i for i in range(n) if i not in plus_local]

            ts_plus  = Y_grp[:, list(plus_local)].sum(axis=1)   # (T,)
            ts_minus = Y_grp[:, minus_local].sum(axis=1)         # (T,)

            score = _r2_demeaned(ts_plus, ts_minus)

            if score < best_score:
                best_score       = score
                best_plus_local  = list(plus_local)
                best_minus_local = minus_local

    # Random orientation: swap plus/minus with probability 0.5
    if rng.random() < 0.5:
        best_plus_local, best_minus_local = best_minus_local, best_plus_local

    return (
        best_score,
        [idx[i] for i in best_plus_local],
        [idx[i] for i in best_minus_local],
    )


# =============================================================================
# CANDIDATE BUILDERS  [§3.2 / Appendix E]
# =============================================================================

def _make_candidate(
    combo: tuple[int, ...],
    Y0: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    """Package a (score, split) pair into a candidate dict."""
    score, g_plus, g_minus = _score_and_split(combo, Y0, rng)
    return {"indices": combo, "score": score, "split": (g_plus, g_minus)}


def _candidates_exhaustive(
    n: int,
    Y0: np.ndarray,
    min_size: int,
    max_size: int,
    rng: np.random.Generator,
) -> list[dict]:
    """
    All C(N, size) subsets for each size in [min_size, max_size].

    Exact but only feasible for N ≲ 50.  N = 210, max_size = 4 would
    produce ~200 M candidates — use a heuristic for production scale.
    """
    candidates = []
    for size in range(min_size, max_size + 1):
        for combo in combinations(range(n), size):
            candidates.append(_make_candidate(combo, Y0, rng))
    return candidates


def _candidates_partition(
    n: int,
    Y0: np.ndarray,
    min_size: int,
    max_size: int,
    n_partitions: int,
    rng: np.random.Generator,
) -> list[dict]:
    """
    Partition heuristic [Appendix E].

    Randomly shuffles all N geos then splits into n_partitions contiguous
    buckets via np.array_split.  Only within-bucket subsets are considered.

    np.array_split on a fresh permutation is used (not a stride slice) so
    bucket membership is independent of any latent ordering in the geo
    indices.

    The heuristic is randomised — running multiple instances with different
    seeds and taking the design with the lowest total loss is the paper's
    recommended usage [Appendix E].
    """
    perm    = rng.permutation(n)
    buckets = [b.tolist() for b in np.array_split(perm, n_partitions)]

    candidates = []
    for bucket in buckets:
        for size in range(min_size, max_size + 1):
            for combo in combinations(bucket, size):
                candidates.append(_make_candidate(combo, Y0, rng))
    return candidates


def _candidates_per_geo(
    n: int,
    Y0: np.ndarray,
    min_size: int,
    max_size: int,
    beta: int,
    alpha: float,
    rng: np.random.Generator,
) -> list[dict]:
    """
    Per-geo heuristic [Appendix E].

    Ranks geos by descending pre-period aggregate Σ_t Y0[t,g] (the largest
    geos are hardest to match in standard pairs design).  For each of the
    beta largest geos:
        • Score every subset of size [min_size, max_size] that contains it.
        • Retain only the alpha-best fraction (lowest scores).
    Remaining small geos, which already match well 1-to-1, keep only
    pairwise (size-2) candidates.

    Duplicates across pools are removed by frozenset-key deduplication.
    """
    Z_agg  = Y0.mean(axis=0)                          # (N,) — used for ranking only
    order  = np.argsort(-Z_agg)
    large_geos = set(order[:beta].tolist())
    small_geos = [g for g in range(n) if g not in large_geos]

    candidates: list[dict] = []

    # Small geos: standard 1-to-1 pairs only
    for combo in combinations(small_geos, 2):
        candidates.append(_make_candidate(combo, Y0, rng))

    # Large geos: alpha-best pool of all containing subsets
    for g in large_geos:
        others = [i for i in range(n) if i != g]
        pool: list[dict] = []
        for size in range(min_size, max_size + 1):
            for rest in combinations(others, size - 1):
                combo = tuple(sorted((g,) + rest))
                pool.append(_make_candidate(combo, Y0, rng))

        pool.sort(key=lambda c: c["score"])
        keep = max(1, int(np.ceil(alpha * len(pool))))
        candidates.extend(pool[:keep])

    # Deduplicate by frozenset of indices
    seen:   set[frozenset] = set()
    unique: list[dict]     = []
    for c in candidates:
        key = frozenset(c["indices"])
        if key not in seen:
            seen.add(key)
            unique.append(c)

    return unique



# =============================================================================
# CLUSTER HEURISTIC  — FPCA embedding + silhouette-optimal KMeans
# =============================================================================

def _spectral_rank(singular_values: np.ndarray, energy_threshold: float = 0.95) -> int:
    """
    Number of singular values whose squared sum meets `energy_threshold` of
    total spectral energy.

    Parameters
    ----------
    singular_values : (K,) float64   Singular values in descending order.
    energy_threshold : float           Fraction of energy to retain, ∈ [0, 1].

    Returns
    -------
    int   Smallest rank r such that Σ_{i<r} σ_i² / Σ_i σ_i² ≥ threshold.
    """
    if energy_threshold == 1.0:
        return len(singular_values)

    sq  = singular_values ** 2
    tot = sq.sum()
    if tot == 0.0:
        return 0

    cumulative = sq.cumsum() / tot
    hits = np.where(cumulative >= energy_threshold)[0]
    if hits.size == 0:
        return len(singular_values)   # numerical edge case
    return int(hits[0]) + 1


def _determine_optimal_clusters(X: np.ndarray) -> int:
    """
    Silhouette-based optimal cluster count for an (N, F) feature matrix.

    Evaluates KMeans for k ∈ [2, min(10, N−1)] and returns the k with the
    highest average silhouette score.  Returns 1 when N < 2 or the range
    is empty.

    Parameters
    ----------
    X : (N, F) float64   Feature matrix, one row per geo.

    Returns
    -------
    int   Optimal number of clusters.
    """
    n = X.shape[0]
    if n < 2:
        return 1

    max_k = min(10, n - 1)
    if max_k < 2:
        return 1

    best_score = -np.inf
    best_k     = 2

    for k in range(2, max_k + 1):
        labels = KMeans(
            n_clusters=k, init="k-means++", n_init="auto", random_state=0
        ).fit_predict(X)

        if len(np.unique(labels)) < 2:
            continue   # degenerate clustering — skip

        score = silhouette_score(X, labels)
        if score > best_score:
            best_score = score
            best_k     = k

    return best_k


def _fpca_features(Y0: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Functional PCA embedding of geos for trajectory-aware clustering.

    Implements the pipeline from the submitted fpca() function:

    1.  B-spline smoothing of each geo's time series.
        Uses degree-3 splines evaluated at equally-spaced normalised time
        points.  Falls back to direct PCA if T ≤ 3 (too few points for a
        cubic spline).
    2.  PCA on the smoothed (N, T) matrix.
    3.  Spectral rank truncation at 95% energy, giving K components.
    4.  Standardisation of the K-dimensional scores (zero mean, unit std).
    5.  Silhouette-optimal cluster count from the standardised scores.

    The standardised scores are the embedding passed to KMeans in
    `_candidates_cluster`; the cluster count is the data-driven default k₀.

    Parameters
    ----------
    Y0 : (T, N) float64   Pre-period outcome panel (time × geos).

    Returns
    -------
    features : (N, K) float64   Standardised FPC scores.
    k_opt : int               Silhouette-optimal cluster count.
    """
    T, N = Y0.shape
    X    = Y0.T                            # (N, T) — geos as rows, as fpca() expects

    spline_degree = 3
    t_norm = np.linspace(0, 1, T)

    if T <= spline_degree:
        # Fallback: direct PCA when too few time points for a cubic spline
        _, S, Vt = np.linalg.svd(X, full_matrices=False)
        K        = _spectral_rank(S)
        scores   = X @ Vt[:K].T           # (N, K)
    else:
        # B-spline smooth each geo's series then PCA on smoothed matrix
        spl     = make_interp_spline(t_norm, X.T, k=spline_degree)  # fits X.T: (T, N)
        X_smooth = spl(t_norm).T           # evaluate → (T, N), transpose → (N, T)

        _, S, Vt = np.linalg.svd(X_smooth, full_matrices=False)
        K        = _spectral_rank(S)
        scores   = X_smooth @ Vt[:K].T    # (N, K) FPC scores

    if K == 0 or scores.shape[1] == 0:
        # Degenerate case: return raw aggregate as single feature
        scores = X.sum(axis=1, keepdims=True).astype(float)

    # Standardise: zero mean, unit std per component (avoids div-by-zero)
    mu  = scores.mean(axis=0)
    std = scores.std(axis=0)
    std = np.where(std == 0.0, 1.0, std)
    features = (scores - mu) / std        # (N, K)

    k_opt = _determine_optimal_clusters(features)
    return features, k_opt


def _candidates_cluster(
    n: int,
    Y0: np.ndarray,
    min_size: int,
    max_size: int,
    n_clusters: int | None,
    n_runs: int,
    noise_scale: float,
    rng: np.random.Generator,
) -> list[dict]:
    """
    Cluster heuristic: embed geos via FPCA, partition by trajectory, then
    generate candidates within each cluster.

    Parameters
    ----------
    n : int            Number of geos
    Y0 : (T, N)         Pre-period panel
    min_size : int            Minimum supergeo pair size
    max_size : int            Maximum supergeo pair size
    n_clusters : int or None    Base cluster count k₀.  When None, the
                                 silhouette-optimal k from FPCA is used.
    n_runs : int            Number of perturbed KMeans runs
    noise_scale : float          Std of Gaussian noise added to FPCA scores
                                 before each KMeans run
    rng : Generator

    Returns
    -------
    list[dict]   Deduplicated candidates, each with keys:
                     "indices", "score", "split"

    Motivation
    ----------
    Random partitioning (the paper's partition heuristic) is agnostic to
    which geos are actually similar.  Clustering on B-spline-smoothed FPC
    scores means geos that genuinely co-move are grouped together before any
    subsets are enumerated, so the candidates reaching the MIP are drawn from
    a much more homogeneous pool.

    Method
    ------
    1.  Embed geos as standardised FPC scores via `_fpca_features`.
        This runs B-spline smoothing → PCA → spectral-rank truncation →
        standardisation, and returns the silhouette-optimal k₀.
    2.  Use k₀ as the base cluster count (or `n_clusters` if supplied).
    3.  Run `n_runs` perturbed KMeans instances.  Each run adds mild
        Gaussian noise to the features and jitters k by ±20%, diversifying
        cluster membership across runs without abandoning trajectory
        structure.
    4.  Collect all within-cluster subsets of size [min_size, max_size].
        Score each via `_make_candidate` → `_score_and_split` →
        `_r2_demeaned`.
    5.  Deduplicate by frozenset of indices; keep first occurrence (score
        is deterministic given the combo and the RNG state at call time).
    """
    features, k_opt = _fpca_features(Y0)                 # (N, K), int
    k_base = n_clusters if n_clusters is not None else k_opt

    candidates: list[dict] = []
    seen: set[frozenset]   = set()

    for _ in range(n_runs):
        # Perturb features to diversify clusterings across runs
        perturbed = features + noise_scale * rng.standard_normal(features.shape)

        # Jitter k by ±20%
        k = int(round(k_base * (1.0 + rng.uniform(-0.2, 0.2))))
        k = max(2, min(n - 1, k))

        labels = KMeans(
            n_clusters=k,
            n_init=10,
            random_state=int(rng.integers(1, 2 ** 31)),
        ).fit_predict(perturbed)

        # Group geo indices by cluster label
        clusters: dict[int, list[int]] = {}
        for geo_idx, label in enumerate(labels):
            clusters.setdefault(int(label), []).append(geo_idx)

        # Generate within-cluster candidates
        for cluster in clusters.values():
            if len(cluster) < min_size:
                # Too small for any valid pair — these geos remain uncovered
                # until a larger run places them in a bigger cluster.  The
                # uncovered-geo guard in solve() catches any that slip through.
                continue

            for size in range(min_size, min(max_size, len(cluster)) + 1):
                for combo in combinations(cluster, size):
                    key = frozenset(combo)
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(_make_candidate(combo, Y0, rng))

    return candidates

def _build_incidence_matrix(n: int, candidates: list[dict]) -> csc_matrix:
    """
    Build the N × M unit-by-candidate incidence matrix A in CSC format.

        A[i, j] = 1  iff unit i ∈ candidate j

    The exact-cover constraint [§3.2] is then A @ x = 1_N, expressed as a
    single matrix constraint in CVXPY.  CVXPY accepts scipy sparse matrices
    directly, so the solver sees a compact constraint structure rather than
    N separate scalar inequalities.
    """
    rows, cols = [], []
    for j, c in enumerate(candidates):
        for i in c["indices"]:
            rows.append(i)
            cols.append(j)

    data = np.ones(len(rows), dtype=np.float64)
    return csc_matrix((data, (rows, cols)), shape=(n, len(candidates)))


# =============================================================================
# POWER ANALYSIS  [Eq. 3 / §4]
# =============================================================================

def _design_se(total_loss: float, budget: float) -> float:
    """
    Standard error of θ̂ from the paper's variance formula [Eq. 3]:

        Var[θ̂] = (1/B²) · Σ_k (Z_{G_k+} − Z_{G_k−})²
                = total_loss / B²

    The design's MIP objective *is* the numerator of the estimator's
    variance.  Minimising loss at design time is equivalent to minimising
    the standard error of the estimator.
    """
    return float(np.sqrt(total_loss)) / budget


def mde(
    total_loss: float,
    budget: float,
    n_pairs: int,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """
    Minimum detectable effect for a supergeo design.

    Uses Student-t with df = n_pairs − 1.  The paper explicitly recommends
    the t-approximation (Section 5.2 of Chen & Au 2022, cited in Appendix D)
    and notes it requires at least a few pairs to be reliable — the normal
    approximation is inappropriate when K is small.

    Parameters
    ----------
    total_loss : float   design["Score"].sum()
    budget : float   Total incremental spend B [Assumption 2]
    n_pairs : int     Number of supergeo pairs K
    alpha : float   Two-sided significance level
    power : float   Target power 1 − β

    Returns
    -------
    float   MDE in the same units as θ (iROAS)
    """
    df      = n_pairs - 1
    se      = _design_se(total_loss, budget)
    t_crit  = t_dist.ppf(1.0 - alpha / 2.0, df=df)
    t_power = t_dist.ppf(power, df=df)
    return (t_crit + t_power) * se


def power_at_effect(
    delta: float,
    total_loss: float,
    budget: float,
    n_pairs: int,
    alpha: float = 0.05,
) -> float:
    """
    Power for a specific true effect size delta.

    Parameters
    ----------
    delta : float   True iROAS deviation from null
    total_loss : float
    budget : float
    n_pairs : int
    alpha : float   Two-sided significance level

    Returns
    -------
    float   Power in [0, 1]
    """
    df     = n_pairs - 1
    se     = _design_se(total_loss, budget)
    t_crit = t_dist.ppf(1.0 - alpha / 2.0, df=df)
    return float(t_dist.cdf(abs(delta) / se - t_crit, df=df))


def power_curve(
    deltas: np.ndarray,
    total_loss: float,
    budget: float,
    n_pairs: int,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Power as a function of effect size.

    The primary use case is the kappa-sweep: solve the MIP for a range of
    kappa values, compute a power curve per design, and overlay them to find
    the kappa that achieves target power at the desired MDE.

    Parameters
    ----------
    deltas : (D,) array-like   Effect sizes to evaluate
    total_loss : float
    budget : float
    n_pairs : int
    alpha : float

    Returns
    -------
    pd.DataFrame with columns ["delta", "power"]
    """
    deltas = np.asarray(deltas)
    powers = [
        power_at_effect(d, total_loss, budget, n_pairs, alpha)
        for d in deltas
    ]
    return pd.DataFrame({"delta": deltas, "power": powers})


# =============================================================================
# SOLVER CLASS
# =============================================================================

class SupergeoSolver:
    """
    Supergeo experimental design solver (Chen et al., 2023).

    Finds a partition of N geos into K supergeo pairs {(G_k+, G_k−)} that
    minimises the total matching loss

        loss = Σ_k score(G_k)    where score ∈ [0,1] is 1 − R² of the
                                  demeaned aggregated time series

    subject to:
        • Every geo belongs to exactly one supergeo pair  (exact cover)
        • Each supergeo pair has total size in [min_size, max_size]
        • At least kappa supergeo pairs are formed         [§3]

    The score directly measures pre-trend co-movement: a score of 0 means
    the two basket time series are perfectly parallel (constant gap allowed);
    a score of 1 means orthogonal deviations.  Minimising total loss
    constructs the design most consistent with parallel trends.

    Parameters
    ----------
    unit_names : list[str]
        Human-readable geo labels, length N.
    Y0 : (T, N) ndarray
        Pre-period outcome matrix.  The full panel is used for matching —
        no temporal information is discarded.
    min_size : int
        Minimum total geos per supergeo pair (ℓ in the paper).  Default 2,
        which recovers standard matched pairs as a special case.
    max_size : int
        Maximum total geos per supergeo pair (u).  Paper uses 4.
    kappa : int
        Minimum number of supergeo pairs (κ).  Default 1.  Larger values
        improve inferential robustness [Appendix D] at the cost of worse
        individual match quality.
    seed : int
        Master RNG seed.  Controls split orientation randomisation and all
        heuristic randomness.  Fix for reproducibility.

    Typical workflow
    ----------------
    >>> solver = SupergeoSolver(unit_names, Y0, min_size=2, max_size=4,
    ...                         kappa=5, seed=42)
    >>> solver.generate_candidates("per_geo", beta=30, alpha=0.10)
    >>> design = solver.solve(budget=1e6)
    >>> solver.summary()
    >>> print(solver.mde())
    >>> aa = solver.run_aa_test(n_sims=10_000)
    """

    def __init__(
        self,
        unit_names: list[str],
        Y0: np.ndarray,
        min_size: int = 2,
        max_size: int = 4,
        kappa: int = 1,
        seed: int = 42,
    ) -> None:
        if Y0.ndim != 2:
            raise ValueError("Y0 must be a 2-D array of shape (T, N).")
        if Y0.shape[1] != len(unit_names):
            raise ValueError(
                f"Y0 has {Y0.shape[1]} columns but unit_names has "
                f"{len(unit_names)} entries."
            )
        if min_size < 2:
            raise ValueError("min_size must be >= 2 (need at least one geo per side).")
        if max_size < min_size:
            raise ValueError("max_size must be >= min_size.")
        if kappa < 1:
            raise ValueError("kappa must be >= 1.")

        self.unit_names: list[str]           = list(unit_names)
        self.Y0:         np.ndarray          = np.asarray(Y0, dtype=float)
        self.n:          int                 = len(unit_names)
        self.min_size:   int                 = min_size
        self.max_size:   int                 = max_size
        self.kappa:      int                 = kappa
        self._rng:       np.random.Generator = np.random.default_rng(seed)

        # Pre-period aggregate — used only for geo ranking in per_geo heuristic
        # and for the AA test statistic.  Matching itself uses the full panel.
        self._Z_agg: np.ndarray = self.Y0.sum(axis=0)  # shape (N,)

        # State set by generate_candidates() and solve()
        self.candidates: list[dict]          = []
        self.design_:    pd.DataFrame | None = None
        self._budget:    float | None        = None

    # ------------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------------

    def generate_candidates(
        self,
        heuristic: Literal["exhaustive", "partition", "per_geo", "cluster"] = "exhaustive",
        *,
        n_partitions: int = 10,
        beta: int | None = None,
        alpha: float = 0.10,
        n_clusters: int | None = None,        n_runs: int = 10,
        noise_scale: float = 0.05,
    ) -> None:
        """
        Populate self.candidates with feasible supergeo groups.

        Parameters
        ----------
        heuristic : {"exhaustive", "partition", "per_geo", "cluster"}

            "exhaustive"
                All C(N, size) subsets for each size in [min_size, max_size].
                Exact; practical only for N ≲ 50.

            "partition"  [Appendix E — Partition heuristic]
                Randomly partition geos into n_partitions contiguous buckets
                and consider only within-bucket subsets.  Run multiple
                instances with different seeds; keep the design with the
                lowest total loss.

            "per_geo"  [Appendix E — Per-geo heuristic]
                The beta largest geos (hardest to match individually) receive
                a curated alpha-best candidate pool.  Small geos keep only
                pairwise candidates.  Combines well with large N.

            "cluster"
                Embed geos via Functional PCA (B-spline smoothing → PCA →
                spectral-rank truncation → standardisation) then cluster in
                that embedding space.  The silhouette method selects k
                automatically from the FPCA scores; n_clusters overrides
                this if supplied.  Run n_runs perturbed KMeans instances to
                diversify coverage.  Recommended for large N with rich
                pre-period panels — trajectory structure is preserved more
                faithfully than random partitioning.

        n_partitions : int
            Number of buckets for the partition heuristic.
        beta : int or None
            Number of large geos for the per-geo heuristic (default N // 2).
        alpha : float
            Candidate retention fraction for the per-geo heuristic.
        n_clusters : int or None
            Base KMeans cluster count for the cluster heuristic.  When None
            (default), the silhouette-optimal k from FPCA is used
            automatically.  Supply an integer to override.
        n_runs : int
            Number of perturbed KMeans runs for the cluster heuristic.
        noise_scale : float
            Std of Gaussian noise added to FPCA scores before each KMeans
            run.  Controls clustering diversity across runs.
        """
        if heuristic == "exhaustive":
            self.candidates = _candidates_exhaustive(
                self.n, self.Y0, self.min_size, self.max_size, self._rng,
            )
        elif heuristic == "partition":
            self.candidates = _candidates_partition(
                self.n, self.Y0, self.min_size, self.max_size,
                n_partitions=n_partitions, rng=self._rng,
            )
        elif heuristic == "per_geo":
            _beta = beta if beta is not None else self.n // 2
            self.candidates = _candidates_per_geo(
                self.n, self.Y0, self.min_size, self.max_size,
                beta=_beta, alpha=alpha, rng=self._rng,
            )
        elif heuristic == "cluster":
            self.candidates = _candidates_cluster(
                self.n, self.Y0, self.min_size, self.max_size,
                n_clusters=n_clusters, n_runs=n_runs, noise_scale=noise_scale,
                rng=self._rng,
            )
        else:
            raise ValueError(
                f"Unknown heuristic '{heuristic}'. "
                "Choose 'exhaustive', 'partition', 'per_geo', or 'cluster'."
            )

        if not self.candidates:
            raise RuntimeError(
                "No candidates generated.  Check min_size / max_size and "
                "heuristic parameters."
            )

    # ------------------------------------------------------------------
    # MIP solver  [§3.2]
    # ------------------------------------------------------------------

    def solve(
        self,
        solver: str = cp.CBC,
        budget: float | None = None,
    ) -> pd.DataFrame:
        """
        Solve the covering MIP and store/return the supergeo design.

        Parameters
        ----------
        solver : str
            CVXPY solver constant supporting MIP.  Default cp.CBC (always
            available).  Use cp.SCIP for better performance if installed.
        budget : float or None
            Total incremental spend B [Assumption 2].  Required for power
            analysis; may also be supplied later via self._budget.

        Returns
        -------
        pd.DataFrame (also stored as self.design_) with columns:
            Pair_ID    int
            Treatment  list[str]   geo names on the treatment side
            Control    list[str]   geo names on the control side
            Score      float       (Z_{G+} − Z_{G−})² for this pair

        MIP formulation [§3.2]
        ----------------------
        Variables : x ∈ {0,1}^M        (M = number of candidates)
        Objective : minimise scores @ x                        [Eq. 7]
        Constraints:
            A @ x = 1_N   (exact cover — one supergeo pair per geo)
            sum(x) ≥ κ    (minimum pairs)

        A is the N × M sparse incidence matrix built by
        _build_incidence_matrix.  Passing it as a single matrix constraint
        rather than N separate scalar constraints eliminates O(N·M) Python
        overhead before the solver starts.

        Status handling
        ---------------
        "optimal"             — accepted silently.
        "optimal_inaccurate"  — accepted with a RuntimeWarning.  This means
                                the solver hit a resource limit; the solution
                                may be suboptimal.  At Samsung scale this
                                warrants investigation before committing the
                                design to a live experiment.
        Anything else         — raises RuntimeError.
        """
        if not self.candidates:
            raise RuntimeError("Call generate_candidates() before solve().")

        if budget is not None:
            self._budget = float(budget)

        m      = len(self.candidates)
        scores = np.array([c["score"] for c in self.candidates], dtype=float)

        x = cp.Variable(m, boolean=True)

        # Sparse exact-cover constraint: A @ x = 1_N
        A = _build_incidence_matrix(self.n, self.candidates)

        # Guard: every geo must appear in at least one candidate
        covered   = np.asarray(A.sum(axis=1)).ravel()
        uncovered = [self.unit_names[i] for i in range(self.n) if covered[i] == 0]
        if uncovered:
            raise RuntimeError(
                f"Geos not covered by any candidate: {uncovered}.  "
                "Increase max_size or relax the heuristic."
            )

        constraints = [
            A @ x == np.ones(self.n, dtype=float),   # exact cover [§3.2]
            cp.sum(x) >= self.kappa,                  # minimum pairs [§3]
        ]

        prob = cp.Problem(cp.Minimize(scores @ x), constraints)
        prob.solve(solver=solver)

        if prob.status == "optimal_inaccurate":
            warnings.warn(
                "Solver returned 'optimal_inaccurate' — solution may be "
                "suboptimal (resource limit reached).  Verify before "
                "committing to a live experiment.",
                RuntimeWarning,
                stacklevel=2,
            )
        elif prob.status not in ("optimal", "feasible"):
            raise RuntimeError(
                f"MIP solver returned status '{prob.status}'.  "
                "Try a different heuristic, increase n_partitions / alpha, "
                "or reduce kappa."
            )

        self.design_ = self._format_solution(x.value)
        return self.design_

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _format_solution(self, x_val: np.ndarray) -> pd.DataFrame:
        """Convert MIP solution vector to a labelled DataFrame."""
        chosen = np.where(x_val > 0.5)[0]
        rows = []
        for pair_id, j in enumerate(chosen, start=1):
            c = self.candidates[j]
            g_plus, g_minus = c["split"]
            rows.append({
                "Pair_ID":   pair_id,
                "Treatment": [self.unit_names[i] for i in g_plus],
                "Control":   [self.unit_names[i] for i in g_minus],
                "Score":     float(c["score"]),
            })
        return pd.DataFrame(rows, columns=["Pair_ID", "Treatment", "Control", "Score"])

    def _require_design(self) -> pd.DataFrame:
        if self.design_ is None:
            raise RuntimeError("Call solve() first.")
        return self.design_

    def _require_budget(self, budget: float | None) -> float:
        b = budget if budget is not None else self._budget
        if b is None:
            raise ValueError("Supply budget= to solve() or to this method.")
        return float(b)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> None:
        """Print a human-readable summary of the solved design."""
        df         = self._require_design()
        total_loss = df["Score"].sum()
        sizes      = df["Treatment"].apply(len) + df["Control"].apply(len)

        print("=" * 62)
        print("  Supergeo Design Summary")
        print("=" * 62)
        print(f"  Geos            : {self.n}")
        print(f"  Supergeo pairs  : {len(df)}")
        print(f"  Size range      : [{self.min_size}, {self.max_size}]")
        print(f"  kappa           : {self.kappa}")
        print(f"  Total loss      : {total_loss:.6f}  (sum of 1-R² scores; 0=perfect)")
        if self._budget is not None:
            se = _design_se(total_loss, self._budget)
            print(f"  SE(theta_hat)   : {se:.6f}  [budget = {self._budget}, heuristic]")
        print(f"  Pair size dist  : {dict(sizes.value_counts().sort_index())}")
        print("=" * 62)
        print(df.to_string(index=False))

    # ------------------------------------------------------------------
    # AA placebo test  [Appendix D]
    # ------------------------------------------------------------------

    def run_aa_test(
        self,
        n_sims: int = 10_000,
        alpha: float = 0.05,
        seed: int = 1,
    ) -> dict:
        """
        Permutation-based AA placebo test [Appendix D].

        Under the sharp null of zero treatment effects, the sign of each
        pair's assignment A_k ∈ {±1} is irrelevant.  We repeatedly re-draw
        signs and compute the null distribution of

            T = Σ_k A_k · (Z_{G_k+} − Z_{G_k−})

        The observed statistic uses the design's stored split orientation.
        The p-value is the fraction of null draws at least as extreme
        (two-sided).

        The xorshift64 Numba kernel streams through the simulation without
        allocating an (n_sims × K) matrix.

        Parameters
        ----------
        n_sims : int    Monte Carlo draws
        alpha : float  Two-sided significance level
        seed : int    Kernel RNG seed (independent of the solver RNG)

        Returns
        -------
        dict:
            "observed_stat"  float         Σ_k (Z_{G_k+} − Z_{G_k−})
            "null_dist"      (n_sims,)     Simulated null statistics
            "p_value"        float         Fraction of nulls >= |observed|
            "reject"         bool          p_value < alpha
            "ci_lower"       float         alpha/2 quantile of null dist
            "ci_upper"       float         1 - alpha/2 quantile of null dist
        """
        df = self._require_design()

        name_to_idx = {name: i for i, name in enumerate(self.unit_names)}

        # For each pair, the AA test statistic contribution is the difference
        # between the demeaned aggregated time series of each side, summarised
        # as the signed L2 norm of the residual.  Under the sharp null,
        # swapping the sign of this contribution (i.e. flipping treatment
        # assignment) is equally likely.
        diffs = np.empty(len(df), dtype=np.float64)
        for k, (_, row) in enumerate(df.iterrows()):
            t_idx = [name_to_idx[g] for g in row["Treatment"]]
            c_idx = [name_to_idx[g] for g in row["Control"]]
            ts_t  = self.Y0[:, t_idx].sum(axis=1)
            ts_c  = self.Y0[:, c_idx].sum(axis=1)
            # Demeaned difference: captures trajectory gap after removing levels
            dm_t  = ts_t - ts_t.mean()
            dm_c  = ts_c - ts_c.mean()
            # Signed norm: positive when treatment side has larger deviations
            diffs[k] = float(np.sum(dm_t - dm_c))

        observed  = float(diffs.sum())
        null_dist = _aa_kernel(diffs, n_sims, seed)

        p_value = float(np.mean(np.abs(null_dist) >= abs(observed)))
        ci_lo   = float(np.quantile(null_dist, alpha / 2.0))
        ci_hi   = float(np.quantile(null_dist, 1.0 - alpha / 2.0))

        return {
            "observed_stat": observed,
            "null_dist":     null_dist,
            "p_value":       p_value,
            "reject":        p_value < alpha,
            "ci_lower":      ci_lo,
            "ci_upper":      ci_hi,
        }

    # ------------------------------------------------------------------
    # Power analysis  [Eq. 3 / §4]
    # ------------------------------------------------------------------

    def mde(
        self,
        budget: float | None = None,
        alpha: float = 0.05,
        power: float = 0.80,
    ) -> float:
        """
        Minimum detectable effect for the solved design.

        Derived from Var[θ̂] = total_loss / B² [Eq. 3], using Student-t
        with K − 1 degrees of freedom.  The paper warns that the
        t-approximation requires at least a few pairs [Appendix D].

        Parameters
        ----------
        budget : float or None   B (can also be supplied to solve())
        alpha : float           Two-sided significance level
        power : float           Target power 1 − β

        Returns
        -------
        float   MDE in iROAS units
        """
        df = self._require_design()
        B  = self._require_budget(budget)
        return mde(
            total_loss=df["Score"].sum(),
            budget=B,
            n_pairs=len(df),
            alpha=alpha,
            power=power,
        )

    def power_at_effect(
        self,
        delta: float,
        budget: float | None = None,
        alpha: float = 0.05,
    ) -> float:
        """
        Power for a specific hypothesised effect size delta.

        Parameters
        ----------
        delta : float   True iROAS deviation from null
        budget : float   B
        alpha : float   Two-sided significance level

        Returns
        -------
        float   Power in [0, 1]
        """
        df = self._require_design()
        B  = self._require_budget(budget)
        return power_at_effect(
            delta=delta,
            total_loss=df["Score"].sum(),
            budget=B,
            n_pairs=len(df),
            alpha=alpha,
        )

    def power_curve(
        self,
        deltas: np.ndarray,
        budget: float | None = None,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """
        Power as a function of effect size.

        Use with a kappa sweep to find the minimum number of pairs that
        achieves target power at the desired MDE.

        Parameters
        ----------
        deltas : array-like   Effect sizes to evaluate
        budget : float        B
        alpha : float        Two-sided significance level

        Returns
        -------
        pd.DataFrame with columns ["delta", "power"]
        """
        df = self._require_design()
        B  = self._require_budget(budget)
        return power_curve(
            deltas=np.asarray(deltas),
            total_loss=df["Score"].sum(),
            budget=B,
            n_pairs=len(df),
            alpha=alpha,
        )


# =============================================================================
# HALF-SYNTHETIC EVALUATION HARNESS  [§4]
# =============================================================================

def empirical_estimator(
    design: pd.DataFrame,
    unit_names: list[str],
    R_test: np.ndarray,
    S_test: np.ndarray,
    theta_true: float,
    r_spend: float,
    n_iter: int = 10_000,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Half-synthetic evaluation of a supergeo design [§4].

    At each iteration a random treatment assignment A_k ∈ {±1} is drawn for
    every pair [§2].  Treatment injection follows Assumption 4 (proportional
    spend heavy-up):

        Control geo g : R_g unchanged,  S_g unchanged
        Treated geo g : R_g -> R_g + theta * r * S_g
                        S_g -> S_g * (1 + r)

    The iROAS is estimated with the empirical estimator [Eq. 1]:

        theta_hat = (sum_{g in T} R_g - sum_{g in C} R_g)
                  / (sum_{g in T} S_g - sum_{g in C} S_g)

    Parameters
    ----------
    design : pd.DataFrame   Output of SupergeoSolver.solve()
    unit_names : list[str]      Geo labels in the same order as Y0 columns
    R_test : (N,) array     Test-phase response per geo (no treatment)
    S_test : (N,) array     Test-phase spend per geo
    theta_true : float          Injected iROAS (ground truth)
    r_spend : float          Heavy-up fraction r [Assumption 4]
    n_iter : int            Monte Carlo iterations
    rng : Generator      Defaults to seed 0

    Returns
    -------
    dict:
        "rmse"       float
        "bias"       float
        "estimates"  (n_iter,) array
    """
    if rng is None:
        rng = np.random.default_rng(0)

    R_test = np.asarray(R_test, dtype=float)
    S_test = np.asarray(S_test, dtype=float)

    name_to_idx = {name: i for i, name in enumerate(unit_names)}
    pairs = [
        (
            [name_to_idx[g] for g in row["Treatment"]],
            [name_to_idx[g] for g in row["Control"]],
        )
        for _, row in design.iterrows()
    ]

    estimates: list[float] = []
    for _ in range(n_iter):
        signs = rng.choice([-1, 1], size=len(pairs))

        sum_R_t = sum_R_c = 0.0
        sum_S_t = sum_S_c = 0.0

        for (g_plus, g_minus), A_k in zip(pairs, signs):
            treated = np.asarray(g_plus if A_k == 1 else g_minus)
            control = np.asarray(g_minus if A_k == 1 else g_plus)

            # Treatment injection [Assumption 4]
            sum_R_t += (R_test[treated] + theta_true * r_spend * S_test[treated]).sum()
            sum_S_t += (S_test[treated] * (1.0 + r_spend)).sum()
            sum_R_c += R_test[control].sum()
            sum_S_c += S_test[control].sum()

        denom = sum_S_t - sum_S_c
        if abs(denom) < 1e-12:
            continue
        estimates.append((sum_R_t - sum_R_c) / denom)   # Eq. 1

    arr = np.array(estimates)
    return {
        "rmse":      float(np.sqrt(np.mean((arr - theta_true) ** 2))),
        "bias":      float(np.mean(arr) - theta_true),
        "estimates": arr,
    }


# =============================================================================
# SMOKE TEST / DEMO
# =============================================================================

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    rng  = np.random.default_rng(0)
    N, T = 20, 70    # panel setting: 70 pre-treatment periods, 20 units
    B    = 50.0

    unit_names = [f"DMA_{i:02d}" for i in range(N)]

    # Simulate correlated panel data with unit-specific trends and levels
    base   = rng.uniform(0.1, 1.0, size=N)
    trend  = rng.uniform(-0.002, 0.002, size=N)
    t_grid = np.arange(T)[:, None]                         # (T, 1)
    Y0     = base[None, :] + trend[None, :] * t_grid + rng.normal(0, 0.05, size=(T, N))
    Y0     = np.clip(Y0, 1e-3, None)

    # --- Build and solve ---
    solver = SupergeoSolver(
        unit_names=unit_names,
        Y0=Y0,
        min_size=2,
        max_size=4,
        kappa=3,
        seed=42,
    )

    print("Generating candidates (exhaustive, N=20, T=70)...")
    solver.generate_candidates("exhaustive")
    print(f"  {len(solver.candidates):,} candidates.")

    print("\nSolving MIP (CBC)...")
    design = solver.solve(solver=cp.CBC, budget=B)
    solver.summary()

    # --- Power analysis (heuristic under demeaned-R² score) ---
    print("\nPower analysis (heuristic):")
    print(f"  MDE  (80% power, alpha=0.05) : {solver.mde():.4f}")
    print(f"  Power at delta=0.5           : {solver.power_at_effect(0.5):.4f}")
    curve = solver.power_curve(np.linspace(0.1, 2.0, 8))
    print("\n  Power curve:")
    print(curve.to_string(index=False))

    # --- AA placebo test ---
    print("\nAA placebo test (5 000 sims):")
    aa = solver.run_aa_test(n_sims=5_000, alpha=0.05, seed=7)
    print(f"  Observed stat : {aa['observed_stat']:.4f}")
    print(f"  p-value       : {aa['p_value']:.4f}")
    print(f"  Reject null   : {aa['reject']}")
    print(f"  95% CI        : [{aa['ci_lower']:.4f}, {aa['ci_upper']:.4f}]")

    # --- Half-synthetic evaluation ---
    print("\nHalf-synthetic evaluation (5 000 iterations)...")
    R_test = Y0.mean(axis=0)
    S_test = rng.uniform(0.1, 1.0, size=N)
    result = empirical_estimator(
        design=design,
        unit_names=unit_names,
        R_test=R_test,
        S_test=S_test,
        theta_true=1.0,
        r_spend=0.2,
        n_iter=5_000,
        rng=rng,
    )
    print(f"  RMSE : {result['rmse']:.4f}")
    print(f"  Bias : {result['bias']:.4f}")
