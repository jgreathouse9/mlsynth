"""
supergeo_solver.py
==================
Supergeo Design: Generalized Matching for Geographic Experiments

    Chen, Doudchenko, Jiang, Stein, Ying (2023)
    arXiv:2301.12044

Every design decision in this module traces directly to a section of the
paper.  Cross-references are written inline as [§N] or [Eq. N].

Architecture
------------

  _aa_kernel(diffs, n_sims, seed)
      Numba-jitted xorshift64 Monte Carlo kernel for the AA placebo test.
      Streams through n_sims sign-flip draws without allocating an
      (n_sims × K) matrix.  [Appendix D]

  _pretest_aggregate(Y0)
      Z_g = Σ_t Y0[t, g].  The paper's proxy for the unobserved
      uninfluenced response.  [§3, "Matching variables"]

  _score_and_split(indices, Z, rng)
      score(G) = min_{G+∪G−=G} (Z_{G+} − Z_{G−})²   [Eq. 7 / §3.1]
      Symmetry reduction halves enumeration; the stored split orientation
      is randomised so neither side is systematically favoured.

  _build_incidence_matrix(n, candidates)
      Sparse N × M CSC matrix A where A[i,j] = 1 iff unit i ∈ candidate j.
      Passed directly to CVXPY so the exact-cover constraint is a single
      matrix expression rather than N separate scalar constraints.  [§3.2]

  mde / power_at_effect / power_curve
      Analytical power analysis from Var[θ̂] = total_loss / B²  [Eq. 3]
      using Student-t with K−1 df to account for small pair counts.

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
from scipy.sparse import csc_matrix
from scipy.stats import t as t_dist


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
    diffs  : (K,) float64   Z_{G_k+} − Z_{G_k−} for each pair k
    n_sims : int            Number of Monte Carlo draws
    seed   : int            Non-zero seed for the xorshift64 state

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
# PRE-PERIOD AGGREGATE  [§3, "Matching variables"]
# =============================================================================

def _pretest_aggregate(Y0: np.ndarray) -> np.ndarray:
    """
    Z_g ≈ R^pre_g = Σ_t R_g[t] — pre-period aggregate per geo.

    The paper uses this as a proxy for the unobserved uninfluenced response
    Z_g.  We match on this scalar throughout.

    Parameters
    ----------
    Y0 : (T, N) array   Pre-period outcome matrix

    Returns
    -------
    Z : (N,) array
    """
    return Y0.sum(axis=0)


# =============================================================================
# SCORE AND SPLIT  [Eq. 7 / §3.1]
# =============================================================================

def _score_and_split(
    indices: tuple[int, ...],
    Z: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, list[int], list[int]]:
    """
    Compute score(G) and a randomly oriented optimal (G+, G−) split.

    Paper definition [Eq. 7 / §3.1]:

        score(G) = min_{G+∪G−=G, G+∩G−=∅}  (Z_{G+} − Z_{G−})²
        where Z_G = Σ_{g∈G} Z_g

    Symmetry reduction
    ------------------
    (Z_{G+} − Z_{G−})² = (Z_{G−} − Z_{G+})², so every bipartition and its
    mirror yield the same score.  By fixing index 0 in the plus set we
    enumerate each unordered bipartition exactly once, halving the work.
    For max_size = 4 the inner loop runs at most
        C(3,0) + C(3,1) + C(3,2) = 7 iterations per candidate.

    Random orientation
    ------------------
    The symmetry reduction is applied only during *scoring*.  Once the
    optimal bipartition is found, the plus/minus label is assigned by an
    unbiased coin flip so that neither side is systematically favoured
    across the candidate pool.  This preserves the validity of the
    downstream per-pair coin-flip randomisation [§2].

    Parameters
    ----------
    indices : tuple[int]      Geo indices forming the candidate group G
    Z       : (N,) float64   Pre-period aggregates
    rng     : Generator       For the random orientation step only

    Returns
    -------
    score   : float
    g_plus  : list[int]   Treatment-side indices
    g_minus : list[int]   Control-side indices
    """
    idx = list(indices)
    n = len(idx)
    z_local = Z[idx]  # local slice — avoids repeated global indexing

    best_score = float("inf")
    best_plus_local: list[int] = []
    best_minus_local: list[int] = []

    # Anchor element 0 in plus; vary elements 1…n-1
    for r in range(0, n - 1):
        for extra in combinations(range(1, n), r):
            plus_local  = (0,) + extra
            minus_local = [i for i in range(n) if i not in plus_local]
            diff = (
                float(z_local[list(plus_local)].sum())
                - float(z_local[minus_local].sum())
            ) ** 2
            if diff < best_score:
                best_score      = diff
                best_plus_local = list(plus_local)
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
    Z: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    """Package a (score, split) pair into a candidate dict."""
    score, g_plus, g_minus = _score_and_split(combo, Z, rng)
    return {"indices": combo, "score": score, "split": (g_plus, g_minus)}


def _candidates_exhaustive(
    n: int,
    Z: np.ndarray,
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
            candidates.append(_make_candidate(combo, Z, rng))
    return candidates


def _candidates_partition(
    n: int,
    Z: np.ndarray,
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
                candidates.append(_make_candidate(combo, Z, rng))
    return candidates


def _candidates_per_geo(
    n: int,
    Z: np.ndarray,
    min_size: int,
    max_size: int,
    beta: int,
    alpha: float,
    rng: np.random.Generator,
) -> list[dict]:
    """
    Per-geo heuristic [Appendix E].

    Ranks geos by descending Z_g (the largest geos are hardest to match in
    standard pairs design).  For each of the beta largest geos:
        • Score every subset of size [min_size, max_size] that contains it.
        • Retain only the alpha-best fraction (lowest scores).
    Remaining small geos, which already match well 1-to-1, keep only
    pairwise (size-2) candidates.

    Duplicates across pools are removed by frozenset-key deduplication.
    """
    order       = np.argsort(-Z)                         # descending by Z_g
    large_geos  = set(order[:beta].tolist())
    small_geos  = [g for g in range(n) if g not in large_geos]

    candidates: list[dict] = []

    # Small geos: standard 1-to-1 pairs only
    for combo in combinations(small_geos, 2):
        candidates.append(_make_candidate(combo, Z, rng))

    # Large geos: alpha-best pool of all containing subsets
    for g in large_geos:
        others = [i for i in range(n) if i != g]
        pool: list[dict] = []
        for size in range(min_size, max_size + 1):
            for rest in combinations(others, size - 1):
                combo = tuple(sorted((g,) + rest))
                pool.append(_make_candidate(combo, Z, rng))

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
# SPARSE INCIDENCE MATRIX  [§3.2]
# =============================================================================

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
    budget     : float   Total incremental spend B [Assumption 2]
    n_pairs    : int     Number of supergeo pairs K
    alpha      : float   Two-sided significance level
    power      : float   Target power 1 − β

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
    delta      : float   True iROAS deviation from null
    total_loss : float
    budget     : float
    n_pairs    : int
    alpha      : float   Two-sided significance level

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
    deltas     : (D,) array-like   Effect sizes to evaluate
    total_loss : float
    budget     : float
    n_pairs    : int
    alpha      : float

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

        loss = Σ_k (Z_{G_k+} − Z_{G_k−})²               [Eq. 7]

    subject to:
        • Every geo belongs to exactly one supergeo pair  (exact cover)
        • Each supergeo pair has total size in [min_size, max_size]
        • At least kappa supergeo pairs are formed         [§3]

    Typical workflow
    ----------------
    >>> solver = SupergeoSolver(unit_names, Y0, min_size=2, max_size=4,
    ...                         kappa=5, seed=42)
    >>> solver.generate_candidates("per_geo", beta=30, alpha=0.10)
    >>> design = solver.solve(budget=1e6)
    >>> solver.summary()
    >>> print(solver.mde())
    >>> aa = solver.run_aa_test(n_sims=10_000)

    Parameters
    ----------
    unit_names : list[str]
        Human-readable geo labels, length N.
    Y0 : (T, N) ndarray
        Pre-period outcome matrix.  Z_g = Σ_t Y0[t, g] is used as the
        proxy for the uninfluenced response [§3, "Matching variables"].
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

        # Z_g ≈ R^pre_g: pre-period aggregate [§3, "Matching variables"]
        self.Z: np.ndarray = _pretest_aggregate(self.Y0)  # shape (N,)

        # State set by generate_candidates() and solve()
        self.candidates: list[dict]          = []
        self.design_:    pd.DataFrame | None = None
        self._budget:    float | None        = None

    # ------------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------------

    def generate_candidates(
        self,
        heuristic: Literal["exhaustive", "partition", "per_geo"] = "exhaustive",
        *,
        n_partitions: int = 10,
        beta: int | None = None,
        alpha: float = 0.10,
    ) -> None:
        """
        Populate self.candidates with feasible supergeo groups.

        Parameters
        ----------
        heuristic : {"exhaustive", "partition", "per_geo"}

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

        n_partitions : int
            Number of buckets for the partition heuristic.
        beta : int or None
            Number of large geos for the per-geo heuristic (default N // 2).
        alpha : float
            Candidate retention fraction for the per-geo heuristic.
        """
        if heuristic == "exhaustive":
            self.candidates = _candidates_exhaustive(
                self.n, self.Z, self.min_size, self.max_size, self._rng,
            )
        elif heuristic == "partition":
            self.candidates = _candidates_partition(
                self.n, self.Z, self.min_size, self.max_size,
                n_partitions=n_partitions, rng=self._rng,
            )
        elif heuristic == "per_geo":
            _beta = beta if beta is not None else self.n // 2
            self.candidates = _candidates_per_geo(
                self.n, self.Z, self.min_size, self.max_size,
                beta=_beta, alpha=alpha, rng=self._rng,
            )
        else:
            raise ValueError(
                f"Unknown heuristic '{heuristic}'. "
                "Choose 'exhaustive', 'partition', or 'per_geo'."
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
        print(f"  Total loss      : {total_loss:.6f}  (= B^2 * Var[theta_hat])")
        if self._budget is not None:
            se = _design_se(total_loss, self._budget)
            print(f"  SE(theta_hat)   : {se:.6f}  [budget = {self._budget}]")
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
        alpha  : float  Two-sided significance level
        seed   : int    Kernel RNG seed (independent of the solver RNG)

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
        diffs = np.array([
            self.Z[[name_to_idx[g] for g in row["Treatment"]]].sum()
            - self.Z[[name_to_idx[g] for g in row["Control"]]].sum()
            for _, row in df.iterrows()
        ], dtype=np.float64)

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
        alpha  : float           Two-sided significance level
        power  : float           Target power 1 − β

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
        delta  : float   True iROAS deviation from null
        budget : float   B
        alpha  : float   Two-sided significance level

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
        alpha  : float        Two-sided significance level

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
    design     : pd.DataFrame   Output of SupergeoSolver.solve()
    unit_names : list[str]      Geo labels in the same order as Y0 columns
    R_test     : (N,) array     Test-phase response per geo (no treatment)
    S_test     : (N,) array     Test-phase spend per geo
    theta_true : float          Injected iROAS (ground truth)
    r_spend    : float          Heavy-up fraction r [Assumption 4]
    n_iter     : int            Monte Carlo iterations
    rng        : Generator      Defaults to seed 0

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
    N, T = 20, 28
    B    = 50.0

    unit_names = [f"DMA_{i:02d}" for i in range(N)]
    base = rng.uniform(0.1, 1.0, size=N)
    Y0   = base[None, :] + rng.normal(0, 0.05, size=(T, N))
    Y0   = np.clip(Y0, 1e-3, None)

    # --- Build and solve ---
    solver = SupergeoSolver(
        unit_names=unit_names,
        Y0=Y0,
        min_size=2,
        max_size=4,
        kappa=3,
        seed=42,
    )

    print("Generating candidates (exhaustive, N=20)...")
    solver.generate_candidates("exhaustive")
    print(f"  {len(solver.candidates):,} candidates.")

    print("\nSolving MIP (CBC)...")
    design = solver.solve(solver=cp.CBC, budget=B)
    solver.summary()

    # --- Power analysis ---
    print("\nPower analysis:")
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
    R_test = Y0.sum(axis=0)
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
    print(f"  Bias : {result['bias']:.4f}")    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Array of placebo test statistics (shape: n_sims,).
    """

    K = diffs.shape[0]
    out = np.empty(n_sims, dtype=np.float64)

    # simple LCG RNG (Numba-friendly; avoids Python RNG overhead)
    state = seed if seed != 0 else 1

    for s in range(n_sims):
        total = 0.0

        for k in range(K):
            # Linear congruential generator
            state = (1103515245 * state + 12345) & 0x7fffffff

            # Map to ±1
            sign = 1.0 if (state & 1) == 0 else -1.0

            total += sign * diffs[k]

        out[s] = total

    return out


def _aggregate(Z: np.ndarray, indices: Tuple[int, ...]) -> float:
    """
    Calculate the aggregate response for a set of units.

    Parameters
    ----------
    Z : np.ndarray
        Array of pre-period characteristics for each unit.
    indices : tuple of int
        The indices of the units to aggregate.

    Returns
    -------
    float
        The sum of Z for the specified indices.
    """
    return float(Z[list(indices)].sum())


def _optimal_split(Z: np.ndarray, indices: Tuple[int, ...]) -> Tuple[List[int], List[int], float]:
    """
    Find the bipartition of a group that minimizes the squared difference.

    This function implements the inner minimization of the score(G) calculation.

    Parameters
    ----------
    Z : np.ndarray
        Array of pre-period characteristics for each unit.
    indices : tuple of int
        The indices of the units in group G.

    Returns
    -------
    best_plus : list of int
        Indices assigned to the treatment side (G+).
    best_minus : list of int
        Indices assigned to the control side (G-).
    best_score : float
        The minimum squared difference (Z_{G+} - Z_{G-})².
    """
    idx = list(indices)
    n = len(idx)
    Z_group = Z[idx]

    best_score = np.inf
    best_plus: List[int] = []
    best_minus: List[int] = []

    if n < 2:
        return list(indices), [], np.inf

    # Enumerate all non-empty proper subsets for G+
    for r in range(1, n):
        for plus_local in combinations(range(n), r):
            minus_local = [i for i in range(n) if i not in plus_local]
            z_plus = Z_group[list(plus_local)].sum()
            z_minus = Z_group[list(minus_local)].sum()
            score = (z_plus - z_minus) ** 2
            if score < best_score:
                best_score = score
                best_plus = [idx[i] for i in plus_local]
                best_minus = [idx[i] for i in minus_local]

    return best_plus, best_minus, best_score


def _candidates_from_index_pool(
    Z: np.ndarray,
    index_pool: List[List[int]],
    min_size: int,
    max_size: int,
) -> List[Dict[str, Any]]:
    """
    Build candidate supergeo pairs from a specified pool of unit indices.

    Parameters
    ----------
    Z : np.ndarray
        Array of pre-period characteristics for each unit.
    index_pool : list of list of int
        A list containing subsets of unit indices to consider for combinations.
    min_size : int
        Minimum size of a supergeo pair.
    max_size : int
        Maximum size of a supergeo pair.

    Returns
    -------
    list of dict
        A list of candidate dictionaries containing 'indices', 'g_plus',
        'g_minus', and 'score'.
    """
    candidates: List[Dict[str, Any]] = []
    seen: set[Tuple[int, ...]] = set()

    for pool in index_pool:
        for size in range(min_size, max_size + 1):
            for combo in combinations(sorted(pool), size):
                if combo in seen:
                    continue
                seen.add(combo)
                g_plus, g_minus, score = _optimal_split(Z, combo)
                candidates.append(
                    dict(indices=combo, g_plus=g_plus, g_minus=g_minus, score=score)
                )

    return candidates


class SupergeoSolver:
    """
    Solver for Supergeo experimental design using Mixed-Integer Programming.

    Parameters
    ----------
    unit_names : list of str
        Names/Labels for the N experimental units.
    Y0 : np.ndarray
        Pre-period outcome matrix of shape (T, N), where T is time points
        and N is units.
    min_size : int, optional
        Minimum size (l) of a supergeo pair, by default 2.
    max_size : int, optional
        Maximum size (u) of a supergeo pair, by default 4.
    kappa : int, optional
        Minimum number of supergeo pairs (k) required, by default 1.

    Attributes
    ----------
    Z : np.ndarray
        Aggregated pre-test response for each unit (proxy for uninfluenced response).
    N : int
        Total number of experimental units.
    """

    def __init__(
        self,
        unit_names: List[str],
        Y0: np.ndarray,
        min_size: int = 2,
        max_size: int = 4,
        kappa: int = 1,
    ) -> None:
        if Y0.ndim != 2:
            raise ValueError("Y0 must be a 2-D array of shape (T, N).")
        if Y0.shape[1] != len(unit_names):
            raise ValueError("Y0.shape[1] must equal len(unit_names).")

        self.unit_names = list(unit_names)
        self.Y0 = Y0
        self.N = len(unit_names)
        self.min_size = min_size
        self.max_size = max_size
        self.kappa = kappa
        self.Z: np.ndarray = Y0.sum(axis=0)
        self._candidates: List[Dict[str, Any]] = []

    def generate_candidates_exhaustive(self) -> None:
        """
        Enumerate all possible unit subsets within the specified size bounds.

        Warning
        -------
        This method is exponential in complexity. It is recommended only
        for small unit counts (N < 50).
        """
        self._candidates = _candidates_from_index_pool(
            self.Z,
            [list(range(self.N))],
            self.min_size,
            self.max_size,
        )

    def generate_candidates_partition(self, n_partitions: int = 10, seed: int = 0) -> None:
        """
        Partition units randomly into buckets to generate candidates.

        This heuristic reduces the number of MIP variables by only matching
        units within the same random partition.

        Parameters
        ----------
        n_partitions : int, optional
            Number of partitions to divide units into, by default 10.
        seed : int, optional
            Random seed for reproducibility, by default 0.
        """
        rng = np.random.default_rng(seed)
        perm = rng.permutation(self.N).tolist()
        partitions = [perm[i::n_partitions] for i in range(n_partitions)]

        self._candidates = _candidates_from_index_pool(
            self.Z, partitions, self.min_size, self.max_size
        )

    def generate_candidates_per_geo(
        self,
        beta: Optional[int] = None,
        alpha: float = 0.05,
    ) -> None:
        """
        Generate candidates using the per-geo pruning heuristic.

        Identifies the largest units and keeps only the best-scoring
        candidate subsets containing them.

        Parameters
        ----------
        beta : int, optional
            Number of largest units to apply pruning to. Defaults to N // 2.
        alpha : float, optional
            The top fraction of best-scoring subsets to retain per unit, by default 0.05.
        """
        if beta is None:
            beta = max(1, self.N // 2)

        order = np.argsort(self.Z)[::-1]
        large_geos = set(order[:beta].tolist())
        small_geos = order[beta:].tolist()

        large_pool = list(large_geos)
        all_large = _candidates_from_index_pool(self.Z, [large_pool], self.min_size, self.max_size)

        kept_indices: set[Tuple[int, ...]] = set()
        for geo in large_geos:
            geo_cands = [c for c in all_large if geo in c["indices"]]
            geo_cands.sort(key=lambda c: c["score"])
            n_keep = max(1, int(np.ceil(alpha * len(geo_cands))))
            for c in geo_cands[:n_keep]:
                kept_indices.add(c["indices"])

        large_candidates = [c for c in all_large if c["indices"] in kept_indices]
        small_candidates = _candidates_from_index_pool(self.Z, [small_geos], self.min_size, self.max_size)

        self._candidates = large_candidates + small_candidates

    def solve(self, solver: str = "CBC") -> pd.DataFrame:
        """
        Solve the covering Mixed-Integer Program to find the optimal design.

        Parameters
        ----------
        solver : str, optional
            The name of the CVXPY-compatible solver, by default "CBC".

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the Pair_ID, Treatment units, Control units,
            aggregated Z values, and individual pair scores.

        Raises
        ------
        RuntimeError
            If a unit is not covered by any candidate or the solver fails
             to find a feasible solution.
        """
        if not self._candidates:
            warnings.warn("No candidates found. Running exhaustive generation.", stacklevel=2)
            self.generate_candidates_exhaustive()

        m = len(self._candidates)
        scores = np.array([c["score"] for c in self._candidates], dtype=float)
        x = cp.Variable(m, boolean=True)

        cover_constraints = []
        for i in range(self.N):
            covering_j = [j for j, c in enumerate(self._candidates) if i in c["indices"]]
            if not covering_j:
                raise RuntimeError(f"Unit {i} is not covered by any candidate.")
            cover_constraints.append(cp.sum(x[covering_j]) == 1)

        constraints = cover_constraints + [cp.sum(x) >= self.kappa]
        prob = cp.Problem(cp.Minimize(scores @ x), constraints)
        prob.solve(solver=solver)

        if prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"MIP solver failed with status '{prob.status}'.")

        return self._format_solution(x.value)

    def _format_solution(self, x_val: np.ndarray) -> pd.DataFrame:
        """
        Convert the solver's boolean vector into a readable DataFrame.

        Parameters
        ----------
        x_val : np.ndarray
            The optimal boolean vector from the solver.

        Returns
        -------
        pd.DataFrame
            Formatted experimental design.
        """
        chosen = np.where(x_val > 0.5)[0]
        rows = []
        for pair_id, j in enumerate(chosen, start=1):
            c = self._candidates[j]
            rows.append(dict(
                Pair_ID=pair_id,
                Treatment=[self.unit_names[i] for i in c["g_plus"]],
                Control=[self.unit_names[i] for i in c["g_minus"]],
                Z_plus=_aggregate(self.Z, tuple(c["g_plus"])),
                Z_minus=_aggregate(self.Z, tuple(c["g_minus"])),
                Score=float(c["score"]),
            ))
        df = pd.DataFrame(rows)
        if not df.empty:
            df["Loss"] = df["Score"].sum()
        return df

    def training_loss(self, design: pd.DataFrame) -> float:
        """
        Calculate the total pre-test loss for a given design.

        Parameters
        ----------
        design : pd.DataFrame
            The design DataFrame returned by the solve() method.

        Returns
        -------
        float
            Total squared difference across all supergeo pairs.
        """
        return float(design["Score"].sum())

    def test_loss(self, design: pd.DataFrame, Y_test: np.ndarray) -> float:
        """
        Evaluate the loss of a design on a held-out test dataset.

        Parameters
        ----------
        design : pd.DataFrame
            The design DataFrame returned by the solve() method.
        Y_test : np.ndarray
            Post-period or held-out pre-period outcome matrix.

        Returns
        -------
        float
            Total squared difference calculated on the test data.
        """
        Z_test = Y_test.sum(axis=0)
        loss = 0.0
        name_to_idx = {name: i for i, name in enumerate(self.unit_names)}

        for _, row in design.iterrows():
            z_plus = sum(Z_test[name_to_idx[n]] for n in row["Treatment"])
            z_minus = sum(Z_test[name_to_idx[n]] for n in row["Control"])
            loss += (z_plus - z_minus) ** 2

        return loss



def estimate_power(
        self, 
        design: pd.DataFrame, 
        expected_lift: float = 0.05, 
        alpha: float = 0.05
    ) -> dict:
        """
        Performs a Power Analysis on the generated design.

        This assumes a T-test framework for Matched Pairs as described
        in the paper's section on inference.

        Parameters
        ----------
        design : pd.DataFrame
            The design returned by solve().
        expected_lift : float
            The percentage lift you want to be able to detect (e.g., 0.05 for 5%).
        alpha : float
            Significance level, by default 0.05.

        Returns
        -------
        dict
            Dictionary containing Power, MDE, and Standard Error.
        """
        # 1. Calculate the standard error of the lift
        # In Supergeo, the variance of the estimator is proportional 
        # to the sum of the squared differences (Scores)
        total_z_treat = design['Z_plus'].sum()
        total_z_control = design['Z_minus'].sum()
        
        # The paper notes that Var(Tau) is estimated using the pairwise differences
        # Diff_k = (Z_k_plus - Z_k_minus)
        diffs = design['Z_plus'] - design['Z_minus']
        n_pairs = len(design)
        
        if n_pairs < 2:
            return {"error": "Power analysis requires at least K=2 pairs for variance estimation."}

        # Calculate standard error of the mean difference
        # This is the "Placebo" standard error (Pre-test noise)
        se_mean = np.std(diffs, ddof=1) / np.sqrt(n_pairs)
        
        # 2. Calculate the Absolute Effect Size
        # We assume the lift applies to the total treatment volume
        delta = expected_lift * total_z_treat
        
        # 3. Power Calculation using T-distribution (n_pairs - 1 degrees of freedom)
        df = n_pairs - 1
        t_alpha = stats.t.ppf(1 - alpha/2, df)
        
        # Non-centrality parameter
        ncp = delta / se_mean
        
        # Power = P(T > t_alpha | alternative is true)
        power = 1 - stats.t.cdf(t_alpha, df, loc=ncp)
        
        # 4. Calculate MDE (Minimum Detectable Effect) at 80% Power
        t_beta = stats.t.ppf(0.80, df)
        mde_abs = (t_alpha + t_beta) * se_mean
        mde_pct = mde_abs / total_z_treat

        return {
            "Total_Treatment_Volume": total_z_treat,
            "Expected_Lift_Tested": expected_lift,
            "Standard_Error": se_mean,
            "Statistical_Power": power,
            "MDE_at_80_percent": mde_pct,
            "Degrees_of_Freedom": df
        }




def run_aa_simulations(
    self,
    n_sims: int = 500,
    seed: int = 0
) -> pd.Series:
    """
    Efficient AA placebo simulations using a JIT-compiled streaming kernel.

    This implementation avoids constructing the full (n_sims × K) sign matrix
    and instead generates random ±1 assignments on-the-fly inside a compiled
    loop (Numba JIT). This yields:

    - O(K) memory usage
    - near C-level performance for large n_sims
    - no Python loop overhead in the inner simulation loop

    Parameters
    ----------
    n_sims : int, optional
        Number of placebo simulations to run, by default 500.
    seed : int, optional
        Random seed for reproducibility, by default 0.

    Returns
    -------
    pd.Series
        Simulated distribution of normalized placebo treatment effects.

    Raises
    ------
    ValueError
        If `solve()` has not been run and `design_` is missing.

    Notes
    -----
    This implementation uses a deterministic linear congruential generator
    inside Numba to avoid Python RNG overhead. While fast and reproducible,
    it is not cryptographically secure (not needed for Monte Carlo inference).
    """

    if not hasattr(self, 'design_'):
        raise ValueError("Run solve() first to generate a design.")

    diffs = (self.design_["Z_plus"] - self.design_["Z_minus"]).to_numpy(dtype=np.float64)
    denom = float(self.design_["Z_plus"].sum())

    raw = _aa_chunk_kernel(diffs, n_sims, seed)

    return pd.Series(raw / denom)
