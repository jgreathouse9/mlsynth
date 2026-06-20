"""Original HCW best-subset panel data approach (Hsiao, Ching & Wan 2012).

The counterfactual for the treated unit is an *unrestricted* OLS regression
(with intercept) of its pre-treatment outcome on a **best-subset**-selected set
of control series (eq. 17), with no simplex / non-negativity constraint. The
subset is chosen in two steps (Section 5): for each model size the best subset
by residual sum of squares, then the size minimising a model-selection
criterion (AICc by default, also AIC / BIC). This is a direct port of the
``pampe`` R package (``leaps::regsubsets`` best-subset + AICc + ``lm`` with
intercept), and reproduces HCW (2012) Table XVI value-for-value.

Best-subset selection is combinatorial. Like ``pampe``'s ``nvmax`` argument, the
search ranges over model sizes up to ``nvmax`` (default: all controls); for a
large donor pool the caller must either cap ``nvmax`` or pre-restrict the pool
(as HCW did, limiting Hong Kong to ten candidate economies), otherwise a
translated error is raised pointing to the scalable ``fs`` / ``LASSO`` / ``l2``
variants.
"""

from __future__ import annotations

from itertools import combinations
from typing import List, Optional, Tuple

import numpy as np

from ....exceptions import MlsynthEstimationError

# Default cap on the number of branch-and-bound nodes explored. HCW's intended
# small pools (a few dozen donors) certify the optimum in far fewer; a large
# pool stops at the budget and returns the best incumbent with a valid
# optimality gap rather than refusing. ~200k nodes is a few seconds at worst.
_DEFAULT_NODE_BUDGET = 200_000

# Past this many donors the discrete first-order warm start is skipped: seeding
# every model size costs more than the pruning it would buy at that scale.
_WARMSTART_MAX_N = 128

_CRITERIA = ("AICc", "AIC", "BIC")


def info_criterion(rss: float, n: int, n_regressors: int, criterion: str) -> float:
    """Model-selection criterion for an OLS fit, in the ``pampe`` convention.

    Parameters
    ----------
    rss : float
        Residual sum of squares of the pre-period OLS fit.
    n : int
        Number of pre-treatment observations.
    n_regressors : int
        Number of regressors including the intercept (i.e. donors + 1).
    criterion : {"AICc", "AIC", "BIC"}
        Which criterion. ``pampe`` (and HCW Table XVI) count the error variance
        as a parameter, so the penalised parameter count is
        ``K = n_regressors + 1`` (donors + intercept + variance).

    Returns
    -------
    float
        The criterion value (lower is better). Uses the concentrated Gaussian
        form ``n * log(rss / n)`` (no ``n log 2*pi + n`` constant), matching the
        magnitudes HCW report (e.g. AICc = -171.771 for the Table XVI model).
    """
    if criterion not in _CRITERIA:
        raise MlsynthEstimationError(
            f"Unknown HCW criterion {criterion!r}; use one of {_CRITERIA}."
        )
    if rss <= 0 or n <= 0:
        return -np.inf
    K = n_regressors + 1                         # + error variance (pampe count)
    ll_term = n * np.log(rss / n)
    if criterion == "AIC":
        return float(ll_term + 2 * K)
    if criterion == "BIC":
        return float(ll_term + K * np.log(n))
    # AICc: the small-sample-corrected AIC.
    if n - K - 1 <= 0:                            # correction undefined; +inf
        return float("inf")
    return float(ll_term + 2 * K + 2 * K * (K + 1) / (n - K - 1))


def _rss(y: np.ndarray, Z: np.ndarray) -> float:
    """OLS residual sum of squares for design ``Z`` (intercept included)."""
    coef, *_ = np.linalg.lstsq(Z, y, rcond=None)
    e = y - Z @ coef
    return float(e @ e)


def _gram(y_pre: np.ndarray, X_pre: np.ndarray):
    """Pre-period sufficient statistics for OLS-by-subset.

    Returns ``(G, Zty, yty)`` for the augmented design ``Z = [1, X_pre]``:
    ``G = Z'Z`` (``(N+1, N+1)``), ``Zty = Z'y`` (``(N+1,)``) and ``yty = y'y``.
    Computed once; every subset's RSS then comes from a small submatrix solve,
    so the best-subset search costs no SVD and is independent of ``T0`` per
    subset.
    """
    Z = np.column_stack([np.ones(len(y_pre)), X_pre])
    return Z.T @ Z, Z.T @ y_pre, float(y_pre @ y_pre)


def _subset_rss(G: np.ndarray, Zty: np.ndarray, yty: float, cols) -> float:
    """RSS of OLS on the intercept + donor columns ``cols`` from the Gram.

    ``RSS = y'y - (Z_S'y)' (Z_S'Z_S)^{+} (Z_S'y)`` over the submatrix indexed by
    the intercept (column 0) and ``cols`` (shifted by +1). The solve goes through
    a symmetric eigendecomposition, dropping the near-null spectrum a collinear
    donor subset produces: solving the singular normal equations with ``lstsq``
    instead keeps a ~machine-epsilon direction and can return a garbage, even
    negative, RSS (platform-dependent). The eigen solve matches the full-rank
    direct solve and the min-norm fit on rank-deficient subsets alike; the result
    is clamped to be non-negative against residual cancellation near a perfect
    fit.
    """
    idx = [0]
    idx.extend(c + 1 for c in cols)
    Gs = G[np.ix_(idx, idx)]
    bs = Zty[idx]
    w, V = np.linalg.eigh(Gs)                     # Gs is symmetric PSD
    tol = max(Gs.shape) * np.finfo(float).eps * max(float(w[-1]), 1.0)
    inv_w = np.where(w > tol, 1.0 / w, 0.0)       # pseudo-inverse: drop ~null dirs
    sol = V @ (inv_w * (V.T @ bs))
    return max(yty - float(bs @ sol), 0.0)


def _sweep(M: np.ndarray, k: int) -> None:
    """Symmetric Gauss-Jordan sweep of ``M`` on pivot ``k``, in place.

    The sweep operator (Goodnight 1979) is the Furnival-Wilson regression
    engine. Sweeping the intercept and a set of donor columns into the augmented
    cross-product matrix ``M = [1, X, y]'[1, X, y]`` leaves, in the (y, y)
    diagonal, the residual sum of squares of regressing ``y`` on those columns.
    A depth-first search can therefore add a donor on the way down (this forward
    sweep) and remove it on the way back up (:func:`_unsweep`, the reverse
    sweep) in ``O(p^2)`` apiece, never re-solving an OLS from scratch. (Pivots
    must be non-degenerate; the caller skips any pivot whose diagonal has
    collapsed to zero, i.e. a donor collinear with the set already swept in.)

    Note the forward sweep is not its own inverse: applying it twice negates the
    pivot row and column, so removal must go through :func:`_unsweep`.
    """
    D = M[k, k]
    col = M[:, k].copy()
    row = M[k, :].copy()
    M -= np.outer(col, row) / D                  # a_ij - a_ik a_kj / a_kk
    M[k, :] = row / D                            # row k:  a_kj / a_kk
    M[:, k] = col / D                            # col k:  a_ik / a_kk
    M[k, k] = -1.0 / D                           # pivot:  -1 / a_kk


def _unsweep(M: np.ndarray, k: int) -> None:
    """Reverse sweep of ``M`` on pivot ``k``, in place: the inverse of :func:`_sweep`.

    Identical to the forward sweep except the pivot row and column are negated,
    which is exactly what undoes a prior :func:`_sweep` on the same pivot
    (``_sweep`` then ``_unsweep`` restores ``M`` to machine precision). This is
    how the search removes a donor when it backtracks.
    """
    D = M[k, k]
    col = M[:, k].copy()
    row = M[k, :].copy()
    M -= np.outer(col, row) / D
    M[k, :] = -row / D
    M[:, k] = -col / D
    M[k, k] = -1.0 / D


def _augmented(G: np.ndarray, Zty: np.ndarray, yty: float, N: int) -> np.ndarray:
    """Assemble the augmented cross-product matrix ``[1, X, y]'[1, X, y]``.

    Intercept at index 0, donors at 1..N, the response at N+1, so a donor with
    column index ``j`` lives at sweep index ``j + 1`` and the response diagonal
    ``M[N+1, N+1]`` holds the running RSS once the predictors are swept in.
    """
    M = np.empty((N + 2, N + 2))
    M[: N + 1, : N + 1] = G
    M[: N + 1, N + 1] = Zty
    M[N + 1, : N + 1] = Zty
    M[N + 1, N + 1] = yty
    return M


def _all_in_rss(M: np.ndarray, rem_idx: List[int], y_idx: int, cur_rss: float) -> float:
    """RSS of adding *every* remaining donor to the currently swept set.

    Given ``M`` swept on the intercept and the chosen donors, the un-swept block
    ``M[rem, rem]`` is the residual cross-product of the remaining donors and
    ``M[rem, y]`` their residual covariance with the response (Schur
    complements). Including all of them drops the RSS by the regression sum of
    squares ``M[y, rem] (M[rem, rem])^{+} M[rem, y]``. This is the smallest RSS
    any descendant subset can reach (RSS is monotone in the variable set), so it
    is the lower-bound ingredient for pruning. The symmetric-PSD eigen solve (as
    in :func:`_subset_rss`) drops the near-null spectrum of a rank-deficient
    remaining block, keeping the bound finite and non-negative.
    """
    if not rem_idx:
        return cur_rss
    A = M[np.ix_(rem_idx, rem_idx)]
    b = M[rem_idx, y_idx]
    w, V = np.linalg.eigh(A)
    tol = max(A.shape) * np.finfo(float).eps * max(float(w[-1]), 1.0)
    inv_w = np.where(w > tol, 1.0 / w, 0.0)
    sol = V @ (inv_w * (V.T @ b))
    return max(cur_rss - float(b @ sol), 0.0)


def _resolve_nvmax(nvmax: Optional[int], N: int, T0: int) -> int:
    """Largest model size to search: bounded by donors and the OLS df."""
    cap = min(N, max(T0 - 2, 0))                 # need n > params for a fit
    if nvmax is None:
        return cap
    if nvmax < 1:
        raise MlsynthEstimationError("HCW nvmax must be a positive integer.")
    return min(nvmax, cap)


def best_subset_select(
    y: np.ndarray,
    X: np.ndarray,
    T0: int,
    *,
    criterion: str = "AICc",
    nvmax: Optional[int] = None,
    node_budget: Optional[int] = _DEFAULT_NODE_BUDGET,
    backend: str = "fw",
    stats: Optional[dict] = None,
) -> List[int]:
    """Best-subset donor selection by ``criterion`` (HCW Section 5 / ``pampe``).

    For every model size ``r = 0, 1, ..., nvmax`` the subset of ``r`` donors with
    the smallest pre-period RSS is found, and the size (and subset) minimising
    ``criterion`` is returned. The default ``backend="fw"`` is the exact
    Furnival-Wilson search; for a large pool it stops at ``node_budget`` and
    returns the best incumbent found with an optimality gap rather than refusing.
    ``backend="scip"`` selects the optional SCIP mixed-integer solver (requires
    ``pyscipopt``), which certifies the optimum at larger pool sizes. Pass a
    ``stats`` dict to receive the search diagnostics (node count / gap /
    certification).

    Returns the selected donor column indices (a possibly empty list).
    """
    if criterion not in _CRITERIA:
        raise MlsynthEstimationError(
            f"Unknown HCW criterion {criterion!r}; use one of {_CRITERIA}."
        )
    y_pre = np.asarray(y, dtype=float)[:T0]
    X_pre = np.asarray(X, dtype=float)[:T0]
    N = X_pre.shape[1]
    r_max = _resolve_nvmax(nvmax, N, T0)

    # Precompute the Gram sufficient statistics once; the Furnival-Wilson engine
    # then reads each subset's RSS off a sweep of the augmented cross-products,
    # never re-solving an OLS from scratch.
    G, Zty, yty = _gram(y_pre, X_pre)

    if backend == "scip":
        try:
            from .scip import best_subset_scip
        except ImportError as exc:                # pyscipopt not installed
            raise MlsynthEstimationError(
                "The HCW 'scip' backend requires pyscipopt, which is not "
                "installed. Install it with 'pip install pyscipopt' (or "
                "'pip install mlsynth[scip]'), or use the default exact "
                "Furnival-Wilson backend (backend='fw')."
            ) from exc
        return best_subset_scip(G, Zty, yty, N, T0, r_max, criterion, stats=stats)
    if backend != "fw":
        raise MlsynthEstimationError(
            f"Unknown HCW backend {backend!r}; use 'fw' or 'scip'."
        )

    return _best_subset_fw(
        G, Zty, yty, N, T0, r_max, criterion,
        node_budget=node_budget, _stats=stats,
    )


def _best_subset_exhaustive(G, Zty, yty, N, n, r_max, criterion) -> List[int]:
    """Brute-force best subset: score every subset up to ``r_max`` by criterion.

    The reference search -- correct by construction and the oracle that
    :func:`_best_subset_bnb` is validated against.
    """
    best_idx: List[int] = []
    best_ic = info_criterion(_subset_rss(G, Zty, yty, ()), n, 1, criterion)
    for r in range(1, r_max + 1):
        for combo in combinations(range(N), r):
            ic = info_criterion(_subset_rss(G, Zty, yty, combo), n, r + 1, criterion)
            if ic < best_ic:
                best_ic, best_idx = ic, list(combo)
    return best_idx


def _best_subset_bnb(G, Zty, yty, N, n, r_max, criterion) -> List[int]:
    """Best subset by branch-and-bound (Furnival-Wilson style pruning).

    Donors are ordered strongest-first (smallest univariate RSS) so the
    incumbent tightens early. At each node the criterion is lower-bounded over
    the whole subtree by the smallest achievable RSS (include *all* remaining
    donors -- RSS is monotone in the variable set) paired with the smallest
    feasible penalty (stop now); if that bound is no better than the incumbent,
    the subtree is pruned. The bound is a true lower bound, so the returned
    optimum is identical to :func:`_best_subset_exhaustive`.
    """
    uni = [_subset_rss(G, Zty, yty, (j,)) for j in range(N)]
    order = sorted(range(N), key=lambda j: uni[j])

    best = {"ic": info_criterion(_subset_rss(G, Zty, yty, ()), n, 1, criterion),
            "idx": []}

    def recurse(chosen: List[int], pos: int) -> None:
        k = len(chosen)
        if k >= 1:
            ic = info_criterion(_subset_rss(G, Zty, yty, chosen), n, k + 1, criterion)
            if ic < best["ic"]:
                best["ic"], best["idx"] = ic, list(chosen)
        if k >= r_max or pos >= N:
            return
        remaining = order[pos:]
        # Lower bound over the subtree: include every remaining donor (minimum
        # RSS) at the smallest feasible model size (minimum penalty).
        rss_lb = _subset_rss(G, Zty, yty, chosen + remaining)
        if info_criterion(rss_lb, n, max(k, 1) + 1, criterion) >= best["ic"]:
            return
        for p in range(pos, N):
            recurse(chosen + [order[p]], p + 1)

    recurse([], 0)
    return best["idx"]


def _centered_gram(G: np.ndarray, Zty: np.ndarray, yty: float, n: int):
    """Mean-centred donor cross-products ``(Sxx, Sxy, syy)`` from the raw Gram.

    Regressing ``y`` on ``[1, X_S]`` is equivalent to regressing the centred
    response on the centred donors, so the discrete first-order warm start works
    on ``Sxx = X'X - (X'1)(1'X)/n``, ``Sxy = X'y - (X'1)(1'y)/n`` and
    ``syy = y'y - (1'y)^2/n`` -- all recovered from the augmented Gram without
    touching the raw data.
    """
    sx = G[0, 1:]                                # column sums of X (= 1'X)
    sy = float(Zty[0])                           # 1'y
    Sxx = G[1:, 1:] - np.outer(sx, sx) / n
    Sxy = Zty[1:] - sx * sy / n
    syy = yty - sy * sy / n
    return Sxx, Sxy, float(syy)


def _ls_on_support(Sxx: np.ndarray, Sxy: np.ndarray, support, N: int):
    """Least-squares coefficients on ``support`` (centred), and the explained SS.

    Returns ``(beta_full, explained)`` where ``beta_full`` is an ``N``-vector
    zero off the support and ``explained = Sxy_S' (Sxx_SS)^{-1} Sxy_S`` is the
    drop in centred RSS from fitting that support (larger is better). A least-
    norm solve covers a collinear support.
    """
    beta = np.zeros(N)
    support = list(support)
    if not support:
        return beta, 0.0
    A = Sxx[np.ix_(support, support)]
    b = Sxy[support]
    try:
        sol = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    beta[support] = sol
    return beta, float(b @ sol)


def _discrete_first_order(
    Sxx: np.ndarray,
    Sxy: np.ndarray,
    k: int,
    L: float,
    *,
    restarts: int = 3,
    max_iter: int = 100,
    rng: Optional[np.random.Generator] = None,
) -> List[int]:
    """Best-subset warm start by the Bertsimas-King-Mazumder (2016) method.

    Algorithm 1 of BKM: projected-gradient hard-thresholding for the
    cardinality-constrained least squares ``min ||y - X beta||^2`` s.t.
    ``||beta||_0 <= k``. From each start we iterate
    ``beta <- H_k(beta - (1/L) (Sxx beta - Sxy))`` -- a gradient step followed by
    the hard-threshold ``H_k`` (keep the ``k`` largest-magnitude entries) -- with
    a least-squares correction on the active set each step (the fully corrective
    variant, which converges in a few iterations). ``L`` must upper-bound the
    largest eigenvalue of ``Sxx``. Several starts (the univariate top-``k`` plus
    random supports) are tried and the support with the largest explained sum of
    squares is returned.

    This is a heuristic: the support need not be optimal. It only seeds the
    incumbent of the exact Furnival-Wilson search, which still certifies the true
    optimum -- so a good warm start prunes more, and a poor one costs nothing but
    correctness is never at risk.
    """
    N = Sxx.shape[0]
    k = min(k, N)
    if k <= 0:
        return []
    if rng is None:
        rng = np.random.default_rng(0)
    if not np.isfinite(L) or L <= 0:
        L = 1.0

    starts = [np.argsort(-np.abs(Sxy))[:k]]      # univariate top-k
    for _ in range(restarts):
        starts.append(rng.choice(N, size=k, replace=False))

    best_support: List[int] = []
    best_expl = -np.inf
    for s0 in starts:
        support = sorted({int(i) for i in s0})
        beta, _ = _ls_on_support(Sxx, Sxy, support, N)
        for _ in range(max_iter):
            grad = Sxx @ beta - Sxy
            c = beta - grad / L
            top = np.arange(N) if k >= N else np.argpartition(-np.abs(c), k - 1)[:k]
            new_support = sorted(int(i) for i in top)
            if new_support == support:
                break
            support = new_support
            beta, _ = _ls_on_support(Sxx, Sxy, support, N)
        _, expl = _ls_on_support(Sxx, Sxy, support, N)
        if expl > best_expl:
            best_expl, best_support = expl, support
    return best_support


def _best_subset_fw(
    G, Zty, yty, N, n, r_max, criterion,
    *,
    warm_start: bool = True,
    seed: int = 0,
    node_budget: Optional[int] = None,
    _stats: Optional[dict] = None,
) -> List[int]:
    """Best subset by the Furnival-Wilson leaps-and-bounds engine.

    The canonical ``leaps::regsubsets`` algorithm: a depth-first search over
    donor subsets driven by the sweep operator, so a donor's residual sum of
    squares is read off the swept augmented matrix in ``O(1)`` (after the
    ``O(p^2)`` descent sweep) rather than re-solved. Donors are ordered
    strongest-first so the incumbent tightens early. Each subtree is lower
    bounded by ``n log(rss_all_in / n) + penalty(k+1)`` -- the smallest RSS the
    subtree can reach (include every remaining donor) paired with the smallest
    penalty any descendant can carry (one more donor than the current node) --
    and pruned when that bound cannot beat the incumbent. The bound is a true
    lower bound on the information criterion, so the returned subset is identical
    to :func:`_best_subset_exhaustive`; the IC optimum is always the best-RSS
    subset at its own size, so this best-subset-then-criterion search returns it.

    With ``warm_start`` the incumbent is seeded from the Bertsimas-King-Mazumder
    discrete first-order method (:func:`_discrete_first_order`) at every size,
    which tightens the bound from the outset and prunes more subtrees. It can
    only lower the incumbent, never raise it, so the certified optimum is
    unchanged. (The seed is skipped past ``_WARMSTART_MAX_N`` donors, where its
    cost would outweigh the pruning it buys.)

    ``node_budget`` caps the number of explored nodes. If the search completes
    within it, the result is the certified optimum. If the budget is hit, the
    search stops early and returns the best incumbent found together with a
    valid lower bound -- the smallest subtree bound it never got to explore --
    so ``optimality_gap = incumbent - lower_bound`` certifies how far the
    returned subset can be from optimal (zero means provably optimal). ``_stats``
    (if given) receives ``nodes``, ``budget_hit``, ``lower_bound``,
    ``incumbent_ic``, ``optimality_gap`` and ``certified``.
    """
    M = _augmented(G, Zty, yty, N)
    y_idx = N + 1
    _sweep(M, 0)                                 # intercept always in
    tss = M[y_idx, y_idx]                         # intercept-only RSS

    # Strongest-first donor order by univariate (post-intercept) RSS reduction;
    # ``d0`` is each donor's standalone residual variance, the scale against
    # which a collapsed pivot signals collinearity with the swept set.
    d0 = np.array([M[j + 1, j + 1] for j in range(N)])
    red = np.array([
        (M[j + 1, y_idx] ** 2 / d0[j]) if d0[j] > 1e-14 else 0.0
        for j in range(N)
    ])
    order = sorted(range(N), key=lambda j: red[j], reverse=True)

    best_ic = [info_criterion(tss, n, 1, criterion)]
    best_idx: List[int] = []
    best_rss = np.full(r_max + 1, np.inf)
    best_rss[0] = tss

    if warm_start and r_max >= 1 and N <= _WARMSTART_MAX_N:
        Sxx, Sxy, _ = _centered_gram(G, Zty, yty, n)
        L = float(np.linalg.eigvalsh(Sxx)[-1]) if N else 1.0
        rng = np.random.default_rng(seed)
        for k_size in range(1, r_max + 1):
            supp = _discrete_first_order(Sxx, Sxy, k_size, L, rng=rng)
            if not supp:                          # pragma: no cover - DFO returns
                continue                          # a non-empty support for k>=1
            rss = _subset_rss(G, Zty, yty, supp)
            ic = info_criterion(rss, n, len(supp) + 1, criterion)
            if ic < best_ic[0]:
                best_ic[0] = ic
                best_idx[:] = supp
            if rss < best_rss[len(supp)]:
                best_rss[len(supp)] = rss

    nodes = [0]
    budget_hit = [False]
    frontier_lb = [np.inf]

    def recurse(chosen: List[int], pos: int, k: int) -> None:
        nodes[0] += 1
        if k >= r_max or pos >= N:
            return
        cur_rss = M[y_idx, y_idx]
        rem_idx = [order[p] + 1 for p in range(pos, N)]
        rss_all = _all_in_rss(M, rem_idx, y_idx, cur_rss)
        # Lower bound over the subtree: descendants have size >= k+1, so pair the
        # all-in RSS with the (k+1)-donor penalty. Prune if it cannot improve.
        lb = info_criterion(rss_all, n, k + 2, criterion)
        if lb >= best_ic[0]:
            return
        # Over budget: leave this (un-prunable) subtree unexplored, but record its
        # lower bound so the reported gap stays a valid certificate.
        if node_budget is not None and nodes[0] >= node_budget:
            budget_hit[0] = True
            frontier_lb[0] = min(frontier_lb[0], lb)
            return
        for p in range(pos, N):
            d = order[p]
            idx = d + 1
            collinear = M[idx, idx] <= 1e-9 * d0[d] or M[idx, idx] <= 1e-14
            if not collinear:
                _sweep(M, idx)
            rss = M[y_idx, y_idx]
            k2 = k + 1
            ic = info_criterion(rss, n, k2 + 1, criterion)
            if ic < best_ic[0]:
                best_ic[0] = ic
                best_idx[:] = chosen + [d]
            if rss < best_rss[k2]:
                best_rss[k2] = rss
            recurse(chosen + [d], p + 1, k2)
            if not collinear:
                _unsweep(M, idx)                 # restore for the next branch

    recurse([], 0, 0)
    if budget_hit[0]:
        lower_bound = min(frontier_lb[0], best_ic[0])
    else:                                        # search completed -> optimum certified
        lower_bound = best_ic[0]
    gap = max(0.0, best_ic[0] - lower_bound)
    if _stats is not None:
        _stats.update(
            nodes=nodes[0],
            budget_hit=budget_hit[0],
            lower_bound=lower_bound,
            incumbent_ic=best_ic[0],
            optimality_gap=gap,
            certified=gap <= 1e-9,
        )
    return sorted(best_idx)


def fit_hcw(
    y: np.ndarray,
    X: np.ndarray,
    T0: int,
    *,
    criterion: str = "AICc",
    nvmax: Optional[int] = None,
    backend: str = "fw",
    select_stats: Optional[dict] = None,
) -> Tuple[List[int], np.ndarray, float, np.ndarray]:
    """HCW best-subset fit: select donors, refit OLS, extrapolate counterfactual.

    Returns ``(selected_indices, beta_full, intercept, counterfactual)`` where
    ``beta_full`` is an ``N``-vector with zeros off the selected support and the
    counterfactual is the OLS extrapolation ``X @ beta_full + intercept`` over
    all periods. ``backend`` selects the search engine ('fw' or 'scip'); pass a
    ``select_stats`` dict to receive the best-subset search diagnostics (node
    count, optimality gap, certification).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    N = X.shape[1]
    selected = best_subset_select(
        y, X, T0, criterion=criterion, nvmax=nvmax, backend=backend,
        stats=select_stats)

    beta_full = np.zeros(N)
    if not selected:                             # intercept-only counterfactual
        intercept_val = float(np.mean(y[:T0]))
        return [], beta_full, intercept_val, np.full(X.shape[0], intercept_val)

    Z = np.column_stack([np.ones(T0), X[:T0, selected]])
    coef, *_ = np.linalg.lstsq(Z, y[:T0], rcond=None)
    intercept_val = float(coef[0])
    beta_full[selected] = coef[1:]
    counterfactual = X @ beta_full + intercept_val
    return selected, beta_full, intercept_val, counterfactual
