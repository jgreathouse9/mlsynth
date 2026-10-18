"""Readable port of Lv & Li (2026): the oracle for an RRMA build.

Jing Lv and Jialiang Li, "Rank-based Model Average Estimator for Average
Treatment Effects", arXiv:2610.15553v1 (11 October 2026). The authors released
no code; they name the R functions they call, and each function below names the
section of the paper it implements and the R function it stands in for.

Notation follows the paper. ``y`` is the treated unit's outcome path, ``X`` the
``T x (N-1)`` donor outcome matrix, ``T1`` the number of pre-treatment periods.
Candidate model ``s`` regresses ``y`` on the donor columns ``groups[s]``.

* ``cov_rob_mve`` is ``MASS::cov.rob(method = "mve")`` (MASS 7.3-60.0.1,
  ``R/lqs.R`` and ``src/lqs.c``), ported line for line. With fewer than 5000
  subsets of size ``p + 1`` it enumerates them in R's order and is
  deterministic; past that it samples, and NumPy's draws are not R's.
* ``wlad`` is the weighted least-absolute-deviation fit Remark 1 sends to
  ``quantreg::rq``. It solves the dual linear programme with HiGHS; the primal
  has one row per pair of periods and is 50 to 180 times slower at the sizes
  the Monte Carlo uses.
* ``kendall_pvalues`` is ``cor.test(method = "kendall")``: exact below 50
  observations without ties, the normal approximation otherwise.

Choices the paper leaves open, each recorded where it is made:

* the Mallows weights ``h`` of a candidate model are computed once from its
  donor columns over all ``T1`` pre-periods and restricted to each training
  window of the growing-window cross-validation. Equation (11) writes the same
  symbol ``b_ll'`` as equation (8), and the weights read only donor outcomes;
* Algorithm 1 nests its loop over ``t`` outside its loop over ``m``, but
  ``Q(m)`` sums over ``t``. The port runs the loop over ``m`` outside: for each
  ``m``, the cross-validated predictions over the whole validation window, then
  the model weights, then ``Q(m)``;
* when fewer donors survive the screen than the smallest group size, the
  candidate set is one model holding every surviving donor, or every donor when
  none survives. Section 4.1 does not cover the case.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from itertools import combinations
from math import comb
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linprog, nnls
from scipy.stats import chi2, kendalltau, norm

from mlsynth.utils.solvers.active_set import solve_simplex_qp

#: Section 4.1: donors whose Kendall p-value exceeds this are dropped.
SCREEN_LEVEL = 0.05
#: Section 2.3: the Mallows cut-off is this quantile of chi-square(p).
MALLOWS_LEVEL = 0.95
#: ``do_one`` in ``src/lqs.c``: the rank tolerance handed to ``dqrdc2``.
_QR_TOL = 1.0e-7
#: Subsets scored per batch in ``cov_rob_mve``, bounding memory.
_CHUNK = 1000


class RobustCovError(ValueError):
    """``cov.rob`` stops: too few rows, an IQR of zero, or every subset singular."""


# ----------------------------------------------------------- robust covariance


def _quantile7(x: np.ndarray, q: float, axis: int = 0) -> np.ndarray:
    """R's default ``quantile`` (type 7), which is NumPy's ``linear``."""
    return np.quantile(x, q, axis=axis, method="linear")


def _inv_like_r(A: np.ndarray) -> np.ndarray:
    """``solve(A)`` as ``mahalanobis`` calls it: stop when ``rcond(A) < eps``.

    A block bootstrap repeats rows, so the rows ``cov.rob`` keeps can all be
    copies of fewer than ``p + 1`` points; R stops there with "system is
    computationally singular", and so does this.
    """
    if not np.all(np.isfinite(A)) or np.linalg.cond(A, 1) * np.finfo(float).eps > 1.0:
        raise RobustCovError("system is computationally singular")
    return np.linalg.inv(A)


def _score_subsets(x: np.ndarray, subsets: np.ndarray, qn: int):
    """``do_one`` and the MVE criterion of ``mve_fitlots`` for each subset.

    Returns ``(crit, lim, d2, singular)``: the criterion ``p log(lim) + 2 det``,
    the ``qn``-th smallest squared distance, the distances of every row from the
    subset's mean in the subset's covariance, and whether ``dqrdc2`` would have
    found the subset rank deficient.
    """
    n, p = x.shape
    nnew = subsets.shape[1]
    Z = x[subsets]                                       # (K, nnew, p)
    means = Z.mean(axis=1)
    Z = Z - means[:, None, :]
    R = np.linalg.qr(Z, mode="r")                        # (K, p, p)
    diag = np.abs(np.diagonal(R, axis1=1, axis2=2))
    # dqrdc2 drops a column whose norm, after projecting out the columns before
    # it, falls below tol times its original norm; the unpivoted |R_kk| is that
    # norm. A zero column has its reference norm set to one, as in dqrdc2.
    colnorm = np.linalg.norm(Z, axis=1)
    colnorm = np.where(colnorm == 0.0, 1.0, colnorm)
    singular = np.any(diag < _QR_TOL * colnorm, axis=1)
    K = subsets.shape[0]
    crit = np.full(K, np.inf)
    lim = np.full(K, np.nan)
    d2 = np.full((K, n), np.nan)
    ok = ~singular
    if ok.any():
        Rk = R[ok]
        resid = x[None, :, :] - means[ok][:, None, :]    # (k, n, p)
        # mah(): solve R' b = (x_i - mean), squared length times (nnew - 1).
        b = np.linalg.solve(np.transpose(Rk, (0, 2, 1)), np.transpose(resid, (0, 2, 1)))
        dist = np.sum(b * b, axis=1) * (nnew - 1)        # (k, n)
        lim_k = np.partition(dist, qn - 1, axis=1)[:, qn - 1]
        det = np.sum(np.log(diag[ok]), axis=1)
        crit[ok] = p * np.log(lim_k) + 2.0 * det
        lim[ok] = lim_k
        d2[ok] = dist
    return crit, lim, d2, singular


def cov_rob_mve(x: np.ndarray, rng: Optional[np.random.Generator] = None):
    """``MASS::cov.rob(x, method = "mve")``, Rousseeuw and van Zomeren (1990).

    Returns ``(center, cov, info)``. ``info`` carries ``exact`` (whether every
    subset was scored), ``nsamp`` and ``singular`` (the subset counts), and
    ``best`` (the 0-based rows ``cov.rob`` reports as ``best``).
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise RobustCovError("x must be a matrix")
    n, p = x.shape
    if not np.all(np.isfinite(x)):
        raise RobustCovError("missing or infinite values are not allowed")
    if n < p + 1:
        raise RobustCovError(f"at least {p + 1} cases are needed")
    qn = (n + p + 1) // 2
    if qn < p + 1:
        raise RobustCovError(f"'quantile' must be at least {p + 1}")
    if qn > n - 1:
        raise RobustCovError(f"'quantile' must be at most {n - 1}")
    divisor = _quantile7(x, 0.75) - _quantile7(x, 0.25)
    if np.any(divisor == 0):
        raise RobustCovError("at least one column has IQR 0")
    xs = x / divisor
    ps = p + 1
    nexact = comb(n, ps)
    exact = nexact < 5000
    if exact:
        nsamp = nexact
    else:
        nsamp = min(500 * ps, 3000)
        rng = np.random.default_rng() if rng is None else rng

    best_crit, best_one, n_sing = np.inf, None, 0
    it = combinations(range(n), ps) if exact else None
    done = 0
    while done < nsamp:
        k = min(_CHUNK, nsamp - done)
        if exact:
            subsets = np.array([next(it) for _ in range(k)], dtype=int)
        else:
            subsets = np.argsort(rng.random((k, n)), axis=1)[:, :ps]
        crit, lim, d2, singular = _score_subsets(xs, subsets, qn)
        n_sing += int(singular.sum())
        # Strict "<" in mve_fitlots: the first subset reaching the minimum wins.
        j = int(np.argmin(crit))
        if crit[j] < best_crit:
            best_crit = crit[j]
            best_one = d2[j] <= lim[j]
        done += k
    if best_one is None or not best_one.any():
        raise RobustCovError("'x' is probably collinear")

    best = np.flatnonzero(best_one)
    means = xs[best].mean(axis=0)
    rcov = np.cov(xs[best], rowvar=False).reshape(p, p) * (1 + 15 / (n - p)) ** 2
    diff = xs - means
    dist = np.einsum("ij,ij->i", diff @ _inv_like_r(rcov), diff)
    cut = chi2.ppf(0.975, p) * _quantile7(dist, qn / n) / chi2.ppf(qn / n, p)
    keep = dist < cut
    if keep.sum() < 2:
        raise RobustCovError("fewer than two rows inside the cut")
    cov = np.cov(xs[keep], rowvar=False).reshape(p, p) * np.outer(divisor, divisor)
    center = xs[keep].mean(axis=0) * divisor
    info = dict(exact=exact, nsamp=int(nsamp), singular=n_sing, best=best)
    return center, cov, info


def mallows(x: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Section 2.3: ``h(x_t) = min{1, nu_p / d2(x_t)}`` from the MVE fit.

    ``nu_p`` is the 0.95 quantile of chi-square(p) and ``d2`` the squared
    Mahalanobis distance of row ``t`` from the MVE centre in the MVE covariance.
    Raises :class:`RobustCovError` where ``cov.rob`` stops.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    center, cov, _ = cov_rob_mve(x, rng)
    diff = x - center
    d2 = np.einsum("ij,ij->i", diff @ _inv_like_r(cov), diff)
    nu = chi2.ppf(MALLOWS_LEVEL, x.shape[1])
    with np.errstate(divide="ignore"):
        return np.minimum(1.0, np.where(d2 > 0, nu / d2, np.inf))


def mallows_or_ones(x: np.ndarray, rng: Optional[np.random.Generator] = None):
    """``(h, True)`` from :func:`mallows`, or ``(ones, False)`` where it stops."""
    try:
        return mallows(x, rng), True
    except RobustCovError:
        return np.ones(np.asarray(x).shape[0]), False


# ------------------------------------------------------------------ the L1 fits


@lru_cache(maxsize=None)
def pairs(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """The ``n(n-1)/2`` index pairs ``t < t'``.

    Equations (8) and (11) sum over ordered pairs ``t != t'``; the term for
    ``(t', t)`` equals the term for ``(t, t')``, so unordered pairs give the same
    minimiser at half the rows.
    """
    i, j = np.triu_indices(n, 1)
    return i, j


#: HiGHS settings tried in turn. The first 500-replication run met one
#: programme on which the default returned status 15 (model status Unknown,
#: primal feasible); the dual simplex without presolve, then the interior-point
#: method, are the retries.
_HIGHS_ATTEMPTS = (
    dict(method="highs"),
    dict(method="highs-ds", options=dict(presolve=False)),
    dict(method="highs-ipm"),
)
#: Detours taken, per process: ``retry`` counts programmes a later HiGHS setting
#: solved, ``pairfree`` counts fits ``rank_fit`` handed to the pair-free solver.
LP_FALLBACKS = {"retry": 0, "pairfree": 0}


def _linprog_retrying(*args, **kwargs):
    """``linprog`` over ``_HIGHS_ATTEMPTS``; the first result with status 0, else the last."""
    for k, attempt in enumerate(_HIGHS_ATTEMPTS):
        res = linprog(*args, **kwargs, **attempt)
        if res.status == 0:
            LP_FALLBACKS["retry"] += k > 0
            return res
    return res


def wlad(y: np.ndarray, D: np.ndarray, c: np.ndarray) -> np.ndarray:
    """``argmin_beta sum_k c_k |y_k - D_k beta|`` (Remark 1's ``rq`` call).

    Solved as the dual programme ``max y'd s.t. D'd = 0, -c <= d <= c``; the
    coefficients are the negated multipliers of the equality rows. Raises
    ``RuntimeError`` when no HiGHS setting returns an optimum.
    """
    res = _linprog_retrying(-y, A_eq=D.T, b_eq=np.zeros(D.shape[1]),
                            bounds=np.column_stack([-c, c]))
    if res.status != 0:
        raise RuntimeError(f"weighted LAD failed: {res.message}")
    return -np.asarray(res.eqlin.marginals, dtype=float)


def simplex_wlad(y: np.ndarray, D: np.ndarray, c: np.ndarray) -> np.ndarray:
    """``argmin_w sum_k c_k |y_k - D_k w|`` over the simplex (Section 2.4's LP).

    Dual: ``max y'd + mu s.t. D'd + mu 1 <= 0, -c <= d <= c``; the weights are
    the negated multipliers of the inequality rows.
    """
    S = D.shape[1]
    if S == 1:
        return np.ones(1)
    cost = -np.r_[y, 1.0]
    A_ub = np.column_stack([D.T, np.ones(S)])
    bounds = np.vstack([np.column_stack([-c, c]), [-np.inf, np.inf]])
    res = _linprog_retrying(cost, A_ub=A_ub, b_ub=np.zeros(S), bounds=bounds)
    if res.status != 0:                                  # pragma: no cover - every HiGHS setting failing
        raise RuntimeError(f"simplex LAD failed: {res.message}")
    w = np.clip(-np.asarray(res.ineqlin.marginals, dtype=float), 0.0, None)
    return w / w.sum()


def rank_fit(y: np.ndarray, X: np.ndarray, h: Optional[np.ndarray] = None):
    """Section 2.3: weighted Wilcoxon slope (8), median intercept.

    ``h`` holds the Mallows weights of the rows; ``b_tt' = h_t h_t'``. ``None``
    is the unweighted Wilcoxon fit. Returns ``(alpha, beta)``. When HiGHS
    fails under every setting the fit goes to ``solver_pairfree``, which
    certifies its own optimum, and ``LP_FALLBACKS["pairfree"]`` counts it.
    """
    n = len(y)
    i, j = pairs(n)
    c = np.ones(len(i)) if h is None else h[i] * h[j]
    try:
        beta = wlad(y[i] - y[j], X[i] - X[j], c)
    except RuntimeError:
        from . import solver_pairfree as PF

        LP_FALLBACKS["pairfree"] += 1
        hh = (np.ones(n) if h is None else h)[None]
        beta = PF.solve(y[None], X[None], hh, PF.crash(y[None], X[None], hh))[0][0]
    alpha = float(np.median(y - X @ beta))
    return alpha, beta


def ols_fit(y: np.ndarray, X: np.ndarray, h: Optional[np.ndarray] = None):
    """Least squares with an intercept; ``h`` is accepted and ignored."""
    Z = np.column_stack([np.ones(len(y)), X])
    coef, *_ = np.linalg.lstsq(Z, y, rcond=None)
    return float(coef[0]), coef[1:]


def mscm_fit(y: np.ndarray, X: np.ndarray):
    """Li (2020): intercept plus non-negative donor weights, no adding-up."""
    Xc, yc = X - X.mean(axis=0), y - y.mean()
    beta, _ = nnls(Xc, yc)
    return float(y.mean() - X.mean(axis=0) @ beta), beta


# -------------------------------------------------------- candidate models (4.1)


def kendall_pvalues(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """``cor.test(y, X[, j], method = "kendall")$p.value`` for each column."""
    n = len(y)
    out = np.empty(X.shape[1])
    y_ties = len(np.unique(y)) < n
    for j in range(X.shape[1]):
        ties = y_ties or len(np.unique(X[:, j])) < n
        method = "exact" if (n < 50 and not ties) else "asymptotic"
        out[j] = kendalltau(y, X[:, j], method=method).pvalue
    return out


def screen(y: np.ndarray, X: np.ndarray, level: float = SCREEN_LEVEL):
    """Equation (13): donors with p <= level, ordered by p (R's stable ``order``).

    Returns ``(order, pvalues, ties)``; ``ties`` counts surviving donors whose
    p-value equals another survivor's, so that their place in ``order`` comes
    from their column position.
    """
    p = kendall_pvalues(y, X)
    keep = np.flatnonzero(p <= level)
    order = keep[np.argsort(p[keep], kind="stable")]
    _, counts = np.unique(p[order], return_counts=True)
    ties = int(counts[counts > 1].sum())
    return order, p, ties


def partition(order: np.ndarray, m: int) -> List[np.ndarray]:
    """Equation (14): ``S = floor(d/m)`` groups of ``m``, the last takes the rest."""
    d = len(order)
    S = d // m
    if S == 0:
        return []
    groups = [order[s * m:(s + 1) * m] for s in range(S - 1)]
    groups.append(order[(S - 1) * m:])
    return groups


def candidate_sets(order: np.ndarray, n_donors: int, m_range: Sequence[int]):
    """``{m: groups}`` over Algorithm 1's grid, with the fallback for small ``d``."""
    sets = {m: g for m in m_range if (g := partition(order, m))}
    if not sets:
        pool = order if len(order) else np.arange(n_donors)
        sets = {len(pool): [pool]}
    return sets


# ---------------------------------------------------------- weights (2.4) and Q


def validation_window(T1: int) -> np.ndarray:
    """Section 2.4: ``t = T1/2 + 1, ..., T1`` (0-based ``T1 // 2 .. T1 - 1``)."""
    return np.arange(T1 // 2, T1)


def gwcv_predictions(y: np.ndarray, X: np.ndarray, T1: int, groups, fit, hs) -> np.ndarray:
    """Equation (11): each model fit on ``1..t-1``, predicting period ``t``."""
    val = validation_window(T1)
    P = np.empty((len(val), len(groups)))
    for k, t in enumerate(val):
        for s, g in enumerate(groups):
            Xs = X[:t][:, g]
            h = None if hs[s] is None else hs[s][:t]
            a, b = fit(y[:t], Xs, h)
            P[k, s] = a + X[t, g] @ b
    return P


def full_predictions(y: np.ndarray, X: np.ndarray, T1: int, groups, fit, hs):
    """Equation (9): each model fit on all pre-periods, predicting every period."""
    M = np.empty((X.shape[0], len(groups)))
    thetas = []
    for s, g in enumerate(groups):
        a, b = fit(y[:T1], X[:T1][:, g], hs[s])
        M[:, s] = a + X[:, g] @ b
        thetas.append((a, b))
    return M, thetas


def weigh_pairwise(y_val: np.ndarray, P: np.ndarray, rng=None):
    """Section 2.4: simplex weights minimising the tau-weighted pairwise L1 loss.

    Returns ``(w, tau_ok)``; ``tau_ok`` is False when ``cov.rob`` stops on the
    predictions and the Mallows weights fall back to one.
    """
    hbar, ok = mallows_or_ones(P, rng) if P.shape[1] > 1 else (np.ones(len(y_val)), True)
    i, j = pairs(len(y_val))
    w = simplex_wlad(y_val[i] - y_val[j], P[i] - P[j], hbar[i] * hbar[j])
    return w, ok


def weigh_level(y_val: np.ndarray, P: np.ndarray, rng=None):
    """Ablation: the same programme on levels, ``sum_t hbar_t |y_t - P_t w|``."""
    hbar, ok = mallows_or_ones(P, rng) if P.shape[1] > 1 else (np.ones(len(y_val)), True)
    return simplex_wlad(y_val, P, hbar), ok


def weigh_ls(y_val: np.ndarray, P: np.ndarray, rng=None):
    """Ablation: simplex weights minimising squared validation error."""
    if P.shape[1] == 1:
        return np.ones(1), True
    return np.asarray(solve_simplex_qp(P, y_val), dtype=float), True


def weigh_equal(y_val: np.ndarray, P: np.ndarray, rng=None):
    """The paper's EW arm: equal weights on the candidate models."""
    return np.full(P.shape[1], 1.0 / P.shape[1]), True


# ---------------------------------------------------------------- Algorithm 1


@dataclass
class Path:
    """One value of ``m``: its candidate models and their predictions."""

    m: int
    groups: List[np.ndarray]
    hs: List[Optional[np.ndarray]]
    P: np.ndarray                 # cross-validated predictions, validation window
    M: np.ndarray                 # full-sample predictions, every period
    thetas: list
    mallows_fallbacks: int = 0


@dataclass
class Choice:
    """Algorithm 1's output under one weighting rule."""

    m: int
    groups: List[np.ndarray]
    w: np.ndarray
    counterfactual: np.ndarray
    Q: Dict[int, float] = field(default_factory=dict)
    tau_fallbacks: int = 0


def candidate_paths(y, X, T1, m_range, fit=rank_fit, mallows_weights=True,
                    rng=None, order=None) -> Dict[int, Path]:
    """The per-``m`` half of Algorithm 1: models, GWCV and full-sample predictions."""
    if order is None:
        order, _, _ = screen(y[:T1], X[:T1])
    paths = {}
    for m, groups in candidate_sets(order, X.shape[1], m_range).items():
        hs, falls = [], 0
        for g in groups:
            if mallows_weights:
                h, ok = mallows_or_ones(X[:T1][:, g], rng)
                falls += not ok
                hs.append(h)
            else:
                hs.append(None)
        P = gwcv_predictions(y, X, T1, groups, fit, hs)
        M, thetas = full_predictions(y, X, T1, groups, fit, hs)
        paths[m] = Path(m, groups, hs, P, M, thetas, falls)
    return paths


def choose(paths: Dict[int, Path], y, T1, weigh, loss: str = "abs", rng=None) -> Choice:
    """The selection half of Algorithm 1: weights per ``m``, then ``m = argmin Q``.

    ``loss`` is ``"abs"`` for the paper's average absolute prediction error and
    ``"sq"`` for the least-squares ablation.
    """
    y_val = y[validation_window(T1)]
    best, Q, falls = None, {}, 0
    for m, path in paths.items():
        w, ok = weigh(y_val, path.P, rng)
        falls += not ok
        e = y_val - path.P @ w
        Q[m] = float(np.mean(np.abs(e)) if loss == "abs" else np.mean(e * e))
        if best is None or Q[m] < Q[best[0]]:
            best = (m, w)
    m, w = best
    return Choice(m, paths[m].groups, w, paths[m].M @ w, Q, falls)


def rrma(y, X, T1, m_range=range(3, 11), rng=None, order=None):
    """RRMA as the paper specifies it: returns ``(choice, paths)``."""
    paths = candidate_paths(y, X, T1, m_range, rank_fit, True, rng, order)
    return choose(paths, y, T1, weigh_pairwise, "abs", rng), paths


# ------------------------------------------------------------ inference (4.2)


def mbb_indices(T1: int, rng: np.random.Generator) -> np.ndarray:
    """Section 4.2: ``k = ceil(T1/l)`` overlapping blocks of ``l = floor(T1^(1/3))``."""
    l = int(np.floor(T1 ** (1 / 3) + 1e-12))
    k = int(np.ceil(T1 / l))
    starts = rng.integers(0, T1 - l + 1, size=k)
    return (starts[:, None] + np.arange(l)[None, :]).ravel()[:T1]


def rrma_variance(y, X, T1, groups, w, rng, B: int = 1000):
    """Section 4.2 and Theorem 3: the variance of ``Delta_1(w)``.

    ``(1/T2) phi^2 H Cov(sqrt(T1) theta) H' + Sigma_v / T2`` with
    ``phi^2 = T2/T1``, ``Cov`` from the moving-block bootstrap of the
    pre-period rows (each model refit by Section 2.3, Mallows weights
    included), ``H^(s) = (w_s, w_s mean_post(X^(s)))`` and
    ``Sigma_v = mean(v_t^2)``, ``v_t`` the post-period gaps minus their mean.
    Returns ``(variance, parts)``.
    """
    T = len(y)
    T2 = T - T1
    hs = [mallows_or_ones(X[:T1][:, g], rng)[0] for g in groups]
    _, thetas = full_predictions(y, X, T1, groups, rank_fit, hs)
    cf = sum(w[s] * (thetas[s][0] + X[:, g] @ thetas[s][1]) for s, g in enumerate(groups))
    gaps = y[T1:] - cf[T1:]
    draws, boot_falls = [], 0
    for _ in range(B):
        idx = mbb_indices(T1, rng)
        yb, Xb = y[idx], X[idx]
        theta_b = []
        for g in groups:
            h, ok = mallows_or_ones(Xb[:, g], rng)
            boot_falls += not ok
            a, b = rank_fit(yb, Xb[:, g], h)
            theta_b.extend([a, *b])
        draws.append(theta_b)
    draws = np.asarray(draws)
    cov_sqrt = T1 * np.cov(draws, rowvar=False, ddof=1)
    H = np.concatenate([np.r_[w[s], w[s] * X[T1:, g].mean(axis=0)] for s, g in enumerate(groups)])
    v = gaps - gaps.mean()
    sigma_v = float(np.mean(v * v))
    term_theta = float(H @ cov_sqrt @ H) * (T2 / T1) / T2
    var = term_theta + sigma_v / T2
    return var, dict(att=float(gaps.mean()), term_theta=term_theta,
                     term_v=sigma_v / T2, boot_mallows_fallbacks=boot_falls)


def normal_ci(att: float, var: float, level: float) -> Tuple[float, float]:
    """Equation (15)."""
    z = norm.ppf(0.5 + level / 2)
    se = np.sqrt(var)
    return att - z * se, att + z * se
