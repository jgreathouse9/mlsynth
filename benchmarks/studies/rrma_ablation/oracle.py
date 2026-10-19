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
* ``rank_fit`` is the weighted Wilcoxon fit Remark 1 sends to
  ``quantreg::rq``, solved by ``solver_pairfree``: NumPy, no array with a pair
  dimension, each optimum certified exactly. The cross-validation fits are
  batched over models of one size and warm-started from window to window.
  Section 2.4's model weights and the weights-on-levels ablation are solved by
  ``lp``. Nothing here imports ``scipy.optimize``; HiGHS stays in
  ``highs_reference`` as the check the tests compare against.
* ``kendall_pvalues`` is ``cor.test(method = "kendall")``: exact below 50
  observations without ties, the normal approximation otherwise. It and the
  chi-square and normal quantiles come from ``special``, so the oracle imports
  nothing from SciPy.

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

from mlsynth.utils.solvers.active_set import solve_simplex_qp
from mlsynth.utils.solvers.nnls import nnls

from . import lp as LP
from . import special as SP
from . import solver_pairfree as PF

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
    cut = SP.chi2_ppf(0.975, p) * _quantile7(dist, qn / n) / SP.chi2_ppf(qn / n, p)
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
    nu = SP.chi2_ppf(MALLOWS_LEVEL, x.shape[1])
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


def rank_fit(y: np.ndarray, X: np.ndarray, h: Optional[np.ndarray] = None):
    """Section 2.3: weighted Wilcoxon slope (8), median intercept.

    ``h`` holds the Mallows weights of the rows; ``b_tt' = h_t h_t'``. ``None``
    is the unweighted Wilcoxon fit. Returns ``(alpha, beta)``.
    """
    hh = (np.ones(len(y)) if h is None else np.asarray(h, dtype=float))[None]
    yy, XX = np.asarray(y, dtype=float)[None], np.asarray(X, dtype=float)[None]
    beta = PF.solve(yy, XX, hh, PF.crash(yy, XX, hh))[0][0]
    return float(np.median(y - X @ beta)), beta


def rank_fits(y: np.ndarray, X: np.ndarray, hs: Sequence[Optional[np.ndarray]], ns: Sequence[int]):
    """``rank_fit`` of each model ``X[:, :, s]`` on the first ``n`` rows, for every ``n`` in ``ns``.

    ``X`` is ``(K, T, p)``: models of one size, solved together. ``ns`` must
    increase; each window starts from the previous window's optimal labelling
    with the new rows as groups of their own, which is a vertex of the larger
    problem. Returns ``{n: (alpha (K,), beta (K, p))}``.
    """
    K, T, p = X.shape
    H = np.stack([np.ones(T) if h is None else np.asarray(h, dtype=float) for h in hs])
    Y = np.broadcast_to(np.asarray(y, dtype=float), (K, len(y)))
    out, labels = {}, None
    for n in ns:
        yy, XX, hh = Y[:, :n], X[:, :n], H[:, :n]
        if labels is None:
            lab0 = PF.crash(yy, XX, hh)
        else:
            new = np.broadcast_to(np.arange(labels.shape[1], n), (K, n - labels.shape[1]))
            lab0 = np.concatenate([labels, new], axis=1)
        beta, labels, _, _ = PF.solve(yy, XX, hh, lab0)
        alpha = np.median(yy - np.einsum("knp,kp->kn", XX, beta), axis=1)
        out[n] = (alpha, beta)
    return out


def ols_fit(y: np.ndarray, X: np.ndarray, h: Optional[np.ndarray] = None):
    """Least squares with an intercept; ``h`` is accepted and ignored."""
    Z = np.column_stack([np.ones(len(y)), X])
    coef, *_ = np.linalg.lstsq(Z, y, rcond=None)
    return float(coef[0]), coef[1:]


def mscm_fit(y: np.ndarray, X: np.ndarray):
    """Li (2020): intercept plus non-negative donor weights, no adding-up."""
    Xc, yc = X - X.mean(axis=0), y - y.mean()
    beta, _ = nnls(Xc, yc)                       # mlsynth's Lawson-Hanson, NumPy
    return float(y.mean() - X.mean(axis=0) @ beta), beta


# -------------------------------------------------------- candidate models (4.1)


def kendall_pvalues(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """``cor.test(y, X[, j], method = "kendall")$p.value`` for each column (``special``)."""
    return np.array([SP.kendall_pvalue(y, X[:, j]) for j in range(X.shape[1])])


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


def _by_size(groups):
    """Indices of ``groups`` grouped by group size, so each batch shares ``p``."""
    sizes = {}
    for s, g in enumerate(groups):
        sizes.setdefault(len(g), []).append(s)
    return sizes


def gwcv_predictions(y: np.ndarray, X: np.ndarray, T1: int, groups, fit, hs) -> np.ndarray:
    """Equation (11): each model fit on ``1..t-1``, predicting period ``t``."""
    val = validation_window(T1)
    P = np.empty((len(val), len(groups)))
    if fit is rank_fit:
        for idx in _by_size(groups).values():
            Xb = np.stack([X[:, groups[s]] for s in idx])
            fits = rank_fits(y, Xb, [hs[s] for s in idx], [int(t) for t in val])
            for k, t in enumerate(val):
                alpha, beta = fits[int(t)]
                P[k, idx] = alpha + np.einsum("kp,kp->k", Xb[:, t], beta)
        return P
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
    thetas = [None] * len(groups)
    if fit is rank_fit:
        for idx in _by_size(groups).values():
            Xb = np.stack([X[:, groups[s]] for s in idx])
            alpha, beta = rank_fits(y, Xb, [hs[s] for s in idx], [T1])[T1]
            for k, s in enumerate(idx):
                thetas[s] = (float(alpha[k]), beta[k])
                M[:, s] = alpha[k] + X[:, groups[s]] @ beta[k]
        return M, thetas
    for s, g in enumerate(groups):
        a, b = fit(y[:T1], X[:T1][:, g], hs[s])
        M[:, s] = a + X[:, g] @ b
        thetas[s] = (a, b)
    return M, thetas


def weigh_pairwise(y_val: np.ndarray, P: np.ndarray, rng=None):
    """Section 2.4: simplex weights minimising the tau-weighted pairwise L1 loss.

    Returns ``(w, tau_ok)``; ``tau_ok`` is False when ``cov.rob`` stops on the
    predictions and the Mallows weights fall back to one.
    """
    hbar, ok = mallows_or_ones(P, rng) if P.shape[1] > 1 else (np.ones(len(y_val)), True)
    return LP.simplex_rank_lad(y_val, P, hbar), ok


def weigh_level(y_val: np.ndarray, P: np.ndarray, rng=None):
    """Ablation: the same programme on levels, ``sum_t hbar_t |y_t - P_t w|``."""
    hbar, ok = mallows_or_ones(P, rng) if P.shape[1] > 1 else (np.ones(len(y_val)), True)
    return LP.simplex_lad(y_val, P, hbar), ok


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
    # The draws and their Mallows weights in the order the generator is read,
    # then each model's B refits solved as one batch.
    idx_b, h_b, boot_falls = [], [[] for _ in groups], 0
    for _ in range(B):
        idx = mbb_indices(T1, rng)
        idx_b.append(idx)
        for s, g in enumerate(groups):
            h, ok = mallows_or_ones(X[idx][:, g], rng)
            boot_falls += not ok
            h_b[s].append(h)
    idx_b = np.asarray(idx_b)
    Yb = y[idx_b]
    cols = []
    for s, g in enumerate(groups):
        Xb, Hb = X[idx_b][:, :, g], np.asarray(h_b[s])
        beta, _, _, _ = PF.solve(Yb, Xb, Hb, PF.crash(Yb, Xb, Hb))
        alpha = np.median(Yb - np.einsum("knp,kp->kn", Xb, beta), axis=1)
        cols.append(np.column_stack([alpha, beta]))
    draws = np.hstack(cols)
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
    z = SP.norm_ppf(0.5 + level / 2)
    se = np.sqrt(var)
    return att - z * se, att + z * se
