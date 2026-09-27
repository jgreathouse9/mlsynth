"""Doubly robust dynamic ATT of Zhang (2026), Algorithm 1 (simultaneous case).

Four variants, matching the paper's simulation columns:

    DR2   two folds, m and pi on the SAME training fold      (standard cross-fit)
    DR3   three folds, m and pi on SEPARATE folds            (the proposal)
    DR2*  DR2 with the true |alpha_i - alpha_j| in place of d-hat   (infeasible)
    DR3*  DR3 with the true |alpha_i - alpha_j| in place of d-hat   (infeasible)

Four things Algorithm 1 leaves open; each is a named option here and each is
reported, because the port has to choose and the paper does not:

  kernel          Assumption 5.1(vi) asks for a bounded, Lipschitz kernel on
                  [-1, 1] with K(0) > 0. Epanechnikov by default.
  bandwidth grid  "two lists of bandwidths", contents unstated.
  scale           K((X_j-X_i)/h) K(d_ij/h) shares ONE h across two arguments on
                  unrelated scales -- here d-hat is ~12.5 |alpha_i-alpha_j| while
                  X is a fertility-rate-like level. "std" divides each argument
                  by its own dispersion first; "raw" is the formula as printed.
  trimming        (1 - pi-hat) sits in a denominator and NW can return pi-hat = 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

Z95 = 1.959963984540054


def epanechnikov(u: np.ndarray) -> np.ndarray:
    out = 1.0 - u * u
    np.maximum(out, 0.0, out=out)
    return 0.75 * out


def uniform_kernel(u: np.ndarray) -> np.ndarray:
    return (np.abs(u) <= 1.0).astype(float)


KERNELS = {"epanechnikov": epanechnikov, "uniform": uniform_kernel}


@dataclass
class Diagnostics:
    h_m: float = np.nan
    h_pi: float = np.nan
    h_m_at_edge: bool = False
    h_pi_at_edge: bool = False
    trimmed_share: float = 0.0
    empty_m_share: float = 0.0
    n_eff_median: float = np.nan
    extras: dict = field(default_factory=dict)


def _fold_plan(n: int, n_folds: int, rng: np.random.Generator):
    """Fold id per unit, plus the (outcome, propensity) source fold for each id.

    Paper footnote 7: i in I_s draws m from I_{1+(s mod 3)} and pi from
    I_{1+((s+1) mod 3)}. Zero-indexed that is (s+1) % 3 and (s+2) % 3. With two
    folds both nuisances come from the other fold, which is the DR2 column.
    """
    fold = rng.permutation(n) % n_folds
    if n_folds == 3:
        src_m = [(s + 1) % 3 for s in range(3)]
        src_pi = [(s + 2) % 3 for s in range(3)]
    else:
        src_m = [(s + 1) % n_folds for s in range(n_folds)]
        src_pi = list(src_m)
    return fold, src_m, src_pi


def _weights(dist: np.ndarray, xdiff: np.ndarray, h: float, kernel) -> np.ndarray:
    W = kernel(xdiff / h) * kernel(dist / h)
    np.fill_diagonal(W, 0.0)          # every sum below excludes the unit itself
    return W


def _select_bandwidths(
    dist, xdiff, D, fold, grid_pi, grid_m, kernel, kappa, q_floor, n,
):
    """Algorithm 1-CV step 3: LOO CV for h_pi, effective-sample-size rule for h_m."""
    own = fold[:, None] == fold[None, :]

    best_pi, best_loss = grid_pi[0], np.inf
    for h in grid_pi:
        W = _weights(dist, xdiff, h, kernel) * own
        denom = W.sum(axis=1)
        num = W @ D
        pred = np.divide(num, denom, out=np.full(n, D.mean()), where=denom > 0)
        loss = float(np.mean((D - pred) ** 2))
        if loss < best_loss:
            best_pi, best_loss = h, loss

    threshold = (kappa * np.log(n)) ** 2
    control = (1 - D).astype(float)
    chosen_m, n_eff_med = grid_m[-1], np.nan
    for h in grid_m:
        W = _weights(dist, xdiff, h, kernel) * own * control[None, :]
        s1 = W.sum(axis=1)
        s2 = (W * W).sum(axis=1)
        n_eff = np.divide(s1 * s1, s2, out=np.zeros(n), where=s2 > 0)
        if np.mean(n_eff > threshold) >= q_floor:
            chosen_m, n_eff_med = h, float(np.median(n_eff))
            break
    return best_pi, chosen_m, n_eff_med


def dr_att(
    panel,
    *,
    horizon: int = 0,
    n_folds: int = 3,
    oracle: bool = False,
    metric: str = "zhang_range",
    scale: str = "std",
    kernel: str = "epanechnikov",
    kappa: float = 0.2,
    q_floor: float = 0.8,
    n_grid: int = 14,
    trim: float = 0.01,
    fallback: str = "nn",
    rng: np.random.Generator,
    dist_cache: np.ndarray | None = None,
) -> tuple[float, float, Diagnostics]:
    """Estimate ATT(T0 + horizon) and its standard error."""
    from distance import pseudo_distance

    Y, D, T0 = panel.Y, panel.D, panel.T0
    n = Y.shape[0]
    kern = KERNELS[kernel]

    if oracle:
        dist = np.abs(panel.alpha[:, None] - panel.alpha[None, :])
    elif dist_cache is not None:
        dist = dist_cache
    else:
        dist = pseudo_distance(Y[:, :T0], metric)

    # X_{i,T0} is the covariate treatment selects on: the last pre-period outcome.
    X = Y[:, T0 - 1]
    xdiff = np.abs(X[:, None] - X[None, :])

    off = ~np.eye(n, dtype=bool)
    if scale == "std":
        sd_d = dist[off].std() or 1.0
        sd_x = xdiff[off].std() or 1.0
        dist, xdiff = dist / sd_d, xdiff / sd_x
        grid = np.geomspace(0.02, 3.0, n_grid)
    elif scale == "raw":
        span = max(dist[off].std(), xdiff[off].std()) or 1.0
        grid = np.geomspace(0.01 * span, 3.0 * span, n_grid)
    else:
        raise ValueError(f"unknown scale {scale!r}")

    fold, src_m, src_pi = _fold_plan(n, n_folds, rng)
    h_pi, h_m, n_eff_med = _select_bandwidths(
        dist, xdiff, D, fold, grid, grid, kern, kappa, q_floor, n
    )

    src_m_of = np.asarray(src_m)[fold]
    src_pi_of = np.asarray(src_pi)[fold]
    in_m = src_m_of[:, None] == fold[None, :]
    in_pi = src_pi_of[:, None] == fold[None, :]

    Wm = _weights(dist, xdiff, h_m, kern) * in_m * (1 - D)[None, :]
    y_out = Y[:, T0 + horizon]
    dm = Wm.sum(axis=1)
    m_hat = np.divide(Wm @ y_out, dm, out=np.zeros(n), where=dm > 0)
    empty = dm <= 0
    if empty.any():
        if fallback == "mean":            # crude: the pooled control mean
            m_hat[empty] = y_out[D == 0].mean()
        else:
            # Nearest eligible control in the product-kernel sense: the smallest
            # radius at which j enters i's kernel support is max(dist, xdiff).
            radius = np.maximum(dist, xdiff)
            blocked = ~(in_m & (D == 0)[None, :])
            radius = np.where(blocked, np.inf, radius)
            nearest = radius[empty].argmin(axis=1)
            m_hat[empty] = y_out[nearest]

    Wp = _weights(dist, xdiff, h_pi, kern) * in_pi
    dp = Wp.sum(axis=1)
    pi_hat = np.divide(Wp @ D.astype(float), dp, out=np.full(n, D.mean()), where=dp > 0)
    trimmed = (pi_hat > 1 - trim) | (pi_hat < trim)
    pi_hat = np.clip(pi_hat, trim, 1 - trim)

    resid = y_out - m_hat
    psi = np.where(D == 1, resid, -pi_hat / (1.0 - pi_hat) * resid)
    n1 = int(D.sum())
    att = float(psi.sum() / n1)
    se = float(np.sqrt(((psi - D * att) ** 2).sum()) / n1)

    diag = Diagnostics(
        h_m=float(h_m),
        h_pi=float(h_pi),
        h_m_at_edge=bool(h_m in (grid[0], grid[-1])),
        h_pi_at_edge=bool(h_pi in (grid[0], grid[-1])),
        trimmed_share=float(trimmed.mean()),
        empty_m_share=float(empty.mean()),
        n_eff_median=n_eff_med,
        extras={"ci_width": 2 * Z95 * se, "max_abs_psi": float(np.abs(psi).max())},
    )
    return att, se, diag
