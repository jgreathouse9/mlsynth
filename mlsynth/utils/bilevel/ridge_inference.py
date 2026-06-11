"""Conformal inference for ridge-augmented SCM (Ben-Michael, Feller & Rothstein
2021, §5.4; augsynth ``inf_type="conformal"``, the package default).

The conformal procedure of Chernozhukov, Wuthrich & Zhu (2019) adapted to ASCM:
for a sharp null ``H0: tau = tau0`` it subtracts ``tau0`` from the treated
post-treatment outcomes, **refits** the ridge ASCM treating the (adjusted) post
periods as additional matching periods, and asks whether the post-treatment
residual "conforms" with the pre-treatment residuals -- comparing the post-block
test statistic ``(sum|x|^q / sqrt(n))^(1/q)`` against permutations of the
residual path.

Two products, mirroring augsynth:

* :func:`conformal_pvalue` -- the joint-null p-value (the ``( p )`` printed next
  to the Average ATT). The treated outcomes are unchanged (``tau0 = 0``), the
  ASCM is refit on **all** periods, and the post-block statistic is compared to
  ``ns`` i.i.d. permutations of the residuals.
* :func:`conformal_intervals` -- per-period confidence intervals (Eq. 29) by
  inverting the test on a grid of nulls for each post period.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .ridge_augment import ridge_augment_weights


def _stat(x: np.ndarray, q: float = 1.0) -> float:
    """Conformal test statistic ``(sum|x|^q / sqrt(len))^(1/q)`` (augsynth q=1)."""
    x = np.asarray(x, dtype=float)
    return float((np.sum(np.abs(x) ** q) / np.sqrt(x.size)) ** (1.0 / q))


def _augmented_gaps(
    y: np.ndarray,
    Y0: np.ndarray,
    Z0: Optional[np.ndarray],
    z1: Optional[np.ndarray],
    ridge_kwargs: Dict[str, Any],
) -> np.ndarray:
    """Refit ridge ASCM matching on the full (given) outcome window; return gaps.

    Passing the *entire* outcome path as the matching window is exactly augsynth's
    conformal refit (``X <- cbind(X, y)``): the ASCM balances every period and
    the residuals are the treated-minus-synthetic gaps over the whole window.
    """
    ra = ridge_augment_weights(y, Y0, Z0=Z0, z1=z1, **ridge_kwargs)
    return np.asarray(y, dtype=float) - np.asarray(Y0, dtype=float) @ ra.W


def conformal_pvalue(
    y: np.ndarray,
    Y0: np.ndarray,
    pre: int,
    *,
    lambda_: Optional[float] = None,
    Z0: Optional[np.ndarray] = None,
    z1: Optional[np.ndarray] = None,
    q: float = 1.0,
    ns: int = 1000,
    seed: Optional[int] = 0,
    ridge_kwargs: Optional[Dict[str, Any]] = None,
) -> float:
    """Conformal p-value for the joint null of no post-treatment effect.

    Parameters
    ----------
    y : np.ndarray, shape (T,)
        Treated outcomes over all periods.
    Y0 : np.ndarray, shape (T, J)
        Donor outcomes over all periods.
    pre : int
        Number of pre-treatment periods.
    lambda_ : float, optional
        Ridge penalty to **reuse** in the refit. augsynth fixes the penalty at
        the originally CV-selected value for every conformal refit (it does not
        re-cross-validate); pass the fitted ``lambda_`` to reproduce it. ``None``
        re-selects by CV on each refit (slower, and slightly anti-conservative).
    Z0, z1 : np.ndarray, optional
        Auxiliary covariates (donor ``(J, K)`` / treated ``(K,)``).
    q : float
        Norm for the test statistic (augsynth default ``1``).
    ns : int
        Number of i.i.d. residual permutations (augsynth default ``1000``).
    seed : int, optional
        RNG seed for the permutations.
    ridge_kwargs : dict, optional
        Other hyper-parameters forwarded to :func:`ridge_augment_weights`.

    Returns
    -------
    float
        The conformal p-value ``mean(stat(observed_post) <= stat(permuted_post))``.
    """
    ridge_kwargs = dict(ridge_kwargs or {})
    if lambda_ is not None:
        ridge_kwargs["lambda_"] = lambda_
    resids = _augmented_gaps(y, Y0, Z0, z1, ridge_kwargs)
    obs = _stat(resids[pre:], q)
    rng = np.random.default_rng(seed)
    perm = np.fromiter(
        (_stat(rng.permutation(resids)[pre:], q) for _ in range(ns)),
        dtype=float, count=ns,
    )
    return float(np.mean(obs <= perm))


@dataclass
class ConformalIntervals:
    """Per-period conformal intervals from :func:`conformal_intervals`.

    Attributes
    ----------
    periods : list
        The post-period indices (0-based, relative to the full series).
    att : np.ndarray
        Point estimate (gap) per post period.
    lower, upper : np.ndarray
        ``1 - alpha`` confidence bounds per post period.
    p_value : np.ndarray
        Per-period p-value for a null of zero effect.
    joint_p_value : float
        The joint-null p-value (:func:`conformal_pvalue`).
    alpha : float
        Level.
    """

    periods: List[int]
    att: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    p_value: np.ndarray
    joint_p_value: float
    alpha: float


def _period_pvalue(
    y: np.ndarray,
    Y0: np.ndarray,
    pre: int,
    j: int,
    tau0: float,
    Z0: Optional[np.ndarray],
    z1: Optional[np.ndarray],
    q: float,
    ns: int,
    seed: Optional[int],
    ridge_kwargs: Dict[str, Any],
) -> float:
    """Conformal p-value for ``H0: tau_j = tau0`` at a single post period ``j``.

    Matches on the pre-periods plus post period ``j`` (with its treated outcome
    adjusted by ``tau0``); the post-block here is the single period ``j``.
    """
    y = np.asarray(y, dtype=float).copy()
    Y0 = np.asarray(Y0, dtype=float)
    # matching window: pre-periods + the single post period j (adjusted by tau0)
    idx = list(range(pre)) + [j]
    y_fit = y[idx].copy()
    y_fit[-1] -= tau0
    Y0_fit = Y0[idx]
    resids = _augmented_gaps(y_fit, Y0_fit, Z0, z1, ridge_kwargs)
    obs = _stat(resids[-1:], q)
    rng = np.random.default_rng(seed)
    perm = np.fromiter(
        (_stat(rng.permutation(resids)[-1:], q) for _ in range(ns)),
        dtype=float, count=ns,
    )
    return float(np.mean(obs <= perm))


def conformal_intervals(
    y: np.ndarray,
    Y0: np.ndarray,
    pre: int,
    *,
    lambda_: Optional[float] = None,
    Z0: Optional[np.ndarray] = None,
    z1: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    q: float = 1.0,
    ns: int = 1000,
    grid_size: int = 50,
    seed: Optional[int] = 0,
    ridge_kwargs: Optional[Dict[str, Any]] = None,
) -> ConformalIntervals:
    """Per-period conformal confidence intervals by test inversion (Eq. 29).

    For each post period a grid of nulls spanning ``+/- 2`` post-RMSE around the
    point estimate is tested; the interval is the range of nulls not rejected at
    level ``alpha``. Also returns the joint-null p-value. ``lambda_`` is reused
    across refits (augsynth's behaviour; pass the fitted penalty).
    """
    ridge_kwargs = dict(ridge_kwargs or {})
    if lambda_ is not None:
        ridge_kwargs["lambda_"] = lambda_
    y = np.asarray(y, dtype=float)
    Y0 = np.asarray(Y0, dtype=float)
    T = y.shape[0]
    post = list(range(pre, T))

    # Point effect from the standard pre-period ASCM fit -- the same
    # counterfactual that is plotted -- so the inversion interval is centered on
    # (and contains) it. The earlier all-period refit over-fits the post period,
    # collapsing the gaps toward zero and mis-centering the search grid.
    ra = ridge_augment_weights(y[:pre], Y0[:pre], Z0=Z0, z1=z1, **ridge_kwargs)
    gap_full = y - Y0 @ ra.W
    att = gap_full[pre:]
    # Grid width from the noise scale (pre-period RMSE), wide enough to bracket
    # the non-rejected region around each period's point effect.
    scale = float(np.sqrt(np.mean(gap_full[:pre] ** 2)))
    half = max(6.0 * scale, 2.0 * float(np.sqrt(np.mean(att ** 2))), 1e-9)

    lower = np.full(len(post), np.nan)
    upper = np.full(len(post), np.nan)
    pvals = np.full(len(post), np.nan)
    for k, j in enumerate(post):
        grid = np.linspace(att[k] - half, att[k] + half, grid_size)
        grid = np.append(grid, 0.0)
        ps = np.array([
            _period_pvalue(y, Y0, pre, j, tau0, Z0, z1, q, ns, seed, ridge_kwargs)
            for tau0 in grid
        ])
        kept = grid[ps >= alpha]
        if kept.size:
            lower[k] = float(kept.min())
            upper[k] = float(kept.max())
        pvals[k] = float(ps[grid == 0.0][0])

    joint_p = conformal_pvalue(
        y, Y0, pre, Z0=Z0, z1=z1, q=q, ns=ns, seed=seed, ridge_kwargs=ridge_kwargs
    )
    return ConformalIntervals(
        periods=post, att=att, lower=lower, upper=upper,
        p_value=pvals, joint_p_value=joint_p, alpha=alpha,
    )
