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


def _reference_stats(resids, post_slice, q, *, conformal_type, ns, seed):
    """Reference statistics for the conformal test (augsynth's permutation set).

    ``"iid"`` draws ``ns`` random permutations of the residual path;
    ``"block"`` uses the ``T`` **cyclic shifts** of the path (the moving-block
    scheme, ``resids[(0:T-1+j) %% T]`` in augsynth) -- deterministic, preserving
    serial dependence. The post-block statistic is taken on ``post_slice`` of
    each permuted/shifted path.
    """
    if conformal_type == "block":
        n = resids.shape[0]
        return np.array(
            [_stat(np.roll(resids, s)[post_slice], q) for s in range(n)],
            dtype=float,
        )
    if conformal_type == "iid":
        # Same ns permutations and draws as the per-permutation loop
        # ``[_stat(rng.permutation(resids)[post_slice], q) for _ in range(ns)]``,
        # but the statistic is computed once over the whole (ns, |post|) block
        # instead of ns times. Bit-identical to the loop (pinned in the tests);
        # only the per-call Python/ufunc overhead is removed.
        resids = np.asarray(resids, dtype=float)
        rng = np.random.default_rng(seed)
        idx = np.arange(resids.shape[0])[post_slice]      # slice or fancy index
        m = idx.size
        P = np.empty((ns, m), dtype=float)
        for i in range(ns):
            P[i] = rng.permutation(resids)[idx]
        return (np.sum(np.abs(P) ** q, axis=1) / np.sqrt(m)) ** (1.0 / q)
    raise ValueError(f"conformal_type must be 'iid' or 'block'; got {conformal_type!r}.")


def _augmented_gaps(
    y: np.ndarray,
    Y0: np.ndarray,
    Z0: Optional[np.ndarray],
    z1: Optional[np.ndarray],
    ridge_kwargs: Dict[str, Any],
    warm_start: Optional[np.ndarray] = None,
    return_base: bool = False,
):
    """Refit ridge ASCM matching on the full (given) outcome window; return gaps.

    Passing the *entire* outcome path as the matching window is exactly augsynth's
    conformal refit (``X <- cbind(X, y)``): the ASCM balances every period and
    the residuals are the treated-minus-synthetic gaps over the whole window.
    ``warm_start`` seeds the base simplex solve (e.g. the previous tau0 in a grid
    sweep); ``return_base`` also returns the base weights so the caller can chain
    the warm start. The default (no warm start, gaps only) is unchanged.
    """
    ra = ridge_augment_weights(y, Y0, Z0=Z0, z1=z1, warm_start=warm_start,
                               **ridge_kwargs)
    gaps = np.asarray(y, dtype=float) - np.asarray(Y0, dtype=float) @ ra.W
    if return_base:
        return gaps, ra.W_base
    return gaps


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
    conformal_type: str = "iid",
    fixed_effects: bool = False,
    finite_sample: bool = False,
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
    finite_sample : bool, default False
        Return ``(1 + #{stat >= obs}) / (1 + ns)`` instead of
        ``mean(obs <= perm)``. The plain mean is augsynth's convention and is
        kept as the default because ``geox_augsynth_geolift`` reproduces
        GeoLift, which inherits it -- but it returns exactly ``0`` when the
        observed statistic beats every permutation, and zero is not a p-value:
        ``ns`` draws cannot evidence more than ``1 / (ns + 1)``.
    fixed_effects : bool, default False
        Unit fixed effects (augsynth ``fixed_effects=TRUE``): demean every unit
        by its mean over the full matching window before the refit, so the donor
        pool cannot absorb a post-period level shift. Use the same value as the
        point fit.
    ridge_kwargs : dict, optional
        Other hyper-parameters forwarded to :func:`ridge_augment_weights`.

    Returns
    -------
    float
        The conformal p-value ``mean(stat(observed_post) <= stat(permuted_post))``,
        or its finite-sample correction when ``finite_sample`` is set.
    """
    ridge_kwargs = dict(ridge_kwargs or {})
    if lambda_ is not None:
        ridge_kwargs["lambda_"] = lambda_
    y = np.asarray(y, dtype=float)
    Y0 = np.asarray(Y0, dtype=float)
    if fixed_effects:
        # augsynth's conformal refit (``cbind(X, y)`` then ``fixedeff``) demeans
        # every unit by its mean over the *whole* matching window -- here the
        # full path -- so the donor pool cannot absorb a post-period level shift.
        y = y - y.mean()
        Y0 = Y0 - Y0.mean(axis=0)
    resids = _augmented_gaps(y, Y0, Z0, z1, ridge_kwargs)
    obs = _stat(resids[pre:], q)
    perm = _reference_stats(
        resids, slice(pre, None), q, conformal_type=conformal_type, ns=ns, seed=seed
    )
    if finite_sample:
        return float((1.0 + np.sum(obs <= perm)) / (1.0 + perm.size))
    return float(np.mean(obs <= perm))




def conformal_att_interval(
    y: np.ndarray,
    Y0: np.ndarray,
    pre: int,
    *,
    alpha: float = 0.05,
    max_span: float = 512.0,
    rtol: float = 0.02,
    max_iter: int = 32,
    **kwargs: Any,
) -> Tuple[float, float]:
    """Interval for a constant ATT, by inverting the joint test.

    ``conformal_intervals`` inverts period by period, which gives a band around
    the path; coverage of the *average* effect needs the set of constant effects
    the joint test does not reject, which is what this returns.

    The search brackets and bisects instead of scanning a grid. A fixed grid
    fails here: the non-rejected set is narrow relative to the post-window gap
    dispersion it would have to be scaled by, so a grid wide enough to be safe
    steps over the set and reports it empty. Bracketing finds each crossing to a
    relative tolerance whatever its width.

    From the point estimate, the effect is doubled outward until the test
    rejects, which brackets a crossing; that bracket is then bisected. Both
    crossings are found separately, so the interval need not be symmetric --
    the test's own asymmetry is preserved.

    The point estimate is where the search starts, not where it gives up. The
    refit reads every period, so shifting the post block by a candidate changes
    the fit as well as the residual and the most conforming candidate can sit
    well away from the estimate -- on Recast A1 seed 23 the estimate is rejected
    at ``p = 0.010`` while the test accepts a band 30% below it. When the
    estimate is rejected the search therefore relocates to the candidate that
    minimises the test statistic, found by ternary search (the statistic is
    unimodal in the candidate), and only calls the set empty if that is rejected
    too.

    Two answers are not numbers, and both are real.

    ``(nan, nan)`` is an empty confidence set: no constant effect conforms, the
    most conforming one included. A multiplicative lift on a trending panel
    produces it, because the null being inverted is a *constant* effect.

    ``-inf`` or ``+inf`` on a side is an unbounded one: the test still accepts
    ``max_span`` pre-period sigmas out, so nothing in that direction is
    excluded. Block conformal makes this reachable by construction -- its
    reference set is the ``T`` cyclic shifts, one of which is the observed path,
    so the p-value cannot fall below ``1 / T`` and at ``alpha < 1 / T`` no
    candidate is ever rejected. Returning ``max_span`` sigmas as if it were a
    bound would invent a rejection and report an interval narrower than the
    truth.
    """
    y = np.asarray(y, dtype=float)
    Y0 = np.asarray(Y0, dtype=float)
    kwargs.setdefault("conformal_type", "block")
    ridge_kwargs = {k: v for k, v in (("lambda_", kwargs.get("lambda_")),)
                    if v is not None}

    # Centre on the pre-period fit, as `conformal_intervals` does: an
    # all-period refit treats the post block as matching periods and pulls its
    # gaps toward zero, which would centre the search away from the effect.
    if kwargs.get("fixed_effects"):
        y_m, Y0_m = y[:pre].mean(), Y0[:pre].mean(axis=0)
        ra = ridge_augment_weights(y[:pre] - y_m, Y0[:pre] - Y0_m, **ridge_kwargs)
        fitted = (y - y_m) - (Y0 - Y0_m) @ ra.W
    else:
        ra = ridge_augment_weights(y[:pre], Y0[:pre], **ridge_kwargs)
        fitted = y - Y0 @ ra.W
    centre = float(np.mean(fitted[pre:]))
    scale = float(np.std(fitted[:pre])) or abs(centre) or 1.0

    def accepts(tau0: float) -> bool:
        shifted = y.copy()
        shifted[pre:] -= tau0
        return conformal_pvalue(shifted, Y0, pre, **kwargs) > alpha

    def statistic(tau0: float) -> float:
        """The observed post-block statistic under ``H0: tau = tau0``.

        The same quantity ``conformal_pvalue`` compares against its reference
        set, without the reference set. It is continuous in the candidate where
        the p-value is a step function of granularity ``1 / T``, so it is what
        the relocation search can descend.
        """
        shifted = y.copy()
        shifted[pre:] -= tau0
        Z = Y0
        if kwargs.get("fixed_effects"):
            shifted = shifted - shifted.mean()
            Z = Z - Z.mean(axis=0)
        gaps = _augmented_gaps(shifted, Z, kwargs.get("Z0"), kwargs.get("z1"),
                               ridge_kwargs)
        return _stat(gaps[pre:], kwargs.get("q", 1.0))

    def most_conforming() -> float:
        """The candidate minimising the statistic, by ternary search."""
        left, right = centre - max_span * scale, centre + max_span * scale
        for _ in range(max_iter * 2):
            if right - left <= rtol * scale:
                break
            a, b = left + (right - left) / 3.0, right - (right - left) / 3.0
            if statistic(a) <= statistic(b):
                right = b
            else:
                left = a
        return 0.5 * (left + right)

    if not accepts(centre):
        centre = most_conforming()
        if not accepts(centre):
            return float("nan"), float("nan")

    def crossing(sign: int) -> float:
        """First rejection outward from the centre, located by bisection."""
        step, far = scale, None
        while step <= max_span * scale:
            if not accepts(centre + sign * step):
                far = centre + sign * step
                break
            step *= 2.0
        if far is None:                     # nothing excluded in this direction
            return sign * float("inf")
        near = centre + sign * (step / 2.0) if step > scale else centre
        for _ in range(max_iter):
            mid = 0.5 * (near + far)
            if abs(far - near) <= rtol * scale:
                break
            if accepts(mid):
                near = mid
            else:
                far = mid
        return float(near)

    return crossing(-1), crossing(+1)


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
    warm_start: Optional[np.ndarray] = None,
    conformal_type: str = "iid",
    fixed_effects: bool = False,
):
    """Conformal p-value for ``H0: tau_j = tau0`` at a single post period ``j``.

    Matches on the pre-periods plus post period ``j`` (with its treated outcome
    adjusted by ``tau0``); the post-block here is the single period ``j``.
    Returns ``(p_value, base_weights)``; ``warm_start`` seeds the base solve and
    the returned base weights chain the warm start across a tau0 grid.
    """
    y = np.asarray(y, dtype=float).copy()
    Y0 = np.asarray(Y0, dtype=float)
    # matching window: pre-periods + the single post period j (adjusted by tau0)
    idx = list(range(pre)) + [j]
    y_fit = y[idx].copy()
    y_fit[-1] -= tau0
    Y0_fit = Y0[idx]
    if fixed_effects:
        # demean by the mean over this matching window (augsynth fixedeff refit)
        y_fit = y_fit - y_fit.mean()
        Y0_fit = Y0_fit - Y0_fit.mean(axis=0)
    resids, w_base = _augmented_gaps(y_fit, Y0_fit, Z0, z1, ridge_kwargs,
                                     warm_start=warm_start, return_base=True)
    obs = _stat(resids[-1:], q)
    perm = _reference_stats(
        resids, slice(-1, None), q, conformal_type=conformal_type, ns=ns, seed=seed
    )
    return float(np.mean(obs <= perm)), w_base


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
    conformal_type: str = "iid",
    fixed_effects: bool = False,
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
    # collapsing the gaps toward zero and mis-centering the search grid. Under
    # fixed effects, demean each unit by its pre-period mean and restore the
    # level (the residual gap is invariant to the treated mean), mirroring
    # ``fit_augsynth_once(fixed_effects=True)``.
    if fixed_effects:
        y_pre_m = y[:pre].mean()
        Y0_pre_m = Y0[:pre].mean(axis=0)
        ra = ridge_augment_weights(y[:pre] - y_pre_m, Y0[:pre] - Y0_pre_m,
                                   Z0=Z0, z1=z1, **ridge_kwargs)
        gap_full = (y - y_pre_m) - (Y0 - Y0_pre_m) @ ra.W
    else:
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
        # Sweep the tau0 grid warm-starting each base solve from the previous
        # (nearly identical) one -- the refits collapse to ~0 pivots.
        ps = np.empty(grid.size)
        w_prev = None
        for gi, tau0 in enumerate(grid):
            ps[gi], w_prev = _period_pvalue(
                y, Y0, pre, j, tau0, Z0, z1, q, ns, seed, ridge_kwargs,
                warm_start=w_prev, conformal_type=conformal_type,
                fixed_effects=fixed_effects,
            )
        kept = grid[ps >= alpha]
        if kept.size:
            lower[k] = float(kept.min())
            upper[k] = float(kept.max())
        pvals[k] = float(ps[grid == 0.0][0])

    joint_p = conformal_pvalue(
        y, Y0, pre, Z0=Z0, z1=z1, q=q, ns=ns, seed=seed,
        conformal_type=conformal_type, fixed_effects=fixed_effects,
        ridge_kwargs=ridge_kwargs,
    )
    return ConformalIntervals(
        periods=post, att=att, lower=lower, upper=upper,
        p_value=pvals, joint_p_value=joint_p, alpha=alpha,
    )
