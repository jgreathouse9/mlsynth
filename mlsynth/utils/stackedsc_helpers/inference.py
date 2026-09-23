"""Placebo inference for STACKEDSC: sampled placebo averages.

With one treated unit the in-space placebo test of Abadie et al. (2010) has an
obvious permutation set -- reassign treatment to each donor in turn and compare
the real gap path to the placebo ones. With many treated units the estimate is
already an average, so the comparison has to be against other *averages*.
Wiltshire (2023) builds them as follows, and ``allsynth`` automates it
(``pvalues(rmspe variance)``, ``stacked(..., sampleavgs(S))``).

For every treated unit ``i`` and every donor ``j`` in that unit's pool, refit
the synthetic control with ``j`` cast as treated at ``i``'s adoption time. That
gives a placebo gap path for each pair. A placebo *average* then draws exactly
one ``j`` per treated unit and averages the drawn paths under the same
``gamma_i`` the estimate uses. The number of distinct averages is the product of
the pool sizes -- 39 donors and 566 treated counties is 39 ** 566 -- so ``S`` of
them are sampled rather than enumerated.

Two statistics come off that distribution:

* RMSPE-ranked p-values. For each post horizon ``E``, the mean squared gap over
  ``e in [0, E]`` divided by the mean squared gap over the pre-window, then the
  share of sampled averages whose ratio reaches the estimate's. Ranking a ratio
  rather than a level discounts averages that fit badly before treatment.
* Placebo-variance p-values and intervals. The spread of the sampled averages at
  each horizon is read as a standard error, following Algorithm 4 of
  Arkhangelsky et al. (2021). This one assumes homoskedasticity across units and
  asymptotic normality; the RMSPE ranking assumes neither.

The donor pool for a placebo
----------------------------

``allsynth`` documents "i and the remaining untreated units collectively
constituting the donor pool for each j", so under the permutation the treated
unit is itself a control. ``placebo_donor_pool="permutation"`` follows that and
is the default. It has two consequences. The placebo path becomes a property of
the pair ``(i, j)`` and not of the cohort, so the cost is the number of treated
units times the pool size. And under the alternative, ``i``'s
post-treatment outcomes carry the effect being tested, so any weight the placebo
puts on ``i`` pulls the placebo gap away from zero and widens the null
distribution. That is the stacked-case power loss Zhang (2019, section 3.1)
describes: as the number of treated units grows, genuinely treated units enter
the null distribution more and more often, moving it away from the null it is
meant to represent.

``placebo_donor_pool="donors-only"`` drops the treated unit. Every unit in a
cohort then faces the same design and the same targets, so one solve per
``(cohort, donor)`` serves the whole cohort, which on the Walmart panel is 6 x
39 instead of 566 x 39.

How the refits are solved
-------------------------

Either pool poses a leave-one-out family: each donor in a pool is cast as
treated and fitted against the rest, which is one matrix's columns fitted
against one another. In Gram form the whole family falls out of a single
``M' M`` -- deleting a column deletes a row and a column of it, and each target
is itself a column -- so
:func:`mlsynth.utils.bilevel.minnorm.solve_simplex_loo_exact` assembles it with
no product with the data per refit. Under ``donors-only`` the matrix is the
cohort's design restricted to the pool and the family is shared by the cohort;
under ``permutation`` the treated unit's column is appended, a donor to every
placebo and a target to none. On the Walmart panel's 22,074 programs that is
101s down to 21s, with every reported statistic unchanged to 1e-11.

The p-values are rank statistics over these fits, so each member is certified
against its own design and the ones that fail are re-solved with the same active
set the loop used.

What is reused instead of refitted
----------------------------------

The predictor weights ``V`` are taken from the cohort's own fit and not
recomputed for each placebo. Recomputing them would mean one regression per
placebo and would change the design matrix the placebos are compared on; holding
``V`` fixed keeps every path in the permutation set on the same scale as the
estimate.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import norm

from ...exceptions import MlsynthDataError
from ..bilevel import bias_corrected_gaps
from ..bilevel.active_set import solve_simplex_qp
from .structures import StackedPlacebo

__all__ = ["cohort_placebo_paths", "sample_placebo_averages",
           "placebo_inference"]

_EPS = 1e-12


def _placebo_weights(A, B, idx, i, donor_pool):
    """Weights for one treated unit's whole placebo pool, in one solve.

    Casting each donor in ``idx`` as treated and refitting against the rest is a
    leave-one-out family over a single matrix, so
    :func:`~mlsynth.utils.bilevel.minnorm.solve_simplex_loo_exact` assembles all
    of it from one Gram. Under ``donors-only`` that matrix is the cohort's design
    restricted to the pool. Under ``permutation`` the treated unit is a control,
    so its column is appended: it is a donor to every placebo and a target to
    none, which is the family's target border.

    Returns ``(len(idx), n_pool)`` with row ``r`` in the column order the caller
    builds its pool in -- ``idx`` minus the pseudo-treated donor, then the
    treated unit under ``permutation`` -- or ``None`` if the family cannot be
    formed, in which case the caller solves one at a time.
    """
    from ..bilevel.minnorm import solve_simplex_loo_exact

    n = len(idx)
    M = (A[:, idx] if donor_pool == "donors-only"
         else np.column_stack([A[:, idx], B[:, i]]))
    try:
        W = solve_simplex_loo_exact(M, n_targets=n)
    except Exception:  # pragma: no cover - fall back to the loop
        return None
    # Row r fits column r, which carries zero weight; drop it so the row lines
    # up with the pool the caller assembles.
    return W[~np.eye(n, W.shape[1], dtype=bool)].reshape(n, W.shape[1] - 1)


def cohort_placebo_paths(
    cohort,
    A: np.ndarray,
    B: np.ndarray,
    take: np.ndarray,
    allowed: Sequence[Sequence[int]],
    *,
    donor_pool: str = "permutation",
    bias_correct: bool = False,
    ridge: float = 1e-2,
) -> Tuple[Dict[str, np.ndarray], int]:
    """Placebo gap paths for one cohort, one row per donor in each unit's pool.

    Parameters
    ----------
    cohort : Cohort
        The cohort, supplying the outcome blocks the gaps are formed on.
    A, B : np.ndarray
        Donor and treated design blocks from :func:`..pipeline._design` -- the
        pre-treatment outcome paths without predictors, the ``sqrt(V)``-scaled
        predictor blocks with them.
    take : np.ndarray
        Boolean row mask selecting the reported event-time horizons.
    allowed : sequence of sequence of int
        Donor column indices available to each treated unit, in the column
        order of ``A``. Equal to every donor unless a donor predicate binds.
    donor_pool : {"permutation", "donors-only"}
        Whether the treated unit joins each placebo's donor pool.
    bias_correct, ridge
        Apply the Abadie-L'Hour correction to the placebo gaps as well, so the
        permutation distribution matches the estimator it is compared against.

    Returns
    -------
    (dict, int)
        ``{unit id as str: (n_allowed, n_horizons)}`` and the number of simplex
        programs actually solved.
    """
    Y, D, X1, X0 = cohort.Y, cohort.D, cohort.X1, cohort.X0
    shared: Dict[Any, np.ndarray] = {}
    batches: Dict[Any, Optional[np.ndarray]] = {}
    paths: Dict[str, np.ndarray] = {}
    n_fits = 0

    for i, unit in enumerate(cohort.units):
        idx = list(allowed[i])
        if len(idx) < 2:
            raise MlsynthDataError(
                f"treated unit {unit!r} has {len(idx)} donor(s) available, so "
                f"no placebo can be formed: casting a donor as treated leaves "
                f"nothing to fit it against. Placebo inference needs at least "
                f"two donors per treated unit.")
        # Under ``donors-only`` the family is a property of the pool, so units
        # sharing one share the solve; under ``permutation`` each unit's own
        # column is in it and no two units share.
        wkey = tuple(idx) if donor_pool == "donors-only" else (tuple(idx), i)
        if wkey not in batches:
            batches[wkey] = _placebo_weights(A, B, idx, i, donor_pool)
        batch = batches[wkey]
        rows: List[np.ndarray] = []
        for r, j in enumerate(idx):
            keep = [k for k in idx if k != j]
            key = (tuple(keep), j) if donor_pool == "donors-only" else None
            if key is not None and key in shared:
                rows.append(shared[key])
                continue
            if donor_pool == "donors-only":
                A_pool, Y_pool = A[:, keep], D[:, keep]
                X_pool = None if X0 is None else X0[:, keep]
            else:
                A_pool = np.column_stack([A[:, keep], B[:, i]])
                Y_pool = np.column_stack([D[:, keep], Y[:, i]])
                X_pool = (None if X0 is None
                          else np.column_stack([X0[:, keep], X1[:, i]]))
            if batch is not None:
                w = batch[r]
            else:
                w = np.asarray(solve_simplex_qp(A_pool, A[:, j]),
                               dtype=float).ravel()
            gap = D[:, j] - Y_pool @ w
            if bias_correct and X_pool is not None:
                gap = bias_corrected_gaps(w, X0[:, j], X_pool, D[:, j], Y_pool,
                                          ridge=ridge)
            n_fits += 1
            g = gap[take]
            if key is not None:
                shared[key] = g
            rows.append(g)
        paths[str(unit)] = np.vstack(rows)
    return paths, n_fits


def sample_placebo_averages(
    paths: Dict[str, np.ndarray],
    units: Sequence[Any],
    gammas: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """``(n_samples, n_horizons)`` placebo averages.

    Each row draws one placebo path per treated unit, independently and
    uniformly from that unit's own pool, and averages them under the estimate's
    aggregation weights.
    """
    first = paths[str(units[0])]
    out = np.zeros((n_samples, first.shape[1]))
    for unit, gamma in zip(units, gammas):
        P = paths[str(unit)]
        out += gamma * P[rng.integers(0, P.shape[0], size=n_samples)]
    return out


def _rmspe(grid: np.ndarray, tau: np.ndarray, placebo: np.ndarray
           ) -> Tuple[Dict[int, float], Dict[int, float]]:
    """RMSPE ratios for the estimate and RMSPE-ranked p-values, by horizon.

    The denominator is the mean squared pre-treatment gap. Under the base-period
    indexing the gap at ``e = -1`` is identically zero, so a pre-window of that
    single period gives ``0 / 0``; the ratios are then undefined and reported as
    NaN rather than as a rank.
    """
    pre, post = grid < 0, grid >= 0
    S = placebo.shape[0]
    actual: Dict[int, float] = {}
    pvals: Dict[int, float] = {}

    denom0 = float(np.mean(tau[pre] ** 2)) if pre.any() else np.nan
    denoms = (np.mean(placebo[:, pre] ** 2, axis=1) if pre.any()
              else np.full(S, np.nan))
    degenerate = ~np.isfinite(denoms) | (denoms <= _EPS)

    for E in grid[post]:
        window = post & (grid <= E)
        num0 = float(np.mean(tau[window] ** 2))
        if not np.isfinite(denom0) or denom0 <= _EPS:
            actual[int(E)] = float("nan")
            pvals[int(E)] = float("nan")
            continue
        stat0 = num0 / denom0
        nums = np.mean(placebo[:, window] ** 2, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            stats = nums / denoms
        # A placebo that reproduces its pre-window exactly has a zero
        # denominator. Its ratio is the limit rather than a value to discard:
        # unbounded if it has any post-treatment gap, and zero if it fits
        # everywhere. Dropping such rows instead would shrink p_E, since the
        # denominator below stays S + 1 either way.
        stats = np.where(degenerate,
                         np.where(nums > _EPS, np.inf, 0.0), stats)
        actual[int(E)] = stat0
        # allsynth's two-sided form: the share of the S sampled averages that
        # reach the estimate's ratio, over S + 1. The estimate itself is not
        # counted in the numerator, so a p-value of exactly zero is attainable.
        pvals[int(E)] = float(np.sum(stats >= stat0) / (S + 1))
    return actual, pvals


def _two_sided(estimate: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Normal two-sided p-value, NaN where the placebo spread is degenerate."""
    est, s = np.atleast_1d(estimate), np.atleast_1d(se)
    out = np.full(est.shape, np.nan)
    ok = np.isfinite(s) & (s > _EPS)
    out[ok] = 2.0 * norm.sf(np.abs(est[ok]) / s[ok])
    return out


def placebo_inference(
    cohorts,
    designs: Sequence[Tuple[np.ndarray, np.ndarray]],
    allowed: Sequence[Sequence[Sequence[int]]],
    grid: np.ndarray,
    tau: np.ndarray,
    gamma_of: Dict[Any, float],
    config,
) -> StackedPlacebo:
    """Run the whole placebo procedure and package the two sets of statistics.

    ``designs`` and ``allowed`` are the per-cohort design blocks and per-unit
    donor index lists the point estimate was fitted from, so the permutation is
    over exactly the pools the estimate used.
    """
    paths: Dict[str, np.ndarray] = {}
    units: List[Any] = []
    n_fits = 0
    for cohort, (A, B), allow in zip(cohorts, designs, allowed):
        e_all = np.arange(cohort.Y.shape[0]) - cohort.t0_index - 1
        take = np.isin(e_all, grid)
        block, fits = cohort_placebo_paths(
            cohort, A, B, take, allow,
            donor_pool=config.placebo_donor_pool,
            bias_correct=config.bias_correct,
            ridge=config.bias_correct_ridge,
        )
        paths.update(block)
        units.extend(cohort.units)
        n_fits += fits

    rng = np.random.default_rng(config.seed)
    gammas = np.array([gamma_of[u] for u in units], dtype=float)
    placebo = sample_placebo_averages(paths, units, gammas,
                                      int(config.n_placebo_samples), rng)

    # Equation (4): the placebo variance is the population variance of the
    # sampled averages, so ddof = 0.
    se = placebo.std(axis=0, ddof=0)
    z = float(norm.ppf(1.0 - config.alpha / 2.0))
    rmspe_actual, rmspe_p = _rmspe(grid, tau, placebo)

    post = grid >= 0
    att = float(np.mean(tau[post])) if post.any() else float("nan")
    att_draws = (placebo[:, post].mean(axis=1) if post.any()
                 else np.full(placebo.shape[0], np.nan))
    att_se = float(np.std(att_draws, ddof=0))
    att_p = float(_two_sided(np.array([att]), np.array([att_se]))[0])

    return StackedPlacebo(
        horizons=grid, n_samples=int(config.n_placebo_samples), n_fits=n_fits,
        donor_pool=config.placebo_donor_pool,
        placebo_averages=placebo, paths=paths,
        se=se, ci_lower=tau - z * se, ci_upper=tau + z * se,
        p_variance=_two_sided(tau, se),
        rmspe_actual=rmspe_actual, rmspe_p=rmspe_p,
        att_se=att_se, att_p_value=att_p,
        att_ci_lower=att - z * att_se, att_ci_upper=att + z * att_se,
        confidence_level=float(1.0 - config.alpha),
    )
