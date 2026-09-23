"""Whether the selected donor cluster can still span the treated unit.

CLUSTERSC selects donors by trajectory similarity and then fits weights
against the survivors. Those two steps answer different questions, and under
a convex-hull weight objective they can disagree: a cluster that is tight in
functional-PC space may have dropped the donors the treated unit needs to be
reachable at all. The clustering objective has no spannability term, so
nothing in the pipeline notices.

West Germany is the case this module was written for. The FPCA cluster drops
the USA, Switzerland and Greece, which together carry 0.549 of the optimal
convex weight on the full donor pool. The best achievable pre-period RMSE
inside the cluster is 522.7 against 60.8 on the pool. That factor of 8.6 is
present in the raw outcomes, before any denoising, and it is invisible under
the default ``nnls`` objective because unconstrained weights absorb it as
extrapolation -- Germany reports 78.2 with ``sum(w) = 1.148``.

The diagnostic compares the best achievable convex fit inside the cluster
against the same quantity on the whole pool. Both are optimisation values,
not fitted residuals, so the ratio measures what the cluster made impossible
and not how well any particular estimator did.
"""

from __future__ import annotations

import warnings
from typing import NamedTuple, Sequence

import cvxpy as cp
import numpy as np

from ...exceptions import MlsynthDataError, MlsynthEstimationError

#: Ratio above which the cluster is reported as having cost real reachability.
#: Germany scores 8.6 and Prop 99 scores 2.2; Basque scores 1.00.
SPANNABILITY_WARN_RATIO: float = 1.5

#: Ratio of the best simplex fit to the best unconstrained fit, against the
#: same donors, above which the convex restriction is reported as binding.
#: The `fgrc_keep="cluster"` blocks score 18.4, 13.6 and 4.1 on Basque,
#: Proposition 99 and West Germany; HSVT and PCP on the same clusters score
#: 1.0 to 2.7; Amjad's full-pool rank-1 blocks score exactly 1.0.
CONVEXITY_WARN_RATIO: float = 2.0


class DenoiseSpannabilityReport(NamedTuple):
    """How much convex reach the denoising step gave up.

    The companion to :class:`SpannabilityReport`, which measures the
    selection step. The two catch opposite failures and neither implies
    the other: West Germany loses a factor of 8.6 at selection and
    nothing at denoising; Basque loses nothing at selection and a factor
    of 3.3 at denoising.
    """

    raw_rmse: float          #: best achievable simplex pre-RMSE on the raw cluster
    denoised_rmse: float     #: the same quantity after denoising
    ratio: float             #: ``denoised_rmse / raw_rmse``; below 1.0 the denoiser helped
    span_rmse: float         #: best UNCONSTRAINED pre-RMSE against the denoised donors
    hull_span_ratio: float   #: ``denoised_rmse / span_rmse``; 1.0 when convexity is free
    weights_identified: bool  #: False when the denoised donors are affinely dependent
    n_donors: int


class SpannabilityReport(NamedTuple):
    """How much convex reach the donor-selection step gave up."""

    cluster_rmse: float    #: best achievable simplex pre-RMSE inside the cluster
    pool_rmse: float       #: the same quantity on the whole donor pool
    ratio: float           #: ``cluster_rmse / pool_rmse``; 1.0 costs nothing
    excluded_mass: float   #: pool-optimal convex weight on donors the cluster dropped
    n_cluster: int
    n_pool: int


def _best_convex_fit(donors: np.ndarray, target: np.ndarray, scale: float):
    """Return ``(weights, rmse)`` for ``min ||target - donors w||`` on the simplex.

    ``scale`` is supplied by the caller so that the cluster and pool solves
    are normalised identically: their residual floors then sit at the same
    place and their ratio is meaningful. Outcome magnitudes across panels span
    five orders (Basque GDP ~10, Prop 99 pack sales ~300, German GDP ~38000),
    and at the top of that range CLARABEL returns ``infeasible`` on a simplex,
    which is never empty. The simplex constraint and the weights are invariant
    to a common factor, so dividing it out changes the answer only by removing
    that failure.
    """
    n_periods, n_donors = donors.shape
    donors_u = donors / scale
    target_u = target / scale
    w = cp.Variable(n_donors)
    problem = cp.Problem(
        cp.Minimize(cp.sum_squares(target_u - donors_u @ w)),
        [w >= 0, cp.sum(w) == 1],
    )
    try:
        problem.solve(solver=cp.CLARABEL)
    except cp.error.SolverError as exc:  # pragma: no cover - solver-install dependent
        raise MlsynthEstimationError(f"Spannability solve failed: {exc}") from exc
    if w.value is None:  # pragma: no cover - CLARABEL returns a point or raises
        raise MlsynthEstimationError(
            f"Spannability solve did not converge (status: {problem.status})."
        )
    weights = np.asarray(w.value, dtype=float)
    rmse = float(np.sqrt(max(float(problem.value), 0.0) / n_periods)) * scale
    return weights, rmse


def assess_spannability(
    donor_pre_pool: np.ndarray,
    treated_pre: np.ndarray,
    cluster_index: Sequence[int],
) -> SpannabilityReport:
    """Measure what restricting to ``cluster_index`` cost in convex reach.

    Parameters
    ----------
    donor_pre_pool : np.ndarray
        Pre-period outcomes for every candidate donor, shape ``(T0, J)``.
    treated_pre : np.ndarray
        Treated unit's pre-period outcomes, shape ``(T0,)``.
    cluster_index : sequence of int
        Column positions of the selected donors within ``donor_pre_pool``.

    Returns
    -------
    SpannabilityReport
        ``ratio`` is 1.0 when the cluster gave up nothing and grows without
        bound as it drops donors the treated unit needs.
    """
    pool = np.asarray(donor_pre_pool, dtype=float)
    target = np.asarray(treated_pre, dtype=float).ravel()
    if pool.ndim != 2:
        raise MlsynthDataError("donor_pre_pool must be 2D (T0, J).")
    if pool.shape[0] != target.shape[0]:
        raise MlsynthDataError(
            f"Pre-period length mismatch: donors have {pool.shape[0]} rows "
            f"but the treated unit has {target.shape[0]}."
        )
    idx = np.asarray(list(cluster_index), dtype=int)
    if idx.size == 0:
        raise MlsynthDataError("cluster_index is empty; a cluster needs a donor.")
    if idx.min() < 0 or idx.max() >= pool.shape[1]:
        raise MlsynthDataError(
            f"cluster_index holds an index outside the donor pool "
            f"(pool has {pool.shape[1]} donors; got {idx.min()}..{idx.max()})."
        )

    scale = max(
        float(np.abs(target).max(initial=0.0)),
        float(np.abs(pool).max(initial=0.0)),
        1e-12,
    )
    pool_weights, pool_rmse = _best_convex_fit(pool, target, scale)
    _cluster_weights, cluster_rmse = _best_convex_fit(pool[:, idx], target, scale)

    kept = np.zeros(pool.shape[1], dtype=bool)
    kept[idx] = True
    excluded_mass = float(np.clip(pool_weights[~kept], 0.0, None).sum())

    # A pool the treated unit already sits on drives pool_rmse to the solver's
    # own tolerance, not to zero -- CLARABEL leaves ~1e-5 relative on an
    # exactly solvable panel -- so the cutoff is that floor, taken against the
    # same scale both solves were normalised by. Below it the ratio is 0/0:
    # report 1.0 when the cluster also reaches the unit and infinity when it
    # does not, instead of dividing by noise.
    negligible = 1e-5 * scale
    if pool_rmse <= negligible:
        ratio = 1.0 if cluster_rmse <= negligible else float("inf")
    else:
        ratio = float(cluster_rmse / pool_rmse)

    return SpannabilityReport(
        cluster_rmse=cluster_rmse,
        pool_rmse=pool_rmse,
        ratio=ratio,
        excluded_mass=excluded_mass,
        n_cluster=int(idx.size),
        n_pool=int(pool.shape[1]),
    )


def warn_if_poorly_spanned(
    report: SpannabilityReport,
    threshold: float = SPANNABILITY_WARN_RATIO,
) -> None:
    """Warn when donor selection cost the treated unit its convex reach."""
    if report.ratio <= threshold:
        return
    warnings.warn(
        f"The selected donor cluster cannot span the treated unit: the best "
        f"achievable convex pre-period RMSE inside the cluster is "
        f"{report.ratio:.1f}x the value on the full donor pool "
        f"({report.cluster_rmse:.4g} vs {report.pool_rmse:.4g}), and donors "
        f"carrying {report.excluded_mass:.2f} of the pool-optimal weight were "
        f"dropped. Under a convex weight objective this shows up as a large "
        f"pre-period error; under an unconstrained one it is absorbed as "
        f"extrapolation (sum(w) > 1) and the fit looks healthy. Widen the "
        f"cluster or fit against the full donor pool.",
        UserWarning,
        stacklevel=3,
    )


def _best_unconstrained_fit(donors: np.ndarray, target: np.ndarray, scale: float) -> float:
    """Return the pre-RMSE of the best fit in the donors' span.

    No sign or adding-up restriction, so this is the floor every linear
    weight objective shares: whatever distance remains here, the simplex,
    the cone and ridge all pay it alike. Comparing it against the simplex
    solve separates the denoiser's effect on the span from the part of the
    distance that convexity alone is responsible for.
    """
    n_periods = donors.shape[0]
    coef, *_ = np.linalg.lstsq(donors / scale, target / scale, rcond=None)
    resid = target / scale - (donors / scale) @ coef
    return float(np.sqrt(float(resid @ resid) / n_periods)) * scale


def assess_denoise_spannability(
    raw_cluster_pre: np.ndarray,
    denoised_cluster_pre: np.ndarray,
    treated_pre: np.ndarray,
) -> DenoiseSpannabilityReport:
    """Measure what the denoising step cost in convex reach.

    :func:`assess_spannability` compares the selected cluster against the
    whole donor pool on raw outcomes, so it answers whether selection
    dropped donors the treated unit needed. It runs before the denoiser and
    is blind to what happens next.

    Basque is the case this function was written for. Its three-donor FPCA
    cluster costs nothing against the full sixteen-donor pool -- the best
    achievable convex pre-period RMSE is 0.0842 either way -- and
    undenoised it reproduces Abadie-Gardeazabal's published weights
    (Cataluna 0.840, Madrid 0.160, zero on Baleares) and their ATT of
    -0.6996 to within 0.002, from outcomes alone. Running the default PCP
    over that cluster moves the best achievable fit to 0.2786, hands
    Baleares -- the outlier -- a plurality of 0.538, and takes the ATT to
    -0.9204. Selection was right and denoising spoiled it.

    Parameters
    ----------
    raw_cluster_pre : np.ndarray
        Pre-period outcomes for the selected donors, shape ``(T0, J)``.
    denoised_cluster_pre : np.ndarray
        The same donors after the denoiser, same shape.
    treated_pre : np.ndarray
        Treated unit's pre-period outcomes, shape ``(T0,)``.

    Returns
    -------
    DenoiseSpannabilityReport
        ``ratio`` is 1.0 when denoising changed nothing, below 1.0 when it
        moved the hull towards the treated unit, and above when it moved
        the hull away. ``weights_identified`` is False when the denoised
        donors are affinely dependent, in which case the fitted weights are
        one arbitrary point of a continuum and should not be read as the
        donor composition.
    """
    raw = np.asarray(raw_cluster_pre, dtype=float)
    den = np.asarray(denoised_cluster_pre, dtype=float)
    target = np.asarray(treated_pre, dtype=float).ravel()
    if raw.ndim != 2 or den.ndim != 2:
        raise MlsynthDataError(
            "raw_cluster_pre and denoised_cluster_pre must be 2D (T0, J)."
        )
    if raw.shape != den.shape:
        raise MlsynthDataError(
            f"The raw and denoised donor blocks must have the same shape; "
            f"got {raw.shape} and {den.shape}."
        )
    if raw.shape[0] != target.shape[0]:
        raise MlsynthDataError(
            f"Pre-period length mismatch: donors have {raw.shape[0]} rows "
            f"but the treated unit has {target.shape[0]}."
        )

    scale = max(
        float(np.abs(target).max(initial=0.0)),
        float(np.abs(raw).max(initial=0.0)),
        float(np.abs(den).max(initial=0.0)),
        1e-12,
    )
    _w_raw, raw_rmse = _best_convex_fit(raw, target, scale)
    _w_den, denoised_rmse = _best_convex_fit(den, target, scale)
    span_rmse = _best_unconstrained_fit(den, target, scale)

    # Affine independence of the denoised donors. The weights solving the
    # simplex program are unique only if no donor is an affine combination
    # of the others; a low-rank denoiser applied to a handful of donors
    # routinely destroys that, and the solver then returns one point of a
    # continuum with nothing to mark it as such.
    n_donors = int(raw.shape[1])
    if n_donors <= 1:
        weights_identified = True
    else:
        diffs = den[:, 1:] - den[:, [0]]
        weights_identified = bool(
            np.linalg.matrix_rank(diffs / scale) == n_donors - 1
        )

    # Same 0/0 guard as `assess_spannability`: a cluster the treated unit
    # already sits on drives the solve to CLARABEL's floor, not to zero.
    negligible = 1e-5 * scale
    if raw_rmse <= negligible:
        ratio = 1.0 if denoised_rmse <= negligible else float("inf")
    else:
        ratio = float(denoised_rmse / raw_rmse)

    # Same 0/0 guard, one level down. A denoised block whose span already
    # contains the treated unit drives the unconstrained solve to the
    # least-squares floor; below it, report 1.0 when the hull reaches the
    # unit too and infinity when only the span does.
    if span_rmse <= negligible:
        hull_span_ratio = 1.0 if denoised_rmse <= negligible else float("inf")
    else:
        hull_span_ratio = float(max(denoised_rmse / span_rmse, 1.0))

    return DenoiseSpannabilityReport(
        raw_rmse=raw_rmse,
        denoised_rmse=denoised_rmse,
        ratio=ratio,
        span_rmse=span_rmse,
        hull_span_ratio=hull_span_ratio,
        weights_identified=weights_identified,
        n_donors=n_donors,
    )


def warn_if_denoising_shrank_the_hull(
    report: DenoiseSpannabilityReport,
    threshold: float = SPANNABILITY_WARN_RATIO,
    convexity_threshold: float = CONVEXITY_WARN_RATIO,
) -> None:
    """Warn when the denoiser moved the donors, and when convexity binds.

    The two are separate questions and each is silent on the other.
    ``ratio`` is a delta: it compares the denoised donors against the raw
    ones and says nothing about whether the simplex costs anything. On the
    raw Basque cluster it reads 1.00 -- correctly, no denoiser ran -- while
    the treated unit sits at 0.0070 from the donors' span and 0.3767 from
    their hull, a convexity cost of 54. ``hull_span_ratio`` is that level,
    and it is the one that distinguishes the weight objectives.

    They also disagree the other way. Amjad's full-pool rank-1 blocks score
    3.72 on Basque and 3.99 on Proposition 99 for ``ratio`` and exactly 1.00
    for ``hull_span_ratio``: rank-1 denoising confines the unconstrained fit
    to the same single direction the hull nearly exhausts, so no objective
    can do better than any other. That is the thesis finding its linear and
    convex controls interchangeable.

    ``weights_identified`` is reported on the result and deliberately does
    not warn. Across the three in-repo panels and the six denoiser and
    clustering combinations it is False in eleven of twelve, because a
    low-rank denoiser applied to ``J`` donors makes them affinely dependent
    whenever the retained rank is below ``J - 1`` -- which is what a
    denoiser is for. It is the normal state of RPCA-SC, not an anomaly, and
    a warning that fires on eleven runs in twelve teaches the reader to
    ignore the one in ``ratio`` that does discriminate: Basque under the
    default PCP scores 3.31 against 0.99 to 1.35 for the configurations
    that leave the hull alone.
    """
    if report.ratio > threshold:
        warnings.warn(
            f"Denoising moved the donors away from the treated unit: the best "
            f"achievable convex pre-period RMSE against the denoised donors is "
            f"{report.ratio:.1f}x the value against the raw ones "
            f"({report.denoised_rmse:.4g} vs {report.raw_rmse:.4g}). Donor "
            f"selection is not the problem -- these are the same donors either "
            f"way -- so widening the cluster will not help. Most of this "
            f"distance is usually to the donors' span, which every weight "
            f"objective pays alike; see `hull_span_ratio` for the part "
            f"convexity is responsible for. Raise the retained rank, weaken "
            f"the penalty, or fit against the raw donors.",
            UserWarning,
            stacklevel=3,
        )
    if report.hull_span_ratio > convexity_threshold:
        warnings.warn(
            f"The convex restriction is binding on these donors: the best "
            f"simplex fit is {report.hull_span_ratio:.1f}x the best "
            f"unconstrained fit against the same denoised donors "
            f"({report.denoised_rmse:.4g} vs {report.span_rmse:.4g}). The "
            f"treated unit lies much closer to the donors' span than to their "
            f"hull. Under `weight_objective=\"simplex\"` that gap is paid as "
            f"pre-period error; under `nnls` it is absorbed as extrapolation "
            f"and the fit looks healthy, so the objective decides whether you "
            f"see it, not whether it is there.",
            UserWarning,
            stacklevel=3,
        )
