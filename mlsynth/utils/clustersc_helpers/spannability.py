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
        f"dropped. With weight_objective='simplex' this appears as a large "
        f"pre-period error; with 'nnls' it is absorbed as extrapolation "
        f"(sum(w) > 1). Widen the cluster or use the full donor pool.",
        UserWarning,
        stacklevel=3,
    )
