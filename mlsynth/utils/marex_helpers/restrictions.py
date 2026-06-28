"""Geographic design restrictions for the MAREX MIP (Abadie & Zhao 2026).

MAREX already carries most geographic structure natively: its ``cluster`` field
is AZ's "region as a distinct experimental design", the per-cluster cardinality
(``m_min`` / ``m_max``) is a stratum quota, ``costs`` / ``budget`` is AZ's cost
bound, and the control synthetic ``v[.,k]`` is built only from cluster-``k``
members, so donors are automatically same-region. This module adds the four
remaining capabilities -- the same vocabulary SYNDES and GEOLIFT expose -- as
linear constraints on the design variables:

* ``to_be_treated``      -> ``sum_k z[j,k] = 1``  (force unit ``j`` treated)
* ``not_to_be_treated`` /
  ``size_col`` band       -> ``sum_k z[j,k] = 0``  (forbid treatment; stays donor)
* ``adjacency`` /
  ``spillover_threshold`` -> ``sum_k z[i,k] + sum_k z[j,k] <= 1`` (no two treated
                            markets border each other)
* ``exclude_bordering_donors`` -> within a cluster ``k``, ``v[j,k] <= 1 - z[i,k]``
                            for every bordering pair ``(i,j)`` (a treated market's
                            neighbours are dropped from its control pool)

The index-level bundle is the shared, estimator-agnostic
:class:`~mlsynth.utils.syndes_helpers.restrictions.DesignRestrictions`, built by
the shared :func:`~mlsynth.utils.syndes_helpers.restrictions.build_restrictions`
(which reuses the LEXSCM conflict graph and the GEOLIFT attribute/size helpers,
all aligned to the unit ``IndexSet``). Only the *applier* differs from SYNDES,
because MAREX's selection variable is the ``(N, K)`` matrix ``z`` and its control
synthetic is per-cluster.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Shared, estimator-agnostic restriction infrastructure (the bundle + builder are
# pure index/label machinery; only the applier below is MAREX-specific).
from ..syndes_helpers.restrictions import (  # noqa: F401  (re-exported for callers)
    DesignRestrictions,
    build_restrictions,
)


def apply_restrictions_marex(z: Any, v: Any, M: np.ndarray,
                             restrictions: DesignRestrictions) -> list:
    """cvxpy constraints encoding ``restrictions`` on MAREX's ``z`` / ``v``.

    Parameters
    ----------
    z : cvxpy.Variable
        Binary selection matrix, shape ``(N, K)`` (``z[j,k]`` = unit ``j`` treated
        in cluster ``k``). A unit's treated indicator is ``sum_k z[j,k]``.
    v : cvxpy.Variable
        Control-weight matrix, shape ``(N, K)``.
    M : np.ndarray of bool
        Unit-to-cluster membership mask, shape ``(N, K)``.
    restrictions : DesignRestrictions
        The index-level bundle (indices aligned to the unit ``IndexSet``).

    Returns
    -------
    list of cvxpy.Constraint
    """
    import cvxpy as cp

    if restrictions is None or restrictions.is_empty:
        return []
    N = M.shape[0]
    cluster_of = {j: int(np.argmax(M[j])) for j in range(N)}   # each unit's cluster
    cons: list = []
    for j in restrictions.forced_in:
        cons.append(cp.sum(z[j, :]) == 1)
    for j in restrictions.forbidden:
        cons.append(cp.sum(z[j, :]) == 0)
    for i, j in restrictions.conflict_pairs:
        cons.append(cp.sum(z[i, :]) + cp.sum(z[j, :]) <= 1)
    # Stratum coverage quotas: a unit's treated indicator is sum_k z[j,k], so the
    # treated count in a stratum is the sum of z over its members across clusters.
    for members, lo, hi in restrictions.strata:
        treated_in_stratum = cp.sum(z[list(members), :])
        if lo is not None:
            cons.append(treated_in_stratum >= lo)
        if hi is not None:
            cons.append(treated_in_stratum <= hi)
    # Donor (control) exclusion: "if i is treated, j may not be its donor". MAREX
    # builds one control synthetic per cluster, so a donor for treated unit i is a
    # control unit in i's cluster; the exclusion only binds when i and j share a
    # cluster (otherwise v[j, cluster(i)] is already 0 by membership).
    for i, j in restrictions.donor_exclusion:
        ki = cluster_of[i]
        if M[j, ki]:
            cons.append(v[j, ki] <= 1 - z[i, ki])
    return cons
