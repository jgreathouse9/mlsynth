"""Design restrictions for the SYNDES MIP (geography / clustering / size / forcing).

SYNDES selects the treated set by a mixed-integer program over a binary
assignment vector ``D`` (length ``N``, in ``unit_index`` order). The same
restriction vocabulary LEXSCM exposes by *filtering* enumerated
candidate sets maps, for a MIP, to linear constraints on ``D``:

* ``to_be_treated``      -> ``D_i = 1``         (force a unit into the treated set)
* ``not_to_be_treated``  -> ``D_i = 0``         (forbid treatment; the unit stays
                                                  available as a control donor)
* ``size_col`` band      -> ``D_i = 0`` for units whose size falls outside
                            ``[min_size, max_size]`` (they remain donors)
* ``cluster_col`` /
  ``adjacency`` /
  ``spillover_threshold`` -> ``D_i + D_j <= 1`` for every conflicting (spillover)
                            pair, so two interfering units are never both treated
* ``stratum_col`` quota  -> ``min_per_stratum <= sum_{i in stratum} D_i`` (on the
                            strata that contain a treatable unit) and
                            ``sum_{i in stratum} D_i <= max_per_stratum``

This module translates the label-based config into an index-level
:class:`DesignRestrictions` bundle (:func:`build_restrictions`) and turns that
bundle into cvxpy constraints (:func:`apply_restrictions`). The conflict graph is
built by the shared :func:`mlsynth.utils.fast_scm_helpers.conflict.build_conflict_matrix`
(the same one LEXSCM uses), aligned to the ``unit_index``; cluster / stratum /
size attributes are read with
:func:`~mlsynth.utils.syndes_helpers.eligibility.unit_attribute_map`
and :func:`~mlsynth.utils.syndes_helpers.eligibility.eligible_by_size`, so the
vocabulary is identical across LEXSCM and SYNDES.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np

from ...exceptions import MlsynthConfigError, MlsynthDataError
from ..fast_scm_helpers.conflict import build_conflict_matrix
from .eligibility import (
    eligible_by_size,
    unit_attribute_map,
)

# A stratum constraint: (member indices, lower bound or None, upper bound or None).
Stratum = Tuple[Tuple[int, ...], Optional[int], Optional[int]]


@dataclass(frozen=True)
class DesignRestrictions:
    """Index-level restrictions for the SYNDES assignment vector ``D``.

    Attributes
    ----------
    forced_in : list of int
        Unit indices fixed to treatment (``D_i = 1``).
    forbidden : list of int
        Unit indices forbidden from treatment (``D_i = 0``) -- the union of
        ``not_to_be_treated`` and any size-ineligible units.
    conflict_pairs : list of (int, int)
        Index pairs ``(i, j)``, ``i < j``, that interfere (``D_i + D_j <= 1``).
    strata : list of (tuple of int, int or None, int or None)
        Per-stratum ``(members, lower, upper)`` coverage quotas.
    donor_exclusion : list of (int, int)
        Directed pairs ``(i, j)``: if treated unit ``i`` is selected, unit ``j``
        may not serve as its donor (control). Couples ``D`` to the control
        weights; see :func:`donor_constraints`.
    """

    forced_in: List[int] = field(default_factory=list)
    forbidden: List[int] = field(default_factory=list)
    conflict_pairs: List[Tuple[int, int]] = field(default_factory=list)
    strata: List[Stratum] = field(default_factory=list)
    donor_exclusion: List[Tuple[int, int]] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """True when no restriction is present (the MIP is unconstrained by this)."""
        return not (self.forced_in or self.forbidden or self.conflict_pairs
                    or self.strata or self.donor_exclusion)


def build_restrictions(
    df: Any,
    unitid: str,
    unit_index: Any,
    *,
    to_be_treated: Optional[List[Any]] = None,
    not_to_be_treated: Optional[List[Any]] = None,
    cluster_col: Optional[str] = None,
    adjacency: Optional[Any] = None,
    spillover_threshold: float = 0.0,
    stratum_col: Optional[str] = None,
    min_per_stratum: Optional[int] = None,
    max_per_stratum: Optional[int] = None,
    size_col: Optional[str] = None,
    min_size: Optional[float] = None,
    max_size: Optional[float] = None,
    donor_region_col: Optional[str] = None,
    exclude_bordering_donors: bool = False,
    donor_exclusion: Optional[Any] = None,
) -> DesignRestrictions:
    """Translate label-based design restrictions into an index-level bundle.

    Returns an (empty) :class:`DesignRestrictions` when no restriction field is
    set. Labels are resolved against ``unit_index`` (the single source of truth
    for unit order), so the resulting indices line up with the MIP's ``D``.

    Raises
    ------
    MlsynthConfigError
        If a forced/forbidden label is not a unit, or a unit is both forced in
        and forbidden.
    MlsynthDataError
        If a cluster / stratum / size column varies within a unit (via
        :func:`unit_attribute_map`) or an adjacency matrix is malformed.
    """
    labels = list(np.asarray(unit_index.labels).tolist())
    pos = {lab: i for i, lab in enumerate(labels)}
    units = set(labels)

    def _to_idx(items, what):
        bad = [u for u in items if u not in units]
        if bad:
            raise MlsynthConfigError(
                f"{what} contains units not in the panel: {sorted(map(str, bad))}."
            )
        return [pos[u] for u in items]

    forced_set = list(to_be_treated or [])
    forbidden_labels = set(not_to_be_treated or [])

    # Size band: units outside [min_size, max_size] lose treatment eligibility
    # (they stay donors), i.e. they join the forbidden set.
    if size_col is not None and (min_size is not None or max_size is not None):
        size_map = unit_attribute_map(df, unitid, size_col)
        eligible = eligible_by_size(labels, size_map,
                                    min_size=min_size, max_size=max_size)
        forbidden_labels |= (units - set(eligible))

    forced_in = sorted(_to_idx(forced_set, "to_be_treated"))
    forbidden = sorted(_to_idx(list(forbidden_labels), "not_to_be_treated"))

    overlap = set(forced_in) & set(forbidden)
    if overlap:
        clash = sorted(labels[i] for i in overlap)
        raise MlsynthConfigError(
            "units cannot be both to_be_treated and forbidden "
            f"(not_to_be_treated / size-ineligible): {list(map(str, clash))}."
        )

    # Spillover / interference conflict graph (shared with LEXSCM).
    cluster_of = (unit_attribute_map(df, unitid, cluster_col)
                  if cluster_col is not None else None)
    conflict_pairs: List[Tuple[int, int]] = []
    conflict = build_conflict_matrix(
        unit_index, cluster_of=cluster_of, adjacency=adjacency,
        spillover_threshold=spillover_threshold,
    )
    if conflict is not None:
        iu, ju = np.where(np.triu(conflict, 1))
        conflict_pairs = list(zip(iu.tolist(), ju.tolist()))

    # Stratum coverage quotas.
    strata: List[Stratum] = []
    if stratum_col is not None and (min_per_stratum is not None
                                    or max_per_stratum is not None):
        stratum_map = unit_attribute_map(df, unitid, stratum_col)
        treatable = set(range(len(labels))) - set(forbidden)
        groups: dict = {}
        for lab in labels:
            s = stratum_map.get(lab)
            if s is None or (isinstance(s, float) and np.isnan(s)):
                continue
            groups.setdefault(s, []).append(pos[lab])
        for s in groups:
            members = tuple(sorted(groups[s]))
            hi = max_per_stratum
            # A minimum applies only to strata that have a treatable member.
            lo = (min_per_stratum
                  if (min_per_stratum is not None
                      and any(m in treatable for m in members))
                  else None)
            if lo is not None or hi is not None:
                strata.append((members, lo, hi))

    # ---- donor-side exclusion: (i, j) means "if i is treated, j is not its
    # donor". Filled by region matching, spillover-neighbour exclusion, and/or
    # an explicit matrix; combined by union.
    donor_pairs: set = set()
    if donor_region_col is not None:
        region_of = unit_attribute_map(df, unitid, donor_region_col)
        regions = [region_of.get(lab) for lab in labels]
        for i in range(len(labels)):
            for j in range(len(labels)):
                if i != j and regions[i] != regions[j]:
                    donor_pairs.add((i, j))
    if exclude_bordering_donors and conflict is not None:
        iu, ju = np.where(conflict)            # both directions (symmetric graph)
        donor_pairs.update(zip(iu.tolist(), ju.tolist()))
    if donor_exclusion is not None:
        M = _align_square(donor_exclusion, labels, "donor_exclusion")
        iu, ju = np.where(M > 0)
        donor_pairs.update((int(i), int(j)) for i, j in zip(iu, ju) if i != j)

    return DesignRestrictions(
        forced_in=forced_in, forbidden=forbidden,
        conflict_pairs=conflict_pairs, strata=strata,
        donor_exclusion=sorted(donor_pairs),
    )


def _align_square(matrix: Any, labels: list, name: str) -> np.ndarray:
    """Reindex a square label-keyed matrix to ``labels`` order, validating cover."""
    import pandas as pd

    if isinstance(matrix, pd.DataFrame):
        missing = [l for l in labels
                   if l not in matrix.index or l not in matrix.columns]
        if missing:
            raise MlsynthDataError(
                f"{name} is missing rows/columns for units: {missing[:5]}"
                + (" ..." if len(missing) > 5 else "")
            )
        return matrix.reindex(index=labels, columns=labels).to_numpy(dtype=float)
    M = np.asarray(matrix, dtype=float)
    if M.shape != (len(labels), len(labels)):
        raise MlsynthConfigError(
            f"{name} has shape {M.shape}, expected "
            f"({len(labels)}, {len(labels)}); pass a DataFrame keyed by unit id "
            "to avoid relying on ordering."
        )
    return M


def apply_restrictions(D: Any, restrictions: DesignRestrictions) -> list:
    """Build the cvxpy constraints encoding ``restrictions`` on assignment ``D``.

    Parameters
    ----------
    D : cvxpy.Variable
        The boolean assignment vector (length ``N``).
    restrictions : DesignRestrictions
        The index-level restriction bundle.

    Returns
    -------
    list of cvxpy.Constraint
        ``D_i == 1`` (forced), ``D_i == 0`` (forbidden), ``D_i + D_j <= 1``
        (conflict), and ``sum`` bounds (stratum quotas).
    """
    import cvxpy as cp

    cons: list = []
    for i in restrictions.forced_in:
        cons.append(D[i] == 1)
    for i in restrictions.forbidden:
        cons.append(D[i] == 0)
    for i, j in restrictions.conflict_pairs:
        cons.append(D[i] + D[j] <= 1)
    for members, lo, hi in restrictions.strata:
        members = list(members)
        if lo is not None:
            cons.append(cp.sum(D[members]) >= lo)
        if hi is not None:
            cons.append(cp.sum(D[members]) <= hi)
    return cons


def donor_constraints(mode: str, variables: dict, D: Any,
                      donor_exclusion: list) -> list:
    """cvxpy constraints forbidding donor ``j`` for treated unit ``i``.

    The control weight of ``j`` (toward ``i``) is forced to zero whenever ``i``
    is treated, encoded per mode against that mode's control variables:

    * ``global_equal_weights`` (one-way): ``c[j] <= 1 - D[i]``;
    * ``global_2way``:                    ``w[j] - q[j] <= 1 - D[i]``
      (``w - q`` is the control simplex);
    * ``per_unit``:                       ``w[i, j] == 0`` (row ``i``'s weight on
      donor ``j``; only binds when ``i`` is treated since ``w[i, j] <= D[i]``).

    Returns an empty list when ``donor_exclusion`` is empty.
    """
    if not donor_exclusion:
        return []
    cons: list = []
    if mode == "global_equal_weights":
        c = variables["c"]
        for i, j in donor_exclusion:
            cons.append(c[j] <= 1 - D[i])
    elif mode == "global_2way":
        w, q = variables["w"], variables["q"]
        for i, j in donor_exclusion:
            cons.append(w[j] - q[j] <= 1 - D[i])
    elif mode == "per_unit":
        w = variables["w"]
        for i, j in donor_exclusion:
            cons.append(w[i, j] == 0)
    else:                                          # pragma: no cover - guarded upstream
        raise MlsynthConfigError(
            f"donor restrictions are not supported for mode {mode!r}."
        )
    return cons
