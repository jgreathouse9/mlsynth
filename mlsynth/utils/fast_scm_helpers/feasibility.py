"""Up-front feasibility audit for LEXSCM treated-unit selection.

Every active selection constraint -- candidate pool, budget, coverage, quota,
spillover -- is checked *before* the search, and any that is individually
infeasible is reported **together** in one :class:`MlsynthConfigError`, each line
in a uniform ``have vs need -> minimal fix`` shape. The audit **reports**, it does
not auto-relax: it never silently loosens a constraint the analyst set.
"""
from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from ...exceptions import MlsynthConfigError
from . import strata as _strata


def _money(x: float) -> str:
    return f"${x:,.0f}"


def audit_feasibility(
    candidate_idx: Sequence[int],
    m: int,
    *,
    unit_costs: Optional[np.ndarray] = None,
    budget: Optional[float] = None,
    conflict: Optional[np.ndarray] = None,
    strata: Optional[np.ndarray] = None,
    min_per_stratum: Optional[int] = None,
    max_per_stratum: Optional[int] = None,
    size_band: Optional[Tuple[Any, Any]] = None,
) -> None:
    """Raise a single, itemised :class:`MlsynthConfigError` if any active
    constraint makes a size-``m`` design impossible. No-op when feasible.
    """
    cand = np.asarray(list(candidate_idx), dtype=int)
    M = len(cand)
    problems: List[str] = []

    # 1. Candidate pool -- the precondition for everything else.
    if M < m:
        within = ""
        if size_band is not None:
            lo, hi = size_band
            within = f" within the size band [{lo}, {hi}]"
        problems.append(
            f"candidate pool: only {M} eligible market(s){within}, but m={m}. "
            f"Widen eligibility / the size band, or reduce m."
        )
        _raise(problems)        # the other checks are meaningless with too few units

    # 2. Budget -- even the m cheapest eligible markets must fit.
    if budget is not None and unit_costs is not None:
        cheapest = float(np.sort(np.asarray(unit_costs, dtype=float)[cand])[:m].sum())
        if cheapest > budget + 1e-9:
            problems.append(
                f"budget: the {m} cheapest eligible markets cost {_money(cheapest)}, "
                f"over the {_money(budget)} budget by {_money(cheapest - budget)}. "
                f"Raise the budget to >= {_money(cheapest)}, reduce m, or relax the "
                f"size band."
            )

    # 3. Coverage / quota.
    if strata is not None:
        problems += _strata.feasibility_problems(
            strata, cand, m, min_per_stratum, max_per_stratum)

    # NB: spillover (max independent set >= m) is NP-hard for general adjacency,
    # and the greedy size is only a *lower* bound -- so it cannot soundly prove
    # infeasibility up front (false positives). It is proven by the search itself
    # (exact under enumeration) and reported by ``select_treated_designs``.

    _raise(problems)


def _raise(problems: List[str]) -> None:
    if problems:
        raise MlsynthConfigError(
            "LEXSCM design is infeasible -- the binding constraint(s):\n  - "
            + "\n  - ".join(problems)
        )
