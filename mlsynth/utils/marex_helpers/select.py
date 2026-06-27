"""Composite power-vs-fit recommendation over a MAREX solution pool.

Mirrors ``utils/syndes_helpers/select.py``: each design is ranked (dense) on fit
(lower ``objective`` better) and on power (lower ``mde_pct`` better); the
composite score is a normalised weighted sum of the two ranks, and the winner is
the lowest-scoring power-feasible design. Also reports the fit-vs-power Pareto
front.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from scipy.stats import rankdata

from .structures import MAREXRecommendation


def _pareto_front(fit: np.ndarray, mde: np.ndarray) -> List[int]:
    """Indices on the non-dominated (fit downwards, mde downwards) front."""
    n = len(fit)
    keep: List[int] = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j:
                continue
            if (fit[j] <= fit[i] and mde[j] <= mde[i]
                    and (fit[j] < fit[i] or mde[j] < mde[i])):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return keep


def recommend_marex(pool: List[Dict[str, Any]], *, power_weight: float = 0.51,
                    fit_weight: float = 0.49,
                    max_shortlist: int = 5) -> MAREXRecommendation:
    """Recommend a design from the pool on a composite power-vs-fit score."""
    s = power_weight + fit_weight
    pw, fw = power_weight / s, fit_weight / s
    fit = np.array([float(e["objective"]) for e in pool], dtype=float)
    mde = np.array([e["mde_pct"] if np.isfinite(e["mde_pct"]) else np.inf
                    for e in pool], dtype=float)
    rank_fit = rankdata(fit, method="dense")
    rank_pow = rankdata(mde, method="dense")
    scores = fw * rank_fit + pw * rank_pow
    order = [int(i) for i in np.argsort(scores, kind="stable")]
    pareto = _pareto_front(fit, mde)

    feasible = [i for i in order if np.isfinite(mde[i])]
    winner_idx = feasible[0] if feasible else order[0]
    status = "OK" if feasible else "POWER_NOT_ESTABLISHED"
    shortlist = [pool[i] for i in order[:max_shortlist]]
    return MAREXRecommendation(
        winner=pool[winner_idx], shortlist=shortlist, pareto=pareto,
        weights={"power": pw, "fit": fw}, status=status,
    )
