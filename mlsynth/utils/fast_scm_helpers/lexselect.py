"""Final design recommendation for LEXSCM (lexicographic selection).

Replaces the old ``select_best_tuple``, whose efficiency score multiplied an
MDE that was often NaN/Inf and whose Pareto frame could empty out and raise an
opaque ``IndexError``.

The selection is *lexicographic*, matching the method's premise (Abadie & Zhao:
approximate **balance first**, then power):

  1. **Validity gate.** Keep designs whose imbalance is within ``imbalance_tol``
     (relative) of the best achievable balance -- the set of "good treated fits".
  2. **Power.** Among the gated designs, pick the smallest MDE (most detectable).
  3. **Tie-breaks.** Then better out-of-sample stability, then lower cost.

If no gated design has a feasible MDE, it still returns the best-by-balance
design and flags ``status='POWER_NOT_ESTABLISHED'`` instead of crashing.  A
Pareto frontier (imbalance vs MDE) is always exposed for transparency.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class DesignMetrics:
    design_id: str
    indices: List[int]
    imbalance: float                 # validity: ||Xbar - sum w_j X_j||
    mde_sd: float = np.inf           # power: MDE in residual-SD units (lower=better)
    mde_abs: float = np.inf
    mde_feasible: bool = False
    stability: float = np.nan        # e.g. nmse_B (out-of-sample fit; lower=better)
    total_cost: float = 0.0
    labels: List[Any] = field(default_factory=list)


@dataclass
class Recommendation:
    winner: Optional[DesignMetrics]
    shortlist: List[DesignMetrics]
    pareto_ids: List[str]
    status: str                      # OK | POWER_NOT_ESTABLISHED | EMPTY
    explanation: str
    table: List[Dict[str, Any]]


def _pareto_front(designs: List[DesignMetrics]) -> List[str]:
    """Pareto-optimal ids on (imbalance ↓, mde_sd ↓); infeasible MDE = +inf."""
    ids = []
    for d in designs:
        dominated = False
        for o in designs:
            if o is d:
                continue
            if (o.imbalance <= d.imbalance and o.mde_sd <= d.mde_sd and
                    (o.imbalance < d.imbalance or o.mde_sd < d.mde_sd)):
                dominated = True
                break
        if not dominated:
            ids.append(d.design_id)
    return ids


def select_design(
    designs: List[DesignMetrics],
    *,
    imbalance_tol: float = 0.25,        # relative slack above best balance
    imbalance_abs: Optional[float] = None,  # or an absolute balance ceiling
    max_shortlist: int = 5,
) -> Recommendation:
    """Lexicographic recommendation: balance gate -> power -> stability -> cost."""
    if not designs:
        return Recommendation(None, [], [], "EMPTY",
                              "No candidate designs supplied.", [])

    best_imb = min(d.imbalance for d in designs)
    ceil_ = imbalance_abs if imbalance_abs is not None else best_imb * (1.0 + imbalance_tol)
    gated = [d for d in designs if d.imbalance <= ceil_ + 1e-12]
    if not gated:                                  # degenerate tol -> keep the best
        gated = [min(designs, key=lambda d: d.imbalance)]

    pareto = _pareto_front(designs)

    feasible = [d for d in gated if d.mde_feasible and np.isfinite(d.mde_sd)]
    if feasible:
        status = "OK"
        ranked = sorted(feasible, key=lambda d: (d.mde_sd,
                                                 d.stability if np.isfinite(d.stability) else np.inf,
                                                 d.total_cost))
        reason = (f"validity gate kept {len(gated)}/{len(designs)} designs within "
                  f"{imbalance_tol:.0%} of the best imbalance ({best_imb:.4f}); "
                  f"among them the most detectable has MDE {ranked[0].mde_sd:.3f} s.d.")
    else:
        status = "POWER_NOT_ESTABLISHED"
        # fall back to best balance, then stability; MDE could not be reached
        ranked = sorted(gated, key=lambda d: (d.imbalance,
                                              d.stability if np.isfinite(d.stability) else np.inf))
        reason = (f"no gated design reached the power target within the effect "
                  f"grid; recommending the best-balanced design "
                  f"(imbalance {ranked[0].imbalance:.4f}). Power not established.")

    winner = ranked[0]
    shortlist = ranked[:max_shortlist]
    table = [{
        "design_id": d.design_id,
        "indices": d.indices,
        "labels": d.labels,
        "imbalance": round(d.imbalance, 6),
        "mde_sd": (round(d.mde_sd, 4) if np.isfinite(d.mde_sd) else None),
        "mde_abs": (round(d.mde_abs, 4) if np.isfinite(d.mde_abs) else None),
        "mde_feasible": d.mde_feasible,
        "stability": (round(d.stability, 6) if np.isfinite(d.stability) else None),
        "total_cost": round(d.total_cost, 4),
        "in_validity_gate": d in gated,
        "pareto": d.design_id in pareto,
        "winner": d.design_id == winner.design_id,
    } for d in sorted(designs, key=lambda d: d.imbalance)]

    explanation = (
        f"--- DESIGN RECOMMENDATION: {winner.design_id} ---\n"
        f"Units: {winner.labels or winner.indices}\n"
        f"Imbalance: {winner.imbalance:.4f}  (best available {best_imb:.4f})\n"
        f"MDE: {winner.mde_sd:.3f} s.d." + (f" ({winner.mde_abs:.4f} abs)\n" if np.isfinite(winner.mde_abs) else " (not reached)\n") +
        f"Status: {status}\n{reason}"
    )
    return Recommendation(winner, shortlist, pareto, status, explanation, table)
