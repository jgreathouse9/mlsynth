"""Pareto recommendation for the SYNDES solution pool.

The SYNDES MIP ranks designs by fit (the mean-squared pre-period imbalance)
alone. But the best-fitting design is rarely the most *detectable* one, and a
manager usually wants a single, defensible recommendation that balances the two.
This module turns the ``top_K`` pool (already re-scored on power and cost by
:func:`mlsynth.estimators.syndes._syndes_pool_menu`) into:

* a Pareto frontier on ``(fit downwards, power downwards)`` -- where fit is the
  pre-period RMSE between the treated group and the weighted average of the
  controls -- the designs for which neither fit nor minimum-detectable-effect can
  be improved without worsening the other (cost enters only as a tie-break),
  always exposed for transparency; and
* a single recommended design chosen by a GeoLift-style composite score: the
  weighted mean of the two dense ranks (fit and power), defaulting to a slight
  preference for power (``0.51``) over fit (``0.49``). Cost, then pre-period
  RMSE, break ties.

The selection never raises: with no feasibly-powered design it degrades to the
best-fitting one and reports ``status="POWER_NOT_ESTABLISHED"``; with an empty
pool it returns ``status="EMPTY"``. The design mirrors the LEXSCM recommender in
:mod:`mlsynth.utils.fast_scm_helpers.lexselect`; the difference is the selection
rule (a weighted score rather than a lexicographic validity gate).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.stats import rankdata


def _fit_value(d: "SYNDESDesignMetrics") -> float:
    """Fit criterion: pre-period RMSE (treated vs weighted control), lower better.

    Falls back to the MIP objective when a design carries no RMSE (e.g. a mode
    that does not populate it), so the recommender is always well-defined.
    """
    if d.pre_fit_rmse is not None and np.isfinite(d.pre_fit_rmse):
        return float(d.pre_fit_rmse)
    return float(d.objective)


@dataclass(frozen=True)
class SYNDESDesignMetrics:
    """Per-design criteria a SYNDES recommendation trades off.

    Parameters
    ----------
    design_id : str
        Stable id ``"D1" .. "Dn"`` in pool order (so ``D1`` is the
        MSE-optimal, best-fitting design).
    markets : list
        Treated-unit labels (the design's arms).
    control_group : list
        Donor labels backing the synthetic control.
    objective : float
        The MIP objective (fit / mean-squared pre-period imbalance); lower is
        better.
    mde_pct : float
        Minimum detectable effect at the realised horizon, as a percent of the
        treated baseline; lower is better. ``inf``/``nan`` means power could not
        be established.
    mde_feasible : bool
        Whether ``mde_pct`` is finite.
    cost : float or None
        Summed cost of the treated set (tie-break), or ``None``.
    pre_fit_rmse : float or None
        Root-mean-square pre-period contrast (secondary tie-break), or ``None``.
    design : Any
        The underlying ``SYNDESDesign`` (carried for deployment), or ``None``.
    """

    design_id: str
    markets: List[Any]
    control_group: List[Any]
    objective: float
    mde_pct: float
    mde_feasible: bool
    cost: Optional[float] = None
    pre_fit_rmse: Optional[float] = None
    design: Any = field(default=None, repr=False)


@dataclass(frozen=True)
class SYNDESRecommendation:
    """Outcome of scoring a SYNDES pool.

    Parameters
    ----------
    winner : SYNDESDesignMetrics or None
        The recommended design (``None`` only when the pool is empty).
    shortlist : list of SYNDESDesignMetrics
        Top designs by composite score (best first), at most ``max_shortlist``.
    pareto_ids : list of str
        Design ids on the ``(fit, power)`` Pareto frontier (always exposed).
    status : str
        ``"OK"`` (a feasibly-powered design was recommended),
        ``"POWER_NOT_ESTABLISHED"`` (no finite MDE; best-fitting design
        returned), or ``"EMPTY"`` (no pool).
    explanation : str
        Human-readable summary of the recommendation.
    table : list of dict
        One row per design with its metrics, composite score, and the
        ``pareto`` / ``winner`` flags -- ready for a DataFrame.
    weights : dict
        The normalised ``{"power": ..., "fit": ...}`` weights used.
    """

    winner: Optional[SYNDESDesignMetrics]
    shortlist: List[SYNDESDesignMetrics]
    pareto_ids: List[str]
    status: str
    explanation: str
    table: List[Dict[str, Any]]
    weights: Dict[str, float]


def _pareto_front(designs: List[SYNDESDesignMetrics]) -> List[str]:
    """Pareto-optimal ids on (fit RMSE downwards, mde_pct downwards).

    Infeasible MDE is treated as ``+inf``, so a design with no established power
    is dominated by any equally- or better-fitting design that does have power.
    """
    ids: List[str] = []
    for d in designs:
        d_fit, d_mde = _fit_value(d), (d.mde_pct if np.isfinite(d.mde_pct) else np.inf)
        dominated = False
        for o in designs:
            if o is d:
                continue
            o_fit = _fit_value(o)
            o_mde = o.mde_pct if np.isfinite(o.mde_pct) else np.inf
            if (o_fit <= d_fit and o_mde <= d_mde
                    and (o_fit < d_fit or o_mde < d_mde)):
                dominated = True
                break
        if not dominated:
            ids.append(d.design_id)
    return ids


def _composite_scores(
    designs: List[SYNDESDesignMetrics], power_weight: float, fit_weight: float
) -> Dict[str, float]:
    """GeoLift-style score: weighted mean of dense ranks (lower = better).

    Each criterion is dense-ranked ascending (best metric -> rank 1), exactly as
    GeoLift's ``compute_rank`` aggregates ``rank_mde`` / ``rank_pvalue``; the two
    ranks are then combined with the power / fit weights. Using ranks rather than
    raw values keeps the score robust to the very different scales of the fit RMSE
    and the percentage MDE.
    """
    fit = np.array([_fit_value(d) for d in designs], dtype=float)
    mde = np.array([d.mde_pct if np.isfinite(d.mde_pct) else np.inf
                    for d in designs], dtype=float)
    rank_fit = rankdata(fit, method="dense")
    rank_power = rankdata(mde, method="dense")
    scores = fit_weight * rank_fit + power_weight * rank_power
    return {d.design_id: float(s) for d, s in zip(designs, scores)}


def recommend_syndes(
    pool: List[Dict[str, Any]],
    *,
    power_weight: float = 0.51,
    fit_weight: float = 0.49,
    max_shortlist: int = 5,
) -> SYNDESRecommendation:
    """Recommend a single SYNDES design from the solution pool.

    Parameters
    ----------
    pool : list of dict
        The ``results.pool`` menu (each entry carries ``markets``,
        ``control_group``, ``objective``, ``mde_pct``, ``cost``,
        ``pre_fit_rmse``, and ``design``).
    power_weight, fit_weight : float
        Relative weight on power versus fit in the composite score. Normalised to
        sum to one internally; defaults ``0.51`` / ``0.49`` (a slight preference
        for power).
    max_shortlist : int
        Maximum number of designs in the returned shortlist.

    Returns
    -------
    SYNDESRecommendation
    """
    total_w = power_weight + fit_weight
    if total_w <= 0:
        raise ValueError("power_weight + fit_weight must be positive.")
    pw, fw = power_weight / total_w, fit_weight / total_w
    weights = {"power": pw, "fit": fw}

    if not pool:
        return SYNDESRecommendation(
            winner=None, shortlist=[], pareto_ids=[], status="EMPTY",
            explanation="No solution pool to recommend from (top_K must be > 1).",
            table=[], weights=weights,
        )

    designs = [
        SYNDESDesignMetrics(
            design_id=f"D{i + 1}",
            markets=list(e.get("markets", [])),
            control_group=list(e.get("control_group", [])),
            objective=float(e["objective"]),
            mde_pct=float(e.get("mde_pct", np.inf)),
            mde_feasible=bool(np.isfinite(e.get("mde_pct", np.inf))),
            cost=(None if e.get("cost") is None else float(e["cost"])),
            pre_fit_rmse=(None if e.get("pre_fit_rmse") is None
                          else float(e["pre_fit_rmse"])),
            design=e.get("design"),
        )
        for i, e in enumerate(pool)
    ]

    scores = _composite_scores(designs, pw, fw)
    pareto_ids = _pareto_front(designs)

    # Tie-break key: composite score, then cost, then the fit RMSE itself.
    def _key(d: SYNDESDesignMetrics):
        return (
            scores[d.design_id],
            d.cost if d.cost is not None else np.inf,
            _fit_value(d),
        )

    feasible = [d for d in designs if d.mde_feasible]
    if feasible:
        status = "OK"
        ranked = sorted(feasible, key=_key)
        winner = ranked[0]
        explanation = (
            f"Recommended {winner.design_id} (markets {winner.markets}) by the "
            f"composite score weighting power {pw:.0%} and fit {fw:.0%}: "
            f"MDE {winner.mde_pct:.3g}% at the realised horizon, fit RMSE "
            f"{_fit_value(winner):.4g}. {len(pareto_ids)} design(s) on the "
            f"fit-power Pareto frontier."
        )
    else:
        status = "POWER_NOT_ESTABLISHED"
        ranked = sorted(designs, key=_fit_value)
        winner = ranked[0]
        explanation = (
            "No pool design had an established (finite) MDE; recommending the "
            f"best-fitting design {winner.design_id} (fit RMSE "
            f"{_fit_value(winner):.4g}). Treat the power ranking as unavailable."
        )

    # Shortlist: best composite scores first (feasible designs ranked ahead).
    shortlist = sorted(
        designs, key=lambda d: (0 if d.mde_feasible else 1, _key(d))
    )[:max_shortlist]

    table = [
        {
            "design_id": d.design_id,
            "markets": d.markets,
            "control_group": d.control_group,
            "fit_rmse": _fit_value(d),
            "objective": d.objective,
            "mde_pct": d.mde_pct,
            "mde_feasible": d.mde_feasible,
            "cost": d.cost,
            "pre_fit_rmse": d.pre_fit_rmse,
            "score": scores[d.design_id],
            "pareto": d.design_id in pareto_ids,
            "winner": winner is not None and d.design_id == winner.design_id,
        }
        for d in designs
    ]

    return SYNDESRecommendation(
        winner=winner, shortlist=shortlist, pareto_ids=pareto_ids,
        status=status, explanation=explanation, table=table, weights=weights,
    )
