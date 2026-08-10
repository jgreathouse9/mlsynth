"""Greedy forward selection for treated-unit design (De Geest & Wang 2025).

Port of the flow chart on p.2 -- the method the prose describes: scan every
candidate treated unit, build a synthetic control for each candidate treated
set, keep the best by pre-treatment fit, add units until the objective stops
improving.

Two departures from the paper as printed, both documented in the review:

* Algorithm 1 (p.2) is NOT this procedure. As printed it fits a fixed target
  ``y_pre`` with a simple average over selected columns of ``X_pre`` -- that is
  FDID's donor step (Table 1c), with no inner synthetic control per candidate
  treated set, and no counterpart to the flow chart's "Construct synthetic
  control(s) for each experiment". We implement the flow chart.
* Table 1(d) averages the treated/control gap over ``t = 1..T1`` (the pre
  periods) and drops the intercept. Li's FDID ATT is the post-period mean of
  ``y_1t - alpha_hat - y_St``. We use mlsynth's validated FDID
  (``did_from_mean`` / ``forward_did_select``), i.e. Li's actual estimator.

The inner estimator is mlsynth's own ``forward_did_select`` so this is a port of
the *design search*, not a second FDID.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from mlsynth.utils.fdid_helpers.estimation import forward_did_select


@dataclass
class GreedyStep:
    """One accepted iteration of the greedy search."""

    treated: List[Any]
    r2: float
    donors: List[Any]
    att: float
    intercept: float


@dataclass
class GreedyDesign:
    """Result of the greedy forward-selection design search."""

    treated: List[Any]
    donors: List[Any]
    r2: float
    att: float
    history: List[GreedyStep] = field(default_factory=list)
    stopped_at: Optional[int] = None
    n_scored: int = 0


def _score_pooled(
    Y: np.ndarray,
    T0: int,
    treated_idx: Sequence[int],
    labels: Sequence[Any],
) -> Dict[str, Any]:
    """Score one candidate treated set under the pooling approach (S4.1).

    The treated aggregate is the simple average of the candidate treated units;
    every remaining unit is a donor. Donor selection and the fit statistic use
    the first ``T0`` rows only, so nothing post-treatment enters the design.
    """
    treated_idx = list(treated_idx)
    donor_idx = [j for j in range(Y.shape[1]) if j not in set(treated_idx)]
    if not donor_idx:
        return {"r2": -np.inf, "donors": [], "att": np.nan, "intercept": np.nan}
    y_T = Y[:, treated_idx].mean(axis=1)
    res = forward_did_select(
        y_T, Y[:, donor_idx], T0, [labels[j] for j in donor_idx]
    )["FDID"]
    return {
        "r2": float(res["Fit"]["R-Squared"]),
        "donors": list(res["selected_names"]),
        "att": float(res["Effects"]["ATT"]),
        "intercept": float(res["Inference"]["Intercept"]),
    }


def _score_separate(
    Y: np.ndarray,
    T0: int,
    treated_idx: Sequence[int],
    labels: Sequence[Any],
) -> Dict[str, Any]:
    """Score one candidate treated set under the separate approach (S4.1).

    One synthetic control per treated unit; the objective is the mean
    pre-treatment R^2 and the effect is the mean of the per-unit ATTs.
    """
    treated_idx = list(treated_idx)
    donor_idx = [j for j in range(Y.shape[1]) if j not in set(treated_idx)]
    if not donor_idx:
        return {"r2": -np.inf, "donors": [], "att": np.nan, "intercept": np.nan}
    r2s, atts, donors = [], [], []
    for i in treated_idx:
        res = forward_did_select(
            Y[:, i], Y[:, donor_idx], T0, [labels[j] for j in donor_idx]
        )["FDID"]
        r2s.append(float(res["Fit"]["R-Squared"]))
        atts.append(float(res["Effects"]["ATT"]))
        donors.extend(res["selected_names"])
    return {
        "r2": float(np.mean(r2s)),
        "donors": sorted(set(donors), key=lambda d: donors.index(d)),
        "att": float(np.mean(atts)),
        "intercept": np.nan,
    }


def greedy_forward_design(
    Y: np.ndarray,
    T0: int,
    labels: Sequence[Any],
    *,
    tol: float = 0.0,
    pooling: bool = True,
    max_treated: Optional[int] = None,
    forced: Sequence[Any] = (),
    forbidden: Sequence[Any] = (),
) -> GreedyDesign:
    """Greedy forward selection over treated units (flow chart, p.2).

    Parameters
    ----------
    Y : np.ndarray
        Outcome panel, shape ``(T, N)``, rows time-ordered.
    T0 : int
        Number of leading rows that are pre-treatment (the design window).
    labels : sequence
        Unit labels, length ``N``, aligned to the columns of ``Y``.
    tol : float, default 0.0
        Stop when the best achievable R^2 improvement falls below this. The
        paper requires a tolerance but never states the value it used.
    pooling : bool, default True
        Pooling (one SC for the treated average) vs separate (one SC per
        treated unit); S4.1.
    max_treated : int, optional
        Hard cap on the treated-set size.
    forced, forbidden : sequence
        Units always / never treated (Practical Note 1, p.4).

    Returns
    -------
    GreedyDesign
    """
    labels = list(labels)
    index = {u: j for j, u in enumerate(labels)}
    score = _score_pooled if pooling else _score_separate

    forbidden_idx = {index[u] for u in forbidden}
    treated: List[int] = [index[u] for u in forced]
    candidates = [
        j for j in range(Y.shape[1])
        if j not in forbidden_idx and j not in set(treated)
    ]
    cap = max_treated if max_treated is not None else Y.shape[1] - 1

    history: List[GreedyStep] = []
    best_r2 = -np.inf
    n_scored = 0
    stopped_at = None

    if treated:  # seed the history with the forced-in set
        s = score(Y, T0, treated, labels)
        n_scored += 1
        best_r2 = s["r2"]
        history.append(GreedyStep([labels[j] for j in treated], s["r2"],
                                  s["donors"], s["att"], s["intercept"]))

    while candidates and len(treated) < cap:
        scored = []
        for j in candidates:
            s = score(Y, T0, treated + [j], labels)
            n_scored += 1
            scored.append((s["r2"], j, s))
        r2_new, j_star, s_star = max(scored, key=lambda t: t[0])

        if history and (r2_new - best_r2) < tol:
            stopped_at = len(treated)
            break

        treated.append(j_star)
        candidates.remove(j_star)
        best_r2 = r2_new
        history.append(GreedyStep([labels[j] for j in treated], r2_new,
                                  s_star["donors"], s_star["att"],
                                  s_star["intercept"]))

    # Paper's line 22: return the iteration with the highest objective value.
    best = max(history, key=lambda h: h.r2)
    return GreedyDesign(
        treated=best.treated, donors=best.donors, r2=best.r2, att=best.att,
        history=history, stopped_at=stopped_at, n_scored=n_scored,
    )


def design_to_contrast(design: GreedyDesign) -> Dict[Any, float]:
    """``w_treated - w_control`` for the common design-comparison currency.

    FDID builds both sides as simple averages, so the treated units carry
    ``+1/|T|`` and the selected donors ``-1/|S|``.
    """
    wt = 1.0 / len(design.treated)
    wc = 1.0 / len(design.donors)
    contrast = {u: wt for u in design.treated}
    for u in design.donors:
        contrast[u] = contrast.get(u, 0.0) - wc
    return contrast
