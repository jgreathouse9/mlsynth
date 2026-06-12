"""Multi-cell GeoLift analysis: per-cell effects + the cross-cell comparison.

Each treatment cell is measured against the **shared control** pool with the
(validated) fixed-effect Augmented SCM + conformal inference -- crucially
excluding the *other* cells' markets from the donor pool, since they are treated
(with a different treatment) and so contaminated (GeoLift's
``filter(!location %in% other_cells)``). The cross-cell winner uses GeoLift's
rule: a cell wins a pairwise comparison when its ATT confidence interval lies
strictly above the other's (non-overlapping CIs).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from mlsynth.utils.geolift_helpers.marketselect.realize import realize_design

from .structures import MultiCellResults


def analyze_multicell(
    Ywide: pd.DataFrame,
    cell_map: Dict[str, List],
    control_units: List,
    pre_periods: int,
    *,
    how: str = "mean",
    augment: Optional[str] = "ridge",
    fixed_effects: bool = True,
    alpha: float = 0.1,
    q: float = 1.0,
    ns: int = 1000,
    seed: int = 0,
    conformal_type: str = "iid",
    cpic: Optional[float] = None,
) -> MultiCellResults:
    """Measure every cell against the shared control pool and compare them.

    Parameters mirror :func:`realize_design`. Returns a :class:`MultiCellResults`
    with the per-cell reports, the pairwise comparison, and the overall winner.
    """
    cells: Dict[str, object] = {}
    cell_ci: Dict[str, tuple] = {}            # label -> (att, lower, upper)

    for label in sorted(cell_map):
        units = list(cell_map[label])
        # Restrict the panel to this cell + the shared control pool, so the
        # other cells are *excluded* from the donor matrix (they are treated).
        sub = Ywide[list(control_units) + units]
        report = realize_design(
            sub, frozenset(units), pre_periods, how=how, augment=augment,
            alpha=alpha, q=q, ns=ns, seed=seed, conformal_type=conformal_type,
            fixed_effects=fixed_effects, cpic=cpic,
        )
        cells[label] = report
        att = float(report.effects.att)
        details = report.inference.details
        # ATT confidence interval = the per-period conformal bounds averaged
        # (contains the ATT, since each period's effect lies in its own band).
        lower = float(np.nanmean(details["lower"]))
        upper = float(np.nanmean(details["upper"]))
        cell_ci[label] = (att, lower, upper)

    labels = sorted(cell_map)
    comparison: List[dict] = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            att_a, lo_a, hi_a = cell_ci[a]
            att_b, lo_b, hi_b = cell_ci[b]
            if lo_a > hi_b:
                win = a
            elif lo_b > hi_a:
                win = b
            else:
                win = None                     # overlapping CIs -> no winner
            comparison.append({
                "cell_a": a, "cell_b": b,
                "att_a": att_a, "att_b": att_b, "att_diff": att_a - att_b,
                "winner": win,
            })

    # Overall winner: the cell that wins every pairwise comparison it is in.
    winner = None
    for label in labels:
        involved = [c for c in comparison if label in (c["cell_a"], c["cell_b"])]
        if involved and all(c["winner"] == label for c in involved):
            winner = label
            break

    primary = winner or max(labels, key=lambda lab: abs(cell_ci[lab][0]))
    return MultiCellResults(
        report=cells[primary],
        cells=cells,
        comparison=comparison,
        winner=winner,
        assignment={label: list(units) for label, units in cell_map.items()},
        selected_units=(list(cell_map[winner]) if winner else None),
    )
