"""Assemble the MAREX solution-pool menu from raw design solves.

For each design returned by :func:`optimization.solve_design_pool`, build a menu
entry mirroring SYNDES's pool: the treated ``markets`` and ``control_group``, the
pre-period fit, and an MDE power curve (the same Newey-West power analysis SYNDES
uses) scored on the design's aggregate contrast.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .power import marex_mde_curve


def build_marex_pool(raws: List[Dict[str, Any]], *, alpha: float = 0.05,
                     power: float = 0.80, n_post: int = 1) -> List[Dict[str, Any]]:
    """Build the pool menu (one entry per raw design) with fit + MDE power."""
    menu: List[Dict[str, Any]] = []
    for raw in raws:
        df = raw["df"]
        w_agg = np.asarray(raw["w_agg"], dtype=float)
        v_agg = np.asarray(raw["v_agg"], dtype=float)
        labels = (list(df.index) if hasattr(df, "index")
                  else list(range(w_agg.size)))
        contrast = w_agg - v_agg
        treated_idx = np.where(w_agg > 1e-8)[0]
        control_idx = np.where(v_agg > 1e-8)[0]
        markets = [str(labels[i]) for i in treated_idx]
        control_group = [str(labels[i]) for i in control_idx]

        Y_fit = np.asarray(raw["Y_fit"], dtype=float)
        g = Y_fit.T @ contrast                       # pre-period contrast series
        pre_fit_rmse = float(np.sqrt(np.mean(g ** 2)))
        baseline = (float(np.mean(Y_fit[treated_idx, :]))
                    if treated_idx.size else float("nan"))
        curve = marex_mde_curve(Y_fit, contrast, horizons=range(1, 13),
                                alpha=alpha, power=power, baseline=baseline)
        h = max(1, min(int(n_post), 12))
        mde_pct = float(curve["mde_pct"][h - 1])

        menu.append({
            "markets": markets, "control_group": control_group,
            "n_treated": len(markets), "n_control": len(control_group),
            "objective": pre_fit_rmse,            # fit metric for ranking
            "pre_fit_rmse": pre_fit_rmse,
            "mip_objective": float(raw.get("objective", float("nan"))),
            "mde_pct": mde_pct, "power_curve": curve,
            "design": raw,
        })
    return menu
