"""Exact SCPI prediction intervals for staggered-adoption VanillaSC.

A from-scratch (MIT) re-derivation of the aggregated causal-predictand prediction
intervals of Cattaneo, Feng, Palomba & Titiunik (2025), Section 4. It does not
import the GPL ``scpi`` package; it has been validated numerically against it.

For each treated unit we already have, from :func:`scpi_intervals`, the exact
per-unit conic in-sample simulation (the worst-case in-sample error per draw, for
every post period and for the time-averaged predictand) and the per-cell
out-of-sample location-scale band. The aggregated predictands -- TSUA (per event
time, averaged over units) and TAUA (overall) -- are linear combinations of those
per-unit pieces, so their intervals are assembled by:

* in-sample: summing the per-draw worst-case errors across the (independent)
  treated units and taking the ``alpha/2`` / ``1 - alpha/2`` quantiles;
* out-of-sample: combining the per-cell location-scale bands under the
  sub-Gaussian concentration bound.

Each band is ``[point - Mbar_in - Mbar_out, point - M_in - M_out]``, matching the
single-unit SCPI convention.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .scpi import scpi_intervals


def unit_components(config, y: np.ndarray, Y0: np.ndarray, pre: int,
                    W: np.ndarray, seed: int) -> Dict[str, Any]:
    """Per-unit SCPI components needed for cross-unit aggregation.

    ``seed`` is unit-specific so the in-sample Gaussian draws are independent
    across treated units (required for the cross-unit in-sample aggregation).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = scpi_intervals(
            y, Y0, pre, W, sims=config.scpi_sims,
            u_alpha=config.alpha, e_alpha=config.alpha,
            e_method=config.scpi_e_method, seed=seed,
        )
    md = sc.metadata
    oos_lb = np.asarray(md["oos_lb_rows"], dtype=float)   # (n_post + 1,)
    oos_ub = np.asarray(md["oos_ub_rows"], dtype=float)
    return {
        "in_lo": np.asarray(md["insample_draws_lo"], dtype=float),   # (S, n_post+1)
        "in_hi": np.asarray(md["insample_draws_hi"], dtype=float),
        "e_mean": (oos_lb + oos_ub) / 2.0,                            # (n_post+1,)
        "e_half": (oos_ub - oos_lb) / 2.0,
        "cf": np.asarray(md["cf"], dtype=float),                      # (n_post,)
        "obs": np.asarray(md["obs"], dtype=float),
        "n_post": int(md["n_post"]),
        "result": sc,
    }


def _aggregate_band(
    terms: Sequence[Tuple[Dict[str, Any], float, int]],
    u_alpha: float,
    *,
    cross_unit: str = "independent",
) -> Tuple[float, float, float]:
    """Assemble one aggregated predictand band.

    ``terms`` is a sequence of ``(component, weight c_i, row)`` where ``row``
    selects the post period (``0..n_post-1``) or the time-averaged predictand
    (``n_post``). Returns ``(point, lower, upper)``.

    ``cross_unit`` controls the out-of-sample combination across units:
    ``"independent"`` adds the per-cell variances (sub-Gaussian quadrature);
    ``"general"`` averages the half-widths (no cross-unit independence assumed).
    """
    # point: weighted sum of per-unit effects at the selected row.
    point = 0.0
    for comp, c, row in terms:
        eff = comp["obs"] - comp["cf"]                       # per-period effect
        val = float(eff.mean()) if row == comp["n_post"] else float(eff[row])
        point += c * val

    # in-sample: per-draw worst case summed across independent units, then quantiled.
    agg_lo = sum(c * comp["in_lo"][:, row] for comp, c, row in terms)
    agg_hi = sum(c * comp["in_hi"][:, row] for comp, c, row in terms)
    with np.errstate(invalid="ignore"):
        m_in = float(np.nanquantile(agg_lo, u_alpha / 2.0))
        mbar_in = float(np.nanquantile(agg_hi, 1.0 - u_alpha / 2.0))

    # out-of-sample: combine per-cell location-scale bands.
    mean = sum(c * comp["e_mean"][row] for comp, c, row in terms)
    halves = np.array([c * comp["e_half"][row] for comp, c, row in terms], dtype=float)
    if cross_unit == "general":
        half = float(np.sum(np.abs(halves)))
    else:  # independent: sub-Gaussian variances add
        half = float(np.sqrt(np.sum(halves ** 2)))
    m_out, mbar_out = mean - half, mean + half

    lower = point - mbar_in - mbar_out
    upper = point - m_in - m_out
    return point, lower, upper
