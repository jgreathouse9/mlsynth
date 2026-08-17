"""Callaway-Sant'Anna group-time ATT and its influence function, vendored.

Transcribed from ``diff-diff`` 3.9.0 (Isaac Gerber, MIT; see
``LICENSE.diff-diff``), commit ``d9cd475fec8e67ceea1772896e2b29501cb90485``,
from ``diff_diff/staggered.py`` (the per-cell estimate and influence function)
and ``diff_diff/staggered_aggregation.py`` (the weight-influence-corrected
aggregation). ``diff-diff`` carries R-parity suites against ``did``, ``fixest``,
``estimatr`` and Stata, which is what makes it usable as a reference here.

Vendored, not depended on, and cut down instead of copied wholesale.
The distribution is 95 modules of estimators mlsynth does not use; what PPSCM's
Callaway-Sant'Anna mode has to agree with is the panel / never-treated /
no-covariate path, which is small enough to read in one sitting. A reference
nobody can read is not a reference, and 240KB of ``staggered.py`` would drift
without anyone noticing.

Restrictions, all of them deliberate and all of them checked by the caller:

* balanced panel, one row per ``(unit, time)``;
* a never-treated comparison group, coded ``first_treat = 0``;
* no covariates, so the doubly-robust estimator collapses to the difference in
  means that ``diff-diff`` computes inline for that case;
* no survey design, no clustering, no anticipation.

Outside those the vendored path is wrong, not approximate, so it raises.

Refreshing this file is a deliberate act: bump the commit above, re-transcribe,
and re-run the parity test. It is a characterisation of a pinned upstream, and
an edit that brings it into agreement with mlsynth stops it being evidence of
anything.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

#: Upstream this file is a transcription of. Recorded here as well as in
#: ``provenance.json`` so a reader of the code sees it without leaving it.
DIFF_DIFF_VERSION = "3.9.0"
DIFF_DIFF_COMMIT = "d9cd475fec8e67ceea1772896e2b29501cb90485"


def _wide(df, outcome, unit, time):
    w = df.pivot(index=unit, columns=time, values=outcome)
    if w.isna().to_numpy().any():
        raise ValueError(
            "The vendored Callaway-Sant'Anna reference needs a balanced panel: "
            "the (unit, time) grid has holes.")
    return w


def group_time_att(
    df: pd.DataFrame, outcome: str, unit: str, time: str, first_treat: str,
) -> Tuple[Dict[Tuple[Any, Any], float], Dict[Tuple[Any, Any], float], Dict[Any, Any]]:
    """``ATT(g, t)``, its standard error, and the per-cell influence functions.

    With a never-treated comparison group and no covariates each cell is a
    two-period, two-group difference in differences against base period
    ``g - 1``. ``diff-diff`` computes exactly this inline
    (``staggered.py``)::

        mu_t = mean(treated_change);  mu_c = mean(control_change)
        att  = mu_t - mu_c
        inf_treated =  (treated_change - mu_t) / n_t
        inf_control = -(control_change - mu_c) / n_c
        se = sqrt(sum(inf_treated**2) + sum(inf_control**2))

    The standard error is ``sqrt(sum(phi^2))`` and not a plug-in with ``ddof=1``:
    the influence contributions are already divided by their group sizes, and
    the DRDID convention is what R's ``reg_did_panel`` matches.
    """
    wide = _wide(df, outcome, unit, time)
    cohort = df.groupby(unit)[first_treat].first().reindex(wide.index)
    never = np.flatnonzero((cohort == 0).to_numpy())
    if never.size == 0:
        raise ValueError(
            "The vendored reference uses a never-treated comparison group and "
            "this panel has none.")
    periods = list(wide.columns)
    Y = wide.to_numpy(dtype=float)

    effects: Dict[Tuple[Any, Any], float] = {}
    ses: Dict[Tuple[Any, Any], float] = {}
    inf: Dict[Tuple[Any, Any], Any] = {}
    for g in sorted({c for c in cohort.unique() if c != 0}):
        if (g - 1) not in periods:
            continue
        base = periods.index(g - 1)
        treated = np.flatnonzero((cohort == g).to_numpy())
        if treated.size == 0:
            continue
        for t in periods:
            if t < g:
                continue
            col = periods.index(t)
            d_tr = Y[treated, col] - Y[treated, base]
            d_co = Y[never, col] - Y[never, base]
            mu_t, mu_c = float(d_tr.mean()), float(d_co.mean())
            att = mu_t - mu_c
            inf_t = (d_tr - mu_t) / treated.size
            inf_c = -(d_co - mu_c) / never.size
            effects[(g, t)] = att
            ses[(g, t)] = float(np.sqrt((inf_t ** 2).sum() + (inf_c ** 2).sum()))
            inf[(g, t)] = {"treated_idx": treated, "control_idx": never,
                           "treated_inf": inf_t, "control_inf": inf_c}
    return effects, ses, inf


def aggregate(
    cells: Iterable[Tuple[Any, Any]],
    effects: Dict[Tuple[Any, Any], float],
    inf: Dict[Any, Any],
    cohort_of_unit: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Tuple[float, float, np.ndarray]:
    """Aggregate selected cells, with the weight-influence correction.

    The correction is the part that is easy to omit and expensive to omit: the
    group-size weights are themselves estimated, so an aggregate's variance
    carries their sampling error too. Dropping it leaves standard errors that
    look entirely plausible and are too small. R's ``did::aggte`` computes

        agg_inf_i = sum_k w_k * inf_i_k + wif_i
        se        = sqrt(sum(agg_inf^2))

    with ``wif`` built from the group shares ``pg`` as transcribed below.

    Returns ``(estimate, standard_error, psi)``; ``psi`` is the per-unit
    combined influence function, which is what a multiplier bootstrap resamples.
    """
    cells = [c for c in cells if c in effects]
    if not cells:
        return float("nan"), float("nan"), np.zeros(0)
    n_units = int(cohort_of_unit.shape[0])
    eff = np.array([effects[c] for c in cells], dtype=float)
    w = (np.full(len(cells), 1.0 / len(cells)) if weights is None
         else np.asarray(weights, dtype=float))

    psi_standard = np.zeros(n_units)
    for k, c in enumerate(cells):
        info = inf[c]
        psi_standard[info["treated_idx"]] += w[k] * info["treated_inf"]
        psi_standard[info["control_idx"]] += w[k] * info["control_inf"]

    groups_for_cell = np.array([c[0] for c in cells])
    uniq = sorted(set(groups_for_cell))
    pg_by_group = {g: float((cohort_of_unit == g).sum()) / n_units for g in uniq}
    pg_keepers = np.array([pg_by_group[g] for g in groups_for_cell])
    sum_pg = float(pg_keepers.sum())
    if sum_pg == 0:
        return float(eff @ w), 0.0, psi_standard

    indicator = (cohort_of_unit[:, None] == groups_for_cell[None, :]).astype(float)
    diff = indicator - pg_keepers
    if1 = diff / sum_pg
    if2 = np.outer(diff.sum(axis=1), pg_keepers) / (sum_pg ** 2)
    psi_wif = ((if1 - if2) @ eff) / n_units

    psi = psi_standard + psi_wif
    return float(eff @ w), float(np.sqrt((psi ** 2).sum())), psi


def multiplier_bootstrap(psi: np.ndarray, n_boot: int, seed: int = 0) -> np.ndarray:
    """Mammen multiplier bootstrap draws of ``sum_i v_i psi_i``.

    Mammen's two-point distribution has mean 0, variance 1 and third moment 1,
    which is what buys the refinement over Rademacher weights here.
    """
    rng = np.random.default_rng(seed)
    k = (np.sqrt(5.0) + 1.0) / 2.0
    p = (np.sqrt(5.0) + 1.0) / (2.0 * np.sqrt(5.0))
    draws = np.where(rng.random((n_boot, psi.shape[0])) < p, 1.0 - k, k)
    return draws @ psi
