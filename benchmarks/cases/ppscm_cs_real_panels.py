"""Cross-validation: PPSCM's Callaway-Sant'Anna mode on ``diff-diff``'s own panels.

The equivalence #465 established was measured on panels this repository
generated. Agreement on synthetic data is agreement on a data-generating
process someone here chose; this case runs it on the datasets the reference
implementation validates against, including Callaway and Sant'Anna's own paper
data.

Provenance
----------
Datasets ship with ``diff-diff`` (Isaac Gerber, MIT) and are loaded from the
installed package, so nothing is vendored and nothing can drift out of step
with the reference:

* ``mpdta`` -- the county teen-employment panel from Callaway & Sant'Anna
  (2021), and the example dataset of R's ``did``. 500 counties x 5 years,
  cohorts 2004/2006/2007, 309 never-treated.
* ``castle_doctrine`` -- castle-doctrine laws, 50 states x 11 years, five
  cohorts, 29 never-treated.
* ``walmart`` -- fourteen cohorts, which is what exercises the cohort-share
  weighting that a three- or five-cohort panel cannot. ``diff-diff`` sources it
  by download and substitutes a synthetic panel of the same schema when the
  host is unreachable, so it is a structural check and not a replication. Its
  metrics carry that in their names.

Every panel's ``attrs["source"]`` is pinned. A silent substitution of the
synthetic fallback -- a network failure during a benchmark run -- would
otherwise let this case report real-data agreement against generated data,
which is the failure mode #473 was about.

Which estimator these panels justify
------------------------------------
This case validates PPSCM's ``callaway_santanna`` mode, and only that mode, on
these panels. Difference in differences is a large-N, short-T estimator, and
``mpdta`` is exactly that shape: 4 pre-periods against 309 never-treated
donors, N0/T0 = 77, with 480 donors carrying weight in the window pool. The
partially-pooled program there fits 480 free parameters to 4 balance
constraints and reaches a global imbalance of 3.4e-06, which is interpolation
and not fit. ``castle_doctrine`` at N0/T0 = 3.2 cannot drive its imbalance to
zero (2.1e-02), which is what a determined problem looks like.

So the agreement recorded below says the Callaway-Sant'Anna estimand is
reachable from PPSCM and reached exactly, on the data the reference is
validated against. It is not a comparison of PPSCM's default against
Callaway-Sant'Anna, and a panel this short does not support one.

The reference is ``diff_diff.CallawaySantAnna(control_group="never_treated",
estimation_method="dr", n_bootstrap=0)`` run live on the same frame, so the
target is the package's number and not a transcription of it.

What is compared, and on which cells
------------------------------------
PPSCM inherits augsynth's rule that the event study runs to the *last* cohort's
horizon, ``n_leads = T - T_0``. On real staggered panels the last cohort tends
to adopt at or near the final period, so that window is short: one lead on
``mpdta`` and ``walmart``, two on ``castle_doctrine``. The comparison is
therefore over the cells PPSCM estimates -- 3 of the 7 post-treatment cells
Callaway-Sant'Anna report for ``mpdta``, 10 of 20 for ``castle_doctrine``,
14 of 105 for ``walmart`` -- and the coverage fraction is reported as a metric
so a change in the windowing rule shows up here.

``mpdta`` also appears with its 2007 cohort dropped. That cohort adopts in the
final period, so removing it moves the last adoption back to 2006 and widens
the window to two leads; it is a subsample of the real panel, not a synthetic
variant, and it exercises the multi-lead path on real data.

The castle_doctrine residual
----------------------------
Its ``homicide_rate`` ships as ``float32`` while the other two panels are
``float64``. As shipped, the two implementations promote to double at different
points and agree to 4.5e-08 -- float32 epsilon against an ATT of 0.62 is
7.4e-08, so that residual is the storage type and not a disagreement. Cast the
column to ``float64`` and the same comparison closes to 1.1e-16. Both are
measured below: ``castle_att_diff`` keeps the shipped dtype, and
``castle_att_diff_float64`` pins the diagnosis.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from benchmarks.compare import BenchmarkSkipped

_SPECS = {
    "mpdta": ("load_mpdta", "countyreal", "year", "lemp", "first_treat"),
    "castle": ("load_castle_doctrine", "state", "year", "homicide_rate", "first_treat"),
    "walmart": ("load_walmart", "cid", "year", "log_retail_emp", "first_year"),
}


def _load(key: str) -> pd.DataFrame:
    try:
        import diff_diff
    except ImportError as exc:                   # pragma: no cover - CI without it
        raise BenchmarkSkipped(
            "diff-diff is not installed; it supplies both the reference "
            "implementation and the datasets for this case.") from exc
    return getattr(diff_diff, _SPECS[key][0])()


def _cs_cells(df, unit, time, y, ft):
    """``diff-diff``'s per-cell estimates and standard errors, from a live fit."""
    from diff_diff import CallawaySantAnna

    fit = CallawaySantAnna(control_group="never_treated", estimation_method="dr",
                           n_bootstrap=0).fit(df, outcome=y, unit=unit,
                                              time=time, first_treat=ft)
    eff = {c: float(v["effect"]) for c, v in fit.group_time_effects.items()}
    ses = {c: float(v["se"]) for c, v in fit.group_time_effects.items()}
    return eff, ses


def _one(key: str, df: pd.DataFrame = None) -> Dict[str, Any]:
    from mlsynth import PPSCM

    _, unit, time, y, ft = _SPECS[key]
    src = df.attrs.get("source") if df is not None else None
    df = (_load(key) if df is None else df).copy()
    if src is not None:
        df.attrs["source"] = src
    df["_d"] = ((df[ft] > 0) & (df[time] >= df[ft])).astype(int)

    eff, ses = _cs_cells(df, unit, time, y, ft)
    per = df.groupby(unit)[ft].first()
    size = {g: int((per == g).sum()) for g in sorted(c for c in per.unique() if c > 0)}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = PPSCM({"df": df[[unit, time, y, "_d"]], "outcome": y, "treat": "_d",
                     "unitid": unit, "time": time, "display_graphs": False,
                     "method": "callaway_santanna", "run_inference": True}).fit()

    leads = len(next(iter(res.per_unit.values())).tau)
    cells = [(g, g + h) for g in size for h in range(leads) if (g, g + h) in eff]
    w = np.array([size[g] for g, _ in cells], dtype=float)
    w /= w.sum()
    target = float(np.array([eff[c] for c in cells]) @ w)

    gta, gse = res.inference_detail.group_time_att, res.inference_detail.group_time_se
    return {
        "source": str(df.attrs.get("source", "unknown")),
        "att_diff": abs(float(res.att) - target),
        "cell_att_diff": max(abs(gta[c] - eff[c]) for c in cells),
        "cell_se_diff": max(abs(gse[c] - ses[c]) for c in cells
                            if np.isfinite(ses[c])),
        "leads": leads,
        "cells": len(cells),
        "post_cells": sum(1 for (g, t) in eff if t >= g),
        "cohorts": len(size),
    }


def run() -> dict:
    mpdta = _one("mpdta")
    full = _load("mpdta")
    trimmed = _one("mpdta", full[full.first_treat != 2007])
    castle = _one("castle")
    c64 = _load("castle")
    c64["homicide_rate"] = c64["homicide_rate"].astype(np.float64)
    castle64 = _one("castle", c64)
    walmart = _one("walmart")

    return {
        "mpdta_att_diff": mpdta["att_diff"],
        "mpdta_cell_att_diff": mpdta["cell_att_diff"],
        "mpdta_cell_se_diff": mpdta["cell_se_diff"],
        "mpdta_cells_covered": mpdta["cells"],
        "mpdta_post_cells": mpdta["post_cells"],
        "mpdta_trimmed_att_diff": trimmed["att_diff"],
        "mpdta_trimmed_leads": trimmed["leads"],
        "castle_att_diff": castle["att_diff"],
        "castle_att_diff_float64": castle64["att_diff"],
        "castle_cell_se_diff_float64": castle64["cell_se_diff"],
        "many_cohort_att_diff": walmart["att_diff"],
        "many_cohort_cell_se_diff": walmart["cell_se_diff"],
        "many_cohort_cohorts": walmart["cohorts"],
        "mpdta_is_authentic": int(mpdta["source"] == "callaway_santanna_mpdta"),
        "castle_is_authentic": int(castle["source"] == "cheng_hoekstra_castle_data"),
    }


def comparison() -> dict:
    """PPSCM's Callaway-Sant'Anna mode against ``diff-diff``, panel by panel.

    Each row is a live ``CallawaySantAnna`` fit on the same frame, aggregated
    over the cells PPSCM's event-study window covers with the cohort-share
    weights PPSCM uses, against PPSCM's own aggregate. Raises
    ``BenchmarkSkipped`` when ``diff-diff`` is absent, since it supplies the
    reference and the data alike.
    """
    from mlsynth import PPSCM

    rows: List[Dict[str, Any]] = []
    jobs: List[Tuple[str, str, pd.DataFrame]] = []
    full = _load("mpdta")
    jobs.append(("mpdta", "mpdta", None))
    jobs.append(("mpdta (2007 cohort dropped)", "mpdta",
                 full[full.first_treat != 2007]))
    jobs.append(("castle_doctrine (float32, as shipped)", "castle", None))
    c64 = _load("castle")
    c64["homicide_rate"] = c64["homicide_rate"].astype(np.float64)
    jobs.append(("castle_doctrine (cast to float64)", "castle", c64))
    jobs.append(("walmart (synthetic fallback here; structural check of 14-cohort weighting)", "walmart", None))

    for label, key, frame in jobs:
        m = _one(key, frame)
        rows.append({"quantity": f"{label}/ATT abs diff",
                     "mlsynth": float(f"{m['att_diff']:.3g}"), "reference": 0.0})
        rows.append({"quantity": f"{label}/per-cell ATT abs diff",
                     "mlsynth": float(f"{m['cell_att_diff']:.3g}"), "reference": 0.0})
        rows.append({"quantity": f"{label}/per-cell SE abs diff",
                     "mlsynth": float(f"{m['cell_se_diff']:.3g}"), "reference": 0.0})
        rows.append({"quantity": f"{label}/cells covered",
                     "mlsynth": m["cells"], "reference": m["post_cells"]})

    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "PPSCM",
                         "config": {"method": "callaway_santanna",
                                    "run_inference": True}},
        "reference": {"impl": "diff_diff.CallawaySantAnna (live run)",
                      "version": "diff-diff 3.9.0"},
    }


# Agreement is exact wherever both sides read float64: the estimators are the
# same object on these conventions, so the only residual is arithmetic. The
# castle_doctrine tolerance as shipped brackets float32 epsilon against an ATT
# of 0.62 (7.4e-08), and the float64 row beside it shows that is the whole of
# it. "cells covered" is a structural count, pinned exactly so a change to
# augsynth's n_leads rule surfaces here instead of silently widening or
# narrowing what was compared.
EXPECTED = {
    "mpdta_att_diff": (0.0, 1e-14),
    "mpdta_cell_att_diff": (0.0, 1e-14),
    "mpdta_cell_se_diff": (0.0, 1e-14),
    "mpdta_cells_covered": (3, 0),
    "mpdta_post_cells": (7, 0),
    "mpdta_trimmed_att_diff": (0.0, 1e-14),
    "mpdta_trimmed_leads": (2, 0),
    "castle_att_diff": (0.0, 1e-06),          # float32 as shipped
    "castle_att_diff_float64": (0.0, 1e-14),
    "castle_cell_se_diff_float64": (0.0, 1e-14),
    "many_cohort_att_diff": (0.0, 1e-14),
    "many_cohort_cell_se_diff": (0.0, 1e-14),
    "many_cohort_cohorts": (14, 0),
    # Provenance, pinned. These are the assertions that keep the case honest:
    # diff-diff falls back to a synthetic panel of the same schema when a
    # dataset host is unreachable, and without these a network failure would
    # turn a real-data replication into a generated-data one with no sign.
    "mpdta_is_authentic": (1, 0),
    "castle_is_authentic": (1, 0),
}
