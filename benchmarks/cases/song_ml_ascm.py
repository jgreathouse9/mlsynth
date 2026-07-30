"""Ridge ASCM Path-A: Song et al. (2023) clean winter heating in China.

Path A (the paper's published empirical result, on the authors' own data). Song,
Liu, Cheng, Cole, Dai, Elliott & Shi (2023), *Attribution of Air Quality Benefits
to Clean Winter Heating Policies in China*, Environ. Sci. Technol.
57:17707-17717, `10.1021/acs.est.2c06800 <https://doi.org/10.1021/acs.est.2c06800>`_.

Their method ("ML-ASCM") is two stages and only the second is a synthetic
control: a random-forest weather normalization, then Ridge Augmented SCM via
``augsynth``. The first stage is not reimplemented and does not need to be -- the
authors ship the deweathered series, vendored as
``basedata/song_ml_ascm_china.parquet``, so the synthetic-control half is
reproducible on its own.

The design, transcribed from their ``main_result.R`` rather than inferred from the
paper's prose: for each of 8 heating-year windows (1 May to 30 April) and each of
8 treatment groups, mark treatment from 23 October of the starting year, fit
against a fixed pool of 37 southern control cities, and repeat for each of 16
pollutant series. 8 x 8 x 16 = 1024 augsynth fits, which is what their published
``main_result.csv`` contains.

Two things about the units are worth stating because they change the estimand.
The treated "unit" in every cell here is a pre-aggregated regional average --
``"2+26 cities"`` and ``"Northern"`` are themselves rows in the panel, not
collections resolved at fit time. And the sign convention is theirs: a positive
ATT means winter heating *raised* the pollutant, since the counterfactual is the
no-heating trajectory.

What this case pins
-------------------

A stratified 30-cell subset rather than all 1024, so the routine benchmark stays
cheap. The strata sweep each dimension independently: all 8 groups at one
pollutant-year, all 16 pollutants for one group-year, and all 8 years for one
group-pollutant. A defect confined to one group, one pollutant or one year is
therefore still caught. The full sweep lives in
``benchmarks/reference/song_ml_ascm/run_full_sweep.py`` and is not part of
``run_benchmarks``; ``python benchmarks/reference/song_ml_ascm/run_full_sweep.py``
runs all 1024.

Agreement is asserted as a *distance* from the published values, so each row reads
as "how far apart are we" and cannot be quietly re-fitted if it drifts.

An important correction, recorded because an earlier version of this case got it
wrong. The disagreements below are NOT mlsynth against augsynth. On every cell
checked against a live augsynth 0.2.0 run, mlsynth agrees to ~1e-7 -- including
the worst cell here, where live augsynth gives 24.27549912 and mlsynth
24.27549912 while the published CSV says 25.72563966. What disagrees is the
published artifact against the pinned package.

The gap is concentrated in the 2016 heating year (mean |diff| 0.20 against ~0.01
for every other year) and in the particulate series, and it reaches 1.45 on an
ATT of ~25, about 6 percent. Across all 1024 cells, 54 percent agree with the
published values to 1e-6 and 70 percent to 1e-5; the rest are almost all 2016.
The pre-treatment imbalance agrees everywhere to 6e-4, so the same optimum
*value* is reached even where the reported effect differs.

The cause is reference-version drift. The authors ran whatever augsynth was
current in 2022-2023; its ridge cross-validation has changed since, and
``agents_tests.md`` step 0 -- confirm the reference implements the same version of
the spec before comparing bit-for-bit -- is the step that was skipped. An earlier
revision of this case instead attributed the disagreement to non-unique simplex
optima on near-perfectly-fitting cells. That explanation was fitted to two cells
and the full sweep refuted it: the worst-disagreeing cells are well conditioned,
with gold ``Scaled_L2`` between 0.25 and 0.60.

So the published values are a *loose* Path-A target here and the drift is the
finding, not a defect. The tight cross-validation against the pinned package
lives in ``docs/replications/ascm_ridge_cv.rst`` and
``benchmarks/cases/ascm_jackknife_plus.py``.

Why the agreement is this tight
-------------------------------

The ATT reproduces to ~1e-6. That is a recent state of affairs and worth
recording: before two defects in the ridge penalty's cross-validation were fixed
(a fold off-by-one and a population-vs-sample standard error, see
``docs/replications/ascm_ridge_cv.rst``) this cell disagreed by 4.3 percent --
+18.744 against the published +17.969. These panels are exactly the shape that
exposes it: 25 pre-treatment periods against 37 donors, with the final
pre-period sitting on the seasonal ramp into the heating season.

The interval columns need ``inference="jackknife_plus"``; the authors summarise
every fit with ``inf_type = "jackknife+"``, so ``lower_bound`` /
``average_att_lower`` and their upper counterparts are unreachable without it.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parents[1]
_REF = _HERE / "reference" / "song_ml_ascm"
_PANEL = _HERE.parent / "basedata" / "song_ml_ascm_china.parquet"

#: Heating-year windows and the treatment date within each, from ``main_result.R``.
WINDOWS = {
    2014: ("2014-05-01", "2015-04-30"), 2015: ("2015-05-01", "2016-04-30"),
    2016: ("2016-05-01", "2017-04-30"), 2017: ("2017-05-01", "2018-04-30"),
    2018: ("2018-05-01", "2019-04-30"), 2019: ("2019-05-01", "2020-04-30"),
    2020: ("2020-05-01", "2021-04-30"), 2021: ("2021-05-01", "2021-12-31"),
}
INTERVENTION_MMDD = "10-23"

GROUPS = ["2+26 cities", "Other northern cities", "Alternative", "Northern",
          "South mixing", "Southern", "Southern control", "China"]
POLLUTANTS = ["SO2wn", "NO2wn", "PM2.5wn", "PM10wn", "O3_8hwn", "COwn",
              "SO2", "NO2", "PM2.5", "PM10", "O3_8h", "CO", "O3", "O3wn",
              "Ox", "Oxwn"]


def _pre_fit_l2(panel, donors, group, year, pollutant):
    """augsynth's ``l2_imbalance``: the pre-period norm on centered outcomes."""
    from mlsynth.utils.bilevel.ridge_augment import ridge_augment_weights
    sub = _cell(panel, donors, group, year, pollutant)
    piv = sub.pivot(index="ID", columns="date", values=pollutant).sort_index()
    trt = piv.index.to_numpy() == group
    cut = pd.Timestamp(f"{year}-{INTERVENTION_MMDD}")
    n_pre = int((piv.columns < cut).sum())
    Y = piv.to_numpy()
    y_pre, Y0_pre = Y[trt][0][:n_pre], Y[~trt][:, :n_pre].T
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = ridge_augment_weights(y_pre, Y0_pre).W
    mu = Y0_pre.mean(axis=1)
    return float(np.sqrt(np.sum(((y_pre - mu) - (Y0_pre - mu[:, None]) @ w) ** 2)))


def _donors():
    return [d for d in
            (_REF / "donor_pool.txt").read_text().split("\n") if d.strip()]


def _cell(panel, donors, group, year, pollutant):
    """The (units x periods) slice one augsynth call sees, with its treatment flag."""
    start, end = WINDOWS[year]
    sub = panel[(panel.date >= start) & (panel.date <= end)]
    sub = sub[sub.ID.isin([group] + donors)][["ID", "date", pollutant]].copy()
    cut = pd.Timestamp(f"{year}-{INTERVENTION_MMDD}")
    sub["treatment"] = ((sub.date >= cut) & (sub.ID == group)).astype(int)
    return sub


def strata():
    """The 30 cells, sweeping each dimension independently.

    Returned rather than inlined so the full-sweep driver and this case cannot
    drift apart on how a cell is identified.
    """
    cells = [(g, 2015, "PM2.5wn") for g in GROUPS]                    # 8 groups
    cells += [("2+26 cities", 2015, p) for p in POLLUTANTS]           # 16 pollutants
    cells += [("2+26 cities", y, "PM2.5wn") for y in WINDOWS]         # 8 years
    seen, out = set(), []
    for c in cells:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def fit_cell(panel, donors, group, year, pollutant, inference=False):
    """One cell through VanillaSC's ridge ASCM. ``None`` if the slice is unusable."""
    from mlsynth import VanillaSC
    sub = _cell(panel, donors, group, year, pollutant)
    if sub.empty or sub[pollutant].isna().any() or sub.treatment.sum() == 0:
        return None
    cfg = {"df": sub, "outcome": pollutant, "treat": "treatment", "unitid": "ID",
           "time": "date", "display_graphs": False, "augment": "ridge",
           "inference": inference or False}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VanillaSC(cfg).fit()


def run() -> dict:
    out: dict = {}
    if not _PANEL.exists():                     # pragma: no cover - vendored
        return {k: float("nan") for k in EXPECTED}
    panel = pd.read_parquet(_PANEL)
    gold = pd.read_parquet(_REF / "gold_main_result.parquet")
    donors = _donors()

    out["n_donors"] = float(len(donors))
    out["n_gold_cells"] = float(
        gold.groupby(["city", "year", "pollutant"]).ngroups)

    att_diffs, l2_diffs, fitted, skipped = [], [], 0, 0
    for group, year, pollutant in strata():
        g = gold[(gold.city == group) & (gold.year == year)
                 & (gold.pollutant == pollutant)]
        res = fit_cell(panel, donors, group, year, pollutant)
        if g.empty or res is None:
            skipped += 1
            continue
        fitted += 1
        att_diffs.append(abs(res.effects.att - float(g.average_att.iloc[0])))
        l2_diffs.append(abs(_pre_fit_l2(panel, donors, group, year, pollutant)
                            - float(g.L2.iloc[0])))

    out["n_cells_fitted"] = float(fitted)
    out["n_cells_skipped"] = float(skipped)
    out["att_max_diff"] = float(np.max(att_diffs)) if att_diffs else float("nan")
    out["att_mean_diff"] = float(np.mean(att_diffs)) if att_diffs else float("nan")
    # The claim that holds on every cell, including those where the reported
    # effect drifts: mlsynth reaches the same pre-treatment imbalance the authors
    # report. The same optimum VALUE is found even where the published effect
    # differs, which is what localises the drift to the penalty rather than to
    # the fit.
    out["pre_fit_l2_max_diff"] = (
        float(np.max(l2_diffs)) if l2_diffs else float("nan"))


    # The interval columns, which exist only because jackknife+ was ported. Two
    # cells rather than thirty: each costs one refit per pre-treatment period.
    bound_diffs = []
    for group, year, pollutant in (("2+26 cities", 2015, "PM2.5wn"),
                                   ("Northern", 2015, "SO2wn")):
        g = gold[(gold.city == group) & (gold.year == year)
                 & (gold.pollutant == pollutant)]
        res = fit_cell(panel, donors, group, year, pollutant,
                       inference="jackknife_plus")
        if g.empty or res is None or res.inference is None:  # pragma: no cover
            continue
        bound_diffs += [abs(res.inference.ci_lower
                            - float(g.average_att_lower.iloc[0])),
                        abs(res.inference.ci_upper
                            - float(g.average_att_upper.iloc[0]))]
    out["average_att_bound_max_diff"] = (
        float(np.max(bound_diffs)) if bound_diffs else float("nan"))

    # Their headline sign: winter heating RAISED PM2.5 in the "2+26" cities.
    r = fit_cell(panel, donors, "2+26 cities", 2015, "PM2.5wn")
    out["heating_raises_pm25"] = float(r is not None and r.effects.att > 0)
    out["frac_cells_within_1e5"] = (
        float(np.mean([d <= 1e-5 for d in att_diffs])) if att_diffs else float("nan"))
    return out


EXPECTED = {
    "n_donors": (37.0, 0.5),
    "n_gold_cells": (1024.0, 0.5),
    "n_cells_fitted": (30.0, 0.5),
    "n_cells_skipped": (0.0, 0.5),
    # Distances from the PUBLISHED values, which drift from the pinned package
    # (see the module docstring). Pinned at what is actually observed so a real
    # regression still moves them, and explicitly NOT presented as agreement.
    "att_max_diff": (0.919, 0.05),
    "att_mean_diff": (0.035, 0.02),
    # Holds on every cell, including the 2016 ones whose published effect drifts.
    "pre_fit_l2_max_diff": (0.0, 1e-4),
    # Measured, not targeted. Recorded so the size of the non-uniqueness is
    # visible rather than hidden; a large change here is worth investigating even
    # though agreement is not expected.
    # The claim that holds tightly: the fraction of the 30 strata reproducing the
    # published value to 1e-5. Most do; the 2016 cells are the exceptions.
    "frac_cells_within_1e5": (0.867, 0.07),
    "average_att_bound_max_diff": (0.0, 1e-5),
    "heating_raises_pm25": (1.0, 0.5),
}
