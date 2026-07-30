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

Every row is a *distance*, so it reads as "how far apart are we" and cannot be
quietly re-fitted if it moves.

The rows come in two families, against two different references, and keeping them
apart is the point. ``live_*`` rows compare mlsynth to a live augsynth 0.2.0 run
at the pinned commit; ``published_*`` rows compare it to the authors'
``main_result.csv``. The first asks whether mlsynth implements this
specification, the second whether the published artifact was produced by this
version of it. A third row, ``published_vs_live_att_max_diff``, compares the two
references to each other and involves no mlsynth code at all.

Collapsing these into one number would force a single tolerance loose enough to
swallow the known drift, and a real regression in mlsynth could then hide inside
it. Split, the arithmetic is checkable: the published distance is bounded by the
live distance plus the drift.

An important correction, recorded because an earlier version of this case got it
wrong. The disagreements below are NOT mlsynth against augsynth. On every cell
checked against a live augsynth 0.2.0 run, mlsynth agrees to ~1e-7 -- including
the worst cell here, where live augsynth gives 24.27549912 and mlsynth
24.27549912 while the published CSV says 25.72563966. What disagrees is the
published artifact against the pinned package.

The gap is concentrated in the 2016 heating year (mean |diff| 0.20 against ~0.01
for every other year) and in the particulate series, and it reaches 1.45 on an
ATT of ~25, about 6 percent. Across all 1024 cells, 70 percent agree with the
published values to 1e-5.

The per-year breakdown is worth stating exactly, because it is sharper than "the
disagreement is concentrated in 2016" suggests and sharper than an earlier note
here claimed. 2016 does not merely disagree more often -- *none* of its 128 cells
reproduces to 1e-5, where every other year sits between 77 and 84 percent. So
there are two things going on, not one: a broad ~20 percent tail present in every
year, and 2016 failing wholesale. Only the second is year-specific, and 2016
accounts for about 40 percent of the non-reproducing cells rather than for
almost all of them.

The pre-treatment imbalance agrees everywhere to 6e-4, so the same optimum
*value* is reached even where the reported effect differs.

The cause is reference-version drift. The authors ran whatever augsynth was
current in 2022-2023; its ridge cross-validation has changed since, and
``agents_tests.md`` step 0 -- confirm the reference implements the same version of
the spec before comparing bit-for-bit -- is the step that was skipped.

Two revisions of this case got that wrong in opposite directions, and both are
worth recording because the second mistake was made while correcting the first.

The first attributed the whole disagreement to non-unique simplex optima on
near-perfectly-fitting cells. That was fitted to two cells, and the full sweep
refuted it as a general explanation: the worst-disagreeing cells are well
conditioned, with gold ``Scaled_L2`` between 0.25 and 0.60, and non-uniqueness
would have produced a geometric pattern where the data shows a temporal one.

The second then declared non-uniqueness refuted outright, which overshot.
Running the live augsynth comparison found exactly one cell of 30 where mlsynth
and the pinned package disagree beyond solver noise -- ``"Southern control"`` at
2015 PM2.5wn, whose ``scaled_l2`` is 6.3e-6 against 6.0e-3 for the next smallest,
a factor of 964. That cell is an aggregate of the donor pool used as its own
treated unit, so it sits inside the donors' convex hull by construction and the
optimum genuinely is not unique. Non-uniqueness is the right explanation there
and the wrong one for the 2016 drift. Both statements hold; neither generalises
to the other.

So the published values are a *loose* Path-A target here and the drift is the
finding, not a defect. The tight cross-validation against the pinned package
lives in ``docs/replications/ascm_ridge_cv.rst`` and
``benchmarks/cases/ascm_jackknife_plus.py``.

What the cross-validation fix bought
------------------------------------

The cells that do reproduce, reproduce to ~1e-6, and that is recent. Before two
defects in the ridge penalty's cross-validation were fixed -- a fold off-by-one
and a population-vs-sample standard error, see
``docs/replications/ascm_ridge_cv.rst`` -- the 2015 PM2.5wn cell disagreed by
4.3 percent, +18.744 against the published +17.969. It now agrees to 1e-7.

These panels are exactly the shape that exposed those defects: 25 pre-treatment
periods against 37 donors, with the final pre-period sitting on the seasonal ramp
into the heating season, so the fold that augsynth never holds out is the one
carrying nine times the average error.

So there are two distinct stories in this case and they should not be conflated.
The CV fix closed a real defect in mlsynth, taking the reproducible cells from
4.3 percent out to ~1e-6. What remains is not ours: the published artifact was
produced by an older augsynth than the one pinned here, and that is what the
2016 rows measure.

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


def _fit_diagnostics(panel, donors, group, year, pollutant):
    """``(l2_imbalance, selected ridge penalty)`` for one cell.

    Both come off the same fit, since augsynth reports both (``l2_imbalance``
    and ``lambda``) and refitting to get the second would double the case's cost.
    The penalty is worth comparing in its own right: it is the quantity that
    drifted between the authors' augsynth and the pinned one, so it localises a
    disagreement that the ATT alone only registers.
    """
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
        fit = ridge_augment_weights(y_pre, Y0_pre)
    mu = Y0_pre.mean(axis=1)
    l2 = float(np.sqrt(np.sum(
        ((y_pre - mu) - (Y0_pre - mu[:, None]) @ fit.W) ** 2)))
    return l2, float(fit.lambda_)


def _pre_fit_l2(panel, donors, group, year, pollutant):
    """augsynth's ``l2_imbalance``: the pre-period norm on centered outcomes."""
    return _fit_diagnostics(panel, donors, group, year, pollutant)[0]


def _donors():
    return [d for d in
            (_REF / "donor_pool.txt").read_text().split("\n") if d.strip()]


def _live_gold():
    """The live augsynth 0.2.0 run over the same 30 cells, or ``None``.

    Regenerate with ``Rscript benchmarks/reference/song_ml_ascm/reference.R``
    after ``bash benchmarks/R/install_augsynth.sh``; the output is committed so
    the case runs without an R toolchain. Returning ``None`` when it is absent
    keeps a checkout that has lost the file honest -- the live rows go to
    ``nan`` and fail their tolerances rather than silently reporting only the
    published comparison.

    That script reads the same ``basedata`` parquet this case does, via
    ``nanoparquet``, and slices it with the same windows and treatment dates.
    Duplicating the arithmetic across two languages is a cost worth paying to
    avoid duplicating the *data*: an earlier comparison in this project ran 33
    donors on the R side against 37 here because each read its own inputs, and
    the ``live_*`` rows are what would catch the arithmetic drifting.
    """
    p = _REF / "gold_live_augsynth.csv"
    if not p.exists():                          # pragma: no cover - committed
        return None
    return pd.read_csv(p)


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

    live = _live_gold()

    att_diffs, l2_diffs, fitted, skipped = [], [], 0, 0
    live_att, live_l2, live_lambda, drift, degenerate = [], [], [], [], []
    for group, year, pollutant in strata():
        g = gold[(gold.city == group) & (gold.year == year)
                 & (gold.pollutant == pollutant)]
        res = fit_cell(panel, donors, group, year, pollutant)
        if g.empty or res is None:
            skipped += 1
            continue
        fitted += 1
        published = float(g.average_att.iloc[0])
        att_diffs.append(abs(res.effects.att - published))
        l2, lam = _fit_diagnostics(panel, donors, group, year, pollutant)
        l2_diffs.append(abs(l2 - float(g.L2.iloc[0])))

        # Basis two: the same cell through a LIVE augsynth 0.2.0 run at the
        # pinned commit. This is the comparison that says whether mlsynth
        # implements the spec; the published one says whether the artifact was
        # produced by this version of it. They are different questions.
        lv = live[(live.group == group) & (live.year == year)
                  & (live.pollutant == pollutant)] if live is not None else None
        if lv is not None and not lv.empty:
            # The penalty is compared on EVERY cell, degenerate ones included,
            # because it is selected before the QP is solved and so cannot be
            # excused by a non-unique optimum.
            live_lambda.append(abs(lam - float(lv["lambda"].iloc[0])))
            # Reference against reference: no mlsynth code is involved in this
            # number at all. It is the drift, stated as a quantity rather than
            # as a claim in a docstring.
            drift.append(abs(float(lv.att.iloc[0]) - published))

            if float(lv.scaled_l2.iloc[0]) < _DEGENERATE_SCALED_L2:
                degenerate.append((l2, float(lv.l2.iloc[0])))
            else:
                live_att.append(abs(res.effects.att - float(lv.att.iloc[0])))
                live_l2.append(abs(l2 - float(lv.l2.iloc[0])))

    out["n_cells_fitted"] = float(fitted)
    out["n_cells_skipped"] = float(skipped)
    out["published_att_max_diff"] = (
        float(np.max(att_diffs)) if att_diffs else float("nan"))
    out["published_att_mean_diff"] = (
        float(np.mean(att_diffs)) if att_diffs else float("nan"))
    # The claim that holds on every cell, including those where the reported
    # effect drifts: mlsynth reaches the same pre-treatment imbalance the authors
    # report. The same optimum VALUE is found even where the published effect
    # differs, which is what localises the drift to the penalty rather than to
    # the fit.
    out["published_l2_max_diff"] = (
        float(np.max(l2_diffs)) if l2_diffs else float("nan"))

    # Basis two, and the decomposition it buys. If the first row is tight and
    # the third is not, the disagreement with the paper is the artifact's and
    # not mlsynth's -- and each row fails independently, so a regression in
    # mlsynth cannot hide behind the drift.
    out["live_att_max_diff"] = (
        float(np.max(live_att)) if live_att else float("nan"))
    out["live_l2_max_diff"] = (
        float(np.max(live_l2)) if live_l2 else float("nan"))
    out["live_lambda_max_diff"] = (
        float(np.max(live_lambda)) if live_lambda else float("nan"))
    out["published_vs_live_att_max_diff"] = (
        float(np.max(drift)) if drift else float("nan"))
    out["n_live_cells"] = float(len(live_att))

    # The degenerate cell, reported rather than dropped. "Southern control" is
    # an aggregate of the donor pool used as its own treated unit, so it lies
    # (essentially) inside the donors' convex hull and the simplex optimum is
    # not unique. Both solvers reach a pre-treatment fit of nothing; they land
    # on different members of the optimal set and so extrapolate differently.
    # Recording each side's own imbalance says which one got closer, instead of
    # reporting a difference that reads as a defect.
    out["n_degenerate_cells"] = float(len(degenerate))
    out["degenerate_ml_l2"] = (
        float(np.max([m for m, _ in degenerate])) if degenerate else float("nan"))
    out["degenerate_live_l2"] = (
        float(np.max([r for _, r in degenerate])) if degenerate else float("nan"))


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


# Below this ``scaled_l2`` the treated unit is inside the donors' convex hull to
# numerical precision, the simplex optimum is not unique, and two exact solvers
# can reach the same objective while extrapolating differently. The threshold is
# not a knob tuned until a cell passed: across these 30 cells the smallest
# ``scaled_l2`` is 6.3e-6 and the next smallest is 6.0e-3, a factor of 964, so
# anything in that gap classifies identically. One cell falls below it --
# "Southern control", which is an aggregate of the donor pool being used as its
# own treated unit, so the degeneracy is structural rather than accidental.
_DEGENERATE_SCALED_L2 = 1e-4

# Tolerances for the live-augsynth rows, set from the observed agreement rather
# than from a round number. The simplex QP is solved by different exact solvers
# on the two sides -- quadprog in augsynth, an active-set method here -- and the
# residual disagreement is that floor, not a modelling difference: the ATT
# agrees to 6.5e-6 at worst with a median of 4.3e-8, and the pre-fit imbalance
# to 9.3e-7 in BOTH directions. Neither implementation is systematically the
# better fit, which is what a solver floor looks like.
_LIVE_ATT_TOL = 2e-5
_LIVE_L2_TOL = 5e-6
# The penalty is selected before the QP is solved, so it has no solver floor and
# no degenerate-cell excuse. It agrees to 5.8e-10 on all 30 cells.
_LIVE_LAMBDA_TOL = 1e-9

# Two reference bases, kept apart on purpose. Rows prefixed ``live_`` compare
# mlsynth against a live augsynth 0.2.0 run at the pinned commit and answer "does
# mlsynth implement this spec"; rows prefixed ``published_`` compare it against
# the authors' main_result.csv and answer "was that artifact produced by this
# version of the spec". Collapsing them into one number would force a single
# tolerance loose enough to hide a real regression in the first question behind
# the known drift in the second.
EXPECTED = {
    "n_donors": (37.0, 0.5),
    "n_gold_cells": (1024.0, 0.5),
    "n_cells_fitted": (30.0, 0.5),
    "n_cells_skipped": (0.0, 0.5),
    "n_live_cells": (29.0, 0.5),

    # --- basis one: live augsynth 0.2.0 at commit 7a90ea48 (cross-validation).
    # Tight, because this is agreement and not an estimate of it. These are the
    # rows that must not be loosened; if one of them moves, mlsynth moved.
    "live_att_max_diff": (0.0, _LIVE_ATT_TOL),
    "live_l2_max_diff": (0.0, _LIVE_L2_TOL),
    # The penalty itself, which is the quantity that drifted between the authors'
    # augsynth and the pinned one. Comparing it localises a disagreement that the
    # ATT alone only registers as a number being different. Unlike the two rows
    # above it covers all 30 cells, degenerate one included.
    "live_lambda_max_diff": (0.0, _LIVE_LAMBDA_TOL),

    # --- the one degenerate cell, reported rather than dropped.
    # Excluding a cell from a comparison and saying nothing else about it is how
    # a real disagreement gets buried, so both sides' own pre-treatment imbalance
    # is pinned instead. mlsynth reaches ~1e-15 where augsynth stops at ~4.9e-5:
    # the ATT gap of 0.076 between them is two solvers picking different members
    # of an optimal set, and on the objective being optimised mlsynth is the
    # closer of the two. The effect is ~0.003 against a panel where the other
    # cells run to 20, so this is a near-null cell in absolute terms too.
    "n_degenerate_cells": (1.0, 0.5),
    "degenerate_ml_l2": (0.0, 1e-9),
    "degenerate_live_l2": (4.871e-5, 1e-6),

    # --- basis two: the authors' published main_result.csv (Path A).
    # Loose, and pinned at what is actually observed rather than at zero, so a
    # real regression still moves them. Explicitly NOT presented as agreement.
    "published_att_max_diff": (0.919, 0.05),
    "published_att_mean_diff": (0.035, 0.02),
    # This one is tight even against the published artifact: mlsynth reaches the
    # same pre-treatment imbalance the authors report on every cell, including
    # the 2016 ones whose reported effect drifts. The same optimum VALUE is
    # found where the reported effect differs, which is what localises the drift
    # to the penalty rather than to the fit.
    "published_l2_max_diff": (0.0, 1e-4),
    # The fraction of the 30 strata reproducing the published value to 1e-5:
    # 28 of 30. The tolerance is one cell either way (1/30 = 0.033) with a
    # little headroom, so a single cell crossing the line is visible rather
    # than absorbed. Re-pinned from a stale 0.867 that predated the ridge
    # cross-validation fix and had been passing only by sitting inside a
    # tolerance wide enough to hide the improvement.
    "frac_cells_within_1e5": (0.933, 0.05),
    "average_att_bound_max_diff": (0.0, 1e-5),

    # --- the drift itself: reference against reference, no mlsynth code in it.
    # This is what makes the two-basis structure worth its cost. With only the
    # published basis, "mlsynth disagrees with the paper" and "the paper's
    # artifact disagrees with its own package" are the same number and cannot be
    # told apart. Here they are three rows, and the arithmetic is checkable:
    # published_att_max_diff is bounded by live_att_max_diff plus this.
    "published_vs_live_att_max_diff": (0.919, 0.05),

    # Their headline finding, which survives all of the above.
    "heating_raises_pm25": (1.0, 0.5),
}
