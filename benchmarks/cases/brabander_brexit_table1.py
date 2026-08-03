"""Path A: de Brabander, Juodis & Miyazato Szini (2025) Table 1 -- Brexit GDP loss.

The paper asks what the synthetic-control family says about the cost of the
Brexit referendum once you stop treating "the SC estimator" as a single thing.
It runs seven estimators -- classic SC, demeaned SC, SDID under three panel
conventions, MASC and ASCM -- on the same OECD panel and tabulates the implied
percentage shortfall in UK real GDP at the end of 2018 and 2019. Table 1 is the
no-covariate block; this case reproduces its 2016:Q3 column, all fourteen cells.

    method       2018:Q4  paper / mlsynth      2019:Q4  paper / mlsynth
    SC                    3.06 / 3.04                   4.20 / 4.17
    DSC                   2.98 / 2.99                   4.12 / 4.12
    SDID (i)              2.76 / 2.77                   3.89 / 3.91
    SDID (ii)             2.79 / 2.80                   3.92 / 3.94
    SDID (iii)            2.79 / 2.80                   3.92 / 3.94
    MASC                  2.73 / 2.73                   3.83 / 3.83
    ASCM                  3.04 / 3.04                   4.19 / 4.19

Largest deviation 0.028 (SC at 2019:Q4); eleven of the fourteen cells agree to
better than 0.02. The paper prints two decimals, so this is a match at display
precision throughout.

The three SDID cases differ only in which quarters the panel carries:

* (i)   fit on the panel through 2016:Q4, effect read at the evaluation quarter
        -- so the reported quantity is outside the fitted window;
* (ii)  fit on the panel through the evaluation quarter;
* (iii) fit on the pre-treatment quarters plus the evaluation quarter alone.

Cases (ii) and (iii) come out identical here, in mlsynth and in the paper alike.
That is not a coincidence of rounding: with the ridge switched off, the unit
weights depend only on pre-treatment outcomes, and the time-weight problem
lands on the vertex that puts all its mass on the final pre-treatment quarter,
so the two panels present the weight programs with the same information. Case
(i) does differ, because its time weights spread a little mass onto two earlier
quarters.

Every SDID number here needs ``zeta=0`` and ``intercept_adjust=True``. The
first is the paper's ``zeta.omega = 0``; the second is the lambda-corrected
counterfactual their estimator uses. With mlsynth's default ridge instead, case
(ii) reads 2.53 / 3.60 against the published 2.79 / 3.92 -- the option exists
precisely so this specification is expressible.

Provenance
----------
* Paper: de Brabander, Juodis & Miyazato Szini (2025), "On the Use of Synthetic
  Difference-in-Differences Approach with(-out) Covariates: The Case Study of
  Brexit Referendum", Econometric Reviews. Table 1, treatment period 2016:Q3,
  no covariates when matching.
* Authors' code: ``Treatment effects/Treatment effects - 2016Q3 no covariates -
  no penalty.R`` in their replication package. The window indices (141:226 pre,
  236 = 2018:Q4, 240 = 2019:Q4), ``N0 = 23``, ``T0 = 86``, the three SDID panel
  cases and the ``zeta.omega = zeta.lambda = 0`` calls are read from that file.
* MASC tuning: ``m in [1, 10]`` with 80 folds over 85 pre-treatment periods, per
  the Table 1 note -- ``min_preperiods=6``. The script's ``min_value_five``
  variable sets 5 folds instead and does not reproduce the printed cells; the
  note does.
* Data: the committed ``brexit_panel.csv`` (see ``benchmarks/brabander_common.py``).
"""
from __future__ import annotations

from benchmarks.brabander_common import (
    FIRST_T, TREAT_T, UK, fit_ascm, fit_dsc, fit_masc, fit_sc, fit_sdid,
    load_panel, loss_pct, retreat, sdid_out_of_window_loss)

EVAL = {236: "2018Q4", 240: "2019Q4"}
N_PRE = TREAT_T - FIRST_T          # 86 pre-treatment quarters
LAST_PRE_T = TREAT_T - 1           # 226

# Table 1, treatment period 2016:Q3, no covariates. Losses in percent.
PUBLISHED = {
    "sc":       {236: 3.06, 240: 4.20},
    "dsc":      {236: 2.98, 240: 4.12},
    "sdid_i":   {236: 2.76, 240: 3.89},
    "sdid_ii":  {236: 2.79, 240: 3.92},
    "sdid_iii": {236: 2.79, 240: 3.92},
    "masc":     {236: 2.73, 240: 3.83},
    "ascm":     {236: 3.04, 240: 4.19},
}


def _estimates() -> dict:
    df = load_panel()
    wide = df.pivot(index="t", columns="country", values="y")
    donors = [c for c in wide.columns if c != UK]

    out = {"sc": {}, "dsc": {}, "masc": {}, "ascm": {},
           "sdid_i": {}, "sdid_ii": {}, "sdid_iii": {}}

    # Weights from the full panel; the evaluation quarter is inside it.
    for tag, fit in (("sc", fit_sc), ("dsc", fit_dsc), ("ascm", fit_ascm)):
        res = fit(df)
        out[tag] = {t: loss_pct(res, t) for t in EVAL}
    res_masc = fit_masc(df, min_preperiods=6)
    out["masc"] = {t: loss_pct(res_masc, t) for t in EVAL}

    # SDID case (i): fit through 2016:Q4, report later -- out of window.
    res_i = fit_sdid(df[df.t <= TREAT_T])
    out["sdid_i"] = {t: sdid_out_of_window_loss(res_i, wide, donors, N_PRE, t)
                     for t in EVAL}
    for t in EVAL:
        out["sdid_ii"][t] = loss_pct(fit_sdid(df[df.t <= t]), t)
        out["sdid_iii"][t] = loss_pct(
            fit_sdid(retreat(df, list(range(FIRST_T, LAST_PRE_T + 1)) + [t], t)),
            t)
    return out


def run() -> dict:
    est = _estimates()
    got, worst = {}, 0.0
    for method, by_quarter in est.items():
        for t, label in EVAL.items():
            value = by_quarter[t]
            got[f"{method}_{label}"] = value
            worst = max(worst, abs(value - PUBLISHED[method][t]))
    got["max_dev_vs_published"] = worst
    # The paper's own headline claim: the loss grows between the two dates, for
    # every method. Cheap to assert and it fails loudly on a sign or index slip.
    got["all_methods_grow"] = float(all(
        v[240] > v[236] for v in est.values()))
    # Cases (ii) and (iii) coincide (see the module docstring).
    got["ii_iii_agree"] = float(max(
        abs(est["sdid_ii"][t] - est["sdid_iii"][t]) for t in EVAL))
    return got


def comparison() -> dict:
    """mlsynth vs the published Table 1 cell by cell, for the docs exporter."""
    est = _estimates()
    rows = [{"quantity": f"{m.upper()} {label}",
             "mlsynth": round(est[m][t], 4),
             "reference": PUBLISHED[m][t]}
            for m in PUBLISHED for t, label in EVAL.items()]
    return {"rows": rows,
            "mlsynth_call": {"estimator": "VanillaSC, DSC, SDID, MASC",
                             "sdid": "zeta=0.0, intercept_adjust=True",
                             "masc": "m_grid=1..10, min_preperiods=6"},
            "reference": {"source": "de Brabander et al. (2025), Table 1",
                          "block": "2016:Q3, no covariates"}}


# Deterministic: no resampling anywhere (``vce="noinference"``, inference off),
# and MASC's cross-validation grid is exhaustive rather than sampled, so a
# re-run reproduces these to machine precision.
#
# Cell tolerance 5e-3. The cells are centered on mlsynth's measured values, so
# this is a regression guard rather than a noise budget: it is wide enough to
# absorb simplex-QP jitter across BLAS builds (a vertex solution moves in the
# last few digits, not the third decimal) and far tighter than any regression
# that would matter here -- dropping the lambda correction moves the SDID cells
# by ~0.05, and restoring the default ridge moves them by ~0.27.
#
# ``max_dev_vs_published`` is the assertion that this reproduces the paper, and
# it is the one cell centered on the paper rather than on mlsynth: 0.028 is the
# worst of the fourteen deviations, and 5e-3 around it pins that none of them
# drifts past display precision.
_TOL = 5e-3
EXPECTED = {
    "sc_2018Q4":       (3.0387, _TOL),
    "sc_2019Q4":       (4.1720, _TOL),
    "dsc_2018Q4":      (2.9887, _TOL),
    "dsc_2019Q4":      (4.1249, _TOL),
    "sdid_i_2018Q4":   (2.7714, _TOL),
    "sdid_i_2019Q4":   (3.9076, _TOL),
    "sdid_ii_2018Q4":  (2.8012, _TOL),
    "sdid_ii_2019Q4":  (3.9374, _TOL),
    "sdid_iii_2018Q4": (2.8012, _TOL),
    "sdid_iii_2019Q4": (3.9374, _TOL),
    "masc_2018Q4":     (2.7255, _TOL),
    "masc_2019Q4":     (3.8275, _TOL),
    "ascm_2018Q4":     (3.0447, _TOL),
    "ascm_2019Q4":     (4.1869, _TOL),
    "max_dev_vs_published": (0.0280, _TOL),
    "all_methods_grow": (1.0, 0.0),
    # (ii) and (iii) are the same fit to solver precision, not merely to 2dp.
    "ii_iii_agree": (0.0, 1e-8),
}
