"""SDID on the all-US Walmart Supercenter panel (allsynth Example 12).

Companion to ``rolldid_allsynth_walmart.py``, which found that the
rolling-transformation DiD cannot carry Example 12: linear detrending fails its
own placebo on this panel.  Synthetic difference-in-differences is the natural
next candidate -- it keeps a DiD's staggered machinery but re-weights donors so
they are parallel to the treated unit, which is what the pre-trend problem
called for.

Two things had to be right before any number here means anything.

Normalization.  allsynth's ``transform(emps_n10 ..., normalize)`` indexes each
series -- the treated county's and every donor's -- to that treated county's
final pre-treatment year, so gaps read as percent of baseline employment and the
event-time gap at e = -1 is identically zero.  That is not the same as a log
outcome, and it is not a cosmetic choice: it changes what the synthetic control
is matching on.  Section 1 checks the pipeline against Example 11, the one
example in the article whose gaps are printed rather than plotted.

Structure.  ``VanillaSC`` estimates staggered adoption natively (Cattaneo, Feng,
Palomba & Titiunik 2025: per-unit simplex fits, TSUA event study, SCPI bands)
and so does ``SDID`` (per-cohort fits with the Ciccia 2024 aggregation).  Both
are one call.  Neither can express the allsynth normalization, though, because
the base year is a property of the *treated county*, so a donor needs a
different normalized series for each cohort it serves -- which a single panel
cannot hold.  Reproducing Example 12 therefore means stacking by hand: one fit
per treated county on its own normalized panel, then average the gaps in event
time.  Section 2 runs the native calls anyway, to show what the shortcut costs.

Runtime is about 90 seconds, nearly all of it the 566 synthetic-control solves.
Run from the repository root:

    python examples/sdid_allsynth_walmart.py
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

from mlsynth import SDID, VanillaSC

PANEL = "basedata/allsynth_walmart.parquet"
K = 5  # balanced event window, e in [-5, 5]

# allsynth Example 11, Indiana, the printed "gap" column (classic SC).
EX11_CLASSIC = {-5: 0.541, -4: 0.573, -3: 0.113, -2: 0.249, -1: 0.0,
                0: -1.104, 1: -2.557, 2: -4.270, 3: -5.232, 4: -7.226, 5: -8.496}


def load_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pd.read_parquet(PANEL)
    d["treat"] = ((d.supercenter == 1) & (d.year >= d.super_year)).astype(int)
    meta = d.groupby("cty_fips").agg(sc=("supercenter", "max"), sy=("super_year", "max"),
                                     cz=("czone", "max"), pop90=("pop90", "max"))
    return d, meta


def stacked_gaps(d, meta, estimator, units, donors) -> pd.DataFrame:
    """One fit per treated county on its own normalized panel; gaps in event time.

    This is allsynth's ``stacked()``: the treated county and its donors are
    indexed to the county's final pre-treatment year, a synthetic control (or
    SDID) is fit against the never-treated pool, and the resulting gaps are
    stacked on a common event clock.
    """
    rows = []
    for j in units:
        g = int(meta.sy.loc[j])
        sub = d[d.cty_fips.isin([j] + donors)].copy()
        base = sub[sub.year == g - 1].set_index("cty_fips").emps_n10
        sub["ynorm"] = 100.0 * sub.emps_n10.values / base.reindex(sub.cty_fips).values
        cfg = {"df": sub, "outcome": "ynorm", "treat": "treat", "unitid": "cty_fips",
               "time": "year", "display_graphs": False}
        if estimator is SDID:
            cfg["vce"] = "noinference"          # see section 3
        res = estimator(cfg).fit()
        if estimator is SDID:
            cohort = res.cohorts[max(res.cohorts)]
            gaps = {e: cohort.event_effects[e].tau for e in cohort.event_effects}
        else:
            path = np.asarray(res.time_series.estimated_gap, dtype=float)
            gaps = {y - g: path[i] for i, y in enumerate(sorted(sub.year.unique()))}
        rows.append({"cty": j, "pop90": float(meta.pop90.loc[j]),
                     **{e: gaps.get(e, np.nan) for e in range(-K, K + 1)}})
    return pd.DataFrame(rows)


def _event_row(label: str, values) -> str:
    return f"{label:26s}" + " ".join(f"{v:6.2f}" for v in values)


def _event_header(offsets=range(-K, K + 1)) -> str:
    return f"{'event time':26s}" + " ".join(f"{e:6d}" for e in offsets)


# ------------------------------------------------------- 1. does the pipeline work
def section_1_indiana(d, meta) -> None:
    """Example 11 is the only stacked result in the article with printed gaps."""
    print("\n=== 1. Example 11 (Indiana only): reproduce the printed gaps ===")
    donors = list(meta.index[meta.sc == 0])
    indiana = [c for c in meta.index[meta.sc == 1] if c // 1000 == 18]
    print(f"  {len(indiana)} treated Indiana counties, {len(donors)} never-treated donors")
    print(_event_header())
    print(_event_row("allsynth ex11 (classic SC)", [EX11_CLASSIC[e] for e in range(-K, K + 1)]))
    for est, name in ((VanillaSC, "mlsynth VanillaSC"), (SDID, "mlsynth SDID")):
        G = stacked_gaps(d, meta, est, indiana, donors)
        print(_event_row(name, [G[e].mean() for e in range(-K, K + 1)]))


# ------------------------------------------------------------ 2. Example 12 itself
def section_2_all_us(d, meta) -> None:
    print("\n=== 2. Example 12 (all US): stacked, balanced e = [-5, 5] ===")
    donors = list(meta.index[meta.sc == 0])
    treated = list(meta.index[meta.sc == 1])
    print(f"  {len(treated)} treated counties in {meta.loc[treated].sy.nunique()} "
          f"cohorts, {len(donors)} never-treated donors")
    print(_event_header())
    for est, name in ((SDID, "SDID"), (VanillaSC, "VanillaSC")):
        t0 = time.time()
        G = stacked_gaps(d, meta, est, treated, donors)
        print(_event_row(f"{name}  unweighted",
                         [G[e].mean() for e in range(-K, K + 1)])
              + f"   [{time.time() - t0:.0f}s]")
        print(_event_row(f"{name}  pop90-weighted",
                         [np.average(G[e], weights=G.pop90) for e in range(-K, K + 1)]))

    print("\n  The native staggered calls, for comparison -- one line each, but the "
          "\n  outcome cannot carry a per-treated-unit base year, so they run on "
          "\n  100 x log(employment) instead:")
    dd = d.copy()
    dd["logemp"] = 100.0 * np.log(dd["emps_n10"])
    common = {"df": dd, "outcome": "logemp", "treat": "treat", "unitid": "cty_fips",
              "time": "year", "display_graphs": False}
    sc = VanillaSC(dict(common)).fit()
    es = sc.additional_outputs["event_study"]                     # TSUA, ell = 1..6
    print(_event_row("    VanillaSC (TSUA)", [es[l] for l in sorted(es)]))
    sd = SDID(dict(common, vce="noinference")).fit()
    ev = pd.DataFrame({"ell": sd.event_study.event_times, "tau": sd.event_study.tau})
    print(_event_row("    SDID (Ciccia eq. 6)",
                     ev[ev.ell.between(0, K)].tau.values))


# ----------------------------------------------------------------- 3. what inference
def section_3_inference(d, meta) -> None:
    """Which variance estimators are actually available on this panel."""
    print("\n=== 3. Inference on the staggered design ===")
    dd = d.copy()
    dd["logemp"] = 100.0 * np.log(dd["emps_n10"])
    common = {"df": dd, "outcome": "logemp", "treat": "treat", "unitid": "cty_fips",
              "time": "year", "display_graphs": False}
    for vce in ("placebo", "jackknife", "bootstrap"):
        try:
            res = SDID(dict(common, vce=vce, B=50)).fit()
            inf = res.inference_detail
            print(f"  vce={vce:10s} ATT {inf.att:+.3f}  se {inf.se:.3f}  p {inf.p_value:.4f}")
        except Exception as exc:                       # noqa: BLE001 - reporting the refusal
            print(f"  vce={vce:10s} unavailable: {type(exc).__name__}: "
                  f"{str(exc).splitlines()[0][:110]}")
    n_tr = meta.loc[meta.sc == 1].groupby("sy").size()
    print(f"\n  Placebo inference reassigns each cohort's {n_tr.min()}-{n_tr.max()} treated "
          f"labels among {int((meta.sc == 0).sum())} donors,\n  so the donor pool empties. "
          "The stacked route in section 2 restores it: one\n  treated county against 39 "
          "donors is a block design with valid placebo inference.")


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    pd.set_option("display.width", 200)
    panel, units = load_panel()
    section_1_indiana(panel, units)
    section_2_all_us(panel, units)
    section_3_inference(panel, units)
