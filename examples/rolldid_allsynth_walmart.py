"""ROLLDID on the all-US Walmart Supercenter panel (allsynth Example 12).

Example 12 of Wiltshire's allsynth Stata article estimates the stacked,
bias-corrected synthetic-control average effect of Walmart Supercenter entry on
aggregate county employment for every treated county in the U.S., weighting
county gaps by 1990 population, restricting each treated county's donor pool to
counties in other commuting zones, and reporting the balanced event window
e = [-5, 5].  This script asks what the rolling-transformation DiD of Lee &
Wooldridge (:class:`mlsynth.ROLLDID`) says about the same panel, and how far its
answer can be pushed toward Example 12's design.

The panel (``basedata/allsynth_walmart.parquet``) is 605 counties x 1990-2005:
566 counties that got their first Supercenter in 1995-2000 (six adoption
cohorts) and 39 never-treated donors where entry was blocked.  ROLLDID's
staggered mode fits this shape directly -- absorbing treatment, a never-treated
comparison group -- so the headline call is a plain ``fit()``.

Three of Example 12's features sit outside what ROLLDID does, and each is
quantified here rather than assumed away:

  1. ``avgweights(pop90)``   -- ROLLDID's cross-sectional regression is
     unweighted; the pop90-weighted contrast is computed directly from the
     transformed values.
  2. ``donorif(czone != cz)`` -- ROLLDID compares every treated unit to the same
     never-treated group; the per-treated-unit commuting-zone exclusion is
     recomputed by hand.
  3. ``balanced`` (e in [-5, 5]) -- ROLLDID's staggered post window runs from
     each cohort's adoption year to the end of the panel, so early cohorts get
     11 post periods and late cohorts 6.  The balanced event-time path is built
     by restricting each cohort's pre-window to [g-5, g-1] and evaluating one
     cross-sectional regression per event year.

Section 4 is the part that decides whether any of this is credible: a placebo
that moves the transformation window back in time and re-estimates over periods
in which nobody is treated yet.

Run from the repository root:

    python examples/rolldid_allsynth_walmart.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from mlsynth import ROLLDID

PANEL = "basedata/allsynth_walmart.parquet"
K = 5  # event-window half-width, matching Example 12's balanced e in [-5, 5]


# --------------------------------------------------------------------------- data
def load_panel() -> pd.DataFrame:
    """County panel with an absorbing Supercenter-entry indicator."""
    d = pd.read_parquet(PANEL)
    d["treat"] = ((d.supercenter == 1) & (d.year >= d.super_year)).astype(int)
    # Example 12 reports effects in percent of baseline employment
    # (transform(..., normalize)); 100 x log is the scale-free analogue.
    d["logemp"] = 100.0 * np.log(d["emps_n10"])
    return d


# ------------------------------------------------- pieces ROLLDID does not expose
def _transform_at(y: pd.Series, pre_years, t: int, mode: str) -> float:
    """Rolling-transformed outcome at period ``t`` given an explicit pre-window.

    Same transformation as ROLLDID's pipeline (demean = Procedure 2.1, detrend =
    Procedure 3.1); the pre-window is passed in so it can be truncated (balanced
    event time) or shifted back (the placebo).
    """
    pre = y.loc[list(pre_years)]
    if mode == "demean":
        return float(y.loc[t] - pre.mean())
    slope, intercept = np.polyfit(np.asarray(pre.index, dtype=float), pre.values, 1)
    return float(y.loc[t] - (intercept + slope * float(t)))


def _contrast(yv: np.ndarray, dv: np.ndarray, weights=None, alpha: float = 0.05) -> dict:
    """Treated-vs-control contrast from ``yv on 1, d`` with HC3 standard errors."""
    X = np.column_stack([np.ones_like(dv, dtype=float), dv.astype(float)])
    w = np.ones_like(dv, dtype=float) if weights is None else np.asarray(weights, float)
    bread = np.linalg.inv(X.T @ (X * w[:, None]))
    beta = bread @ (X.T @ (w * yv))
    resid = yv - X @ beta
    lev = np.einsum("ij,jk,ik->i", X, bread, X * w[:, None])
    scaled = X * (w * resid / (1.0 - lev))[:, None]
    var = bread @ (scaled.T @ scaled) @ bread
    se = float(np.sqrt(var[1, 1]))
    dfree = len(yv) - 2
    crit = float(stats.t.ppf(1.0 - alpha / 2.0, dfree))
    att = float(beta[1])
    return {"att": att, "se": se,
            "p": float(2.0 * stats.t.cdf(-abs(att / se), dfree)),
            "ci_lo": att - crit * se, "ci_hi": att + crit * se}


class Panel:
    """Wide outcome matrix plus the cohort bookkeeping the transformations need."""

    def __init__(self, d: pd.DataFrame, outcome: str = "logemp") -> None:
        self.Y = d.pivot(index="year", columns="cty_fips", values=outcome).sort_index()
        unit = d.groupby("cty_fips").agg(sc=("supercenter", "max"),
                                         sy=("super_year", "max"),
                                         cz=("czone", "max"),
                                         pop90=("pop90", "max"))
        self.unit = unit
        self.treated = list(unit.index[unit.sc == 1])
        self.donors = list(unit.index[unit.sc == 0])
        self.cohort_of = {j: int(unit.loc[j, "sy"]) for j in self.treated}
        self.cohorts = sorted(set(self.cohort_of.values()))
        self.n_g = {g: sum(1 for j in self.cohort_of if self.cohort_of[j] == g)
                    for g in self.cohorts}
        self.omega = {g: self.n_g[g] / len(self.treated) for g in self.cohorts}

    def collapsed(self, mode: str, pre_offsets, post_offsets):
        """Per-unit regressand: eq. 7.18 with explicit pre/post offsets.

        A treated unit is transformed against its own cohort; a never-treated
        unit contributes the cohort-share-weighted average over all cohorts.
        """
        def cell(j, g, e):
            return _transform_at(self.Y[j], [g + o for o in pre_offsets], g + e, mode)

        vals = []
        for j in self.treated:
            g = self.cohort_of[j]
            vals.append(float(np.mean([cell(j, g, e) for e in post_offsets])))
        for j in self.donors:
            vals.append(float(sum(self.omega[g] * np.mean([cell(j, g, e) for e in post_offsets])
                                  for g in self.cohorts)))
        return np.array(vals)

    @property
    def d_vector(self) -> np.ndarray:
        return np.array([1.0] * len(self.treated) + [0.0] * len(self.donors))

    @property
    def pop_weights(self) -> np.ndarray:
        """pop90 on treated units; controls held at the treated-side mean weight."""
        pop = np.array([float(self.unit.loc[j, "pop90"])
                        for j in self.treated + self.donors])
        d = self.d_vector
        return np.where(d == 1, pop, pop[d == 1].mean())


# ------------------------------------------------------------------- 1. plain fit
def section_1_out_of_the_box(d: pd.DataFrame) -> None:
    print("\n=== 1. ROLLDID as shipped: staggered, full post window (g .. 2005) ===")
    for rolling in ("demean", "detrend"):
        res = ROLLDID({
            "df": d, "outcome": "logemp", "treat": "treat", "unitid": "cty_fips",
            "time": "year", "rolling": rolling, "inference": "hc3",
            "display_graphs": False,
        }).fit()
        print(f"\n  {rolling}: N1={res.n_treated} N0={res.n_control}   "
              f"ATT {res.effects.att:+7.3f}%  se {res.inference.standard_error:.3f}  "
              f"p {res.inference.p_value:.4f}  "
              f"95% CI [{res.inference.ci_lower:+.3f}, {res.inference.ci_upper:+.3f}]")
        print(res.per_cohort.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))


# ------------------------------------------------- 2. the Example 12 design levers
def section_2_design_levers(p: Panel) -> None:
    print("\n=== 2. Example 12's weighting and donor restriction ===")
    print("  (the unweighted / all-donor rows reproduce section 1's ATT, which "
          "checks\n   this hand-rolled transformation against the shipped estimator)")
    # Same windows as section 1 (all pre-periods, post through the panel end), so
    # the only thing moving is the weighting / donor pool.
    first, last = int(p.Y.index.min()), int(p.Y.index.max())
    pre_years = {g: range(first, g) for g in p.cohorts}
    post_years = {g: range(g, last + 1) for g in p.cohorts}
    for mode in ("demean", "detrend"):
        treated_vals, control_cells = {}, {}
        for j in p.treated:
            g = p.cohort_of[j]
            treated_vals[j] = float(np.mean(
                [_transform_at(p.Y[j], pre_years[g], t, mode) for t in post_years[g]]))
        for j in p.donors:
            control_cells[j] = {
                g: float(np.mean([_transform_at(p.Y[j], pre_years[g], t, mode)
                                  for t in post_years[g]]))
                for g in p.cohorts}
        control_agg = {j: sum(p.omega[g] * control_cells[j][g] for g in p.cohorts)
                       for j in p.donors}

        t_arr = np.array([treated_vals[j] for j in p.treated])
        w = np.array([float(p.unit.loc[j, "pop90"]) for j in p.treated])
        base = float(np.mean(list(control_agg.values())))

        # per-treated-unit commuting-zone exclusion (allsynth's donorif)
        cz_gaps = []
        for j in p.treated:
            cz = p.unit.loc[j, "cz"]
            keep = [k for k in p.donors if p.unit.loc[k, "cz"] != cz]
            cz_gaps.append(treated_vals[j]
                           - float(np.mean([control_agg[k] for k in keep])))
        cz_gaps = np.array(cz_gaps)

        print(f"\n  {mode}:")
        print(f"    all donors, unweighted        {t_arr.mean() - base:+7.3f}%")
        print(f"    all donors, pop90-weighted    {np.average(t_arr, weights=w) - base:+7.3f}%")
        print(f"    other-CZ donors, unweighted   {cz_gaps.mean():+7.3f}%")
        print(f"    other-CZ donors, pop90-wtd    {np.average(cz_gaps, weights=w):+7.3f}%")
    shared = sum(1 for j in p.treated
                 if (p.unit.loc[p.donors, "cz"] == p.unit.loc[j, "cz"]).any())
    print(f"\n  (the CZ rule binds for {shared}/{len(p.treated)} treated counties, "
          f"each losing 1 of {len(p.donors)} donors)")


# ------------------------------------------------------ 3. balanced event-time path
def section_3_event_time(p: Panel) -> None:
    print(f"\n=== 3. Balanced event time, e = 0..{K} (pre-window [g-{K}, g-1]) ===")
    pre = list(range(-K, 0))
    for mode in ("demean", "detrend"):
        rows = []
        for e in range(0, K + 1):
            yv = p.collapsed(mode, pre, [e])
            unw = _contrast(yv, p.d_vector)
            wtd = _contrast(yv, p.d_vector, weights=p.pop_weights)
            rows.append({"e": e, **unw, "att_pop90": wtd["att"], "se_pop90": wtd["se"]})
        tab = pd.DataFrame(rows)
        print(f"\n  {mode}:")
        print(tab.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
        print(f"    mean over e=0..{K}: unweighted {tab.att.mean():+7.3f}%   "
              f"pop90-weighted {tab.att_pop90.mean():+7.3f}%")


# ---------------------------------------------------------------- 4. placebo check
def section_4_placebo(p: Panel) -> None:
    """Re-run the design on periods in which nobody is treated yet.

    Estimate the transformation on [g-10, g-6] and read off "effects" at
    e = -5..-1.  Only the 2000 cohort has enough lead-in for the full
    five-and-five geometry of the headline spec, so the placebo is run on that
    cohort against the same 39 donors, and its true post-period path is printed
    beside it on identical geometry.
    """
    print("\n=== 4. Placebo: same geometry, shifted five years before adoption ===")
    g = max(p.cohorts)
    tr = [j for j in p.treated if p.cohort_of[j] == g]
    units = tr + p.donors
    dv = np.array([1.0] * len(tr) + [0.0] * len(p.donors))
    windows = {
        f"placebo (fit {g-10}-{g-6}, read {g-5}-{g-1})": (range(g - 10, g - 5), range(g - 5, g)),
        f"actual  (fit {g-5}-{g-1},  read {g}-{g+5})": (range(g - 5, g), range(g, g + K + 1)),
    }
    for mode in ("demean", "detrend"):
        print(f"\n  {mode}  (cohort {g}: {len(tr)} treated, {len(p.donors)} donors)")
        for label, (fit_years, read_years) in windows.items():
            rows = []
            for t in read_years:
                yv = np.array([_transform_at(p.Y[j], fit_years, t, mode) for j in units])
                r = _contrast(yv, dv)
                rows.append({"year": t, "e": t - g, **r})
            tab = pd.DataFrame(rows)
            print(f"    {label}")
            print(tab.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    panel = load_panel()
    prep = Panel(panel)
    section_1_out_of_the_box(panel)
    section_2_design_levers(prep)
    section_3_event_time(prep)
    section_4_placebo(prep)
