"""Observed against fitted counterfactual for Hsiao & Zhou's Table 9.

    MLSYNTH_HZ_DATA=<dir> python -m \\
        benchmarks.studies.hsiao_zhou_counterfactuals.plot_empirics

One panel per method. The solid dark line is the treated unit's observed
series, the blue line is this replication's counterfactual and the dashed
orange line is the path printed in Table 9. The paper publishes counterfactuals
for the treatment period only, so the pre-period shows the fit and the
post-period shows the fit and the published path side by side.

With nothing configured this draws the panel from `standin.py` and omits the
Table 9 series, since there is nothing to compare a drawn panel against. Point
MLSYNTH_HZ_DATA at the paper's replication directory for the real figure.

``plot_table9`` returns its Figure and does not show or save it; the caller
decides, per the separation the repository keeps between computing and
presenting.
"""
from __future__ import annotations

import os
import pathlib
import warnings

import numpy as np

from .empirics import build, mab, run_cell
from .standin import resolve_panels

warnings.filterwarnings("ignore")

#: Table 9, the counterfactual columns over 1989-2000.
PAPER = {
    "SCM":  [88.8, 86.7, 81.8, 81.3, 81.1, 80.7, 78.0, 77.1, 77.3, 73.7, 73.1, 66.8],
    "PCA":  [81.3, 72.9, 68.1, 64.9, 59.2, 51.7, 47.6, 42.9, 39.8, 40.6, 35.5, 29.6],
    "CCE":  [86.4, 81.5, 74.3, 72.3, 70.9, 65.3, 65.9, 66.7, 66.2, 64.2, 61.9, 57.5],
    "CPDA": [86.8, 81.9, 75.2, 72.4, 70.7, 67.9, 67.9, 65.8, 65.1, 64.6, 62.7, 57.5],
    "PDA":  [89.7, 84.9, 78.4, 76.6, 75.3, 73.2, 73.0, 70.6, 70.3, 71.7, 69.2, 62.9],
    "MA":   [86.6, 81.0, 75.5, 73.2, 71.3, 67.0, 66.1, 64.0, 63.0, 62.4, 59.6, 54.1],
}
PAPER_MAB = dict(SCM=18.5, PCA=7.46, CCE=9.12, CPDA=9.56, PDA=14.3, MA=8.33)

# Validated on the light chart surface #fcfcfb: adjacent CVD dE 24.7, normal
# vision 33.6, both slots >= 3:1 contrast. Slots 1 and 2 of the categorical order.
INK = "#0b0b0b"
INK_MUTED = "#52514e"
MINE = "#2a78d6"
PAPER_C = "#eb6834"
GRID = "#dedddb"
# Legend and annotation text wears ink, never the series colour: at these
# sizes the hues sit under the 4.5:1 text floor, and the legend's own marks
# carry identity.
SURFACE = "#fcfcfb"


def _panel(ax, years, observed, mine, paper, T0, title, note):
    ax.set_facecolor(SURFACE)
    ax.axvspan(years[T0] - 0.5, years[-1] + 0.5, color="#f1f0ee", zorder=0)
    ax.axvline(years[T0] - 0.5, color=INK_MUTED, lw=1.0, ls=(0, (4, 3)),
               zorder=1)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.plot(years, observed, color=INK, lw=2.2, zorder=4, label="observed")
    ax.plot(years, mine, color=MINE, lw=2.0, zorder=3,
            label="fitted, this replication")
    if paper is not None:
        ax.plot(years[T0:], paper, color=PAPER_C, lw=2.0, ls=(0, (5, 2)),
                zorder=3, label="fitted, Table 9")
    ax.set_title(title, fontsize=11.5, color=INK, loc="left", pad=21)
    ax.text(0.0, 1.028, note, transform=ax.transAxes, fontsize=8.5,
            color=INK_MUTED, va="bottom", ha="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=3)


def plot_table9(years, observed, fitted, T0, real=True):
    """Six method panels. Returns the Figure; does not show or save it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["SCM", "PDA", "CCE", "CPDA", "MA", "PCA"]
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.4), sharex=True,
                             sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, m in zip(axes.ravel(), order):
        v = mab(observed, fitted[m], T0)
        note = (f"mean |effect|  {v:.2f}   ·   Table 9  {PAPER_MAB[m]:.2f}"
                if real else f"mean |effect|  {v:.2f}")
        _panel(ax, years, observed, fitted[m], PAPER[m] if real else None,
               T0, m, note)

    lo = min(float(np.min(observed)),
             min(float(np.min(fitted[m])) for m in order))
    if real:
        lo = min(lo, min(min(PAPER[m]) for m in order))
    hi = max(float(np.max(observed)),
             max(float(np.max(fitted[m])) for m in order))
    pad = 0.07 * (hi - lo)
    axes[0, 0].set_ylim(lo - pad, hi + pad)
    axes[0, 0].set_xlim(years[0] - 0.5, years[-1] + 0.5)
    unit = ("cigarette sales per capita (packs)" if real
            else "outcome (drawn, no unit)")
    axes[0, 0].set_ylabel(unit, fontsize=9.5,
                          color=INK_MUTED)
    axes[1, 0].set_ylabel(unit, fontsize=9.5,
                          color=INK_MUTED)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, -0.005),
               labelcolor=INK_MUTED)
    head = ("California cigarette consumption" if real
            else "Stand-in consumption panel")
    fig.suptitle(f"{head}: observed against the fitted counterfactual",
                 fontsize=13.5, color=INK, x=0.011, ha="left", y=0.985)
    sub = ("Shaded from 1989, the first treated year. Proposition 99 passed "
           "in November 1988. 38 control states, 19 pre-treatment years."
           if real else
           "Drawn by standin.py, not the paper's panel, so no published path "
           "is shown. 38 controls, 19 pre-treatment periods.")
    fig.text(0.011, 0.935, sub, fontsize=9.5, color=INK_MUTED, ha="left")
    fig.tight_layout(rect=(0, 0.045, 1, 0.925))
    return fig


def main() -> int:
    panels = resolve_panels(os.environ.get("MLSYNTH_HZ_DATA"))
    real = panels["real"]
    df = panels["consumption"]
    cut = 1989 if real else int(df.year.min()) + 19
    if not real:
        print("STAND-IN panel; set MLSYNTH_HZ_DATA for the real figure")
    Y, X, years, T0 = build(df, "state", "year", "cigsale",
                            ["lnincome", "EduAttain", "Poverty"], 1, cut)
    fitted, _, _ = run_cell(Y, X, T0, 2, "cce")

    from mlsynth import VanillaSC
    long = df.rename(columns={"state": "unit", "year": "time",
                              "cigsale": "y"}).copy()
    long["D"] = ((long.unit == 1) & (long.time >= cut)).astype(int)
    sc = VanillaSC(dict(df=long, outcome="y", treat="D", unitid="unit",
                        time="time", display_graphs=False)).fit()
    fitted["SCM"] = np.asarray(sc.time_series.counterfactual_outcome,
                               dtype=float).ravel()

    fig = plot_table9(years, Y[:, 0], fitted, T0, real=real)
    name = ("table9_observed_vs_fitted.png" if real
            else "table9_observed_vs_fitted_standin.png")
    out = pathlib.Path(__file__).parent / "results" / name
    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
