"""Prop 99: each Graph-SC arm overlaid on classic SC, and the paper's own plotted curve.

Run::

    python -m benchmarks.studies.graphsc_replicate.figure

writes ``results/prop99_overlay.png``. Every curve except the last panel's is an
exact solve from :mod:`.oracle` on the unstandardized panel. The last panel's
curve is the Graph-SC line of the paper's Figure 2(a), read off the image in the
arXiv source (``results/paper_figure2a_digitized.json`` records how); the
digitization recovers SC's known pre-RMSPE as 1.69 against 1.656.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import oracle as G
from .run import ARMS, TABLE1, prop99

HERE = Path(__file__).resolve().parent
DIGITIZED = HERE / "results" / "paper_figure2a_digitized.json"
OUT = HERE / "results" / "prop99_overlay.png"

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
SC_C, OURS_C, PAPER_C = "#2a78d6", "#eb6834", "#1baf7a"


def curves():
    """``(years, T0, y, {name: counterfactual})``: SC, the three arms in both readings, the paper's plotted line."""
    y, X, T0 = prop99()
    x1, X0 = y[:T0], X[:T0]
    L, _ = G.knn_laplacian(X0.T, 5)
    cf = {"SC": X @ G.simplex_sc(x1, X0)}
    for arm, lam in ARMS.items():
        cf[f"{arm}, signed"] = X @ G.solve_signed(x1, X0, L, *lam)
        cf[f"{arm}, w >= 0"] = X @ G.solve_exact(x1, X0, L, *lam)
    d = json.loads(DIGITIZED.read_text())
    cf["Paper's Figure 2(a)"] = np.array([np.nan if v is None else v for v in d["Graph-SC"]])
    return np.arange(1970, 1970 + len(y)), T0, y, cf


def scores(y, cf, T0):
    """Pre-RMSPE and mean post-period gap ``y - cf``, over the years where ``cf`` is observed."""
    ok = ~np.isnan(cf)
    pre, post = ok.copy(), ok.copy()
    pre[T0:], post[:T0] = False, False
    return float(np.sqrt(np.mean((y[pre] - cf[pre]) ** 2))), float(np.mean(y[post] - cf[post]))


def _frame(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9, length=0, pad=4)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


PANELS = (
    ("Graph-SC, signed weights", "Graph-SC, signed", OURS_C, "Table 1 prints 0.089 and −18.62"),
    ("Graph-SC, w ≥ 0", "Graph-SC, w >= 0", OURS_C, "Algorithm 2's reading"),
    ("Sparse-only, signed weights", "Sparse-only, signed", OURS_C, "matches Table 1 to the printed digit"),
    ("Graph-only, signed weights", "Graph-only, signed", OURS_C, "matches Table 1 to the printed digit"),
    ("The paper's plotted Graph-SC", "Paper's Figure 2(a)", PAPER_C, "digitized from the arXiv source"),
)


def overlay_figure():
    """The figure: five trajectory panels over SC and a summary of the mean post-period gaps."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    years, T0, y, cf = curves()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.8), facecolor=SURFACE)
    sc_pre, sc_gap = scores(y, cf["SC"], T0)
    for ax, (title, key, colour, note) in zip(axes.flat, PANELS):
        _frame(ax)
        ax.axvline(years[T0] - 0.5, color=AXIS, linewidth=0.8)
        ax.text(years[T0] - 0.2, 133, "Prop 99", color=MUTED, fontsize=8, va="top")
        ax.plot(years, y, color=INK, linewidth=1.8, solid_capstyle="round")
        ax.plot(years, cf["SC"], color=SC_C, linewidth=1.6, solid_capstyle="round")
        ax.plot(years, cf[key], color=colour, linewidth=1.6, solid_capstyle="round")
        pre, gap = scores(y, cf[key], T0)
        ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8)
        ax.text(1970.5, 54, f"this curve: pre-RMSPE {pre:.3f}, gap {gap:.2f}".replace("-", "−"),
                fontsize=8.5, color=INK)
        ax.text(1970.5, 49, f"SC: pre-RMSPE {sc_pre:.3f}, gap {sc_gap:.2f}".replace("-", "−"),
                fontsize=8.5, color=INK2)
        ax.text(1970.5, 44, note, fontsize=8.5, color=MUTED)
        ax.set_ylim(38, 135)
        ax.set_xlim(1969, 2001)
    for ax in axes[:, 0]:
        ax.set_ylabel("Cigarette packs per capita", color=INK2, fontsize=9)

    # Mean post-period gap, |California - synthetic|, as Table 1 reports it.
    ax = axes[1, 2]
    _frame(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    rows = ("SC", "Sparse-only", "Graph-only", "Graph-SC")
    ours = {"SC": cf["SC"], **{a: cf[f"{a}, signed"] for a in rows[1:]}}
    for i, arm in enumerate(rows):
        g = abs(scores(y, ours[arm], T0)[1])
        ax.plot([TABLE1[arm][2], g], [i, i], color=GRID, linewidth=1.2, zorder=1)
        ax.scatter(TABLE1[arm][2], i, s=150, facecolor="none", edgecolor=PAPER_C, linewidth=1.6, zorder=3)
        ax.scatter(g, i, s=45, color=OURS_C, edgecolor=SURFACE, linewidth=1.2, zorder=4)
        if arm != "Graph-SC":
            ax.text(g + 0.35, i, f"{g:.2f}, both", fontsize=8, color=INK, va="center")
    fig_gap = abs(scores(y, cf["Paper's Figure 2(a)"], T0)[1])
    ax.scatter(fig_gap, 3, s=70, marker="D", facecolor="none", edgecolor=PAPER_C, linewidth=1.6, zorder=5)
    g_ours = abs(scores(y, ours["Graph-SC"], T0)[1])
    ax.text(g_ours - 0.2, 3, f"ours {g_ours:.2f}", fontsize=8, color=INK, ha="right", va="center")
    ax.text(fig_gap, 3.42, f"plot {fig_gap:.2f}", fontsize=8, color=INK, ha="center", va="center")
    ax.text(TABLE1["Graph-SC"][2] + 0.25, 3, f"table {TABLE1['Graph-SC'][2]:.2f}", fontsize=8, color=INK,
            va="center")
    cut = 0.95 * TABLE1["SC"][2]
    ax.axvline(cut, color=AXIS, linewidth=0.8)
    ax.text(cut - 0.1, -0.5, "95% of SC's gap,\nthe paper's criterion", fontsize=8, color=MUTED, ha="right",
            va="center")
    ax.set_yticks(range(len(rows)), rows)
    ax.set_ylim(-1.0, 3.7)
    ax.invert_yaxis()
    ax.set_xlim(11.5, 20.5)
    ax.set_xlabel("Size of the gap, packs per capita", color=INK2, fontsize=9)
    ax.set_title("Gap: our solves against the paper", loc="left", fontsize=11, color=INK, pad=8)

    keys = [Line2D([], [], color=INK, linewidth=1.8), Line2D([], [], color=SC_C, linewidth=1.6),
            Line2D([], [], color=OURS_C, linewidth=1.6), Line2D([], [], color=PAPER_C, linewidth=1.6),
            Line2D([], [], color=OURS_C, marker="o", linestyle="", markersize=6),
            Line2D([], [], markeredgecolor=PAPER_C, markerfacecolor="none", marker="o", linestyle="",
                   markersize=10, markeredgewidth=1.6),
            Line2D([], [], markeredgecolor=PAPER_C, markerfacecolor="none", marker="D", linestyle="",
                   markersize=7, markeredgewidth=1.6)]
    labels = ["California", "Classic SC", "Our exact solve of the panel's arm", "The paper's plotted Graph-SC",
              "Gap, our solve", "Gap, paper's Table 1", "Gap, paper's Figure 2(a)"]
    fig.legend(keys, labels, loc="upper left", bbox_to_anchor=(0.04, 0.995), ncol=7, frameon=False,
               fontsize=9, labelcolor=INK, handlelength=1.8, columnspacing=1.6)
    fig.text(0.04, 0.935, "Prop 99: each Graph-SC arm overlaid on classic SC", fontsize=13, color=INK)
    fig.text(0.04, 0.912, "gap: mean over 1989-2000 of California minus the synthetic control, packs per capita",
             fontsize=9, color=INK2)
    fig.tight_layout(rect=(0.02, 0.0, 1.0, 0.895), h_pad=2.5, w_pad=2.0)
    return fig


def main():
    fig = overlay_figure()
    fig.savefig(OUT, dpi=150, facecolor=SURFACE)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
