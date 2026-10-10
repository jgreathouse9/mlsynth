"""What carryover, saturation and lag do to the delivery projection.

    python response_shape_plot.py results/response_shape.csv
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#b9b8b4"
COL = {"surrogate_raw": "#2a78d6", "surrogate_fitted": "#eb6834",
       "surrogate_oracle": "#1baf7a", "known": "#eda100"}
LAB = {"surrogate_raw": "projected on raw delivery",
       "surrogate_fitted": "transform chosen by fit",
       "surrogate_oracle": "true transform",
       "known": "repair told the market"}
ORDER = ["linear", "lagged", "saturating", "carryover", "realistic"]
VAR = {"linear": 1.00, "lagged": 1.00, "saturating": 0.64,
       "carryover": 0.42, "realistic": 0.29}


def main(path):
    warnings.filterwarnings("ignore")
    d = pd.read_csv(path)
    rms = lambda x: float(np.sqrt(np.mean(np.asarray(x, float) ** 2)))
    z = d[d["corr"] == 0]

    vals = {k: [rms(z[z.regime == r][k] - z[z.regime == r].true_att) for r in ORDER]
            for k in COL}
    naive = rms(z.naive - z.true_att)
    oracle = rms(z.oracle - z.true_att)

    fig, ax = plt.subplots(figsize=(11.5, 6.0), facecolor=SURF)
    ax.set_facecolor(SURF)
    ax.grid(True, axis="y", color="#e6e5e2", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#d9d8d4")
    ax.tick_params(colors=INK2, labelsize=10)

    x = np.arange(len(ORDER))
    w = 0.2
    for i, k in enumerate(COL):
        ax.bar(x + (i - 1.5) * w, vals[k], w * 0.92, color=COL[k], zorder=3,
               edgecolor=SURF, linewidth=2)

    ax.axhline(naive, color=INK2, lw=2, ls="--", zorder=4)
    ax.annotate(f"make no correction  ({naive:.2f})", (len(ORDER) - 0.45, naive),
                xytext=(0, 7), textcoords="offset points", ha="right",
                color=INK, fontsize=10.5)
    ax.axhline(oracle, color=MUTED, lw=1.5, ls=":", zorder=2)
    ax.annotate(f"oracle ({oracle:.2f})", (-0.45, oracle), xytext=(0, -16),
                textcoords="offset points", color=INK2, fontsize=10)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r}\nvariation {VAR[r]:.2f}" for r in ORDER],
                       color=INK2, fontsize=10.5)
    ax.set_ylabel("RMSE against the true effect  (log)", color=INK2, fontsize=10.5)
    ax.set_title("The delivery projection under a realistic response\n"
                 "contamination present but uncorrelated with delivery; "
                 "bars below the dashed line beat doing nothing",
                 color=INK, fontsize=13, loc="left", pad=14)

    h = [plt.Rectangle((0, 0), 1, 1, color=COL[k]) for k in COL]
    ax.legend(h, [LAB[k] for k in COL], loc="upper left", frameon=False,
              fontsize=10.5, labelcolor=INK, ncol=2)
    fig.tight_layout()
    out = "results/response_shape.png"
    fig.savefig(out, dpi=170, facecolor=SURF, bbox_inches="tight")
    print("saved", out)
    print(pd.DataFrame(vals, index=ORDER).assign(naive=naive, oracle=oracle).round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
