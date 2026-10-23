"""The surrogate projection: the mechanism, and the two ways it fails.

    python surrogate_plot.py results/surrogate.csv
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mlsynth.utils.solvers.simplex import simplex_lstsq
from dgps import DGPS
from run import design
import surrogate as S

SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#b9b8b4"
COL = {"naive": "#2a78d6", "surrogate": "#eb6834", "known": "#1baf7a"}


def style(a):
    a.set_facecolor(SURF)
    a.grid(True, color="#e6e5e2", lw=0.8, zorder=0)
    a.set_axisbelow(True)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        a.spines[sp].set_color("#d9d8d4")
    a.tick_params(colors=INK2, labelsize=10)


def main(path):
    warnings.filterwarnings("ignore")
    d = pd.read_csv(path)
    rms = lambda x: float(np.sqrt(np.mean(np.asarray(x, float) ** 2)))

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 5.0), facecolor=SURF)
    for a in ax:
        style(a)

    # ---- A: the mechanism on one draw ------------------------------------
    name, seed = "rank_shift_ok", 0
    YN, _, T0 = DGPS[name](seed)
    T, J = YN.shape
    Tp = T - T0
    w, v = design(YN, T0, max(2, J // 6))
    treated = np.flatnonzero(w > 1e-8)
    controls = np.flatnonzero(v > 1e-12)
    kstar = int(np.argmax(v))
    rng = np.random.default_rng(50_000 + seed)
    scale = float(np.median(YN[:T0].std(axis=0)))
    rho = S.flighted(Tp, rng, flat=False)
    theta = scale
    post = slice(T0, T)
    Y = YN.copy(); Y[post, treated] += (theta * rho)[:, None]
    gam = S.PI_MULT * scale * (1.0 + S.VAR_FRAC * S.profile_with_corr(rho, 0.0, rng))
    Yc = Y.copy(); Yc[post, kstar] += gam
    g = (Yc @ w)[post] - Yc[post] @ v
    X = np.column_stack([np.ones(Tp), rho])
    b = np.linalg.lstsq(X, g, rcond=None)[0]

    ax[0].plot(rho, g, ls="none", marker="o", ms=9, mfc=COL["surrogate"],
               mec=SURF, mew=2, zorder=3)
    xs = np.linspace(rho.min() * 0.95, rho.max() * 1.05, 50)
    ax[0].plot(xs, b[0] + b[1] * xs, color=COL["surrogate"], lw=2, zorder=2)
    ax[0].axhline(float(g.mean()), color=COL["naive"], lw=2, ls="--", zorder=2)
    ax[0].annotate(f"naive = mean gap = {g.mean():.2f}",
                   (rho.min(), g.mean()), xytext=(2, 8), textcoords="offset points",
                   color=INK, fontsize=10)
    ax[0].annotate(f"slope {b[1]:.2f}  (true θ {theta:.2f})",
                   (xs[-1], b[0] + b[1] * xs[-1]), xytext=(-8, -18),
                   textcoords="offset points", ha="right", color=INK, fontsize=10)
    ax[0].set_title("A.  One draw: the gap against delivery\nthe level is contamination, the slope is the effect",
                    color=INK, fontsize=12, loc="left", pad=12)
    ax[0].set_xlabel("delivery intensity ρₜ", color=INK2, fontsize=10)
    ax[0].set_ylabel("post-period gap gₜ", color=INK2, fontsize=10)

    # ---- B, C: RMSE against corr(gamma, rho) ------------------------------
    for i, (flat, lab) in enumerate(((False, "B.  Flighted campaign"),
                                     (True, "C.  Near-flat campaign")), start=1):
        g_ = d[d.flat == flat]
        rows = [dict(corr=c,
                     **{k: rms(h[k] - h.true_att) for k in COL},
                     oracle=rms(h.oracle - h.true_att))
                for c, h in g_.groupby("corr")]
        t = pd.DataFrame(rows)
        for k in COL:
            ax[i].plot(t["corr"], t[k], color=COL[k], lw=2, marker="o", ms=8,
                       mec=SURF, mew=2, zorder=3)
        ax[i].plot(t["corr"], t["oracle"], color=MUTED, lw=1.5, ls="--", zorder=1)
        ax[i].set_yscale("log")
        ax[i].set_xlabel("corr(γₜ, ρₜ)", color=INK2, fontsize=10)
        ax[i].set_ylabel("RMSE against the true effect", color=INK2, fontsize=10)
        sd = g_.rho_sd.mean()
        ax[i].set_title(f"{lab}  (sd ρ̂ = {sd:.2f})", color=INK,
                        fontsize=12, loc="left", pad=12)
        for k, dy in (("naive", 9), ("surrogate", -18), ("known", 9)):
            ax[i].annotate(k, (t["corr"].iloc[-1], t[k].iloc[-1]), xytext=(-6, dy),
                           textcoords="offset points", ha="right", color=INK,
                           fontsize=10)
        ax[i].annotate("oracle", (t["corr"].iloc[0], t["oracle"].iloc[0]),
                       xytext=(4, -16), textcoords="offset points", color=INK2,
                       fontsize=10)

    lo = min(ax[1].get_ylim()[0], ax[2].get_ylim()[0])
    hi = max(ax[1].get_ylim()[1], ax[2].get_ylim()[1])
    ax[1].set_ylim(lo, hi); ax[2].set_ylim(lo, hi)

    h = [plt.Line2D([], [], color=COL[k], lw=2, marker="o", ms=8, mec=SURF, mew=1.5)
         for k in COL] + [plt.Line2D([], [], color=MUTED, lw=1.5, ls="--")]
    fig.legend(h, ["naive", "surrogate projection", "repair told the market", "oracle"],
               loc="lower center", ncol=4, frameon=False, fontsize=11,
               labelcolor=INK, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Correcting contamination through the delivery profile  "
                 f"({d.seed.nunique()} draws × {d.dgp.nunique()} panels; "
                 "panels B and C share a log scale)",
                 color=INK, fontsize=13, y=1.00, x=0.012, ha="left")
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    out = "results/surrogate.png"
    fig.savefig(out, dpi=170, facecolor=SURF, bbox_inches="tight")
    print("saved", out)

    for flat, gg in d.groupby("flat"):
        print(f"\nflat={flat}  (sd rho_hat {gg.rho_sd.mean():.3f})")
        print(pd.DataFrame([
            dict(corr=c, naive=rms(h.naive - h.true_att),
                 surrogate=rms(h.surrogate - h.true_att),
                 known=rms(h.known - h.true_att),
                 oracle=rms(h.oracle - h.true_att))
            for c, h in gg.groupby("corr")]).round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
