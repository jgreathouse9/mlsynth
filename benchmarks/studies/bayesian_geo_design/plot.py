"""Bayesian synthetic control on Meta's GeoLift test panel.

Treated = chicago + portland (GeoLift's own #1 design, and MVBBSC's), summed as
GeoLift aggregates them. Treatment starts period 91. Bands are the posterior
PREDICTIVE with the pre-period AR(1) carried into the shock -- the version that
covers 87.5% of a nominal 90% on this panel's placebo test.
"""
from __future__ import annotations
import os; os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from . import engines as bayes

S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"      # validated categorical slots
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8983"
SURF, GRID = "#fcfcfb", "#e6e5e0"
T0 = 90

W = pd.read_csv("basedata/geolift_test_data.csv").pivot(
    index="date", columns="location", values="Y").sort_index()
TR = ["chicago", "portland"]
y = W[TR].sum(axis=1).to_numpy(float)
X = W.drop(columns=TR).to_numpy(float)
T = len(y)
fit = bayes.mvbbsc_fit_once(y, X, T0, T0, T - 1, 1, seed=0)
f = fit.extras
print("MVBBSC: max r-hat %.3f | divergences %d | donors %d | pre %d post %d"
      % (f["max_rhat"], f["n_divergent"], X.shape[1], T0, T - T0))

# predictive counterfactual path, AR(1)-corrected (same process as att_draws)
mu, sd = f["mu"], f["sd"]                          # (D,T), (D,)
rho = float(f["rho"])
D = mu.shape[0]
rng = np.random.default_rng(11)
e = np.empty((D, T))
e[:, 0] = rng.standard_normal(D) * sd / np.sqrt(max(1 - rho ** 2, 1e-6))
for t in range(1, T):
    e[:, t] = rho * e[:, t - 1] + rng.standard_normal(D) * sd
CF = mu + e                                        # (D,T) posterior predictive
gap = y[None, :] - CF
att = bayes.att_posterior(fit, T0, T - 1, autocorr=True)
lo, hi = np.percentile(att, [5, 95])
base = float(np.mean(y[T0:]))
print("ATT %.0f per period (%.1f%% of treated volume) | 90%% CI [%.0f, %.0f] -> [%.1f%%, %.1f%%]"
      % (att.mean(), 100 * att.mean() / base, lo, hi, 100 * lo / base, 100 * hi / base))
print("P(effect > 0) = %.3f | AR(1) rho = %.2f" % (float(np.mean(att > 0)), rho))

t = np.arange(1, T + 1)
q = lambda A, p: np.percentile(A, p, axis=0)
fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                       gridspec_kw={"height_ratios": [1.5, 1], "hspace": 0.26})
fig.patch.set_facecolor(SURF)

def frame(a, ylab):
    a.set_facecolor(SURF); a.grid(True, color=GRID, lw=0.8); a.set_axisbelow(True)
    for s in ("top", "right"): a.spines[s].set_visible(False)
    for s in ("left", "bottom"): a.spines[s].set_color(GRID)
    a.tick_params(colors=INK2, labelsize=9, length=0)
    a.set_ylabel(ylab, color=INK2, fontsize=10)
    a.axvline(T0 + 0.5, color=INK3, lw=1.4)
    a.set_xlim(1, T)

frame(ax[0], "Y  (chicago + portland)")
ax[0].fill_between(t, q(CF, 2.5), q(CF, 97.5), color=S1, alpha=0.15, lw=0)
ax[0].fill_between(t, q(CF, 25), q(CF, 75), color=S1, alpha=0.30, lw=0)
ax[0].plot(t, q(CF, 50), color=S1, lw=2)
ax[0].plot(t, y, color=S2, lw=1.8, ls=(0, (5, 2.5)))
ax[0].set_title("Bayesian synthetic control — MVBBSC on Meta's GeoLift test panel\n"
                "treated = chicago + portland (GeoLift's top design); 38 donors; "
                "treatment starts period 91",
                color=INK, fontsize=12.5, loc="left", pad=12)
ax[0].legend(handles=[
    Line2D([], [], color=S2, lw=1.8, ls=(0, (5, 2.5)), label="observed treated"),
    Line2D([], [], color=S1, lw=2, label="posterior counterfactual (median)"),
    Patch(facecolor=S1, alpha=0.30, label="50% credible"),
    Patch(facecolor=S1, alpha=0.15, label="95% credible")],
    loc="upper left", frameon=False, fontsize=9, labelcolor=INK, ncol=2)
ax[0].annotate("treatment", (T0 + 2, ax[0].get_ylim()[1] * 0.97), color=INK2,
               fontsize=9, va="top")

frame(ax[1], "lift  (observed − counterfactual)")
ax[1].fill_between(t, q(gap, 2.5), q(gap, 97.5), color=S3, alpha=0.15, lw=0)
ax[1].fill_between(t, q(gap, 25), q(gap, 75), color=S3, alpha=0.30, lw=0)
ax[1].plot(t, q(gap, 50), color=S3, lw=1.8)
ax[1].axhline(0, color=INK3, lw=1)
ax[1].set_xlabel("period", color=INK2, fontsize=10)
ax[1].set_title("The lift, with honest uncertainty. Pre-period straddles zero; "
                "post-period does not.", color=INK, fontsize=11, loc="left", pad=8)
ax[1].annotate("ATT %+.0f per period  (%+.1f%%)\n90%% credible [%+.1f%%, %+.1f%%]   "
               "P(effect>0) = %.2f" % (att.mean(), 100 * att.mean() / base,
                                       100 * lo / base, 100 * hi / base,
                                       float(np.mean(att > 0))),
               (0.985, 0.06), xycoords="axes fraction", ha="right", va="bottom",
               color=INK, fontsize=9.5)
out = "benchmarks/studies/bayesian_geo_design/results/bayes_sc_geolift.png"
fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=SURF)
print("saved", out)
