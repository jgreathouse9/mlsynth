"""
Bayesian synthetic control: what the simplex buys you
=====================================================

``CausalImpact`` is the Bayesian counterfactual method most applied researchers
reach for, and often the only one they have heard of. The Bayesian
synthetic-control literature is considerably larger than that, and several of its
methods answer the same question while respecting a constraint ``CausalImpact``
does not impose: that the counterfactual be a convex combination of units that
actually exist.

This example puts the two side by side on three panels.

:doc:`bvss` is Bayesian Variable Selection with a Soft Simplex constraint (Xu and
Zhou 2025, arXiv:2503.06454). It places a spike-and-slab prior on the donor
weights, so the posterior says which donors enter, on top of a soft simplex prior,
so the weights that do enter are non-negative and sum to one. :doc:`bscm` (Kim,
Lee and Gupta 2020) is the unconstrained comparison: a Bayesian regression of the
treated unit on the donor pool under a shrinkage prior, with no simplex. That is
the shape of the model ``CausalImpact`` fits -- a Bayesian regression on controls,
with variable selection, and coefficients free to take any sign.

The weights are where the two part company, and the estimated effect is not.
A reader who compares only the headline number will conclude the choice does not
matter.

The panels are chosen so that one of them settles the question and the other two
show what happens past it. Write :math:`C` for the number of donors and
:math:`n` for the number of pre-treatment periods:

- Abadie and Gardeazabal's Basque Country, treated from the 1975 terrorism onset:
  16 donor regions against 20 pre-periods, so :math:`C < n` and the unconstrained
  least-squares problem has a unique solution. Catalonia is the comparison a
  practitioner would pick by hand.
- Proposition 99: 38 donor states against 19 pre-periods, so :math:`C > n`.
- The China anti-corruption luxury-watch panel: 87 donor categories against 35
  pre-periods, again :math:`C > n`.

The Basque panel is the one that settles it. With a unique solution available,
any extrapolation the unconstrained fit does is the missing constraint and
nothing else. On the other two there is no unique answer to begin with, and the
simplex plus the spike-and-slab is what leaves a well-posed estimate standing.
"""

# sphinx_gallery_thumbnail_number = 2

# %%
# Three panels
# ------------
# Each is read from GitHub so a downloaded notebook runs as-is. The Basque file
# carries Spain as a whole as its own row, which is an aggregate of the others
# and is dropped. The watch panel already ships in long form with its treatment
# column.

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mlsynth import BVSS, BSCM

BASE = "https://raw.githubusercontent.com/jgreathouse9/mlsynth/refs/heads/main/basedata/"

basque = pd.read_csv(BASE + "basque_mscmt.csv")
basque = basque[basque["regionname"] != "Spain (Espana)"].copy()
basque["treat"] = ((basque["regionname"] == "Basque Country (Pais Vasco)")
                   & (basque["year"] >= 1975)).astype(int)

prop99 = pd.read_csv(BASE + "smoking_data.csv")
prop99["treat"] = ((prop99["state"] == "California")
                   & (prop99["year"] >= 1989)).astype(int)

watches = pd.read_csv(BASE + "china_watches_long.csv")

PANELS = [
    ("Basque Country", dict(df=basque, outcome="gdpcap", unitid="regionname",
                            time="year", treat="treat"), {},
     "GDP per capita"),
    ("Proposition 99", dict(df=prop99, outcome="cigsale", unitid="state",
                            time="year", treat="treat"), {},
     "packs per capita"),
    # The watch panel is p > n, and the cross-validated benchmark case runs the
    # sampler at this short chain; the settings are kept identical to it.
    ("China luxury watches", dict(df=watches, outcome="y", unitid="unit",
                                  time="time", treat="treat"),
     dict(n_iter=50, burn_in=25), "import growth"),
]
for name, cfg, _, _ in PANELS:
    donors = cfg["df"][cfg["unitid"]].nunique() - 1
    pre = int((cfg["df"].groupby(cfg["time"])[cfg["treat"]].max() == 0).sum())
    regime = "C < n, identified" if donors < pre else "C > n, rank-deficient"
    print(f"{name:<22} {donors:3d} donors, {pre:3d} pre-periods   {regime}")

# %%
# Fitting both on each panel
# --------------------------
# BVSS runs at its defaults except on the watch panel. BSCM runs at its defaults
# throughout. Both are seeded.

fits = {}
for name, cfg, extra, _ in PANELS:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fits[name] = {
            "BVSS": BVSS({**cfg, "seed": 1, "display_graphs": False, **extra}).fit(),
            "BSCM": BSCM({**cfg, "seed": 1, "display_graphs": False}).fit(),
        }

rows = []
for name, _, _, _ in PANELS:
    for method, res in fits[name].items():
        w = np.array([float(v) for v in (res.weights.donor_weights or {}).values()])
        rows.append({"panel": name, "method": method,
                     "ATT": float(res.effects.att),
                     "negative weights": int((w < -1e-6).sum()),
                     "negative mass": round(float(w[w < 0].sum()), 3),
                     "positive mass": round(float(w[w > 0].sum()), 3),
                     "donors used": int((np.abs(w) > 0.01).sum()),
                     "weights sum to": round(float(w.sum()), 3)})
summary = pd.DataFrame(rows)
print(summary.to_string(index=False))

# %%
# The counterfactual paths
# ------------------------
# Observed against each method's counterfactual, for the two panels whose outcome
# is a level. The two counterfactuals are close enough to be hard to separate on
# Basque, and part on Proposition 99 by far less than the weights below would lead
# you to expect. The watch panel's outcome is a monthly growth rate, so its path is
# three overlapping noisy series and nothing is learned by eye; that panel's
# comparison is in the weights.

from mlsynth.utils.plotting import Plotter, mlsynth_style

plotter = Plotter()
with mlsynth_style():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (name, cfg, _, ylab) in zip(axes, PANELS[:2]):
        res_bvss, res_bscm = fits[name]["BVSS"], fits[name]["BSCM"]
        times = np.asarray(res_bvss.time_series.time_periods)
        t0 = int(np.asarray(cfg["df"].groupby(cfg["time"])[cfg["treat"]].max() == 0).sum())
        plotter.observed_vs_counterfactual(
            times,
            np.asarray(res_bvss.time_series.observed_outcome, dtype=float).ravel(),
            [np.asarray(res_bvss.time_series.counterfactual_outcome, dtype=float).ravel(),
             np.asarray(res_bscm.time_series.counterfactual_outcome, dtype=float).ravel()],
            labels=["BVSS (simplex)", "BSCM (unconstrained)"],
            intervention=times[t0] if t0 < len(times) else None,
            outcome=ylab, time="", title=name, ax=ax,
        )
        handles, labels = ax.get_legend_handles_labels()
        labels[0] = "Observed"
        ax.legend(handles, labels, loc="best", fontsize=9)
    fig.tight_layout()

# %%
# The weights
# -----------
# The same fits, as sorted weight profiles: every donor's weight, ordered from
# largest to smallest, with the region below zero shaded. BVSS traces a short
# positive spine that falls to zero and stays there. The unconstrained fit crosses
# into negative territory on every panel.
#
# The count of negative weights is the wrong summary, and it runs opposite to the
# severity, so each legend carries the negative mass as well. On Basque the
# unconstrained fit spends 2.91 of positive weight against 1.91 of negative. The
# watch panel has the most negative weights, thirty-eight, and the mildest, none
# below -0.03, totalling -0.22.
#
# The Basque weights also sum to 1.000 under both methods. The constraint is not
# visible in the total; it is visible in what the total is made of.

with mlsynth_style():
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (name, _, _, _) in zip(axes2, PANELS):
        for method, color in (("BVSS", "red"), ("BSCM", "blue")):
            w = np.array([float(v) for v in
                          (fits[name][method].weights.donor_weights or {}).values()])
            n_neg = int((w < -1e-6).sum())
            lab = f"{method}: {n_neg} negative"
            if n_neg:
                lab += f", mass {w[w < 0].sum():.2f}"
            ax.plot(np.arange(1, w.size + 1), np.sort(w)[::-1], color=color, lw=1.8,
                    label=lab)
        ax.axhline(0.0, color="grey", lw=1)
        lo = ax.get_ylim()[0]
        ax.axhspan(lo, 0.0, color="grey", alpha=0.08, lw=0)
        ax.set_ylim(bottom=lo)
        ax.set_title(name)
        ax.set_xlabel("donor, ranked by weight")
        ax.set_ylabel("posterior mean weight")
        ax.legend(loc="best", fontsize=9)
    fig2.tight_layout()

# %%
# On the Basque panel BVSS puts its largest weight on Catalonia, the donor most
# people would have chosen by hand, and reaches it without being told. The
# unconstrained fit also keeps Catalonia, under weights nobody selected.

for name in ("Basque Country",):
    for method in ("BVSS", "BSCM"):
        w = {k: float(v) for k, v in
             (fits[name][method].weights.donor_weights or {}).items()}
        top = sorted(w.items(), key=lambda kv: -abs(kv[1]))[:4]
        print(f"{method:<5} " + "  ".join(f"{k.split(' (')[0]} {v:+.3f}" for k, v in top))

# %%
# What to take from this
# ----------------------
# The estimated effects are close on every panel -- -0.84 against -0.90 on Basque,
# and agreeing to three decimals on the watches -- so a comparison that stopped at
# the ATT would report that the constraint makes no difference.
#
# The Basque panel says otherwise, and it says it in the regime where nothing else
# can be blamed. Sixteen donors, twenty pre-periods, a unique least-squares
# solution available, and the treated region sitting inside the range of the
# others. The unconstrained fit still assembles its counterfactual from 2.91 of
# positive weight and 1.91 of negative. That is a Basque Country built by adding
# three Spains and subtracting two, and its close pre-period fit is no evidence
# about how such a thing behaves after 1975.
#
# Proposition 99 and the watches make the second point. Both have more donors than
# pre-periods, so the unconstrained problem has no unique solution and only the
# prior picks among the minimisers. BVSS returns a sparse, non-negative weight
# vector summing to one on both.
#
# This is the argument for reading past ``CausalImpact`` when the question is a
# synthetic control. Being Bayesian is not the distinguishing feature -- both
# methods here are Bayesian, both carry full posteriors, and both do variable
# selection. The simplex is the distinguishing feature, and it is the part
# ``CausalImpact`` leaves out.
#
# BVSS is cross-validated against the authors' own Gibbs sampler on the watch
# panel in ``benchmarks/cases/bvss_watches.py``, which is where its numbers are
# pinned.
