"""
The Basque Country: synthetic control, the MSCMT way
====================================================

This is the ground-floor example of the whole toolbox: one treated unit, a pool
of donor units, and a single question. In 1968 the separatist group ETA began a
campaign of terrorism concentrated in the Basque Country, one of Spain's
seventeen regions. Abadie and Gardeazabal (2003) asked what that violence cost
the regional economy. There is no parallel Basque Country that escaped the
conflict to compare against, so they built one: a synthetic Basque Country,
assembled as a weighted average of the other Spanish regions, chosen so that the
blend tracks the real region's economy before the violence began. The gap that
opens up afterwards is the estimated cost.

We reproduce their study through :class:`~mlsynth.VanillaSC` with
``backend="mscmt"``. The name refers to the R package MSCMT (Becker and
Klossner, 2018), whose vignette re-runs Abadie and Gardeazabal on their
thirteen-predictor specification; mlsynth's ``mscmt`` backend matches that
package's donor weights to four decimals (see the :doc:`Verification section of
the VanillaSC page </vanillasc>`). The outcome is real per-capita GDP; the donor
pool is the remaining Spanish regions; the intervention year is 1970.
"""

# %%
# Load the panel
# --------------
# The data ship with mlsynth as ``basedata/basque_mscmt.csv`` -- the MSCMT
# transform of the classic Basque panel, one row per region-year from 1955 to
# 1997, carrying GDP and the thirteen predictors. We read it straight from
# GitHub so this notebook runs anywhere, then add the treatment column:
# Basque Country from 1970 onward. Spain-as-a-whole is a row in the file and is
# dropped from the donor pool downstream by the estimator.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mlsynth import VanillaSC

url = (
    "https://raw.githubusercontent.com/jgreathouse9/mlsynth/"
    "refs/heads/main/basedata/basque_mscmt.csv"
)
panel = pd.read_csv(url)
panel["treat"] = (
    (panel["regionname"] == "Basque Country (Pais Vasco)")
    & (panel["year"] >= 1970)
).astype(int)

panel.head()

# %%
# Fit the synthetic control
# -------------------------
# The classical estimator matches on covariates as well as the outcome, and it
# learns two sets of weights at once: donor weights ``W`` (which regions to
# average) and predictor weights ``V`` (how much each covariate matters for the
# match). That nested problem is what ``backend="mscmt"`` solves, by a seeded
# differential-evolution search over ``V``.
#
# Three settings pin the specification to Abadie and Gardeazabal's:
# ``covariates`` lists the thirteen predictors; ``covariate_windows`` gives each
# one the pre-period years it is averaged over; and ``fit_window=(1960, 1969)``
# is the decade the outcome fit is optimised on, even though the panel reaches
# back to 1955. The ``seed`` makes the search reproducible.

covariates = [
    "school.illit", "school.prim", "school.med", "school.higher", "invest",
    "gdpcap", "sec.agriculture", "sec.energy", "sec.industry",
    "sec.construction", "sec.services.venta", "sec.services.nonventa",
    "popdens",
]
schooling_and_investment = (1964, 1969)
sector_shares = (1961, 1969)
covariate_windows = {
    "school.illit": schooling_and_investment,
    "school.prim": schooling_and_investment,
    "school.med": schooling_and_investment,
    "school.higher": schooling_and_investment,
    "invest": schooling_and_investment,
    "gdpcap": (1960, 1969),
    "sec.agriculture": sector_shares,
    "sec.energy": sector_shares,
    "sec.industry": sector_shares,
    "sec.construction": sector_shares,
    "sec.services.venta": sector_shares,
    "sec.services.nonventa": sector_shares,
    "popdens": (1969, 1969),
}

result = VanillaSC({
    "df": panel,
    "outcome": "gdpcap",
    "treat": "treat",
    "unitid": "regionname",
    "time": "year",
    "backend": "mscmt",
    "canonical_v": "min.loss.w",
    "covariates": covariates,
    "covariate_windows": covariate_windows,
    "fit_window": (1960, 1969),
    "mscmt_maxiter": 400,
    "mscmt_popsize": 20,
    "seed": 42,
    "display_graphs": False,
}).fit()

# %%
# Who is in the synthetic Basque Country?
# ---------------------------------------
# Three regions carry essentially all the weight. Cataluna and Madrid are the
# other industrialised regions; Baleares fills in the rest. These are the same
# weights the MSCMT R package reports, to four decimals.

weights = {
    region: float(w)
    for region, w in result.weights.donor_weights.items()
    if float(w) > 1e-4
}
for region, w in sorted(weights.items(), key=lambda kv: -kv[1]):
    print(f"{region:28s} {w:.4f}")

# %%
# The estimated cost
# ------------------
# The average treatment effect on the treated is the mean gap between the real
# Basque Country and its synthetic twin over the post-1970 period, in the units
# of the outcome (thousands of 1986 USD of per-capita GDP).

print(f"ATT = {result.effects.att:+.3f}  (per-capita GDP, thousands of USD)")

# %%
# Treated versus synthetic
# ------------------------
# The picture is the whole argument. Before 1970 the synthetic Basque Country
# tracks the real one closely -- that is the match the estimator optimised. After
# 1970 the two diverge: the real region's GDP falls below the counterfactual, and
# the shaded distance between the lines is the estimated economic cost of the
# conflict.

ts = result.time_series
years = np.asarray(ts.time_periods).ravel()
observed = np.asarray(ts.observed_outcome, dtype=float).ravel()
synthetic = np.asarray(ts.counterfactual_outcome, dtype=float).ravel()

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(years, observed, color="black", lw=2, label="Basque Country (observed)")
ax.plot(years, synthetic, color="crimson", lw=2, ls="--",
        label="Synthetic Basque Country")
post = years >= 1970
ax.fill_between(years[post], observed[post], synthetic[post],
                color="crimson", alpha=0.12)
ax.axvline(1970, color="gray", lw=1, ls=":")
ax.annotate("ETA campaign\nintensifies", xy=(1970, ax.get_ylim()[1]),
            xytext=(1971, observed.max() * 0.98), fontsize=8, color="gray")
ax.set_xlabel("Year")
ax.set_ylabel("Real per-capita GDP (thousands of 1986 USD)")
ax.set_title("The economic cost of conflict in the Basque Country")
ax.legend(frameon=False, loc="lower right")
fig.tight_layout()

# %%
# What to take away
# -----------------
# This is the entire synthetic-control workflow in one screen: express the panel
# as a long DataFrame, name the outcome, treatment, unit, and time columns, pick
# an estimator, and read the ATT and the donor weights off the result object.
# Every other estimator in mlsynth follows the same shape; what changes between
# them is the assumption each one is willing to make about how the donors relate
# to the treated unit. See :doc:`/choose` to find the one your problem calls for.
