"""
CMBSTS: a price cut, and what it did to the competitor
======================================================

A supermarket chain cut the price of one of its own store-brand products and
left it cut. The question is how many extra units that sold -- and, because the
direct competitor sits on the same shelf, whether those units came out of the
competitor's sales. The second question is what makes this awkward for an
ordinary synthetic control: the competitor is the most informative comparison
available and is also the series most likely to react to the treatment, so it
cannot serve as a clean control.

CMBSTS, the causal multivariate Bayesian structural time series model of
Menchetti and Bojinov (2022, *Annals of Applied Statistics* 16(1):414-435),
answers both at once. The treated product and its competitor are modelled as a
two-dimensional outcome vector sharing a trend, a weekly seasonal component and
a spike-and-slab regression on genuinely untreated series. Each gets its own
counterfactual and its own credible band, so the effect on the treated brand and
the spillover onto the rival are estimated inside one model instead of being
assumed away.

The data are daily sales from the authors' replication package: one
store-competitor pair (pair 10 of the three the paper studies), ten wine-category
control series picked by dynamic time warping, the two brands' prices, and
calendar dummies for Saturdays, Sundays and holidays. Sales are put on a
per-opening-hour basis because Sunday trading hours are shorter. The price
reduction takes effect on 2018-10-04.

Two details carry the design. The store's own post-intervention price is the
thing the treatment changes, so feeding it to the model would absorb the effect;
it is frozen at its last pre-intervention value, which keeps the regressor
exogenous as the paper's Assumption 3 requires. And the effect is summarised over
a one-month horizon ending 2018-11-04 instead of the full year of post-period
data. A structural forecast's uncertainty grows with the horizon, so a band drawn
across the whole remaining panel says little; the month after the cut is the
window the decision was about, and it is the window plotted below.
"""

# sphinx_gallery_thumbnail_number = 1

# %%
# Building the panel
# ------------------
# Six files from the replication package: the two brands' daily sales and prices,
# the wine-category control series, and the calendar dummies. Sales are divided by
# opening hours (5 on most Sundays, 13 otherwise), and the store price is frozen
# from the intervention date onward. The panel CMBSTS wants is long: one row per
# series per day, with ``treated`` marking the store brand after the cut.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mlsynth import CMBSTS

BASE = (
    "https://raw.githubusercontent.com/jgreathouse9/mlsynth/"
    "refs/heads/main/basedata/cmbsts_supermarket/"
)
INTERVENTION = pd.Timestamp("2018-10-04")
HORIZON_END = pd.Timestamp("2018-11-04")          # the paper's one-month horizon
PAIR = 10
CONTROLS = [0, 129, 14, 173, 182, 230, 253, 33, 54, 86]   # DTW-selected wines

dv = pd.read_csv(BASE + "dummy_var.csv", sep=r"\s+")
# That file carries a leading row-number column, so pandas takes it as a 1-based
# index; everything below is positional, so drop it before using the columns.
dv = dv.reset_index(drop=True)
dates = pd.to_datetime(dv["dates"])
dates_arr = dates.to_numpy()
sat, sun_d, hol = (dv[c].to_numpy() for c in ("sat", "sun", "hol"))
store = pd.read_csv(BASE + "store_sales.csv", sep=";")
comp = pd.read_csv(BASE + "competitor_sales.csv", sep=";")
store_price = pd.read_csv(BASE + "store_price.csv", sep=";").to_numpy(float)
comp_price = pd.read_csv(BASE + "competitor_price.csv", sep=";").to_numpy(float)
wines = pd.read_csv(BASE + "wines.csv", sep=";")

sunday = ((dates.dt.day_name() == "Sunday") & (dates.dt.month != 12)
          & (~dates.isin([pd.Timestamp("2017-11-26"), pd.Timestamp("2018-11-25")])))
hours = np.where(sunday, 5.0, 13.0)
h_store = store.div(hours, axis=0).to_numpy()
h_comp = comp.div(hours, axis=0).to_numpy()
h_wines = wines.div(hours, axis=0).to_numpy()

post = (dates >= INTERVENTION).to_numpy()
store_price_frozen = store_price.copy()
store_price_frozen[post] = store_price_frozen[~post][-1]

excl = dv["excl.dates"].to_numpy()
names = [f"w{j}" for j in CONTROLS]
rows = []
for t in range(len(dates)):
    shared = {"week": t, "excl": excl[t]}
    rows.append({"item": "store", "sales": h_store[t, PAIR - 1], "treated": int(post[t]),
                 "sat": sat[t], "sun": sun_d[t], "hol": hol[t],
                 "sprice": store_price_frozen[t, PAIR - 1],
                 "cprice": comp_price[t, PAIR - 1], **shared})
    rows.append({"item": "comp", "sales": h_comp[t, PAIR - 1], "treated": 0,
                 "sat": np.nan, "sun": np.nan, "hol": np.nan,
                 "sprice": np.nan, "cprice": np.nan, **shared})
    for j, name in zip(CONTROLS, names):
        rows.append({"item": name, "sales": h_wines[t, j], "treated": 0,
                     "sat": np.nan, "sun": np.nan, "hol": np.nan,
                     "sprice": np.nan, "cprice": np.nan, **shared})
panel = pd.DataFrame(rows)

horizon = int(((dates >= INTERVENTION) & (dates <= HORIZON_END) & (excl == 0)).sum())
print(f"{len(dates)} days, {horizon} non-excluded days in the one-month horizon")

# %%
# Fitting the model
# -----------------
# ``group_units`` holds the series that may react to the treatment -- here the
# competitor -- and ``control_units`` holds the ones that cannot. That split is
# the partial-interference assumption, stated as configuration. The prior
# cross-series correlation ``prior_rho=-0.8`` encodes the substitution the study
# expects between a brand and its direct rival; ``horizon`` confines the reported
# effect to the month after the cut.

result = CMBSTS({
    "df": panel, "outcome": "sales", "unitid": "item", "time": "week",
    "treat": "treated",
    "group_units": ["comp"], "control_units": names,
    "covariates": ["sat", "sun", "hol", "sprice", "cprice"],
    "components": ["trend", "seasonal"], "seas_period": 7,
    "excl_dates": "excl", "horizon": horizon,
    "prior_scale": 1.0, "prior_rho": -0.8,
    "niter": 1000, "burn": 200, "seed": 1, "display_graphs": False,
}).fit()

detail = result.inference_detail
for k, name in enumerate(result.inputs.series_names):
    lo, hi = float(detail.att_lower[k]), float(detail.att_upper[k])
    print(f"{name:>6}: {float(detail.att_mean[k]):7.2f} units/hour per day   "
          f"95% CI [{lo:7.2f}, {hi:7.2f}]")

# %%
# The effect over the reported horizon
# ------------------------------------
# Observed sales against the counterfactual CMBSTS forecasts for each series, from
# two months before the cut through the end of the one-month horizon. The shaded
# region is the 95% credible band on the counterfactual. Reading the two panels
# together is the point of the model: the store brand separates from its
# counterfactual and stays above it, while the competitor's path remains inside
# its band, which is what an absent spillover looks like.

from mlsynth.utils.plotting import Plotter, mlsynth_style

T0 = result.inputs.T0
counterfactual = detail.counterfactual_full
observed = result.inputs.Y

window = ((dates >= INTERVENTION - pd.Timedelta(days=60))
          & (dates <= HORIZON_END)).to_numpy()
idx = np.flatnonzero(window)
in_post = idx >= T0
band_rows = idx[in_post] - T0                  # the effect arrays start at T0

titles = {"store": "Store brand (price cut)", "comp": "Competitor (untreated)"}
plotter = Plotter()

with mlsynth_style():
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for k, (ax, name) in enumerate(zip(axes, result.inputs.series_names)):
        # The band is on the counterfactual, so subtract the effect bounds from
        # the observed path; the pre-period stays NaN and is left unshaded.
        lower = np.full(idx.size, np.nan)
        upper = np.full(idx.size, np.nan)
        lower[in_post] = observed[idx[in_post], k] - detail.effect_upper[band_rows, k]
        upper[in_post] = observed[idx[in_post], k] - detail.effect_lower[band_rows, k]

        plotter.observed_vs_counterfactual(
            dates_arr[idx],
            observed[idx, k],
            counterfactual[idx, k],
            labels=["Counterfactual"],
            intervention=INTERVENTION,
            interval=(lower, upper),
            interval_label="95% credible band",
            outcome="units sold per hour",
            time="Date" if k == len(axes) - 1 else "",
            title=titles.get(str(name), str(name)),
            ax=ax,
        )
        # The archetype labels its first series "Treated (...)"; only the store
        # brand is treated here, so name each series for what it is.
        handles, labels = ax.get_legend_handles_labels()
        labels[0] = f"Observed ({name})"
        ax.legend(handles, labels, loc="upper left")

        att, lo, hi = (float(detail.att_mean[k]), float(detail.att_lower[k]),
                       float(detail.att_upper[k]))
        ax.annotate(f"ATT {att:.2f}  [{lo:.2f}, {hi:.2f}]",
                    xy=(0.985, 0.06), xycoords="axes fraction", ha="right")
    fig.tight_layout()

# %%
# The store brand's effect is positive and its credible interval excludes zero;
# the competitor's is a small negative number whose interval comfortably contains
# it. That is the paper's Table 3 finding for this pair -- a real gain on the
# discounted brand, no measurable cannibalisation of the rival -- and it is
# cross-validated against the authors' ``CausalMBSTS`` package in
# ``benchmarks/cases/cmbsts_supermarket.py``.
#
# The band widens as the forecast runs on, which is the behaviour of any
# structural time-series counterfactual: the further from the last observed
# pre-period data, the less the model commits. Confining the summary to the
# month the decision was about is what keeps the reported interval informative,
# and the ``horizon`` argument is how CMBSTS expresses that.
