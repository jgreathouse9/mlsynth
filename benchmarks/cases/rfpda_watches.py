"""rfPDA against its reference implementation on the luxury-watch panel.

Liu, Long and Luo (2025, J. Applied Econometrics 40(5):591-607) apply rfPDA to
the panel Shi and Huang (2023) built for China's anti-corruption campaign: the
monthly import growth rate of "watches with cases of, or clad with, precious
metal" against 87 other commodities, February 2010 to December 2016, with the
campaign starting in January 2013. They report an average effect of -2.66% with
an approximate R-squared of 0.78 over seven selected controls, at a p value of
0.063.

What this case can and cannot pin
---------------------------------
Running their ``RF.R`` on their own ``china_import.rda`` at their own seed
reproduces all four of those numbers: ATE -0.0266, R-squared 0.7771, 7 controls,
p 0.0634. That is a Path A match in R, and it is recorded here as the reference.
It is not a target a Python port can be held to pointwise. The selected set is a
function of ``randomForest``'s tree construction and its RNG stream, and no
scikit-learn forest reproduces either, so a port that agreed on the donor list
would be agreeing by accident.

What the two implementations can be held to is the distribution. The selection
is seed-dependent to a degree the paper does not report -- over twenty seeds the
reference's own ATE on this panel runs from -0.063 to -0.003 around its
published -0.0266, with a mean pairwise selection overlap of 0.230 -- so the
estimand a single seed reports is a draw, and comparing draws across two RNGs
measures nothing. This case compares the draws' distribution instead, and pins
separately the one component that is deterministic: the long-run variance.

The three checks
----------------
1. The West (1997) long-run variance of Equation (9)-(10), against the
   reference's ``HAC_function`` on a fixed deterministic error series. This is
   exact arithmetic on both sides apart from the MA(1) fit, where R's ``arima``
   and ``statsmodels``' ``ARIMA`` optimise the same likelihood from different
   starting points; the residual disagreement is the tolerance below.
2. The ATE distribution over seeds under the released configuration -- the
   random pre-period split, out-of-bag importance, no cap on the selected set --
   against the reference's distribution over the same seeds.
3. The cap. The released search runs the prefix length to ``n - 1`` with nothing
   tying it to the pre-period length, against Assumption 3's requirement that
   the selected set be small relative to it. Where it over-selects, the OLS fit
   interpolates the pre-period, the residual falls to rounding and the long-run
   variance collapses with it. The reference does this in 4 of 20 seeds on this
   panel; the default cap at ``T0 - 2`` has to do it in none.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlsynth.utils.pda_helpers.rf import rf_select, rf_ate_inference, west_lrvar

_CASE = "rfpda_watches"
N_SEEDS = 10
N_TREES = 300

# The reference's HAC_function on e_t = sin(t) + 0.3 cos(3t), t = 1..60, split
# at T1 = 40, with MA(1) on each window. Deterministic: no RNG on either side.
R_HAC = {"before": 0.436241897207, "after": 0.850073133287, "z": -0.096661351544}

# RF.R on china_import.rda over seeds 0-19 at its own settings
# (per_train = 0.7, ntree = 1000, no cap), and its published point estimate.
R_RELEASED = {"att_mean": -0.0199, "att_sd": 0.0128,
              "att_min": -0.0629, "att_max": -0.0033,
              "k_min": 2, "k_max": 87, "n_degenerate": 4, "jaccard": 0.230}
PAPER = {"att": -0.0266, "r2": 0.78, "n_selected": 7, "p_value": 0.063}


def _panel():
    d = pd.read_csv("basedata/china_watches_long.csv")
    wide = d.pivot(index="time", columns="unit", values="y").sort_index()
    T0 = int(d.loc[(d["unit"] == "watches") & (d["treat"] == 1), "time"].min())
    y = wide["watches"].to_numpy(float)
    X = wide.drop(columns=["watches"]).to_numpy(float)
    return y, X, T0


def _sweep(y, X, T0, **kw):
    atts, sizes, degenerate, sets = [], [], 0, []
    for seed in range(N_SEEDS):
        sel, _, _, cf, _ = rf_select(y, X, T0, seed=seed, n_estimators=N_TREES, **kw)
        att, _, _, _ = rf_ate_inference(y, cf, T0)
        atts.append(att)
        sizes.append(len(sel))
        sets.append(set(sel))
        # An interpolating pre-period fit is the failure the cap is there to
        # stop: the residual variance falls to rounding and the test degenerates.
        if np.var((y - cf)[:T0]) < 1e-12 * max(np.var(y[:T0]), 1e-30):
            degenerate += 1
    pairs = [len(a & b) / max(len(a | b), 1)
             for i, a in enumerate(sets) for b in sets[i + 1:]]
    a = np.asarray(atts, dtype=float)
    return {"att_mean": float(a.mean()), "att_sd": float(a.std(ddof=1)),
            "att_min": float(a.min()), "att_max": float(a.max()),
            "k_min": int(min(sizes)), "k_max": int(max(sizes)),
            "n_degenerate": int(degenerate),
            "jaccard": float(np.mean(pairs)) if pairs else 1.0}


def run() -> dict:
    got: dict = {}

    # 1. The deterministic component: the West long-run variance.
    e = np.sin(np.arange(1, 61)) + 0.3 * np.cos(3 * np.arange(1, 61))
    lrv = west_lrvar(e[:40], e[40:], T1=40, T2=20, q1=1, q2=1)
    z = float(np.sqrt(20) * e[40:].mean() / np.sqrt(lrv["before"] + lrv["after"]))
    got["lrvar_before_rel_dev"] = abs(lrv["before"] / R_HAC["before"] - 1.0)
    got["lrvar_after_rel_dev"] = abs(lrv["after"] / R_HAC["after"] - 1.0)
    got["lrvar_z_rel_dev"] = abs(z / R_HAC["z"] - 1.0)

    y, X, T0 = _panel()
    got["n_donors"] = float(X.shape[1])
    got["pre_periods"] = float(T0)

    # 2. The released configuration, against the reference's own spread.
    released = _sweep(y, X, T0, split="random", train_fraction=0.7,
                      importance="oob", k_max=X.shape[1])
    got["released_att_mean"] = released["att_mean"]
    got["released_att_sd"] = released["att_sd"]
    got["released_att_min"] = released["att_min"]
    got["released_att_max"] = released["att_max"]
    got["released_jaccard"] = released["jaccard"]
    # The published point estimate has to lie inside the spread the port draws
    # from, which is the sense in which a seed-dependent estimate replicates.
    got["paper_att_inside_port_range"] = float(
        released["att_min"] <= PAPER["att"] <= released["att_max"])
    got["reference_att_mean_inside_port_range"] = float(
        released["att_min"] <= R_RELEASED["att_mean"] <= released["att_max"])

    # 3. The cap: uncapped over-selects and degenerates, capped does not.
    got["released_n_degenerate_positive"] = float(released["n_degenerate"] > 0)
    got["released_k_max_exceeds_pre_periods"] = float(released["k_max"] > T0)

    capped = _sweep(y, X, T0, split="random", train_fraction=0.7, importance="oob")
    got["capped_n_degenerate"] = float(capped["n_degenerate"])
    got["capped_k_max"] = float(capped["k_max"])
    got["capped_k_within_cap"] = float(capped["k_max"] <= T0 - 2)

    paper_cfg = _sweep(y, X, T0, split="temporal")
    got["temporal_n_degenerate"] = float(paper_cfg["n_degenerate"])
    got["temporal_att_mean"] = paper_cfg["att_mean"]
    got["temporal_att_sign_negative"] = float(paper_cfg["att_mean"] < 0)
    return got


# Tolerances.
#
# The long-run variance deviations are the MA(1) optimiser's, not the
# estimator's: R's arima and statsmodels' ARIMA maximise the same likelihood
# from different starts, and the two components land within 0.2% of each other
# with the test statistic within 0.07%. Anything larger is a change to the
# construction, not to the optimiser.
#
# The distribution checks carry wide bounds on purpose. Two forest
# implementations drawing different trees produce different draws from the same
# estimand, so the pins are on the shape: the mean lands in the same region and
# with the same sign, the spread is of the same order, the selection overlap is
# low in both. The two containment indicators are the sharper statement -- the
# published estimate and the reference's own mean both have to lie inside the
# range the port draws, which is what a seed-dependent replication can claim.
#
# The cap indicators are structural and carry no tolerance. The released search
# must be seen to over-select and degenerate on this panel, and the default cap
# must be seen to prevent it, or the cap has nothing to justify it.
EXPECTED = {
    "n_donors": (87.0, 0.0),
    "pre_periods": (35.0, 0.0),

    "lrvar_before_rel_dev": (0.0, 0.01),
    "lrvar_after_rel_dev": (0.0, 0.01),
    "lrvar_z_rel_dev": (0.0, 0.005),

    "paper_att_inside_port_range": (1.0, 0.0),
    "reference_att_mean_inside_port_range": (1.0, 0.0),
    "released_n_degenerate_positive": (1.0, 0.0),
    "released_k_max_exceeds_pre_periods": (1.0, 0.0),

    # The distribution itself, recorded so drift is visible. The bounds are
    # wide because they have to contain both implementations' draws: the port
    # reports mean -0.0271 (sd 0.0142) over ten seeds against the reference's
    # -0.0199 (sd 0.0128) over twenty, and the two selection overlaps are 0.220
    # and 0.230. A move outside these is a change to the selection rule.
    "released_att_mean": (-0.0235, 0.015),
    "released_att_sd": (0.0135, 0.010),
    "released_jaccard": (0.225, 0.100),
    "temporal_att_mean": (-0.0185, 0.015),

    "capped_n_degenerate": (0.0, 0.0),
    "capped_k_within_cap": (1.0, 0.0),
    "temporal_n_degenerate": (0.0, 0.0),
    "temporal_att_sign_negative": (1.0, 0.0),
}
