"""Cross-validation: GEOX's augsynth engine against the Recast simulation study.

`getrecast/geolift-simulation-study <https://github.com/getrecast/geolift-simulation-study>`_
compares four geo-experiment tools over 1000 Monte Carlo iterations per cell
across four stress scenarios, and publishes each tool's bias, coverage and
false-positive rate (``results/aggregated/metrics.csv``). One of the four is
Meta's GeoLift, configured there as augmented SC (ridge) with block conformal
inference at ``alpha = 0.05`` -- which is what GEOX runs under
``engine="augsynth", inference="conformal", conformal_type="block"``.

That makes their GeoLift row an external referent for this engine's inference.
It is a stronger position than the design's other calibration case: ``geox_mc``
imposes ground truth on a process defined in the repository and asks whether
GEOX is internally consistent, which cannot detect an error shared between the
estimator and the DGP. Here the DGP, the tool configuration and the target
numbers all come from outside.

Scope. The study evaluates *estimation* -- a fixed treated geo, a known effect,
how well each tool recovers it -- and not market selection. So the comparison
runs the engine's readout with the treated geo fixed, which is the like-for-like
operation. GEOX's selection loop is validated elsewhere
(``geox_augsynth_geolift`` against R ``GeoLiftMarketSelection``).

The DGP is ported in
:func:`~mlsynth.utils.geox_helpers.simulate.simulate_recast_panel` from their
``src/R/generate_panels.R``. The scenarios are theirs:

===  ==========================  ====================================
A1   Textbook                    21 geos, 90 pre-days, clean
A2   Outlier (5x)                treated geo inflated, convex hull violated
A3   Small pool                  10 geos total
A4   Short pre-treatment         30 pre-days instead of 90
===  ==========================  ====================================

What agreement looks like here. The distinctive number in their table is A2: the
outlier scenario biases GeoLift by +3.22 percentage points, an order of
magnitude more than any other cell, because inflating the treated geo puts it
outside the donors' convex hull and the ridge augmentation extrapolates to
reach it. Reproducing the *pattern* -- small bias in A1/A3/A4 and a large one in
A2 -- is what indicates the engine and the port both match, and it is why the
per-scenario bias is pinned individually and not pooled.

Coverage is their headline metric and it has two rows here, because coverage
alone is a weak check: two procedures can both cover at 0.93 with intervals of
wildly different length. So the interval's average width is pinned against
theirs as well, and it is the sharper of the two.
"""
from __future__ import annotations

import warnings

import numpy as np

from mlsynth.utils.geox_helpers.engines import resolve_engine
from mlsynth.utils.geox_helpers.simulate import simulate_recast_panel

N_REPS = 120                # per cell; the study uses 1000
ALPHA, NS, EFFECT = 0.05, 400, 0.075
SCENARIOS = ("A1", "A2", "A3", "A4")

# The study's published GeoLift row, from results/aggregated/metrics.csv at
# commit 4d2afd4 of facebookincubator/GeoLift with augsynth 65c5a6f.
RECAST_GEOLIFT = {
    "A1": {"bias_pp": 0.22, "fpr": 0.046, "power": 1 - 0.913,
           "coverage": 0.922, "coverage_null": 0.954, "ci_width": 2020.07},
    "A2": {"bias_pp": 3.22, "fpr": 0.042, "power": 1 - 0.910,
           "coverage": 0.936, "coverage_null": 0.958, "ci_width": 9954.52},
    "A3": {"bias_pp": 1.03, "fpr": 0.049, "power": 1 - 0.893,
           "coverage": 0.925, "coverage_null": 0.951, "ci_width": 1926.35},
    "A4": {"bias_pp": -0.39, "fpr": 0.033, "power": 1 - 0.957,
           "coverage": 0.951, "coverage_null": 0.967, "ci_width": 5574.99},
}


def _cell(scenario: str, effect: float):
    """One cell's ``N_REPS`` panels: effect, test and interval on each.

    Returns ``(att_pct, p_value, covered, width, empty)``. ``covered`` is
    against the exact true ATT, which the simulator returns as the treated
    geo's untreated counterfactual, so it is not approximated by the injected
    percentage.
    """
    eng = resolve_engine("augsynth")
    att, pvals, covered, widths, empty = [], [], [], [], []
    for seed in range(N_REPS):
        wide, treated, cf, pre = simulate_recast_panel(scenario, effect, seed)
        y = wide[treated].to_numpy()
        Y0 = wide.drop(columns=[treated]).to_numpy()
        end = wide.shape[0] - 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = eng.fit_once(y, Y0, pre, pre, end, 1,
                               augment="ridge", fixed_effects=True)
            p, details = eng.point_inference(fit, y, Y0, pre, pre, end,
                                             inference="conformal",
                                             conformal_type="block", ns=NS,
                                             seed=0, alpha=ALPHA)
        # Percentage scale against the true counterfactual, their convention:
        # att_pct = mean(y - cf) / mean(cf) over the post window.
        att.append(eng.att(fit, y, pre, end) / float(np.mean(cf[pre:])))
        pvals.append(p)

        true_att = float(np.mean(y[pre:] - cf[pre:]))
        lo, hi = details["ci_lower"], details["ci_upper"]
        gone = lo is None or hi is None or np.isnan(lo) or np.isnan(hi)
        empty.append(gone)
        covered.append(False if gone else bool(lo <= true_att <= hi))
        # An unbounded side has no width; averaging it in would report
        # infinity for the cell and hide the finite intervals.
        widths.append(np.nan if gone or not (np.isfinite(lo) and np.isfinite(hi))
                      else hi - lo)
    return (np.asarray(att), np.asarray(pvals), np.asarray(covered),
            np.asarray(widths), np.asarray(empty))


def run() -> dict:
    out: dict = {}
    worst_bias = worst_fpr = worst_cov = 0.0
    worst_width_ratio = 1.0
    for sc in SCENARIOS:
        null_att, null_p, null_cov, _, _ = _cell(sc, 0.0)
        eff_att, eff_p, eff_cov, eff_w, eff_empty = _cell(sc, EFFECT)

        bias_pp = float(null_att.mean() * 100.0)      # truth is 0 in the null cell
        fpr = float(np.mean(null_p <= ALPHA))
        power = float(np.mean(eff_p <= ALPHA))
        coverage = float(eff_cov.mean())
        width = float(np.nanmean(eff_w))
        ref = RECAST_GEOLIFT[sc]

        out[f"{sc}_bias_pp"] = bias_pp
        out[f"{sc}_fpr"] = fpr
        out[f"{sc}_power"] = power
        out[f"{sc}_coverage"] = coverage
        out[f"{sc}_coverage_null"] = float(null_cov.mean())
        out[f"{sc}_ci_width"] = width
        out[f"{sc}_empty_rate"] = float(eff_empty.mean())
        worst_bias = max(worst_bias, abs(bias_pp - ref["bias_pp"]))
        worst_fpr = max(worst_fpr, abs(fpr - ref["fpr"]))
        worst_cov = max(worst_cov, abs(coverage - ref["coverage"]))
        r = width / ref["ci_width"]
        worst_width_ratio = max(worst_width_ratio, max(r, 1.0 / r))

        # The effect cell must recover the injected lift, not merely detect it.
        out[f"{sc}_att_pct_at_effect"] = float(eff_att.mean() * 100.0)

    out["max_bias_gap_vs_recast_pp"] = worst_bias
    out["max_fpr_gap_vs_recast"] = worst_fpr
    out["max_coverage_gap_vs_recast"] = worst_cov
    out["max_ci_width_ratio_vs_recast"] = worst_width_ratio
    # The signature: A2 must stand well clear of the other three, or the
    # convex-hull failure the scenario exists to produce is not being produced.
    out["a2_bias_excess_pp"] = float(
        out["A2_bias_pp"] - max(out[f"{s}_bias_pp"] for s in ("A1", "A3", "A4")))
    return out


# Tolerances.
#
# The case is fully seeded -- panels are seeds 0..N_REPS-1 and the conformal
# permutation seed is fixed -- so a re-run reproduces every value exactly. The
# bands therefore guard against a real change in behaviour and against solver
# drift on another BLAS, not against Monte-Carlo noise between runs.
#
# Sizing them still needs the noise, because the pinned values are themselves
# Monte-Carlo averages over N_REPS = 120 panels against the study's 1000. The
# ATT's cross-panel standard deviation is 7.2-9.3 percentage points on this DGP,
# so a cell's bias carries a standard error near 0.7 pp and a rate near 0.05
# carries about 0.02. The bands are roughly two of each.
#
# Every scenario's bias agrees with the study's published GeoLift value within
# |z| <= 1.1 -- A1 +0.4, A2 +0.4, A3 -0.4, A4 +1.1 -- so the gap rows are pinned
# as agreement and not as a tolerated discrepancy.
#
# `a2_bias_excess_pp` pins the pattern, not a level: the outlier
# scenario's bias must stand clear of the largest of the other three, or the
# convex-hull failure the scenario exists to produce is not being produced.
# 1.2 is roughly the floor at which it is still unambiguous.
#
# Power is low in every cell -- 0.05 to 0.18 against a 7.5% lift -- and the
# study's GeoLift is equally low (0.043 to 0.107). That is the design being
# under-powered at this effect and panel length, reproduced, not a defect.
#
# Coverage carries a standard error near 0.025 at this count, so its bands are
# 0.06. The width bands are 12% of each pinned value, which is the sharper
# comparison of the two: coverage is a rate and two procedures can both hit
# 0.93 with intervals of very different length, while the width says the two
# tests reject in the same places. Three of the four agree to 2%.
#
# `A4_ci_width` is the one that does not, at 1.14x theirs, and its band is
# proportionally wider. A4 gives the test 30 pre-days against 15 post ones, so
# a third of the block reference set overlaps the post window and the p-value
# floor rises above 1/T; some panels are not bounded at all. Those are excluded
# from the average, and the ones that remain are the wider tail.
#
# `empty_rate` pins that an empty confidence set stays rare. It is a real
# answer -- the null inverted is a constant effect, and the injected lift is
# multiplicative -- but it is also what a search failure would look like, so it
# is bounded, not left unwatched.
EXPECTED = {
    # bias against a truth of zero, percentage points
    "A1_bias_pp": (0.480, 1.5),
    "A2_bias_pp": (3.511, 1.5),
    "A3_bias_pp": (0.733, 1.5),
    "A4_bias_pp": (0.566, 1.8),
    # false-positive rate at alpha = 0.05
    "A1_fpr": (0.0667, 0.045),
    "A2_fpr": (0.0417, 0.045),
    "A3_fpr": (0.0333, 0.045),
    "A4_fpr": (0.0500, 0.045),
    # detection rate at the study's 7.5% lift
    "A1_power": (0.1750, 0.090),
    "A2_power": (0.1250, 0.090),
    "A3_power": (0.1167, 0.090),
    "A4_power": (0.0500, 0.090),
    # the injected lift must be recovered, not merely detected (truth 7.5)
    "A1_att_pct_at_effect": (7.980, 1.5),
    "A2_att_pct_at_effect": (11.011, 1.5),
    "A3_att_pct_at_effect": (8.233, 1.5),
    "A4_att_pct_at_effect": (8.066, 1.8),
    # coverage of the exact true ATT, at the 7.5% lift and at the null
    "A1_coverage": (0.8917, 0.06),
    "A2_coverage": (0.9167, 0.06),
    "A3_coverage": (0.9250, 0.06),
    "A4_coverage": (0.9250, 0.06),
    "A1_coverage_null": (0.9333, 0.06),
    "A2_coverage_null": (0.9500, 0.06),
    "A3_coverage_null": (0.9667, 0.06),
    "A4_coverage_null": (0.9500, 0.06),
    # interval width in outcome levels, over the bounded intervals
    "A1_ci_width": (2062.1, 250.0),
    "A2_ci_width": (10182.0, 1250.0),
    "A3_ci_width": (1968.1, 250.0),
    "A4_ci_width": (6338.5, 800.0),
    # an empty confidence set stays rare
    "A1_empty_rate": (0.0667, 0.05),
    "A2_empty_rate": (0.0333, 0.05),
    "A3_empty_rate": (0.0250, 0.05),
    "A4_empty_rate": (0.0333, 0.05),
    # agreement with the published GeoLift row
    "max_bias_gap_vs_recast_pp": (0.956, 1.5),
    "max_fpr_gap_vs_recast": (0.0207, 0.045),
    "max_coverage_gap_vs_recast": (0.0303, 0.06),
    "max_ci_width_ratio_vs_recast": (1.137, 0.25),
    "a2_bias_excess_pp": (2.778, 1.2),
}
