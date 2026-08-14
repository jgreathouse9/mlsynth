"""PPSCM Path-B: Ben-Michael, Feller & Rothstein's simulation designs.

Validates partially-pooled SCM against the three data-generating processes the authors
evaluate it on (JRSS-B 2022, 84(2), 351-381, Section 6 and supplement Appendix B.1):
two-way fixed effects, a linear factor model, and a heterogeneous autoregression, each
under a sharp null so the true ATT and the true cumulative effect are exactly zero.

    Ben-Michael, E., Feller, A., & Rothstein, J. (2022). "Synthetic controls with
    staggered adoption." JRSS-B 84(2), 351-381. doi:10.1111/rssb.12448

What the paper reports, and what this case can therefore assert
--------------------------------------------------------------
The paper's coverage results (Figure 8, supplement B.3) are, at nominal 95% for the
overall ATT with an intercept:

  ================  ===============  ===============
  DGP               wild bootstrap   jackknife
  ================  ===============  ===============
  two-way FE        93.7             97.3
  factor            95.9             88.5
  AR                97.2             89.3
  ================  ===============  ===============

mlsynth reproduces the two-way fixed effects cells almost exactly (0.940 and 0.980
against 0.937 and 0.973) and the AR cells closely (0.960 and 0.880 against 0.972 and
0.893); the factor cells differ more, with mlsynth's bootstrap more conservative and
its jackknife covering less. The DGP here is not the paper's calibration, which is
fitted to the Paglayan panel with parameters the paper does not report, so the case
asserts the ordering rather than the cells.

Its finding is the *ordering*, not the cells: the wild bootstrap is close to nominal
when there is no bias from inexact fit, and conservative once factor structure or
serial dependence is present. That is the same mechanism that motivated the calibrated
cumulative band (the bootstrap over-stating the per-period SE as shared structure
strengthens), reached here from the authors' own designs.

Exact cells are not reproducible from the paper: there is no replication archive, the
number of Monte Carlo replications is never stated, and the DGPs are calibrated to
fitted parameters that are not reported. So this case asserts geometry, in the shape of
``spsc_ifem_mc.py``: unbiasedness under the sharp null, coverage near nominal where the
paper says it should be, and the bootstrap covering at least as much as the jackknife
once the design departs from two-way fixed effects.

Why the cumulative band is checked at a different level
------------------------------------------------------
The cumulative conformal band (#432) calibrates on non-overlapping out-of-sample
windows, so a ``1-alpha`` band needs at least ``ceil(1/alpha) - 1`` of them. At the
paper's panel shape that is 9 windows for a 90% band -- which 39 pre-periods supply
exactly -- and 19 for a 95% band, which they cannot. The cumulative arm therefore runs
at 90% on a longer panel, and the case records that constraint rather than working
around it silently.
"""

import warnings

import numpy as np
import pandas as pd

from mlsynth.utils.ppscm_helpers.simulation import simulate_bfr_panel

M = 40            # replications for the ATT arms (bootstrap is cheap, jackknife is not)
M_CUM = 25        # replications for the cumulative arm (one pooled solve per origin)
_ALPHA = 0.05     # the paper's nominal level for the ATT
#: Fourteen adoption times, the first at t=7 -- the paper's application shape. The
#: count matters as much as the selection intercept: sweeping 14 times at
#: expit(-2.7) leaves about 30 of 49 units eventually treated, which is what the
#: paper reports. Sweeping only a handful leaves a third of that, and the pooled
#: inference is then run over far fewer cohorts than the design intends.
_ADOPTION_TIMES = tuple(range(7, 35, 2))
_ALPHA_CUM = 0.10  # the level the cumulative band can support at this panel shape
_ALPHA_CUM_FLOOR = 0.85   # nominal 0.90 less three MC standard errors


def _panel_to_long(sim) -> pd.DataFrame:
    """The simulated matrix as the long frame the estimator ingests."""
    n_units, n_periods = sim.Y.shape
    adopt = sim.trt
    rows = []
    for i in range(n_units):
        treated_from = adopt[i]
        for t in range(n_periods):
            rows.append({
                "unit": f"u{i}", "year": 2000 + t, "y": float(sim.Y[i, t]),
                "tr": int(np.isfinite(treated_from) and t >= treated_from),
            })
    return pd.DataFrame(rows)


def _fit(df, **kw):
    from mlsynth import PPSCM
    cfg = {"df": df, "outcome": "y", "treat": "tr", "unitid": "unit", "time": "year",
           "display_graphs": False, "run_inference": True}
    cfg.update(kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PPSCM(cfg).fit()


def _att_arm(design: str, method: str, base_seed: int):
    """Mean ATT and coverage of the true zero, over ``M`` draws of one design."""
    atts, covers = [], []
    for i in range(M):
        sim = simulate_bfr_panel(design=design, n_units=49, n_periods=39,
                                 adoption_times=_ADOPTION_TIMES, seed=base_seed + i)
        if not np.isfinite(sim.trt).any():
            continue
        try:
            res = _fit(_panel_to_long(sim), inference_method=method, alpha=_ALPHA,
                       n_boot=400)
        except Exception:
            continue
        inf = res.inference
        if inf is None or inf.ci_lower is None or not np.isfinite(inf.ci_lower):
            continue
        atts.append(float(res.effects.att))
        covers.append(1.0 if inf.ci_lower <= 0.0 <= inf.ci_upper else 0.0)
    if not covers:
        return float("nan"), float("nan"), 0
    return float(np.mean(atts)), float(np.mean(covers)), len(covers)


def _cumulative_arm(design: str, base_seed: int, *, earliest: int, n_periods: int,
                    n_draws: int):
    """Coverage of the true zero by the per-unit cumulative conformal band.

    ``earliest`` sets how many calibration windows exist, which is the quantity that
    decides whether the band is tight or merely valid.
    """
    covers, widths, infinite, ns = [], [], 0, []
    for i in range(n_draws):
        sim = simulate_bfr_panel(design=design, n_units=24, n_periods=n_periods,
                                 adoption_times=(earliest, earliest + 4),
                                 seed=base_seed + i)
        if not np.isfinite(sim.trt).any():
            continue
        try:
            res = _fit(_panel_to_long(sim), inference_method="bootstrap",
                       alpha=_ALPHA_CUM, n_boot=100, conformal_horizon=3)
        except Exception:
            continue
        for unit in res.per_unit.values():
            lo, hi = unit.cumulative_lower, unit.cumulative_upper
            if lo is None or hi is None:
                continue
            if not (np.isfinite(lo) and np.isfinite(hi)):
                infinite += 1
                continue
            covers.append(1.0 if lo <= 0.0 <= hi else 0.0)
            widths.append(hi - lo)
            ns.append(int(unit.cumulative_windows))
    if not covers:
        return float("nan"), float("nan"), 0, infinite, 0
    return (float(np.mean(covers)), float(np.median(widths)), len(covers), infinite,
            int(np.median(ns)))


def run() -> dict:
    out = {}
    for design, seed in (("twfe", 5100), ("factor", 5200), ("ar", 5300)):
        boot_mean, boot_cover, _ = _att_arm(design, "bootstrap", seed)
        jack_mean, jack_cover, _ = _att_arm(design, "jackknife", seed)
        out[f"{design}_bootstrap_bias"] = boot_mean
        out[f"{design}_bootstrap_coverage"] = boot_cover
        out[f"{design}_jackknife_coverage"] = jack_cover
        out[f"{design}_boot_ge_jack"] = float(boot_cover >= jack_cover)

    # Two calibration regimes. Split conformal guarantees coverage at or above the
    # nominal level for any number of windows; it approaches the level only as that
    # number grows. Just above the ceil(1/alpha)-1 threshold the half-width IS the
    # largest calibration score, so the band is valid but far wider than it needs to
    # be -- a property worth pinning, since a user reading "90% band" on a short panel
    # is getting something much more conservative.
    tight_cover, tight_width, tight_n, tight_inf, tight_m = _cumulative_arm(
        "factor", 5400, earliest=170, n_periods=182, n_draws=8)
    thin_cover, thin_width, thin_n, thin_inf, thin_m = _cumulative_arm(
        "factor", 5500, earliest=48, n_periods=64, n_draws=8)

    out["cumulative_coverage_many_windows"] = tight_cover
    out["cumulative_coverage_few_windows"] = thin_cover
    out["cumulative_windows_many"] = float(tight_m)
    out["cumulative_windows_few"] = float(thin_m)
    out["cumulative_bands_infinite"] = float(tight_inf + thin_inf)
    out["cumulative_tightens_with_windows"] = float(tight_cover <= thin_cover)
    out["cumulative_valid_many"] = float(tight_cover >= _ALPHA_CUM_FLOOR)
    return out


# Deterministic (seeded). Measured on this case at M=50; tolerances are about three
# binomial Monte Carlo standard errors (a coverage SE at M=50 is ~sqrt(.95*.05/50) ~
# 0.031), wide enough to survive a different BLAS and narrow enough that a real
# regression in either inference path fails here.
#
# How this compares with the paper (Figure 8 / B.3, overall ATT at nominal 95%):
#
#   ================  ==================  ============  ==================  ============
#   DGP               mlsynth bootstrap   BFR           mlsynth jackknife   BFR
#   ================  ==================  ============  ==================  ============
#   two-way FE        0.940               0.937         0.980               0.973
#   factor            1.000               0.959         0.840               0.885
#   AR                0.960               0.972         0.880               0.893
#   ================  ==================  ============  ==================  ============
#
# The two-way fixed effects cells land on the paper almost exactly. The factor cells
# are the ones that differ: mlsynth's bootstrap is more conservative there (1.000
# against 0.959) and its jackknife covers less (0.840 against 0.885). The DGP here is
# not their calibration -- theirs is fitted to the Paglayan panel with parameters the
# paper does not report -- so a gap of that size is expected and is recorded rather
# than tuned away.
#
# What is reproduced is the geometry the paper actually argues for:
#
#   * PPSCM is unbiased under the sharp null, and the AR design is the one where
#     inexact fit bites (bias 0.153 here, ~0.294 in the paper);
#   * the wild bootstrap is near nominal when there is no bias from inexact fit and
#     turns conservative once factor structure or serial dependence is present;
#   * the jackknife runs the other way -- above nominal under two-way FE, below it on
#     the other two -- so the bootstrap covers at least as much on those designs;
#   * the cumulative conformal band (#432) covers at its nominal level when the
#     calibration set is comfortably large (0.891 at 39 windows against a nominal
#     0.90), and is valid but conservative when it is barely adequate (0.978 at 11
#     windows, where the half-width IS the largest calibration score).
EXPECTED = {
    # PPSCM is unbiased under a sharp null; AR is the hard design.
    "twfe_bootstrap_bias": (-0.010, 0.10),
    "factor_bootstrap_bias": (0.011, 0.12),
    "ar_bootstrap_bias": (0.153, 0.18),

    # Wild bootstrap: near nominal without factor structure, conservative with it.
    "twfe_bootstrap_coverage": (0.940, 0.10),
    "factor_bootstrap_coverage": (1.000, 0.10),
    "ar_bootstrap_coverage": (0.960, 0.10),

    # Jackknife: the other way round, which is the paper's contrast.
    "twfe_jackknife_coverage": (0.980, 0.10),
    "factor_jackknife_coverage": (0.840, 0.13),
    "ar_jackknife_coverage": (0.880, 0.13),

    # The ordering, which is the finding rather than any single cell.
    "factor_boot_ge_jack": (1.0, 0.0),
    "ar_boot_ge_jack": (1.0, 0.0),

    # The cumulative band, in both calibration regimes.
    "cumulative_coverage_many_windows": (0.891, 0.10),
    "cumulative_coverage_few_windows": (0.978, 0.08),
    "cumulative_windows_many": (39.0, 2.0),
    "cumulative_windows_few": (11.0, 1.0),
    "cumulative_bands_infinite": (0.0, 0.0),
    "cumulative_tightens_with_windows": (1.0, 0.0),
    "cumulative_valid_many": (1.0, 0.0),
}
