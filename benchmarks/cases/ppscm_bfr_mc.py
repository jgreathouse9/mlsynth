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
        sim = simulate_bfr_panel(design=design, n_units=40, n_periods=39,
                                 adoption_times=(14, 20, 26), seed=base_seed + i)
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


# PROVISIONAL -- the values below are read from the paper's Figure 8 / B.3, not yet
# from a run of this case. They are being calibrated against what mlsynth actually
# produces, and any cell that disagrees with the paper's ordering will be recorded as a
# finding rather than fitted away.
#
# Deterministic (seeded). Tolerances absorb binomial Monte Carlo noise: at M=40 a
# coverage SE is ~sqrt(.95*.05/40) ~ 0.035, so a +/-0.12 window is about three SEs.
# The facts reproduced are the paper's ordering, not its cells, which are not
# recoverable from the paper (no replication archive, no stated replication count,
# unreported calibration parameters).
#
#   * the sharp null means PPSCM should be essentially unbiased on every design;
#   * the wild bootstrap covers near nominal under two-way fixed effects, the case
#     with no bias from inexact fit (paper: 93.7);
#   * once factor structure or serial dependence is present the bootstrap does not
#     under-cover -- the paper reports it turning conservative (95.9 and 97.2) while
#     the jackknife falls away (88.5 and 89.3), so the bootstrap covers at least as
#     much as the jackknife on those two designs;
#   * the cumulative conformal band covers near its 90% nominal level, and every band
#     it reports is finite at this panel shape (the windows are there to support it).
EXPECTED = {
    "twfe_bootstrap_bias": (0.0, 0.20),
    "factor_bootstrap_bias": (0.0, 0.30),
    "ar_bootstrap_bias": (0.0, 0.30),

    "twfe_bootstrap_coverage": (0.94, 0.12),
    "factor_bootstrap_coverage": (0.96, 0.12),
    "ar_bootstrap_coverage": (0.97, 0.12),

    "factor_boot_ge_jack": (1.0, 0.0),
    "ar_boot_ge_jack": (1.0, 0.0),

    "cumulative_coverage": (0.90, 0.12),
    "cumulative_bands_infinite": (0.0, 0.0),
}
