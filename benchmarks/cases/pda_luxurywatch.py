"""PDA Path-A: Shi & Huang (2023) China luxury-watch import study (fsPDA).

Reproduces the paper's Example 2 / Section 5 empirical result -- the effect of
China's 2012 anti-corruption campaign (the "Eight-Point Policy", effective
January 2013) on the **monthly growth rate** of luxury-watch imports -- using
forward-selected PDA on ``basedata/china_watches_long.csv`` (the treated
``watches`` series plus 87 UN-Comtrade commodity-import growth-rate controls,
Feb 2010 - Dec 2015, 35 pre / 36 post periods).

Run as the authors' own application script does
(``app1_luxury_watch/fsPDA.R``): forward selection **with an intercept**
(``fs_intercept=True``) and the **prewhitened Newey-West** long-run variance for
the post-selection t-test (mlsynth's default). The point estimate and fit match
that script cell-for-cell:

  =====================  ===============  =====================
  Quantity               mlsynth fs       Shi & Huang fsPDA.R
  =====================  ===============  =====================
  selected controls      3                3
  in-sample R-squared    0.777            0.7768
  monthly ATE            -3.09%           -3.09%
  t-statistic            -2.51            -2.458
  =====================  ===============  =====================

The reference side is a live captured run of Shi & Huang's own application
script (``benchmarks/reference/pda_luxurywatch/fsPDA_app.R``, vendored verbatim
from github.com/zhentaoshi/fsPDA, MIT), executed on the same
``china_watches_long.csv`` panel and captured with its provenance pinned -- not
numbers transcribed from the paper. The t-statistic differs by ~0.05 only
because mlsynth's native prewhitened-NW estimator and R's ``sandwich::lrvar``
(which the script calls) differ in their exact bandwidth/recolor internals; both
prewhiten the strongly mean-reverting growth-rate effects (lag-1 autocorrelation
~ -0.45), which a plain Bartlett kernel cannot handle. The *packaged*
``est.fsPDA``, which does *not* prewhiten, instead reports an insignificant
t ~ -1.15 on this panel -- so the application script, which mlsynth ports, is
the faithful reference for the paper's significant result.

Path A (scenario 3): the data is the authors' own; forward selection and the
prewhitened-NW inference are ported from the application script. Deterministic
(greedy selection, closed-form variance).
"""
from __future__ import annotations

import os
import warnings

from benchmarks.reference import reference_value

_DATA = os.path.join(os.path.dirname(__file__), "..", "..",
                     "basedata", "china_watches_long.csv")


def run() -> dict:
    import numpy as np
    import pandas as pd
    from mlsynth import PDA

    df = pd.read_csv(os.path.abspath(_DATA))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = PDA({
            "df": df, "outcome": "y", "treat": "treat",
            "unitid": "unit", "time": "time",
            "methods": ["fs"], "fs_intercept": True,
            "display_graphs": False,
        }).fit()

    fit = res.fits["fs"]
    T0 = int((df[df.unit == "watches"]["treat"] == 0).sum())
    y = df[df.unit == "watches"].sort_values("time")["y"].to_numpy()
    pre_r2 = 1.0 - np.var(fit.gap[:T0]) / np.var(y[:T0])
    t_stat = fit.att / fit.att_se

    return {
        "fs_ate_pct": float(fit.att * 100.0),
        "fs_pre_r2": float(pre_r2),
        "fs_n_controls": float(len(fit.selected_donors)),
        "fs_abs_tstat": float(abs(t_stat)),
        "fs_significant_5pct": 1.0 if fit.p_value < 0.05 else 0.0,
    }


def comparison() -> dict:
    """mlsynth ``PDA(methods=["fs"])`` vs Shi & Huang's own ``fsPDA.R``.

    Pairs mlsynth's forward-selected PDA fit on the luxury-watch panel against
    the live captured run of the authors' application script
    (``benchmarks/reference/pda_luxurywatch/``): the monthly ATE, the
    pre-period R^2, the number of selected controls, and the prewhitened-NW
    t-statistic, side by side.
    """
    import numpy as np
    import pandas as pd
    from mlsynth import PDA

    df = pd.read_csv(os.path.abspath(_DATA))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = PDA({
            "df": df, "outcome": "y", "treat": "treat", "unitid": "unit",
            "time": "time", "methods": ["fs"], "fs_intercept": True,
            "display_graphs": False,
        }).fit()
    fit = res.fits["fs"]
    T0 = int((df[df.unit == "watches"]["treat"] == 0).sum())
    y = df[df.unit == "watches"].sort_values("time")["y"].to_numpy()
    pre_r2 = 1.0 - np.var(fit.gap[:T0]) / np.var(y[:T0])
    pairs = [
        ("ATE_pct", float(fit.att * 100.0), "fs_ate_pct"),
        ("pre_R2", float(pre_r2), "fs_pre_r2"),
        ("n_controls", float(len(fit.selected_donors)), "fs_n_controls"),
        ("abs_t_stat", float(abs(fit.att / fit.att_se)), "fs_abs_tstat"),
    ]
    rows = [{"quantity": q, "mlsynth": round(v, 6),
             "reference": round(reference_value("pda_luxurywatch", k), 6)}
            for q, v, k in pairs]
    cfg = {"outcome": "y", "treat": "treat", "unitid": "unit", "time": "time",
           "methods": ["fs"], "fs_intercept": True}
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "PDA", "config": cfg},
        "reference": {"impl": "Shi & Huang fsPDA application script (zhentaoshi/fsPDA, live run, captured)",
                      "version": "Shi & Huang (2023), J. Econometrics; fsPDA app1_luxury_watch"},
    }


# Deterministic (greedy forward selection + closed-form prewhitened-NW variance).
# Targets are pinned from the live captured fsPDA.R run
# (benchmarks/reference/pda_luxurywatch/) via reference_value. The estimate/fit
# cells match the authors' script exactly; the t-statistic tolerance covers the
# small gap between mlsynth's native prewhitened-NW and R's sandwich::lrvar
# (mlsynth -2.51 vs script -2.458) while still requiring 5% significance.
_lw = lambda k: reference_value("pda_luxurywatch", k)
EXPECTED = {
    "fs_ate_pct": (_lw("fs_ate_pct"), 0.10),
    "fs_pre_r2": (_lw("fs_pre_r2"), 0.02),
    "fs_n_controls": (_lw("fs_n_controls"), 0.0),
    "fs_abs_tstat": (_lw("fs_abs_tstat"), 0.45),
    "fs_significant_5pct": (1.0, 0.0),    # script p = 0.0140
}
