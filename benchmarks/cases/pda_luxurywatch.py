"""PDA Path-A: Shi & Huang (2023) China luxury-watch import study (fsPDA).

Reproduces the paper's Example 2 / Section 5 empirical result -- the effect of
China's 2012 anti-corruption campaign (the "Eight-Point Policy", effective
January 2013) on the **monthly growth rate** of luxury-watch imports -- using
forward-selected PDA on ``basedata/china_watches_long.csv`` (the treated
``watches`` series plus 87 UN-Comtrade commodity-import growth-rate controls,
Feb 2010 - Dec 2015, 35 pre / 36 post periods).

Run as the released ``fsPDA`` package's application script does
(``app1_luxury_watch/fsPDA.R``): forward selection **with an intercept**
(``fs_intercept=True``) and the **prewhitened Newey-West** long-run variance for
the post-selection t-test (mlsynth's default). The point estimate and fit match
the paper cell-for-cell:

  =====================  ===============  =====================
  Quantity               mlsynth fs       Shi & Huang (Sec. 5)
  =====================  ===============  =====================
  selected controls      3                3
  in-sample R-squared    0.777            0.7785
  monthly ATE            -3.09%           -3.09%
  t-statistic            -2.51            -2.457
  =====================  ===============  =====================

The t-statistic differs by ~0.06 only because mlsynth's native prewhitened-NW
estimator and R's ``sandwich::lrvar`` differ in their exact bandwidth/recolor
internals; both prewhiten the strongly mean-reverting growth-rate effects
(lag-1 autocorrelation ~ -0.45), which a plain Bartlett kernel cannot handle --
the released ``est.fsPDA`` package, which does *not* prewhiten, instead reports
an insignificant t ~ -1.15 on this panel.

Path A (scenario 3): the data is the authors' own; forward selection and the
prewhitened-NW inference are ported from the application script. Deterministic
(greedy selection, closed-form variance).
"""
from __future__ import annotations

import os
import warnings

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


# Deterministic (greedy forward selection + closed-form prewhitened-NW variance).
# The estimate/fit cells match the paper exactly; the t-statistic is pinned with a
# tolerance that covers the small gap to the paper's sandwich::lrvar internals
# (mlsynth -2.51 vs paper -2.457) while still requiring 5% significance.
EXPECTED = {
    "fs_ate_pct": (-3.09, 0.10),          # paper -3.09%
    "fs_pre_r2": (0.777, 0.02),           # paper 0.7785
    "fs_n_controls": (3.0, 0.0),          # paper 3
    "fs_abs_tstat": (2.51, 0.45),         # paper 2.457
    "fs_significant_5pct": (1.0, 0.0),    # paper p = 0.0140
}
