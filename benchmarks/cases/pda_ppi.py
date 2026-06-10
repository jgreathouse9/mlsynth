"""PDA Path-A: Shi & Wang (2024) China PPI / real-estate regulation (L2-relaxation).

Reproduces the headline single-treated-unit application of the L2-relaxation PDA
paper (Section 6.1): the effect of China's 2020-21 real-estate tightening (the
"Three Red Lines" policy) on China's **monthly YoY PPI growth rate**, using
``l2`` against ``basedata/china_ppi_long.csv`` (treated ``China`` PPI + 64
country-control PPI series; pre-period before Aug 2020, T1=115; post-period from
Jun 2021, T2=43; the Aug-2020-May-2021 policy-rollout gap is excluded).

The series are **standardised** before solving (``l2_standardize=True``, the
default, matching the authors' released ``L2relax``). mlsynth recovers the
paper's finding -- a large, significant negative PPI effect:

  =====================  ===============  ========================
  Quantity               mlsynth l2       Shi & Wang (Sec. 6.1)
  =====================  ===============  ========================
  monthly ATE            -5.95%           -6.40%
  t-statistic            -4.48            -3.61  (p = 0.0003)
  =====================  ===============  ========================

Two deliberate method differences, both documented:

* **Cross-validation.** mlsynth tunes ``tau`` by **time-respecting** sequential
  out-of-sample validation (fit on the earlier window, validate on the recent
  tail). The authors' ``L2relax.CV`` uses a 5-block K-fold that trains on both
  past *and* future of each validation block -- it leaks future information into
  the counterfactual fit. The sequential CV selects slightly less regularisation,
  giving -5.95% rather than the paper's -6.40% (with their leaky CV mlsynth
  reproduces -6.40% exactly). The sign, magnitude, and 1%-significance agree.
* **Long-run variance.** mlsynth's Newey-West bandwidth (``newey_west_lag``)
  differs from R ``sandwich``'s automatic one, so the t-statistic (-4.48) differs
  from the paper's -3.61; both reject the zero-ATE null well below 1%.

Path A (scenario 3): the data and method are the authors'; the estimate is
deterministic (convex solve + deterministic CV).
"""
from __future__ import annotations

import os
import warnings

_DATA = os.path.join(os.path.dirname(__file__), "..", "..",
                     "basedata", "china_ppi_long.csv")


def run() -> dict:
    import pandas as pd
    from mlsynth import PDA

    df = pd.read_csv(os.path.abspath(_DATA))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = PDA({
            "df": df, "outcome": "y", "treat": "treat",
            "unitid": "unit", "time": "time",
            "methods": ["l2"], "l2_standardize": True,
            "display_graphs": False,
        }).fit()

    fit = res.fits["l2"]
    return {
        "l2_ate_pct": float(fit.att),
        "l2_abs_tstat": float(abs(fit.att / fit.att_se)),
        "l2_significant_5pct": 1.0 if fit.p_value < 0.05 else 0.0,
    }


# Deterministic (convex L2-relaxation solve + deterministic sequential CV). The
# ATE is pinned to mlsynth's time-respecting-CV value (-5.95%, vs the paper's
# -6.40% under its future-leaking K-fold); the tolerance keeps it close to the
# paper while guarding regressions. The t-stat is pinned loosely (the NW
# bandwidth differs from sandwich's) but must stay well past 5% significance.
EXPECTED = {
    "l2_ate_pct": (-5.948, 0.6),          # paper -6.40 (their leaky 5-block CV)
    "l2_abs_tstat": (4.48, 1.2),          # paper 3.61 (NW-bandwidth difference)
    "l2_significant_5pct": (1.0, 0.0),    # paper p = 0.0003
}
