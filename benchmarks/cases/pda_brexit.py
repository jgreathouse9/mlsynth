"""PDA Path-A: Shi & Wang (2024) Brexit study (multiple-treated-units L2-relaxation).

Reproduces the L2-relaxation PDA paper's *multiple-treated-units* application
(Section 6.2): the effect of the 23 June 2016 Brexit referendum on the daily
stock returns of **UK firms**, treated as a cross-section against a pool of
non-UK/non-EU control firms, on ``basedata/brexit_long.csv`` (52 UK treated +
300 control series; 253 pre + 21 post trading days).

Each UK firm's counterfactual is fit by standardised L2-relaxation against the
shared control pool -- all 52 fits run through **one OSQP factorisation**
(:func:`mlsynth.utils.pda_helpers.l2.batch.l2_relax_batch`), since ``Sigma`` is
shared -- and the effects are aggregated into a per-period cross-sectional ATE
with a covariance-based SE (:func:`run_pda_multitreat`).

The headline is the first post-referendum trading day (24 Jun 2016):

  =====================  ===============  ====================
  Quantity (24 Jun 2016) mlsynth          Shi & Wang (Sec. 6.2)
  =====================  ===============  ====================
  UK return ATE          -4.50%           -4.31%
  s.e.                   0.0060           0.0058
  t-statistic            -7.54            -7.39
  =====================  ===============  ====================

The small ATE gap (-4.50 vs -4.31) is mlsynth's time-respecting per-firm CV vs
the paper's future-leaking 5-block K-fold; the strong, highly-significant
negative shock on the day after the vote reproduces. Path A (scenario 3): the
data and method are the authors'; deterministic.
"""
from __future__ import annotations

import os
import warnings

import numpy as np

_DATA = os.path.join(os.path.dirname(__file__), "..", "..",
                     "basedata", "brexit_long.csv")
_GRID = np.exp(np.linspace(np.log(1e-2), np.log(1.0), 8))


def run() -> dict:
    import pandas as pd
    from mlsynth.utils.pda_helpers.multitreat import run_pda_multitreat

    df = pd.read_csv(os.path.abspath(_DATA))
    w = df.pivot(index="time", columns="unit", values="y").sort_index()
    grp = df.drop_duplicates("unit").set_index("unit")["group"]
    uk = [u for u in w.columns if grp[u] == "UK"]
    ct = [u for u in w.columns if grp[u] == "control"]
    treated = df[df.unit == uk[0]].sort_values("time")
    T0 = int((treated["treat"] == 0).sum())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = run_pda_multitreat(w[uk].to_numpy(float), w[ct].to_numpy(float),
                                 T0, _GRID)

    return {
        "brexit_day1_ate_pct": float(res.ate[0] * 100.0),
        "brexit_day1_se": float(res.se),
        "brexit_day1_abs_tstat": float(abs(res.tstat[0])),
        "brexit_day1_significant": 1.0 if res.pvalue[0] < 0.05 else 0.0,
    }


# Deterministic (OSQP solve + deterministic time-respecting CV). The day-1
# (post-referendum) effect is pinned to mlsynth's value (-4.50%, vs the paper's
# -4.31% under its leaky K-fold CV); tolerances keep it close while guarding
# regressions. The test must stay overwhelmingly significant (paper p ~ 1e-13).
EXPECTED = {
    "brexit_day1_ate_pct": (-4.50, 0.6),       # paper -4.31
    "brexit_day1_se": (0.00597, 0.0015),       # paper 0.00583
    "brexit_day1_abs_tstat": (7.54, 1.5),      # paper 7.39
    "brexit_day1_significant": (1.0, 0.0),     # paper p = 1.5e-13
}
