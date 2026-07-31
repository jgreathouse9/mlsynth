"""Shared panel loading and estimator calls for the de Brabander et al. (2025) cases.

Both Brexit cases -- the headline treatment effects (Table 1) and the in-sample
placebo sweep (Table 7) -- run the same seven estimators on the same panel and
read the same quantity off each. That shared machinery lives here so the two
case modules differ only in what they vary and what they assert.

The panel is ``benchmarks/reference/brabander_sdid_match/brexit_panel.csv``,
built from the authors' ``brexit_data_raw_eo_nov2018.mat`` by
``benchmarks/reference/brabander_sdid_match/reference.R``: log real GDP for 24
countries (23 donors + the UK) over 100 quarters, index ``t = 141..240``, with
the twelve countries the authors drop for missing data already removed. The
treatment column marks the UK from ``t = 227`` (2016:Q3).

One convention decides every SDID number here and is easy to get wrong. The
paper's SDID estimator subtracts the lambda-weighted pre-treatment gap

.. math:: \\hat\\tau = y_{1,T} - y_{0,T}'\\omega
          - \\lambda'(Y_{1,pre} - Y_{0,pre}\\omega),

which is mlsynth's ``intercept_adjust=True`` counterfactual. Reading the
default (un-adjusted) counterfactual instead compares a different estimator to
the paper's number -- on Table 1 that is worth about 0.05 percentage points,
enough to look like a small disagreement rather than the wrong quantity.
"""
from __future__ import annotations

import os
import warnings

import numpy as np

_PANEL = os.path.join(os.path.dirname(__file__), "..", "reference",
                      "brabander_sdid_match", "brexit_panel.csv")
UK = "United Kingdom"
TREAT_T = 227          # 2016:Q3
FIRST_T = 141

# Panel-data interface shared by every estimator below.
_BASE = dict(outcome="y", treat="treat", unitid="country", time="t",
             display_graphs=False)
# zeta.omega = 0 and the lambda-corrected counterfactual: the paper's SDID.
_SDID = dict(_BASE, vce="noinference", zeta=0.0, intercept_adjust=True)


def load_panel():
    import pandas as pd

    from benchmarks.compare import BenchmarkSkipped
    if not os.path.exists(_PANEL):  # pragma: no cover - the panel is committed
        raise BenchmarkSkipped(f"missing {_PANEL}")
    return pd.read_csv(_PANEL)


def retreat(df, rows, treat_from):
    """Panel restricted to `rows`, with the UK treated from `treat_from` on.

    The placebo sweep needs the treatment moved to a pre-Brexit quarter, so the
    treatment column is rebuilt rather than filtered -- filtering alone would
    leave the real 2016:Q3 flag in place on the longer windows.
    """
    d = df[df.t.isin(list(rows))].copy()
    d["treat"] = ((d.country == UK) & (d.t >= treat_from)).astype(int)
    return d


def gap_at(res, t):
    """Treated minus counterfactual at quarter `t`, off the standardized series.

    Read through ``res.time_series`` rather than any estimator-private field, so
    a result-contract change fails this loudly instead of silently.
    """
    ts = res.time_series
    periods = np.asarray(ts.time_periods).ravel().tolist()
    gaps = np.asarray(ts.estimated_gap, dtype=float).ravel().tolist()
    return float(dict(zip(periods, gaps))[t])


def loss_pct(res, t):
    """The paper's reported quantity: counterfactual minus the UK, in percent.

    The outcome is log real GDP, so the difference in logs times 100 is the
    percentage shortfall the paper tabulates as a loss (hence the sign flip).
    """
    return -100.0 * gap_at(res, t)


def _fit(cls, df, **cfg):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cls({"df": df, **cfg}).fit()


def fit_sc(df):
    """Classic simplex SC -- TSSC's ``SC`` variant, inference switched off."""
    from mlsynth import TSSC

    return _fit(TSSC, df, **_BASE, method="SC", inference=False)


def fit_dsc(df):
    """Demeaned SC (Ferman-Pinto) -- TSSC's ``MSCa`` variant.

    The paper's DSC adds an intercept to the SC weight problem and subtracts the
    mean pre-treatment gap; that is exactly the ``MSCa`` variant, so no manual
    correction term is applied on top of it.
    """
    from mlsynth import TSSC

    return _fit(TSSC, df, **_BASE, method="MSCa", inference=False)


def fit_sdid(df):
    from mlsynth import SDID

    return _fit(SDID, df, **_SDID)


def fit_masc(df, min_preperiods):
    """MASC with the paper's tuning grid.

    ``min_preperiods`` is the smallest cross-validation fold, so the fold count
    is ``T_pre - min_preperiods``; the paper's table notes state the fold count,
    which is what the published numbers use (see each case for the value).
    """
    from mlsynth import MASC

    return _fit(MASC, df, **_BASE, m_grid=list(range(1, 11)),
                min_preperiods=min_preperiods)


def fit_ascm(df):
    """Ridge-augmented SCM -- the paper's ASCM, augsynth's ``progfunc="Ridge"``."""
    from mlsynth import VanillaSC

    return _fit(VanillaSC, df, **_BASE, augment="ridge", inference=False)


def sdid_out_of_window_loss(res, wide, donors, n_pre, t):
    """SDID case (i): weights fit on one window, effect read outside it.

    Case (i) stops the panel one quarter after treatment but still reports the
    effect at 2018:Q4 / 2019:Q4, so the quantity lies beyond the fitted series
    and no time-indexed accessor can supply it. It is rebuilt from the
    standardized ``donor_weights`` plus the cohort's time weights, which is the
    only part of this file that reaches past ``time_series``.
    """
    w = np.array([res.donor_weights[c] for c in donors], dtype=float)
    lam = np.asarray(next(iter(res.cohorts.values())).time_weights, dtype=float)
    pre = wide.index[:n_pre]
    correction = float(lam @ (wide.loc[pre, UK].to_numpy()
                              - wide.loc[pre, donors].to_numpy() @ w))
    effect = float(wide.loc[t, UK] - wide.loc[t, donors].to_numpy() @ w
                   - correction)
    return -100.0 * effect
