"""SPILLSYNTH (Cao-Dowd) Path-A: Proposition 99 with spillover.

Cross-validates mlsynth's ``SPILLSYNTH`` (``method="cd"``) against the authors'
committed MATLAB output (``jcao0/synthetic-control-spillover``, pinned commit
``60bbebe``) for Cao & Dowd, *"Estimation and Inference for Synthetic Control
Methods with Spillover Effects."* On the 51-unit Proposition-99 panel (50 states
+ DC, pre-period 1970-1988, post 1989-2000) with the authors' 13 declared
spillover-affected states, mlsynth reproduces the committed California
spillover-adjusted ATT path (``spillover.csv``, the ``CA`` row) **to four
decimals**:

  ======  =============  =============
  Year    mlsynth SP     reference
  ======  =============  =============
  1989    +0.0827        +0.0827
  1994    -10.9137       -10.9137
  2000    -15.4901       -15.4900
  ======  =============  =============

The headline finding reproduces: the spillover-adjusted ATT (avg -9.44) is
attenuated relative to vanilla SCM (-10.81), because spillover to neighbouring
states (Nevada, Oregon, DC) inflates the classical estimate -- starkly so in the
first four post years, where SP is near zero (-0.85).

Path A (scenario 3): the data and reference are the authors'; cross-validation
is mandatory and done here. The case **skips gracefully** when the reference
clone is unavailable.
"""
from __future__ import annotations

import os
import warnings

import numpy as np

_DATA = os.path.join(os.path.dirname(__file__), "..", "..", "basedata",
                     "prop99_with_dc.csv")

# Cao-Dowd Section 5, footnote 5: states flagged as exposed to Prop-99 spillover.
_AFFECTED = [
    "Alaska", "Arizona", "District of Columbia", "Florida", "Hawaii",
    "Massachusetts", "Maryland", "Michigan", "New Jersey", "Nevada",
    "New York", "Oregon", "Washington",
]


def _fit():
    import pandas as pd

    from mlsynth import SPILLSYNTH

    df = pd.read_csv(os.path.abspath(_DATA))
    df = df[(df["year"] >= 1970) & (df["year"] <= 2000)].copy()
    df["treat"] = ((df["state"] == "California") & (df["year"] >= 1989)).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SPILLSYNTH({
            "df": df, "outcome": "cigsale", "treat": "treat",
            "unitid": "state", "time": "year", "method": "cd",
            "affected_units": _AFFECTED, "display_graphs": False,
        }).fit()
    return res


def run() -> dict:
    from benchmarks.reference.clone_spillover import reference_ca_alpha

    ref_ca = reference_ca_alpha()             # skips if reference unavailable
    res = _fit()
    ca = np.asarray(res.cd.alpha[0, :], dtype=float)   # row 0 is the treated unit

    return {
        "n_post_years": float(ca.size),
        "ca_max_abs_diff": float(np.max(np.abs(ca - ref_ca))),
        "sp_avg": float(ca.mean()),
        "scm_avg": float(res.att_scm),
        "sp_early_avg": float(ca[:4].mean()),       # 1989-1992, near zero
        "sp_attenuates_scm": float(abs(ca.mean()) < abs(res.att_scm)),
    }


def comparison() -> dict:
    """mlsynth SPILLSYNTH vs the committed Cao-Dowd MATLAB reference, year by year.

    Pairs mlsynth's California spillover-adjusted ATT path against the authors'
    committed ``spillover.csv`` (the ``CA`` row), one row per post-treatment year
    1989-2000. Triggers the reference load first so a blocked clone propagates
    ``BenchmarkSkipped`` rather than half-populating the table.
    """
    from benchmarks.reference.clone_spillover import _COMMIT, reference_ca_alpha

    ref_ca = reference_ca_alpha()             # skips if reference unavailable
    res = _fit()
    ca = np.asarray(res.cd.alpha[0, :], dtype=float)   # row 0 is the treated unit
    years = list(range(1989, 1989 + ca.size))

    rows = [{"quantity": f"CA_ATT[{yr}]", "mlsynth": round(float(m), 6),
             "reference": round(float(r), 6)}
            for yr, m, r in zip(years, ca, ref_ca)]
    rows.append({"quantity": "CA_ATT[avg]", "mlsynth": round(float(ca.mean()), 6),
                 "reference": round(float(ref_ca.mean()), 6)})
    cfg = {"outcome": "cigsale", "treat": "treat", "unitid": "state",
           "time": "year", "method": "cd", "affected_units": _AFFECTED}
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "SPILLSYNTH", "config": cfg},
        "reference": {"impl": "jcao0/synthetic-control-spillover MATLAB "
                              "spillover.csv (CA row)",
                      "version": f"commit {_COMMIT[:7]}"},
    }


# Deterministic (closed-form SCM/spillover weights, no RNG). ``ca_max_abs_diff``
# pins mlsynth to the committed Cao-Dowd MATLAB ``spillover.csv``; the averages
# pin the paper's headline: spillover attenuates vanilla SCM's effect, with the
# spillover-adjusted ATT near zero in the first four post years.
EXPECTED = {
    "n_post_years": (12.0, 0.0),
    "ca_max_abs_diff": (0.0001, 0.001),     # reproduces spillover.csv to ~4 dp
    "sp_avg": (-9.44, 0.05),                # spillover-adjusted ATT
    "scm_avg": (-10.81, 0.05),              # vanilla SCM (inflated by spillover)
    "sp_early_avg": (-0.85, 0.10),          # 1989-1992 near zero
    "sp_attenuates_scm": (1.0, 0.0),
}
