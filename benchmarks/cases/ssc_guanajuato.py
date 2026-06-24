"""SSC Path-A: staggered synthetic control on the criminality panel.

Cross-validates mlsynth's ``SSC`` against the authors' committed reference output
(``jcao0/staggered_synthetic_control``, pinned commit ``74e77d4``) for Cao, Lu &
Wu, *"Synthetic Control Inference for Staggered Adoption,"* Section 4
("Intergovernmental coordination and criminality"): the effect of a municipal
police-reform on seven crime / cartel outcomes in Guanajuato (Alcocer 2024),
estimated with staggered adoption over an ``S``-period event window.

mlsynth reproduces the committed reference value-for-value:

* the **event-time ATT** path for all seven outcomes (357 cells) matches
  ``results_ssc.csv`` to :math:`\\sim 10^{-4}` (the short annual cartel outcomes
  to :math:`\\sim 10^{-3}`; the residual is the simplex solver, cvxpy here vs.
  the reference ``fmincon``);
* the **smallest eigenvalue** of the SSC Gram matrix for each outcome matches the
  reference ``Table1_eigenvalue.csv`` (the design-rank diagnostic) to
  :math:`\\sim 10^{-4}`.

Path A (scenario 3): the data and reference are the authors'; cross-validation is
mandatory and done here. The case **skips gracefully** when the reference clone
is unavailable.
"""
from __future__ import annotations

import os
import warnings

import numpy as np

_CRIME = os.path.join(os.path.dirname(__file__), "..", "..", "basedata",
                      "guanajuato_crime_ssc.csv")
_CARTEL = os.path.join(os.path.dirname(__file__), "..", "..", "basedata",
                       "guanajuato_cartel_ssc.csv")


def _fit_all():
    """Fit SSC for the seven outcomes; return (att_rows, eigenvalues)."""
    import pandas as pd

    from mlsynth.estimators.ssc import SSC
    from mlsynth.utils.ssc_helpers.replication import GUANAJUATO_SPEC

    frames = {"crime": pd.read_csv(os.path.abspath(_CRIME)),
              "cartel": pd.read_csv(os.path.abspath(_CARTEL))}
    rows, eig = [], {}
    for outcome, (which, unit, time, treat, window) in GUANAJUATO_SPEC.items():
        df = frames[which]
        if window is not None:
            df = df.query(window)
        df = df[[unit, time, treat, outcome]].copy()
        df[outcome] = pd.to_numeric(df[outcome], errors="coerce")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = SSC({"df": df, "outcome": outcome, "treat": treat,
                       "unitid": unit, "time": time, "inference": False,
                       "display_graphs": False}).fit()
        for e in sorted(res.event_att):
            rows.append({"outcome": outcome, "event_time": e + 1,
                         "att": float(res.event_att[e])})
        eig[outcome] = float(res.metadata["gram_min_eigenvalue"])
    return pd.DataFrame(rows), eig


def run() -> dict:
    import pandas as pd

    from benchmarks.reference.clone_ssc import (
        reference_att, reference_eigenvalues)

    ref_att = reference_att()                 # skips if reference unavailable
    ref_eig = reference_eigenvalues()
    est, eig = _fit_all()

    m = est.merge(ref_att, on=["outcome", "event_time"])
    m["d"] = (m["att"] - m["ref_att"]).abs()
    eig_diff = max(abs(eig[k] - ref_eig[k]) for k in ref_eig)

    return {
        "n_att_cells": float(len(m)),
        "att_max_abs_diff": float(m["d"].max()),
        "att_max_abs_diff_rate_outcomes": float(
            m[~m["outcome"].isin(["co_num", "presence_strength", "war"])]["d"].max()),
        "eig_max_abs_diff": float(eig_diff),
        # headline cells (homicide ATT^e_1, the paper's lead estimate)
        "hom_att_e1": float(est.query("outcome=='hom_all_rate' and event_time==1")["att"].iloc[0]),
        "hom_min_eig": eig["hom_all_rate"],
    }


def comparison() -> dict:
    """mlsynth ``SSC`` vs the committed jcao0 reference, quantity by quantity.

    Pairs every event-time ATT cell (``att[<outcome>/e<k>]``, 1-based event time,
    357 cells across the seven outcomes) and every per-outcome SSC Gram
    min-eigenvalue (``min_eig[<outcome>]``) against the authors' committed output
    (``results_ssc.csv`` / ``Table1_eigenvalue.csv``). Loads the reference first
    so a blocked clone propagates ``BenchmarkSkipped`` before mlsynth is run.
    """
    from benchmarks.reference.clone_ssc import (
        _COMMIT, reference_att, reference_eigenvalues)

    ref_att = reference_att()                 # skips if reference unavailable
    ref_eig = reference_eigenvalues()
    est, eig = _fit_all()

    m = est.merge(ref_att, on=["outcome", "event_time"])
    rows = [{"quantity": f"att[{r.outcome}/e{int(r.event_time)}]",
             "mlsynth": round(float(r.att), 6),
             "reference": round(float(r.ref_att), 6)} for r in m.itertuples()]
    for outcome in ref_eig:
        rows.append({"quantity": f"min_eig[{outcome}]",
                     "mlsynth": round(eig[outcome], 6),
                     "reference": round(ref_eig[outcome], 6)})
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "SSC",
                         "config": {"inference": False, "display_graphs": False}},
        "reference": {"impl": "jcao0/staggered_synthetic_control "
                              "(committed results_ssc.csv / Table1_eigenvalue.csv)",
                      "version": f"github jcao0/staggered_synthetic_control @ {_COMMIT[:7]}"},
    }


# Deterministic (closed-form simplex weights, no RNG). The ``*_diff`` cells pin
# mlsynth to the committed jcao0 reference; the headline cells pin the paper's
# lead homicide estimate / design diagnostic as a regression guard.
EXPECTED = {
    "n_att_cells": (357.0, 0.0),
    "att_max_abs_diff": (0.001, 0.0015),          # cartel co_num worst cell
    "att_max_abs_diff_rate_outcomes": (0.0002, 0.0004),  # long-pre rate outcomes
    "eig_max_abs_diff": (0.0006, 0.0008),         # Gram min-eigenvalue diagnostic
    "hom_att_e1": (0.0743, 0.002),                # paper / reference 0.0743
    "hom_min_eig": (0.5711, 0.002),               # reference Table 1 (a)
}
