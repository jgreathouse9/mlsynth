"""Path A + cross-validation: ROLLDID reproduces Lee & Wooldridge (2026) and
matches the ``lwdid`` package -- Prop 99 + castle.

Reproduces both empirical applications of the rolling-transformation DiD paper
("Simple Approaches to Inference with DiD ... Small Cross-Sectional Sample
Sizes") to the reported precision, and cross-validates the common-timing /
staggered-overall point estimates against the authors' own ``lwdid`` Python
package (PyPI ``lwdid``). mlsynth's ``ROLLDID`` is clean-room from the paper
equations and shares no code with ``lwdid``.

Two layers, kept separate:

1. Paper Table 3 / §7.2 (clean-room Path-A reproduction). The ``EXPECTED``
   entries with literal published constants (e.g. demean ATT ``-0.422``) pin
   mlsynth's run against the numbers printed in the paper, at display-rounding
   tolerances.
2. Live ``lwdid`` cross-validation (captured bundle). A ``lwdid.lwdid`` run on
   the same panels and the same demean/detrend + common-timing / staggered
   specs is captured in ``benchmarks/reference/rolldid_lw/`` and pinned via
   :func:`reference_value`. The ``*_xval`` entries in ``EXPECTED`` hold
   mlsynth's run against those captured ``lwdid`` values to ~5e-7 (the capture
   is 6-decimal; tolerance 1e-5). lwdid is now a live captured reference, not a
   skipped black-box oracle. Regenerate with
   ``python benchmarks/reference/generate.py rolldid_lw``.

Spec alignment with ``lwdid``
-----------------------------
* Prop 99 (common timing): mlsynth's headline ATT is the coefficient on a
  unit-level treatment indicator in an OLS of the transformed post-average on a
  constant and that indicator (homoskedastic exact-t). The ``lwdid`` call uses
  the time-invariant ``d`` (California = 1), ``post`` = (year >= 1989), and
  ``vce=None``.
* Castle (staggered): mlsynth aggregates cohorts vs. never-treated (eq.
  7.18-7.19) into one OLS; demean uses exact-t, detrend uses HC3. The ``lwdid``
  call uses ``gvar`` (first treatment year; never-treated = 0),
  ``control_group="never_treated"``, ``aggregate="overall"``, ``vce`` =
  None / "hc3" -- ``att_overall``/``se_overall``. The two independent
  implementations agree to ~5e-7 on every quantity.

Provenance
----------
* California Prop 99: ``basedata/smoking_data.csv`` (Abadie et al. 2010 panel,
  39 states x 1970-2000, California treated 1989), outcome = log per-capita
  cigarette sales. Paper Table 3: demean ATT -0.422 (se 0.121), detrend -0.227
  (se 0.094), detrend exact-p 0.021.
* Castle laws: ``basedata/castle.csv`` (Cunningham 2021, 50 states 2000-2010,
  21 staggered-treated / 29 never-treated), outcome = log homicides. Paper
  §7.2: demean aggregate 0.092 (OLS se 0.057), detrend 0.067 (HC3 se 0.055).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.reference import reference_value

_BASE = Path(__file__).resolve().parents[2] / "basedata"

# Live captured lwdid values (read, not transcribed) from
# benchmarks/reference/rolldid_lw/.
_lw = lambda k: reference_value("rolldid_lw", k)


def _smoking() -> pd.DataFrame:
    d = pd.read_csv(_BASE / "smoking_data.csv")
    d["logcig"] = np.log(d["cigsale"])
    d["treat"] = d["Proposition 99"].astype(int)
    return d


def _castle() -> pd.DataFrame:
    d = pd.read_csv(_BASE / "castle.csv")
    d["W"] = ((d["effyear"].notna()) & (d["year"] >= d["effyear"])).astype(int)
    return d


def _fit_all() -> dict:
    """The shared mlsynth ``ROLLDID`` fits (run() and comparison() both use)."""
    from mlsynth import ROLLDID

    out: dict = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        sm = _smoking()
        dm = ROLLDID({"df": sm, "outcome": "logcig", "treat": "treat",
                      "unitid": "state", "time": "year", "rolling": "demean",
                      "inference": "exact", "display_graphs": False}).fit()
        dt = ROLLDID({"df": sm, "outcome": "logcig", "treat": "treat",
                      "unitid": "state", "time": "year", "rolling": "detrend",
                      "inference": "exact", "display_graphs": False}).fit()
        out["prop99_demean_att"] = float(dm.effects.att)
        out["prop99_demean_se"] = float(dm.inference.standard_error)
        out["prop99_detrend_att"] = float(dt.effects.att)
        out["prop99_detrend_se"] = float(dt.inference.standard_error)
        out["prop99_detrend_exact_p"] = float(dt.inference.p_value)

        ca = _castle()
        cdm = ROLLDID({"df": ca, "outcome": "l_homicide", "treat": "W",
                       "unitid": "state", "time": "year", "rolling": "demean",
                       "inference": "exact", "display_graphs": False}).fit()
        cdt = ROLLDID({"df": ca, "outcome": "l_homicide", "treat": "W",
                       "unitid": "state", "time": "year", "rolling": "detrend",
                       "inference": "hc3", "display_graphs": False}).fit()
        out["castle_demean_att"] = float(cdm.effects.att)
        out["castle_demean_se"] = float(cdm.inference.standard_error)
        out["castle_detrend_att"] = float(cdt.effects.att)
        out["castle_detrend_hc3_se"] = float(cdt.inference.standard_error)
    return out


def run() -> dict:
    out = _fit_all()
    # Mirror the headline mlsynth quantities under *_xval keys so the live
    # lwdid cross-validation pins are separate, named rows in EXPECTED. Same
    # mlsynth computation -- no recomputation, no second spec.
    out["prop99_demean_att_xval"] = out["prop99_demean_att"]
    out["prop99_demean_se_xval"] = out["prop99_demean_se"]
    out["prop99_detrend_att_xval"] = out["prop99_detrend_att"]
    out["prop99_detrend_se_xval"] = out["prop99_detrend_se"]
    out["prop99_detrend_p_xval"] = out["prop99_detrend_exact_p"]
    out["castle_demean_att_xval"] = out["castle_demean_att"]
    out["castle_demean_se_xval"] = out["castle_demean_se"]
    out["castle_detrend_att_xval"] = out["castle_detrend_att"]
    out["castle_detrend_se_xval"] = out["castle_detrend_hc3_se"]
    return out


def comparison() -> dict:
    """mlsynth ``ROLLDID`` vs ``lwdid``, quantity by quantity.

    Lays mlsynth's rolling-DiD fits against the authors' own ``lwdid`` package on
    the same Prop-99 (common timing) and castle (staggered) panels, same
    demean/detrend transformations and inference. The reference side is the live
    captured ``lwdid.lwdid`` run in ``benchmarks/reference/rolldid_lw/`` (not
    transcribed). Returns ``{"rows": [...], "mlsynth_call": {...},
    "reference": {...}}``.
    """
    out = _fit_all()
    pairs = [
        ("prop99 demean ATT", "prop99_demean_att", "prop99_demean_att"),
        ("prop99 demean SE", "prop99_demean_se", "prop99_demean_se"),
        ("prop99 detrend ATT", "prop99_detrend_att", "prop99_detrend_att"),
        ("prop99 detrend SE", "prop99_detrend_se", "prop99_detrend_se"),
        ("prop99 detrend p", "prop99_detrend_exact_p", "prop99_detrend_p"),
        ("castle demean ATT", "castle_demean_att", "castle_demean_att"),
        ("castle demean SE", "castle_demean_se", "castle_demean_se"),
        ("castle detrend ATT", "castle_detrend_att", "castle_detrend_att"),
        ("castle detrend HC3 SE", "castle_detrend_hc3_se", "castle_detrend_se"),
    ]
    rows = [{"quantity": q,
             "mlsynth": round(out[mk], 6),
             "reference": round(_lw(rk), 6)} for q, mk, rk in pairs]
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "ROLLDID",
                         "config": {"rolling": "demean/detrend",
                                    "inference": "exact (prop99/castle-demean), "
                                                 "hc3 (castle-detrend)"}},
        "reference": {"impl": "lwdid.lwdid (Lee & Wooldridge DiD, live run, "
                              "captured): prop99 common-timing (d, post, "
                              "vce=None); castle staggered (gvar, "
                              "control_group='never_treated', "
                              "aggregate='overall', vce=None/hc3)",
                      "version": "lwdid 0.2.3 "
                                 "(benchmarks/reference/rolldid_lw/)"},
    }


# --- Layer 1: paper Table 3 / §7.2 (literal published constants) ----------
# Clean-room Path-A reproduction against the *published* numbers, display-
# rounding tolerances.
EXPECTED = {
    "prop99_demean_att": (-0.422, 5e-3),
    "prop99_demean_se": (0.121, 5e-3),
    "prop99_detrend_att": (-0.227, 5e-3),
    "prop99_detrend_se": (0.094, 5e-3),
    "prop99_detrend_exact_p": (0.021, 2e-3),
    "castle_demean_att": (0.092, 3e-3),
    "castle_demean_se": (0.057, 3e-3),
    "castle_detrend_att": (0.067, 3e-3),
    "castle_detrend_hc3_se": (0.055, 3e-3),
}

# --- Layer 2: live lwdid cross-validation (captured bundle) ---------------
# Independent implementation of the same transformations + specs. The two agree
# to ~5e-7 on every quantity (the capture is 6-decimal; tolerance 1e-5 reflects
# genuine agreement, not slack). Pinned via reference_value so EXPECTED and the
# captured run are the same object and cannot silently drift.
EXPECTED.update({
    "prop99_demean_att_xval": (_lw("prop99_demean_att"), 1e-5),
    "prop99_demean_se_xval": (_lw("prop99_demean_se"), 1e-5),
    "prop99_detrend_att_xval": (_lw("prop99_detrend_att"), 1e-5),
    "prop99_detrend_se_xval": (_lw("prop99_detrend_se"), 1e-5),
    "prop99_detrend_p_xval": (_lw("prop99_detrend_p"), 1e-5),
    "castle_demean_att_xval": (_lw("castle_demean_att"), 1e-5),
    "castle_demean_se_xval": (_lw("castle_demean_se"), 1e-5),
    "castle_detrend_att_xval": (_lw("castle_detrend_att"), 1e-5),
    "castle_detrend_se_xval": (_lw("castle_detrend_se"), 1e-5),
})
