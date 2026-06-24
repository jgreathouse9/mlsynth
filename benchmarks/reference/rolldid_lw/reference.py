#!/usr/bin/env python3
"""Live captured run of the ``lwdid`` package on the ROLLDID empirical cases.

Runs Lee & Wooldridge's own Python implementation of the rolling-transformation
DiD estimator (``lwdid`` on PyPI; ``lwdid.lwdid``) on the two empirical panels
mlsynth's ``ROLLDID`` reproduces -- California Proposition 99 (common timing) and
the castle-doctrine laws (staggered adoption) -- for the ``demean`` and
``detrend`` transformations, and prints the common-timing / overall ATTs and
standard errors. mlsynth's ``ROLLDID`` is clean-room from the paper equations and
shares no code with ``lwdid``; this captured run is the independent
cross-validation oracle.

Spec alignment with mlsynth's ``ROLLDID``
-----------------------------------------
* Proposition 99 (common timing). mlsynth's headline ATT is the coefficient on a
  unit-level treatment indicator in an OLS of the transformed post-treatment
  average on a constant and that indicator, with homoskedastic exact-t inference
  (``inference="exact"``). The matching ``lwdid`` call passes the time-invariant
  unit-level ``d`` (California = 1), the ``post`` indicator (year >= 1989), and
  ``vce=None`` (homoskedastic OLS) -- so ``att``/``se_att``/``pvalue`` are the
  exact-t quantities. (``lwdid`` rejects the time-varying W_it that mlsynth keeps
  in its ``treat`` column, so we collapse it to D_i first, exactly the unit-level
  indicator mlsynth's regression uses.)
* Castle (staggered). mlsynth aggregates over cohorts vs. never-treated units
  (eq. 7.18-7.19) into a single OLS; the demean variant uses homoskedastic
  exact-t and the detrend variant uses HC3. The matching ``lwdid`` call uses
  ``gvar`` (first treatment year; never-treated = 0), ``control_group=
  "never_treated"``, ``aggregate="overall"``, and ``vce`` = None / "hc3"
  respectively -- ``att_overall``/``se_overall``.

Prints the ``== REFERENCE VALUES ==`` block that ``generate.py`` parses,
followed by a ``== SESSION INFO ==`` block recording Python / lwdid / numpy /
pandas versions.

Run from the repository root::

    python benchmarks/reference/rolldid_lw/reference.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_BASE = ROOT / "basedata"
_PROP99_TREAT_YEAR = 1989


def _smoking() -> pd.DataFrame:
    d = pd.read_csv(_BASE / "smoking_data.csv")
    d["logcig"] = np.log(d["cigsale"])
    # mlsynth keeps the time-varying W_it in "treat"; lwdid wants the unit-level
    # D_i (time-invariant) -- the same indicator mlsynth's collapsed regression
    # uses. Collapse to per-unit max and supply the post indicator separately.
    d["treat"] = d["Proposition 99"].astype(int)
    d["D_i"] = d.groupby("state")["treat"].transform("max")
    d["post"] = (d["year"] >= _PROP99_TREAT_YEAR).astype(int)
    return d


def _castle() -> pd.DataFrame:
    d = pd.read_csv(_BASE / "castle.csv")
    # gvar: first treatment year; never-treated -> 0.
    d["gvar"] = d["effyear"].fillna(0).astype(int)
    return d


def _run() -> dict:
    import lwdid

    vals: dict = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        sm = _smoking()
        for roll in ("demean", "detrend"):
            r = lwdid.lwdid(sm, y="logcig", d="D_i", ivar="state", tvar="year",
                            post="post", rolling=roll, vce=None)
            vals[f"prop99_{roll}_att"] = float(r.att)
            vals[f"prop99_{roll}_se"] = float(r.se_att)
        # detrend exact-t two-sided p-value (homoskedastic), matching mlsynth's
        # inference="exact".
        rd = lwdid.lwdid(sm, y="logcig", d="D_i", ivar="state", tvar="year",
                         post="post", rolling="detrend", vce=None)
        vals["prop99_detrend_p"] = float(rd.pvalue)

        ca = _castle()
        for roll, vce in (("demean", None), ("detrend", "hc3")):
            r = lwdid.lwdid(ca, y="l_homicide", ivar="state", tvar="year",
                            gvar="gvar", rolling=roll,
                            control_group="never_treated", aggregate="overall",
                            vce=vce)
            vals[f"castle_{roll}_att"] = float(r.att_overall)
            vals[f"castle_{roll}_se"] = float(r.se_overall)
    return vals


def main() -> int:
    vals = _run()

    print("== REFERENCE VALUES ==")
    for k in ("prop99_demean_att", "prop99_demean_se",
              "prop99_detrend_att", "prop99_detrend_se", "prop99_detrend_p",
              "castle_demean_att", "castle_demean_se",
              "castle_detrend_att", "castle_detrend_se"):
        print(f"{k}\t{vals[k]:.6f}")

    print("== SESSION INFO ==")
    import platform

    import lwdid
    print(f"python {platform.python_version()}")
    print(f"lwdid {lwdid.__version__}")
    print(f"numpy {np.__version__}")
    print(f"pandas {pd.__version__}")
    print("reference: lwdid (Lee & Wooldridge DiD), PyPI lwdid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
