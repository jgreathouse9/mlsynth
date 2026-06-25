"""SPSC Path-A / cross-validation: the Panic of 1907 (paper Example 2).

Park & Tchetgen Tchetgen (2025), Section 5, revisit Fohlin & Lu's Panic of 1907
study: the treated unit is the *average* log stock price of the two trusts the
Panic is hypothesised to have struck -- the Knickerbocker Trust and the Trust
Company of America -- and the donor pool is the trusts conjectured immune. SPSC
views the donor trusts as a single proxy for the treated trusts'
treatment-free price.

This case builds that averaged-treated panel from ``basedata/trust.dta``
(Knickerbocker = ID 34, Trust Co. of America = ID 57; donors = the ``normal``
trusts; the Panic falls after period 229) and cross-checks mlsynth's SPSC --
both the detrended (SPSC-DT) and undetrended (SPSC-NoDT) variants, ridge
``lambda`` fixed at ``10**-2`` for a deterministic live run -- against the
``qkrcks0218/SPSC`` R package run on the identical panel. The ATTs match the
reference value-for-value (SPSC-NoDT -0.813, SPSC-DT -0.804). Path A /
cross-validation (scenario 3). Skips gracefully when ``Rscript`` or the SPSC
clone is unavailable.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

_DATA = os.path.join(os.path.dirname(__file__), "..", "..", "basedata",
                     "trust.dta")
_KNICKERBOCKER, _TCA = 34, 57
_T0 = 229
_LAMBDA = -2.0


def _panel():
    df = pd.read_stata(os.path.abspath(_DATA))
    df = df[df["ID"] != 1]                                   # drop unbalanced unit
    piv = df.pivot_table(index="time", columns="ID", values="prc_log")
    treated = piv[[_KNICKERBOCKER, _TCA]].mean(axis=1).to_numpy()
    donor_ids = sorted(df[df["type"] == "normal"]["ID"].unique())
    W = piv[donor_ids].to_numpy()
    return treated, W, donor_ids


def run() -> dict:
    from benchmarks.reference.clone_spsc import run_reference
    from mlsynth.utils.proximal_helpers.spsc.estimation import estimate_spsc

    y, W, donors = _panel()
    out = {"n_donors": float(len(donors))}
    for detrend, tag in ((False, "nodt"), (True, "dt")):
        ref = run_reference(y, W, _T0, detrend=detrend, att_degree=0,
                            ridge_lambda=_LAMBDA)             # skips if no R
        mls = estimate_spsc(y, W, _T0, detrend=detrend, ridge_lambda=_LAMBDA)
        out[f"{tag}_att"] = float(mls[2])
        out[f"{tag}_att_vs_ref"] = float(abs(mls[2] - ref["effect_path"][0]))
    return out


# Averaged-treated (Knickerbocker + Trust Co. of America) vs the normal-trust
# donor pool, ridge lambda = 10**-2. Validated value-for-value against
# qkrcks0218/SPSC @ 054f1fbb: SPSC-NoDT ATT -0.8129, SPSC-DT ATT -0.8035 --
# mlsynth reproduces both to solver tolerance.
EXPECTED = {
    "n_donors": (48.0, 0.0),
    "nodt_att": (-0.8129, 0.01),
    "dt_att": (-0.8035, 0.01),
    "nodt_att_vs_ref": (0.0, 1e-3),       # bit-for-bit vs the R package
    "dt_att_vs_ref": (0.0, 1e-3),
}
