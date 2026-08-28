"""Path A benchmark: Li (2024) Forward DiD, Hong Kong GDP empirical result.

Li's headline application uses a confidential retailer panel, but the author
released a public companion replication on the Hsiao, Ching & Wan (2012)
Hong Kong GDP panel (the political/economic integration of Hong Kong with
mainland China). This case reproduces that released result cell by cell:
mlsynth's :class:`~mlsynth.FDID` on ``basedata/HongKong.csv`` against the
ATT / %ATT / pre-period R^2 / selected-control-count produced by the author's
own Forward DiD code.

The reference side is two live captured runs of the author's own released
code (Marketing Science replication package, DOI 10.1287/mksc.2022.0212),
vendored and executed with their provenance pinned (tool version, data
checksum) -- not numbers transcribed from the replication readme.

* ``benchmarks/reference/fdid_hongkong/`` runs her ``Fun_FDID.R``, which
  returns the fit, the R^2 path, the selected controls and the ATT.
* ``benchmarks/reference/fdid_hongkong_matlab/`` runs her ``FDID_Matlab.m``
  under GNU Octave. ``Fun_FDID.R`` computes no inference at all; the MATLAB
  driver computes the standard error, the 95% interval, the p-value and the
  standardised ATT, so this bundle is what pins mlsynth's inference path
  against the author instead of against a reading of Proposition 2.1.

Forward selection is deterministic, so both captured runs are exact re-runs,
they agree with each other on every quantity both compute, and mlsynth
reproduces them (tolerances absorb only the estimator's 3-4 dp display
rounding).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from mlsynth import FDID

from benchmarks.reference import reference_value

# basedata/HongKong.csv lives at the repo root.
_DATA = Path(__file__).resolve().parents[2] / "basedata" / "HongKong.csv"

# The two captured reference bundles: the author's R (fit and selection) and
# her MATLAB run under Octave (inference).
_r = lambda k: reference_value("fdid_hongkong", k)
_m = lambda k: reference_value("fdid_hongkong_matlab", k)


def _fit():
    df = pd.read_csv(_DATA)
    return FDID(
        {
            "df": df,
            "outcome": "GDP",
            "treat": "Integration",
            "unitid": "Country",
            "time": "Time",
            "display_graphs": False,
            "verbose": False,
        }
    ).fit()


# Quantities both reference bundles compute, used to check that the author's
# R and MATLAB code agree with each other before either is used to score
# mlsynth. Disagreement there would mean one bundle had gone stale.
_SHARED = ("fdid_att", "fdid_att_pct", "fdid_r2_pre",
           "did_att", "did_att_pct", "did_r2_pre")


def run() -> dict:
    res = _fit()
    f, d = res.fdid, res.did
    out = {
        "fdid_att": float(f.att),
        "fdid_att_pct": float(f.att_percent),
        "fdid_r2_pre": float(f.r_squared),
        "fdid_n_controls": float(len(f.selected_names)),
        "did_att": float(d.att),
        "did_att_pct": float(d.att_percent),
        "did_r2_pre": float(d.r_squared),
    }
    # The inference path, against FDID_Matlab.m.
    for tag, m in (("fdid", f), ("did", d)):
        out[f"{tag}_se"] = float(m.att_se)
        out[f"{tag}_att_std"] = float(m.satt)
        out[f"{tag}_p_value"] = float(m.p_value)
        out[f"{tag}_ci_low"] = float(m.ci[0])
        out[f"{tag}_ci_high"] = float(m.ci[1])
    out["ref_r_vs_matlab_max_gap"] = max(
        abs(_r(k) - _m(k)) for k in _SHARED)
    return out


def comparison() -> dict:
    """mlsynth ``FDID`` vs Li's own released code, quantity by quantity.

    Pairs the mlsynth Forward-DiD (and conventional-DiD) fit on the Hong Kong
    GDP panel against live captured runs of the author's code: the fit,
    selection and ATT against ``Fun_FDID.R``, and the standard error, interval,
    p-value and standardised ATT against ``FDID_Matlab.m`` under Octave.
    """
    res = _fit()
    f, d = res.fdid, res.did
    # (label, mlsynth value, reference key, which bundle)
    pairs = [
        ("FDID/ATT", float(f.att), "fdid_att", _r),
        ("FDID/%ATT", float(f.att_percent), "fdid_att_pct", _r),
        ("FDID/R2_pre", float(f.r_squared), "fdid_r2_pre", _r),
        ("FDID/n_controls", float(len(f.selected_names)), "fdid_n_controls", _r),
        ("FDID/SE", float(f.att_se), "fdid_se", _m),
        ("FDID/ATT_std", float(f.satt), "fdid_att_std", _m),
        ("FDID/p_value", float(f.p_value), "fdid_p_value", _m),
        ("FDID/CI_low", float(f.ci[0]), "fdid_ci_low", _m),
        ("FDID/CI_high", float(f.ci[1]), "fdid_ci_high", _m),
        ("DID/ATT", float(d.att), "did_att", _r),
        ("DID/%ATT", float(d.att_percent), "did_att_pct", _r),
        ("DID/R2_pre", float(d.r_squared), "did_r2_pre", _r),
        ("DID/SE", float(d.att_se), "did_se", _m),
        ("DID/ATT_std", float(d.satt), "did_att_std", _m),
        ("DID/p_value", float(d.p_value), "did_p_value", _m),
        ("DID/CI_low", float(d.ci[0]), "did_ci_low", _m),
        ("DID/CI_high", float(d.ci[1]), "did_ci_high", _m),
    ]
    rows = [{"quantity": q, "mlsynth": round(v, 6),
             "reference": round(ref(k), 6)}
            for q, v, k, ref in pairs]
    cfg = {"outcome": "GDP", "treat": "Integration", "unitid": "Country",
           "time": "Time"}
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "FDID", "config": cfg},
        "reference": {"impl": "Kathleen T. Li's Fun_FDID.R (fit) and FDID_Matlab.m "
                              "under GNU Octave (inference), MKSC replication, live runs, captured",
                      "version": "Li (2024), Marketing Science, DOI 10.1287/mksc.2022.0212"},
    }


# Forward selection is deterministic (no randomness), so both captured runs are
# exact re-runs. Targets are pinned from those live runs via reference_value;
# tolerances absorb the estimator's display rounding only.
#
# The inference tolerances are half the last place the result object prints:
# att_se, p_value and the interval bounds are rounded to 4 dp (5e-5), the
# standardised ATT to 3 dp (5e-4). The interval carries one further difference,
# far below that: mlsynth uses norm.ppf(0.975) = 1.959964 where the author
# writes 1.96, which moves a bound by 2e-7 at this standard error.
_fd = _r
EXPECTED = {
    # --- fit and selection, against Fun_FDID.R -------------------------------
    "fdid_att": (_r("fdid_att"), 5e-4),
    "fdid_att_pct": (_r("fdid_att_pct"), 0.1),
    "fdid_r2_pre": (_r("fdid_r2_pre"), 2e-3),
    "fdid_n_controls": (_r("fdid_n_controls"), 0.0),
    "did_att": (_r("did_att"), 5e-4),
    "did_att_pct": (_r("did_att_pct"), 0.1),
    "did_r2_pre": (_r("did_r2_pre"), 2e-3),
    # --- inference, against FDID_Matlab.m -----------------------------------
    "fdid_se": (_m("fdid_se"), 5e-5),
    "fdid_att_std": (_m("fdid_att_std"), 5e-4),
    "fdid_p_value": (_m("fdid_p_value"), 5e-5),
    "fdid_ci_low": (_m("fdid_ci_low"), 5e-5),
    "fdid_ci_high": (_m("fdid_ci_high"), 5e-5),
    "did_se": (_m("did_se"), 5e-5),
    "did_att_std": (_m("did_att_std"), 5e-4),
    "did_p_value": (_m("did_p_value"), 5e-5),
    "did_ci_low": (_m("did_ci_low"), 5e-5),
    "did_ci_high": (_m("did_ci_high"), 5e-5),
    # --- the two references against each other ------------------------------
    # Both compute the ATT, %ATT and R^2. They are independent implementations
    # in different languages, so agreement to 1e-6 says neither bundle has gone
    # stale against the panel or against the author's own algorithm.
    "ref_r_vs_matlab_max_gap": (0.0, 1e-6),
}
