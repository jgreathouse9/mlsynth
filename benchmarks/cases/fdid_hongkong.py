"""Path A benchmark: Li (2024) Forward DiD, Hong Kong GDP empirical result.

Li's headline application uses a confidential retailer panel, but the author
released a public companion replication on the Hsiao, Ching & Wan (2012)
Hong Kong GDP panel (the political/economic integration of Hong Kong with
mainland China). This case reproduces that released result cell by cell:
mlsynth's :class:`~mlsynth.FDID` on ``basedata/HongKong.csv`` against the
ATT / %ATT / pre-period R^2 / selected-control-count produced by the author's
own Forward DiD code.

The reference side is a live captured run of Kathleen Li's released
``Fun_FDID.R`` (the Marketing Science replication package, DOI
10.1287/mksc.2022.0212), vendored and executed under
``benchmarks/reference/fdid_hongkong/`` with its provenance pinned (R version,
data checksum) -- not numbers transcribed from the replication readme. Forward
selection is deterministic, so the captured run reproduces the readme's printed
output to the digit, and mlsynth's FDID reproduces the captured run (tolerances
absorb only the estimator's 3-4 dp display rounding).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from mlsynth import FDID

from benchmarks.reference import reference_value

# basedata/HongKong.csv lives at the repo root.
_DATA = Path(__file__).resolve().parents[2] / "basedata" / "HongKong.csv"


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


def run() -> dict:
    res = _fit()
    f, d = res.fdid, res.did
    return {
        "fdid_att": float(f.att),
        "fdid_att_pct": float(f.att_percent),
        "fdid_r2_pre": float(f.r_squared),
        "fdid_n_controls": float(len(f.selected_names)),
        "did_att": float(d.att),
        "did_att_pct": float(d.att_percent),
        "did_r2_pre": float(d.r_squared),
    }


def comparison() -> dict:
    """mlsynth ``FDID`` vs Li's released ``Fun_FDID.R``, quantity by quantity.

    Pairs the mlsynth Forward-DiD (and conventional-DiD) fit on the Hong Kong
    GDP panel against the live captured run of the author's own code
    (``benchmarks/reference/fdid_hongkong/``): the ATT, %ATT, pre-period R^2 and
    the forward-selected control count, for both estimators.
    """
    res = _fit()
    f, d = res.fdid, res.did
    pairs = [
        ("FDID/ATT", float(f.att), "fdid_att"),
        ("FDID/%ATT", float(f.att_percent), "fdid_att_pct"),
        ("FDID/R2_pre", float(f.r_squared), "fdid_r2_pre"),
        ("FDID/n_controls", float(len(f.selected_names)), "fdid_n_controls"),
        ("DID/ATT", float(d.att), "did_att"),
        ("DID/%ATT", float(d.att_percent), "did_att_pct"),
        ("DID/R2_pre", float(d.r_squared), "did_r2_pre"),
    ]
    rows = [{"quantity": q, "mlsynth": round(v, 6),
             "reference": round(reference_value("fdid_hongkong", k), 6)}
            for q, v, k in pairs]
    cfg = {"outcome": "GDP", "treat": "Integration", "unitid": "Country",
           "time": "Time"}
    return {
        "rows": rows,
        "mlsynth_call": {"estimator": "FDID", "config": cfg},
        "reference": {"impl": "Kathleen T. Li's Fun_FDID.R (MKSC replication, live run, captured)",
                      "version": "Li (2024), Marketing Science, DOI 10.1287/mksc.2022.0212"},
    }


# Forward selection is deterministic (no randomness), so the captured FDID.R run
# is an exact re-run. Targets are pinned from that live run
# (benchmarks/reference/fdid_hongkong/) via reference_value; tolerances absorb
# the estimator's 3-4 dp display rounding only.
_fd = lambda k: reference_value("fdid_hongkong", k)
EXPECTED = {
    "fdid_att": (_fd("fdid_att"), 5e-4),
    "fdid_att_pct": (_fd("fdid_att_pct"), 0.1),
    "fdid_r2_pre": (_fd("fdid_r2_pre"), 2e-3),
    "fdid_n_controls": (_fd("fdid_n_controls"), 0.0),
    "did_att": (_fd("did_att"), 5e-4),
    "did_att_pct": (_fd("did_att_pct"), 0.1),
    "did_r2_pre": (_fd("did_r2_pre"), 2e-3),
}
