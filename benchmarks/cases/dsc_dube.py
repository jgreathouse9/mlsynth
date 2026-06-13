"""Path-A benchmark: DSC on the Dube (2019) minimum-wage panel (Gunsilius 2023).

DSC is mlsynth's port of Distributional Synthetic Controls (Gunsilius 2023); the
authors' reference is the `DiSCo` R package
(`Davidvandijcke/DiSCos <https://github.com/Davidvandijcke/DiSCos>`_), whose
vignette analyses the Dube (2019) minimum-wage data. This case reproduces that
application: it fits DSC on the **micro-level** county income panel with Alaska
(``fips = 2``) as the treated unit and a 2003 intervention, exactly the vignette's
setup (``id_col.target = 2``, ``t0 = 2003``), and checks DSC's distributional fit
and placebo inference.

The single quantitative claim the vignette states in text is the headline
cross-check: the placebo-permutation **p-value exceeds 0.05** ("no spurious
effect"). DSC reproduces it, alongside a small pre-period Wasserstein fit and the
expected donor pool.

Provenance / scope
------------------
* Data: ``basedata/dube_minwage.csv`` -- the ``DiSCo`` package's ``dube`` dataset
  (Dube 2019; ``adj0contpov`` by state-year), exported from ``dube.rda`` and
  **subsampled to 250 observations per state-year cell** (fixed seed) so the
  micro-panel is ~1 MB rather than 15 MB. 34 states (33 donors) x 7 years
  (1998-2004); each cell is a distribution.
* No live ``DiSCo`` cross-validation: the R package does not install on this
  environment's R version, and the vignette's weight/QTE numbers live in figures,
  not text. So this is Path A on the authors' dataset/setup with mlsynth's own
  output pinned, anchored to the vignette's stated ``p > 0.05`` result. The
  one-time machine-precision DiSCo check is documented on the DSC docs page.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_DATA = Path(__file__).resolve().parents[2] / "basedata" / "dube_minwage.csv"


def run() -> dict:
    from mlsynth import DSC

    df = pd.read_csv(_DATA)
    df["treat"] = ((df.id_col == 2) & (df.time_col >= 2003)).astype(int)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = DSC({
            "df": df, "outcome": "y_col", "treat": "treat", "unitid": "id_col",
            "time": "time_col", "compute_inference": True, "display_graphs": False,
        }).fit()

    p_values = dict(res.inference.p_values)            # {post_year: placebo p-value}
    pre_w = float(np.mean(np.asarray(res.pre_period_wasserstein, dtype=float)))
    return {
        "dsc_att": float(res.att),
        # the vignette's stated headline: placebo permutation fails to reject
        "dsc_no_spurious_effect": float(min(p_values.values()) > 0.05),
        "dsc_pvalue_2003": float(p_values[2003]),
        "dsc_pvalue_2004": float(p_values[2004]),
        "dsc_pre_wasserstein": pre_w,
        "n_donors": float(len(res.donor_weights)),
    }


# Deterministic on the fixed-seed subsample. DSC fits the treated distribution
# from a simplex over donor quantile functions; on the Dube panel it tracks
# closely pre-period (small Wasserstein) and the placebo permutation test fails
# to reject at both post years (p > 0.05) -- the vignette's "no spurious effect".
EXPECTED = {
    "dsc_att": (-0.1515, 0.05),
    "dsc_no_spurious_effect": (1.0, 0.0),     # cross-check vs the DiSCo vignette
    "dsc_pvalue_2003": (0.91, 0.25),
    "dsc_pvalue_2004": (0.32, 0.25),
    "dsc_pre_wasserstein": (0.129, 0.06),
    "n_donors": (33.0, 0.0),
}
