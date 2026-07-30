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
* Data: ``basedata/dube_minwage.parquet`` -- the ``DiSCo`` package's ``dube``
  dataset (Dube 2019; ``adj0contpov`` by state-year), converted from the
  package's ``data/dube.rda`` with every column verified bit-identical on
  round-trip. 652,870 rows, 34 states (33 donors) x 7 years (1998-2004); each
  cell is a distribution. 2.0 MB as zstd parquet.

  This is the authors' complete analysis dataset, not a sample of it.
  ``data-raw/dube.R`` in ``DiSCos`` builds it from the Dube replication package
  by filtering to under-65s, 1998-2004, and Alaska plus the 33 states with no
  minimum-wage change over that window.

  An earlier revision of this case used a 250-observations-per-cell subsample,
  retaining 9.1 percent of the rows, to keep the file small. For most estimators
  that would be an ordinary size-fidelity trade; for a *distributional* method it
  is not. The estimand here is the within-cell distribution, and true cell sizes
  run from 1,118 to 9,516 -- an eight-fold spread flattened to a constant 250.
  That does not merely add sampling noise, it distorts the object being matched
  and equalises precision across donors that genuinely differ in it. Restoring
  the full data cut the pre-period 2-Wasserstein fit from 0.129 to 0.038.

* Cross-validation status: pinned values below are still mlsynth's own output,
  so five of the six rows are regression pins rather than external checks. The
  one externally anchored quantity is the vignette's stated ``p > 0.05`` ("no
  spurious effect"), which is a single bit -- a materially wrong DSC that still
  failed to reject would satisfy it.

  ``DiSCos`` 0.1.4 *is* now installable here (``benchmarks/R/install_discos.sh``),
  and running it on this same file shows donor weights disagreeing with mlsynth
  by up to 0.074, with the two implementations reaching pre-period objective
  values 4 percent apart. That is tracked in issue #304 and is why these rows are
  not yet cross-validated: pinning agreement before the disagreement is
  understood would pin the wrong thing.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_DATA = Path(__file__).resolve().parents[2] / "basedata" / "dube_minwage.parquet"


def run() -> dict:
    from mlsynth import DSC

    df = pd.read_parquet(_DATA)
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


# Deterministic: repeat runs reproduce every value to <1e-12. DSC fits the
# treated distribution from a simplex over donor quantile functions; on the Dube
# panel it tracks closely pre-period and the placebo permutation test fails to
# reject at both post years -- the vignette's "no spurious effect".
#
# These values MOVED when the full data replaced the 250-per-cell subsample.
# That is the point of the change rather than a regression to explain away: the
# pre-period fit improved 3.4-fold, which is the direct evidence that thinning
# the cells was distorting the distributions being matched. The one externally
# anchored row, dsc_no_spurious_effect, is unchanged -- the vignette's claim
# survives the data change, which is the reassurance worth having.
EXPECTED = {
    # -0.2618 on the full data, against -0.1515 on the subsample.
    "dsc_att": (-0.2618, 0.02),
    "dsc_no_spurious_effect": (1.0, 0.0),     # cross-check vs the DiSCo vignette
    # Permutation p-values over 33 donors + the treated unit are multiples of
    # 1/34 = 0.0294, so the tolerance is ~3 permutation steps: tight enough that
    # a real shift in the placebo ranking fails, loose enough to survive one or
    # two donors swapping order.
    "dsc_pvalue_2003": (0.5000, 0.09),        # 17/34
    "dsc_pvalue_2004": (0.1176, 0.09),        # 4/34
    # 0.0384 on the full data against 0.129 on the subsample. Do NOT loosen this
    # back toward the old value: a rise here means the distributional fit has
    # degraded, which is the failure this case exists to catch.
    "dsc_pre_wasserstein": (0.0384, 0.004),
    "n_donors": (33.0, 0.0),
}
