"""Cross-validation: DSC against the published ``disco`` Stata Journal numbers.

The only fully external reference mlsynth's ``DSC`` has. Gunsilius & Van Dijcke's
Stata Journal article ships a replication repository
(`Davidvandijcke/disco_stata_journal <https://github.com/Davidvandijcke/disco_stata_journal>`_,
MIT) containing an anonymised employee-tenure panel and a log with printed
results -- donor weights and a quantile-effects table.

Why this case exists alongside ``dsc_dube``
-------------------------------------------

``dsc_dube`` pins mlsynth's own output: the DiSCo vignette is built with
``eval=FALSE`` and publishes no numbers, so its only external anchor is the
qualitative ``p > 0.05``. That is one bit of information, and a materially wrong
DSC could satisfy it.

The R package cannot supply more, because its weights are not deterministic:
``DiSCo_weights_reg`` draws the quadrature points with ``runif``, so the fitted
weights are a Monte Carlo estimate with ``O(M^-1/2)`` error. Measured on the Dube
panel it disagrees with *itself* across seeds by up to 0.119, where mlsynth sits
0.044 from any one of them -- so no tolerance against a single R run means
anything.

The Stata implementation is deterministic. ``disco_prob_grid``
(``src/disco_utils.mata``) builds an evenly spaced closed grid and the seed is
consumed only by the bootstrap. So its published weights are a real target, and
this case pins them.

What it took to match
---------------------

Reproducing the printed weights required three alignments, in descending order
of effect:

* the quadrature grid must be ``linspace(0, 1, M)`` with endpoints *included*,
  evaluating the sample minimum and maximum -- excluding them costs 0.0869;
* the simplex weights must come from an exact QP, not projected gradient
  (0.0047) -- FISTA matches the objective to 0.00 percent while its argmin
  differs, and the argmin is the reported quantity;
* the empirical quantile must be type 7, not type 1 (0.0026).

With all three, every published weight matches to the printed precision. See
issue #304 for the investigation, including the hypotheses that were wrong.

Scope
-----

The weights are deterministic and pinned exactly. The log's standard errors and
confidence intervals come from a 300-replication bootstrap under ``seed(12143)``
and are *not* pinned: reproducing them would require reproducing Stata's RNG
stream, which is not a property of the estimator.

Provenance: ``basedata/disco_tenure.parquet``, converted from
``examples/tenure_anonymized.dta`` in that repository with every column verified
identical on round-trip. 32 firms x 3 periods, 1,027,406 individual observations.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_DATA = Path(__file__).resolve().parents[2] / "basedata" / "disco_tenure.parquet"

#: Verbatim from ``results/tenure_example.log``. Stata prints four decimals.
PUBLISHED_WEIGHTS = {
    "amazon": 0.2203,
    "autodesk": 0.1271,
    "cisco": 0.1066,
    "dell technologies": 0.0991,
    "slalom consulting": 0.0962,
}

TARGET_ID = 2      # idtarget(2)
T0_PERIOD = 3      # t0(3): treatment in period 3, pre-periods {1, 2}
M_POINTS = 100     # m(100)


def run() -> dict:
    from mlsynth import DSC

    if not _DATA.exists():                      # pragma: no cover - vendored
        return {k: float("nan") for k in EXPECTED}
    df = pd.read_parquet(_DATA)
    d = df.copy()
    d["treat"] = ((d.id_col == TARGET_ID) & (d.time_col >= T0_PERIOD)).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = DSC({
            "df": d, "outcome": "y_col", "treat": "treat", "unitid": "id_col",
            "time": "time_col", "M": M_POINTS, "compute_inference": False,
            "display_graphs": False,
        }).fit()

    names = df.groupby("id_col").company_name.first()
    w = {names[int(k)]: float(v) for k, v in res.donor_weights.items()}
    diffs = [abs(w[n] - v) for n, v in PUBLISHED_WEIGHTS.items() if n in w]
    ranked = sorted(w, key=w.get, reverse=True)[:5]

    out = {
        "n_donors": float(len(w)),
        "weights_sum_to_one": float(abs(sum(w.values()) - 1.0) < 1e-9),
        # The headline: distance from the printed table, so the row reads as
        # "how far apart are we" and cannot be quietly re-fitted.
        "published_weight_max_diff": float(max(diffs)) if diffs else float("nan"),
        # Ordering as well as values -- a permutation would satisfy the
        # per-donor distances while reporting a different donor ranking.
        "top5_order_matches": float(ranked == list(PUBLISHED_WEIGHTS)),
    }
    for name in PUBLISHED_WEIGHTS:
        key = "w_" + name.split()[0]
        out[key] = float(w.get(name, float("nan")))
    return out


# Deterministic: the Stata grid is evenly spaced and the weight solve is an
# exact QP, so repeat runs are bit-identical and the reference has no RNG in the
# quantity being compared.
EXPECTED = {
    "n_donors": (31.0, 0.0),
    "weights_sum_to_one": (1.0, 0.0),
    # 5e-5 is the printed precision of the reference: Stata reports four
    # decimals, so this is as tight as the published table can be checked, and
    # tightening it further would be pinning rounding noise.
    "published_weight_max_diff": (0.0, 5e-5),
    "top5_order_matches": (1.0, 0.0),
    "w_amazon": (0.2203, 5e-5),
    "w_autodesk": (0.1271, 5e-5),
    "w_cisco": (0.1066, 5e-5),
    "w_dell": (0.0991, 5e-5),
    "w_slalom": (0.0962, 5e-5),
}
