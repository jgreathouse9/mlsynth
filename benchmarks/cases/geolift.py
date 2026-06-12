"""GEOLIFT cross-validation vs GeoLift/augsynth: the GeoLift_Walkthrough.

Cross-validation (scenario: match an authoritative reference implementation).
Meta's GeoLift package walkthrough runs Ben-Michael, Feller & Rothstein's
Augmented SCM ([BMFR2021]) through augsynth with ``fixed_effects=TRUE`` (the
package default) and Chernozhukov-Wuthrich-Zhu conformal inference, and reports a
realized effect for the ``chicago`` + ``portland`` test markets over the last 15
of 105 days (``GeoLift_Test``: 40 markets, the other 38 as donors):

* Average ATT (per treated unit, per period): ``155.556``
* Percent Lift: ``5.4%``
* Incremental Y (summed over both units, 15 periods): ``4667``
* Conformal p-value: ``0.01``

This case reproduces those numbers through mlsynth's fixed-effect ridge ASCM and
all-period conformal refit -- the same fit/inference ``realize_design`` runs, here
without the per-period interval grid so the case is fast. The match requires all
four ingredients (unit fixed effects, mean-of-units fit target, the all-period
conformal refit, augsynth's period-space ridge ASCM); see
``docs/replications/geolift.rst``.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

from mlsynth.utils.datautils import geoex_dataprep
from mlsynth.utils.bilevel.ridge_inference import conformal_pvalue
from mlsynth.utils.geolift_helpers.marketselect.helpers.fit import fit_augsynth_once
from mlsynth.utils.geolift_helpers.marketselect.helpers.shaping import (
    aggregate_treated, donor_matrix,
)

_DATA = os.path.join(os.path.dirname(__file__), "..", "..",
                     "basedata", "geolift_test_data.csv")
_TREATED = frozenset({"chicago", "portland"})
_PRE = 90
_NS = 2000


def run() -> dict:
    df = pd.read_csv(os.path.abspath(_DATA))
    Ywide = geoex_dataprep(df, "location", "date", "Y")["Ywide"]
    n_units = len(_TREATED)

    # augsynth fixedeff fits the *mean* of the treated units (colMeans).
    treated = aggregate_treated(Ywide, _TREATED, how="mean").to_numpy()
    donors = donor_matrix(Ywide, _TREATED)
    Y0 = donors.to_numpy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_augsynth_once(
            treated[:_PRE], Y0[:_PRE], augment="ridge", fixed_effects=True,
            donor_names=[str(c) for c in donors.columns],
        )
        gap = treated - fit.predict(Y0)                       # per-unit gap
        p = conformal_pvalue(treated, Y0, _PRE, lambda_=fit.lambda_,
                             ns=_NS, seed=0, fixed_effects=True)

    att_per_unit = float(np.mean(gap[_PRE:]))
    cf_post = float(np.mean(fit.predict(Y0)[_PRE:]))
    return {
        "att_per_unit": att_per_unit,                         # GeoLift 155.556
        "pct_lift": 100.0 * att_per_unit / cf_post,           # GeoLift 5.4
        "incremental": float(np.sum(gap[_PRE:])) * n_units,   # GeoLift 4667
        "conformal_p": float(p),                              # GeoLift 0.01
        "significant": float(p < 0.05),
    }


# Deterministic (fixed CV lambda, fixed seed/ns). The augsynth/GeoLift target is
# in the comment; tolerances accept the small numerical gap (mlsynth hits
# 156.8 / 5.47% / 4704 / 0.011) while pinning the value-for-value match.
EXPECTED = {
    "att_per_unit": (155.556, 5.0),     # GeoLift 155.556
    "pct_lift": (5.4, 0.5),             # GeoLift 5.4%
    "incremental": (4667.0, 150.0),     # GeoLift 4667
    "conformal_p": (0.01, 0.02),        # GeoLift 0.01 (deterministic at seed=0)
    "significant": (1.0, 0.5),
}
