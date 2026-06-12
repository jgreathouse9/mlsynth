"""Cross-validation of GEOLIFT's realized effect report against GeoLift/augsynth.

Reproduces Meta's GeoLift package walkthrough (``GeoLift_Walkthrough``): the
``GeoLift_Test`` panel (40 markets x 105 days), ``chicago`` + ``portland``
treated over the last 15 periods, ``fixed_effects=TRUE``. GeoLift (which uses
augsynth's ridge ASCM + conformal inference under the hood) reports:

* Average ATT (per treated unit, per period): ``155.556``
* Percent Lift: ``5.4%``
* Incremental Y (summed over both units, 15 periods): ``4667`` (=> ~311/period)
* Conformal p-value: ``0.01``

The match requires augsynth's ``fixed_effects`` (per-unit demeaning) **and** the
mean-of-units treated aggregate -- the two ingredients realized in
``realize_design(fixed_effects=True)``. This test pins that parity so a
regression in the ridge ASCM, the fixed-effect demeaning, or the conformal refit
is caught.
"""

import os

import numpy as np
import pandas as pd
import pytest

from mlsynth.utils.datautils import geoex_dataprep
from mlsynth.utils.geolift_helpers.marketselect.realize import realize_design

_DATA = os.path.join(
    os.path.dirname(__file__), "..", "..", "basedata", "geolift_test_data.csv"
)
_TREATED = frozenset({"chicago", "portland"})
_PRE = 90


@pytest.fixture(scope="module")
def ywide():
    df = pd.read_csv(os.path.abspath(_DATA))
    return geoex_dataprep(df, "location", "date", "Y")["Ywide"]


def _report(ywide, how, fixed_effects=True, ns=2000):
    return realize_design(
        ywide, _TREATED, _PRE, how=how, augment="ridge",
        alpha=0.1, ns=ns, seed=0, conformal_type="iid",
        fixed_effects=fixed_effects,
    )


def test_public_geolift_estimator_reproduces_walkthrough():
    """The public ``GEOLIFT(...).fit()`` API reaches the walkthrough numbers.

    Mirrors GeoLift's ``GeoLift(locations=c("chicago","portland"),
    treatment_start_time=91, ...)`` -> ``summary()``: the two markets are pinned
    via ``to_be_treated`` + ``treatment_size`` and the post window via
    ``post_col``; ``res.report`` is the analogue of ``summary(GeoLift_Test)``.
    """
    from mlsynth import GEOLIFT

    df = pd.read_csv(os.path.abspath(_DATA))
    dates = sorted(df["date"].unique())
    df["post"] = df["date"].isin(set(dates[_PRE:])).astype(int)
    res = GEOLIFT({
        "df": df, "outcome": "Y", "unitid": "location", "time": "date",
        "treatment_size": 2, "to_be_treated": ["chicago", "portland"],
        "durations": [len(dates) - _PRE], "effect_sizes": [0.0, 0.10],
        "lookback_window": 1, "post_col": "post", "how": "mean",
        "fixed_effects": True, "alpha": 0.1, "ns": 2000, "seed": 0,
        "display_graphs": False,
    }).fit()
    assert set(res.selected_units) == {"chicago", "portland"}
    rep = res.report
    cf_post = rep.time_series.counterfactual_outcome[_PRE:].mean()
    assert rep.effects.att == pytest.approx(155.6, abs=4.0)        # per-unit ATT
    assert 100.0 * rep.effects.att / cf_post == pytest.approx(5.4, abs=0.4)
    assert rep.inference.p_value < 0.05                            # GeoLift 0.01
    assert rep.inference.method == "conformal"


def test_walkthrough_per_unit_att_and_pvalue(ywide):
    """how='mean' reproduces GeoLift's per-unit ATT (155.6), lift (5.4%), p (0.01)."""
    rep = _report(ywide, how="mean")
    cf_post = rep.time_series.counterfactual_outcome[_PRE:].mean()
    lift = 100.0 * rep.effects.att / cf_post
    assert rep.effects.att == pytest.approx(155.6, abs=4.0)     # per-unit ATT
    assert lift == pytest.approx(5.4, abs=0.4)                  # percent lift
    assert rep.inference.p_value < 0.05                         # significant (GeoLift 0.01)


def test_walkthrough_summed_incremental(ywide):
    """how='sum' reports the summed effect (~311/period); p is scale-invariant."""
    rep = _report(ywide, how="sum")
    incremental = float(np.sum(rep.time_series.estimated_gap[_PRE:]))
    assert rep.effects.att == pytest.approx(311.0, abs=8.0)     # summed per-period
    assert incremental == pytest.approx(4667.0, rel=0.05)       # total incremental
    assert rep.inference.p_value < 0.05


def test_walkthrough_pvalue_invariant_to_reporting_scale(ywide):
    """The conformal p-value is identical for sum/mean (a global reporting scale)."""
    assert _report(ywide, how="mean").inference.p_value == pytest.approx(
        _report(ywide, how="sum").inference.p_value, abs=1e-12
    )


def test_original_default_without_fixed_effects_is_the_bug(ywide):
    """Guard the fix by reproducing the pre-fix behaviour.

    The historical GEOLIFT default (summed treated aggregate, level-matching SCM
    with no per-unit demeaning) lets the donor pool absorb the post-period level
    shift in the conformal refit: the ATT is far from GeoLift's 311/period and
    the effect is *not* significant. The fixed-effect path corrects both.
    """
    rep = _report(ywide, how="sum", fixed_effects=False)
    assert rep.effects.att == pytest.approx(209.0, abs=10.0)   # wrong (vs 311)
    assert rep.inference.p_value > 0.10                        # absorbed -> not significant
