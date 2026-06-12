import numpy as np
import pandas as pd

from mlsynth.utils.geolift_helpers.marketselect.realize import realize_design
from mlsynth.config_models import BaseEstimatorResults


def _wide(T=22, J=4, seed=2):
    rng = np.random.default_rng(seed)
    base = np.arange(T) * 0.3
    cols = {u: base + rng.normal(scale=0.5, size=T) + i for i, u in enumerate("ABCDE")}
    Ywide = pd.DataFrame(cols, index=pd.Index(range(T), name="time"))
    Ywide.columns.name = "unit"
    return Ywide


def test_realize_design_null_when_no_effect():
    """No injected effect -> ATT near zero and a non-significant joint p-value."""
    Ywide = _wide()                                   # no treatment effect anywhere
    rep = realize_design(Ywide, frozenset({"A", "B"}), pre_periods=18,
                         how="mean", augment="ridge", ns=50, seed=0)
    assert isinstance(rep, BaseEstimatorResults)
    assert abs(rep.effects.att) < 5.0                 # small relative to the series scale
    assert rep.inference.p_value > 0.10               # null not rejected
    # time series cover the full panel; intervention at the split
    assert rep.time_series.observed_outcome.shape == (22,)
    assert rep.time_series.counterfactual_outcome.shape == (22,)
    assert rep.time_series.intervention_time == 18


def test_realize_design_conformal_details_present():
    Ywide = _wide()
    rep = realize_design(Ywide, frozenset({"A"}), pre_periods=18,
                         how="mean", augment="ridge", ns=50, seed=0)
    d = rep.inference.details
    assert rep.inference.method == "conformal"
    # one interval per post period (22 - 18 = 4)
    assert len(d["periods"]) == 4
    assert d["lower"].shape == (4,) and d["upper"].shape == (4,)
    # the per-period effect lies within its own [lower, upper]
    finite = np.isfinite(d["lower"]) & np.isfinite(d["upper"])
    assert np.all(d["att"][finite] >= d["lower"][finite] - 1e-6)
    assert np.all(d["att"][finite] <= d["upper"][finite] + 1e-6)
