import numpy as np
import pandas as pd

from mlsynth.utils.geolift_helpers.config import GeoLiftConfig
from mlsynth.utils.geolift_helpers.marketselect.orchestration import (
    run_design,
    GEOLIFTResults,
    MarketSelectSearch,
)
from mlsynth.config_models import WeightsResults, TimeSeriesResults


def _long_panel(n_units=6, T=25, seed=7):
    rng = np.random.default_rng(seed)
    base = np.arange(T) * 0.3
    rows = []
    for i in range(n_units):
        series = base + rng.normal(scale=1.0, size=T) + i
        for t in range(T):
            rows.append({"unit": f"u{i}", "time": t, "Y": float(series[t])})
    return pd.DataFrame(rows)


def test_run_design_smoke_end_to_end():
    cfg = GeoLiftConfig(
        df=_long_panel(), outcome="Y", unitid="unit", time="time",
        treatment_size=2, durations=[4], effect_sizes=[0.0, 0.5],
        lookback_window=2, ns=25, seed=0,
    )
    res = run_design(cfg)
    assert isinstance(res, GEOLIFTResults)
    assert isinstance(res.search, MarketSelectSearch)
    assert len(res.search.candidates) >= 1
    for cd in res.search.candidates:
        assert isinstance(cd.weights, WeightsResults)
        assert isinstance(cd.time_series, TimeSeriesResults)
        assert isinstance(cd.intercept, float)
    # shortlist is a DataFrame; metadata records the candidate count
    assert isinstance(res.power, pd.DataFrame)
    assert res.metadata["n_candidates"] == len(res.search.candidates)


def test_run_design_winner_surfaces_design_weights():
    cfg = GeoLiftConfig(
        df=_long_panel(), outcome="Y", unitid="unit", time="time",
        treatment_size=2, durations=[4], effect_sizes=[0.0, 0.5],
        lookback_window=2, ns=25, seed=0,
    )
    res = run_design(cfg)
    if res.search.winner is not None:               # a candidate cleared 0.8 power
        assert res.design_weights is not None
        assert res.selected_units is not None and len(res.selected_units) == 2
        assert res.search.winner.rank is not None


def test_geolift_config_rejects_bad_treatment_size():
    import pytest
    from mlsynth.exceptions import MlsynthConfigError
    with pytest.raises(MlsynthConfigError, match="treatment_size"):
        GeoLiftConfig(df=_long_panel(), outcome="Y", unitid="unit", time="time",
                      treatment_size=0, durations=[4], effect_sizes=[0.0])
