import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest

from mlsynth import GEOLIFT
from mlsynth.utils.geolift_helpers.marketselect.orchestration import GEOLIFTResults
from mlsynth.config_models import BaseEstimatorResults
from mlsynth.exceptions import MlsynthConfigError


def _panel(n_units=6, T=26, seed=11):
    rng = np.random.default_rng(seed)
    base = np.arange(T) * 0.3
    post = {t: int(t >= T - 6) for t in range(T)}      # last 6 periods are post
    rows = []
    for i in range(n_units):
        series = base + rng.normal(scale=0.5, size=T) + i
        for t in range(T):
            rows.append({"unit": f"u{i}", "time": t, "Y": float(series[t]), "post": post[t]})
    return pd.DataFrame(rows)


def _cfg(**over):
    base = dict(df=_panel(), outcome="Y", unitid="unit", time="time",
                treatment_size=2, durations=[4], effect_sizes=[0.0, 0.5],
                lookback_window=2, ns=25, seed=0)
    base.update(over)
    return base


def test_geolift_fit_returns_design_result():
    res = GEOLIFT(_cfg()).fit()
    assert isinstance(res, GEOLIFTResults)
    assert res.search is not None and len(res.search.candidates) >= 1


def test_geolift_realize_and_plot_post_phase():
    est = GEOLIFT(_cfg(post_col="post"))
    est.fit()
    if est._result.search.winner is None:
        pytest.skip("no detectable design in this tiny synthetic")
    report = est.realize()
    assert isinstance(report, BaseEstimatorResults)
    assert report.inference.method == "conformal"
    fig = est.plot()                 # post phase (report populated)
    assert len(fig.axes) == 2


def test_geolift_plot_design_phase():
    est = GEOLIFT(_cfg())
    est.fit()
    if est._result.search.winner is None:
        pytest.skip("no detectable design")
    fig = est.plot()                 # design phase (no report)
    assert len(fig.axes) == 1


def test_geolift_bad_config_raises():
    with pytest.raises(MlsynthConfigError):
        GEOLIFT("not a config")
